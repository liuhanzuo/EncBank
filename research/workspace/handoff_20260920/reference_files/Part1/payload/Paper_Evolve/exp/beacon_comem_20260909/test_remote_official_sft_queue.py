"""CPU-only queue completion, input identity and lock ownership regression tests."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import remote_official_sft_queue as queue
from remote_official_sft_queue import (available_gpus, completed_job, qasper_complete,
    trainer_command, stage_expectations, launch_trainer, digest, ARMS)
from train_official_sft import evaluation_signature, object_digest


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def qasper_fixture(root):
    source = [{'id': str(i), 'document_id': 'doc'+str(i//3), 'source': 'qasper',
               'split': 'dev', 'question': 'Question '+str(i), 'answer': 'answer',
               'answers': ['answer', 'alias']} for i in range(100)]
    dev = root/'dev.jsonl'
    dev.write_text(''.join(json.dumps(row)+'\n' for row in source), encoding='utf-8')
    source.sort(key=lambda row: hashlib.sha256(f"42:{row['id']}".encode()).hexdigest())
    records = [{**{key: row[key] for key in ('id', 'document_id', 'source', 'split', 'question')},
        'references': ['answer', 'alias'], 'prediction': 'answer', 'generated_ids': [7, 2],
        'generated_tokens': 2, 'finish_reason': 'eos', 'exact_match': 1., 'token_f1': 1.,
        'selected_chunk_indices': [0], 'selected_context_tokens': 20, 'original_context_tokens': 30,
        'prompt_tokens': 10, 'answer_truncated': False, 'answer_ce_tokens': 2,
        'answer_ce_sum': 1., 'answer_ce': .5} for row in source]
    result = {'records': records, 'summary': {'examples': 100, 'exact_match': 1., 'token_f1': 1.,
        'answer_ce_tokens': 200, 'answer_ce': .5, 'score_scale': '0-to-1',
        'protocol': {'decoding': 'greedy', 'max_new_tokens': 128, 'eos_token_ids': [2],
            'enable_thinking': False, 'answer_ce': True, 'ce_reference_index': 0,
            'ce_targets': 'prepared-answer-ids-including-eos-only-when-not-truncated',
            'hardware_timing': 'not-collected'}}}
    write(root/'queue.json', {'complete': True})
    for arm in ARMS:
        (root/arm).mkdir()
        write(root/arm/'status.json', {'complete': True, 'step': 500, 'target_steps': 500})
        write(root/arm/'metadata.json', {'recipe': {'steps': 500, 'seed': 42,
            'dev_sha256': digest(dev), 'final_eval_limit': 100, 'dev_examples': 100, 'max_new_tokens': 128}})
        write(root/arm/'eval_step500.json', result)
    return dev, result


def official_fixture(root):
    recipe = {'steps': 2, 'grad_accum': 1, 'arm': 'beacon4', 'final_budget': {'target_tokens': 8},
        'train_prepared_sha256': 'current-train', 'dev_prepared_sha256': 'current-dev',
        'selection_sha256': 'current-selection', 'selected_train_ids': ['b', 'a'],
        'selected_dev_ids': ['d'], 'eval_ce_ids': ['d'], 'eval_generation_ids': ['d'],
        'qa_max_new_tokens': 512, 'summary_max_new_tokens': 2048}
    expected = {'recipe': recipe, 'file_identities': {}}
    status = {'complete': True, 'step': 2, 'target_steps': 2, 'cursor': 2,
        'budget': recipe['final_budget'], 'gradient_check': {'reader_norm': 1., 'beacon_norm': 2.,
            'frozen_base_has_grad': False}, 'model_state_id': 'weights-a'}
    result = {'complete': True, 'signature': evaluation_signature(2, 'weights-a', object_digest(recipe),
        ['d'], ['d'], 512, 2048), 'ce_records': [{'conversation_id': 'd', 'target_tokens': 4, 'ce_sum': 2.}],
        'generation_records': [{'conversation_id': 'd', 'generation_completed': True,
            'generated_ids': [7, 2], 'generated_tokens': 2, 'assistant_turn_index': 0,
            'references': ['answer'], 'finish_reason': 'eos', 'exact_match': 1., 'token_f1': 1.}]}
    write(root/'metadata.json', {'recipe': recipe}); write(root/'status.json', status)
    write(root/'eval_step2.json', result)
    return expected, status, result


class QueueChecks(unittest.TestCase):
    def test_graphics_python_is_busy_and_512_mib_is_not_free(self):
        def gpu(used, process='', kind='NVIDIA RTX 3090'):
            return f'<gpu><product_name>{kind}</product_name><fb_memory_usage><used>{used} MiB</used></fb_memory_usage><processes>{process}</processes></gpu>'
        process = '<process_info><type>G</type><process_name>python</process_name></process_info>'
        xml = '<nvidia_smi_log>'+gpu(10, process)+gpu(511)+gpu(512)+gpu(10, kind='RTX 5090')+'</nvidia_smi_log>'
        self.assertEqual(available_gpus(xml), [1])

    def test_qasper_accepts_old_generated_schema_and_rejects_missing_work(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); dev, result = qasper_fixture(root)
            self.assertTrue(qasper_complete(root, dev))
            for field in ('answer_ce', 'generated_ids', 'prediction'):
                with self.subTest(field=field):
                    bad = deepcopy(result); bad['records'][0].pop(field)
                    write(root/ARMS[0]/'eval_step500.json', bad)
                    self.assertFalse(qasper_complete(root, dev))
            write(root/ARMS[0]/'eval_step500.json', {'summary': {'examples': 100},
                'records': [{'id': str(i)} for i in range(100)]})
            self.assertFalse(qasper_complete(root, dev))

    def test_qasper_binds_actual_fixed_ids_references_and_generation_end(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); dev, result = qasper_fixture(root)
            for mutation in ('foreign_id', 'changed_reference', 'invalid_eos', 'fake_ce', 'changed_input'):
                with self.subTest(mutation=mutation):
                    bad = deepcopy(result); row = bad['records'][0]
                    if mutation == 'foreign_id': row['id'] = 'unrelated'
                    if mutation == 'changed_reference': row['references'] = ['leaked answer']
                    if mutation == 'invalid_eos': row['generated_ids'][-1] = 8
                    if mutation == 'fake_ce': row['answer_ce_sum'] = float('nan')
                    if mutation == 'changed_input': row['selected_chunk_indices'] = [1]
                    write(root/ARMS[0]/'eval_step500.json', bad)
                    self.assertFalse(qasper_complete(root, dev))
            write(root/ARMS[0]/'eval_step500.json', result)
            dev.write_text(dev.read_text()+'\n', encoding='utf-8')
            self.assertFalse(qasper_complete(root, dev))

    def test_official_completion_requires_real_evaluation_current_weights_and_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); expected, status, result = official_fixture(root)
            self.assertTrue(completed_job(root, 2, expected=expected))
            for mutation in ('weights', 'missing_ce', 'no_tokens', 'missing_score', 'foreign_eval_id'):
                with self.subTest(mutation=mutation):
                    bad = deepcopy(result)
                    if mutation == 'weights': bad['signature']['model_state_id'] = 'weights-b'
                    if mutation == 'missing_ce': bad['ce_records'][0].pop('ce_sum')
                    if mutation == 'no_tokens': bad['generation_records'][0]['generated_ids'] = []
                    if mutation == 'missing_score': bad['generation_records'][0].pop('token_f1')
                    if mutation == 'foreign_eval_id': bad['signature']['ce_conversation_ids'] = ['other']
                    write(root/'eval_step2.json', bad)
                    self.assertFalse(completed_job(root, 2, expected=expected))
            write(root/'eval_step2.json', result)
            for field, value in [('cursor', 1), ('budget', {'target_tokens': 9}),
                                 ('gradient_check', {'reader_norm': 1, 'beacon_norm': 0, 'frozen_base_has_grad': False})]:
                bad = {**status, field: value}; write(root/'status.json', bad)
                self.assertFalse(completed_job(root, 2, expected=expected))

    def test_same_steps_cannot_reuse_other_data_order_or_recipe(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); expected, _, _ = official_fixture(root)
            for key, value in [('train_prepared_sha256', 'new'), ('dev_prepared_sha256', 'new'),
                               ('selected_train_ids', ['a', 'b']), ('selection_sha256', 'new'),
                               ('qa_max_new_tokens', 128)]:
                changed = deepcopy(expected); changed['recipe'][key] = value
                self.assertFalse(completed_job(root, 2, expected=changed))
            data = root/'data.jsonl'; data.write_text('original')
            expected['file_identities'] = {str(data): queue.file_identity(data)}
            self.assertTrue(completed_job(root, 2, expected=expected))
            data.write_text('modified and longer')
            self.assertFalse(completed_job(root, 2, expected=expected))

    def test_probe_and_training_pass_flat_selection_and_resume_only_existing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            first = trainer_command(root, root, 'beacon4', 'probe', True, steps=9, grad_accum=1)
            self.assertIn(str(root/'probe_train.jsonl'), first)
            self.assertEqual(first[first.index('--selection-json')+1], str(root/'probe_selection.json'))
            self.assertEqual(first[first.index('--sample-order')+1], 'selection_order')
            self.assertIn('--skip-initial-eval', first); self.assertNotIn('--resume', first)
            (root/'last.pt').write_bytes(b'checkpoint')
            second = trainer_command(root, root, 'beacon4', 'train', False, steps=250, grad_accum=4)
            self.assertEqual(second[second.index('--selection-json')+1], str(root/'main_selection.json'))
            self.assertIn('--resume', second); self.assertNotIn('--stop-after', second)

    def test_current_expected_recipe_matches_actual_cpu_trainer_and_probe_order(self):
        from prepare_official_pool import tokenizer_receipt
        from test_train_official_sft import row, write_rows
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); model = root/'models/Qwen3-8B'; model.mkdir(parents=True)
            write(model/'tokenizer.json', {}); write(model/'tokenizer_config.json', {})
            adapter = root/'outputs/8b_j12_pub_4k/final/adapter.pt'; adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b'CPU validation only, never loaded')
            fingerprint = tokenizer_receipt(model)['tokenizer_fingerprint']
            train = [row('z-max-first', 'zdoc'), row('a-smaller-second', 'adoc')]
            dev = [row('d', 'devdoc', 'dev')]
            for item in train+dev: item['tokenizer_fingerprint'] = fingerprint
            write_rows(root/'probe_train.jsonl', train); write_rows(root/'probe_dev.jsonl', dev)
            flat = {'train_ids': [r['id'] for r in train], 'dev_ids': ['d']}
            write(root/'probe_selection.json', flat)
            receipt = {'selection': {'probe_'+key: list(value) for key, value in flat.items()}, 'files': {}}
            for split, count in [('train', 2), ('dev', 1)]:
                path = root/f'probe_{split}.jsonl'
                receipt['files']['probe_'+split] = {'path': path.name, 'sha256': digest(path), 'conversations': count}
            write(root/'selection.json', receipt)
            for chunk_tokens in (0, 512):
                with patch.object(queue, 'ROOT', root):
                    expected = stage_expectations(root, 'probe', steps=2, grad_accum=1,
                        receipt=receipt, mlp_chunk_tokens=chunk_tokens)['beacon4']
                    command = trainer_command(root, root/'validation', 'beacon4', 'probe', False,
                        steps=2, grad_accum=1, mlp_chunk_tokens=chunk_tokens)
                command[2] = str(Path(__file__).with_name('train_official_sft.py'))
                command.append('--validate-only')
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = json.loads((root/'validation/input_validation.json').read_text())['recipe']
                self.assertEqual(expected['recipe'], actual)
                self.assertEqual(actual['selected_train_ids'], ['z-max-first', 'a-smaller-second'])
                self.assertEqual(actual['schedule_sha256'], object_digest(actual['selected_train_ids']))
                if chunk_tokens:
                    self.assertEqual(actual['mlp_chunk_tokens'], chunk_tokens)
                    self.assertEqual(actual['mlp_execution'], 'whole-gated-mlp-token-block-checkpoint-v1')
                else:
                    self.assertNotIn('mlp_chunk_tokens', actual)
            flat['train_ids'].reverse(); write(root/'probe_selection.json', flat)
            with patch.object(queue, 'ROOT', root), self.assertRaisesRegex(ValueError, 'receipt ID order'):
                stage_expectations(root, 'probe', steps=2, grad_accum=1, receipt=receipt)

    def test_only_gpu_lock_descriptor_is_passed_to_child(self):
        with tempfile.TemporaryFile() as lock, tempfile.TemporaryFile() as log:
            with patch.object(queue.subprocess, 'Popen') as popen:
                launch_trainer(['python', 'cpu-placeholder'], gpu_lock=lock, log=log, env={'CUDA_VISIBLE_DEVICES': '1'})
            self.assertEqual(popen.call_args.kwargs['pass_fds'], (lock.fileno(),))
            self.assertIs(popen.call_args.kwargs['start_new_session'], True)

    @unittest.skipIf(sys.platform == 'win32', 'flock lifetime is Linux-specific; mock descriptor test runs everywhere')
    def test_gpu_lock_survives_controller_descriptor_close(self):
        import fcntl
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root/'workspace').mkdir()
            lock = (root/'gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            with tempfile.TemporaryFile() as log, patch.object(queue, 'ROOT', root):
                process = launch_trainer([sys.executable, '-c', 'import time; time.sleep(30)'],
                    gpu_lock=lock, log=log, env=os.environ.copy())
            lock.close()
            try:
                with (root/'gpu.lock').open('a') as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                process.terminate(); process.wait(timeout=10)
            with (root/'gpu.lock').open('a') as contender:
                fcntl.flock(contender, fcntl.LOCK_EX|fcntl.LOCK_NB)


if __name__ == '__main__':
    unittest.main(verbosity=2)
