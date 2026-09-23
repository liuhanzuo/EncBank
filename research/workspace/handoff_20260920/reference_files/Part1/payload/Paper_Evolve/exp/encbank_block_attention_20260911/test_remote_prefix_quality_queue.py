"""Stdlib-only fixtures for priority, source, completeness and child ownership."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import remote_prefix_quality_queue as queue
from prefix_quality_protocol import build_prefix_quality_plan, summarize_prefix_quality
from test_prefix_quality_protocol import prepared, records_for


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class PriorityTests(unittest.TestCase):
    def test_training_priority_stops_before_backend_reads(self):
        with patch.object(queue.backend_queue, 'training_priority_clear', return_value=(False, 'training_queued')), \
             patch.object(queue, 'backend_priority_clear') as backend:
            self.assertEqual(queue.priorities_clear(), (False, 'training_queued'))
            backend.assert_not_called()

    def test_backend_pending_failed_or_dead_blocks_prefix(self):
        owner = dict(pid=17, start_ticks=99, argv=['controller'])
        for phase in ('queued', 'running', 'launching', 'failed'):
            state = dict(controller=owner, jobs={mode: dict(phase=phase) for mode in queue.MODES})
            with self.subTest(phase=phase), patch.object(queue, 'read_json', return_value=state), \
                 patch.object(queue, 'process_identity', return_value=owner):
                self.assertEqual(queue.backend_priority_clear(), (False, 'backend_quality_pending_or_failed'))
            with patch.object(queue, 'read_json', return_value=state), \
                 patch.object(queue, 'process_identity', return_value=None):
                self.assertEqual(queue.backend_priority_clear(), (False, 'backend_quality_controller_not_active'))
        with patch.object(queue, 'read_json', return_value={}):
            self.assertFalse(queue.backend_priority_clear()[0])

    def test_completed_backend_requires_actual_completion_source_and_cpu_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend = root / 'outputs/backend'
            receipt_path = root / 'cpu.json'
            write_json(receipt_path, {'fixture': True})
            source = {'runtime.py': 'a' * 64}
            state = dict(controller={}, config=dict(cpu_receipt=str(receipt_path),
                cpu_receipt_sha256=queue.runner.digest(receipt_path), source_sha256=source),
                jobs={mode:dict(phase='complete', out=str(backend/mode), returncode=0) for mode in queue.MODES})
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'BACKEND_OUT', backend), \
                 patch.object(queue, 'read_json', return_value=state), \
                 patch.object(queue.backend_queue, 'validate_cpu_receipt', return_value={'source_sha256':source}), \
                 patch('evaluate_backend_quality.source_hashes', return_value=source), \
                 patch.object(queue.backend_queue, 'completed_quality', return_value=True) as complete:
                self.assertTrue(queue.backend_priority_clear()[0])
                self.assertEqual(complete.call_count, 2)
                complete.return_value = False
                self.assertFalse(queue.backend_priority_clear()[0])
                complete.return_value = True
                state['jobs']['full']['returncode'] = 1
                self.assertFalse(queue.backend_priority_clear()[0])
                state['jobs']['full']['returncode'] = 0
                state['config']['source_sha256'] = {'old.py':'b'*64}
                self.assertFalse(queue.backend_priority_clear()[0])
                state['config']['source_sha256'] = source
                state['config']['cpu_receipt_sha256'] = '0'*64
                self.assertFalse(queue.backend_priority_clear()[0])


class ReceiptTests(unittest.TestCase):
    def test_cpu_receipt_requires_sources_no_cuda_correct_threads_and_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = root/'code'
            code.mkdir()
            for name in queue.VALIDATION_DEPENDENCIES + ('extra_test_dependency.py',):
                (code/name).write_text('# CPU fixture\n', encoding='utf-8')
            source = {'runtime.py':'b'*64}
            bound = {**source, **{p.name:queue.runner.digest(p) for p in code.iterdir()}}
            receipt = dict(passed=True, device='cpu', cuda_visible_devices='', cuda_initialized=False,
                cpu_threads=2, cpu_interop_threads=16, tests_run=9, errors=0, failures=0, skipped=0,
                source_sha256=bound)
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'CODE', code), \
                 patch.object(queue.runner, 'source_hashes', return_value=source), \
                 patch.object(queue, 'read_json', return_value=receipt):
                self.assertEqual(queue.validate_cpu_receipt(root/'receipt.json'), receipt)
                for field, wrong in (('passed',False),('cuda_initialized',True),('cpu_threads',8),
                                     ('cpu_interop_threads',2),('tests_run',0),('skipped',1)):
                    old = receipt[field]
                    receipt[field] = wrong
                    with self.subTest(field=field), self.assertRaises(RuntimeError):
                        queue.validate_cpu_receipt(root/'receipt.json')
                    receipt[field] = old
                receipt['source_sha256']['runtime.py'] = 'c'*64
                with self.assertRaises(RuntimeError):
                    queue.validate_cpu_receipt(root/'receipt.json')
                receipt['source_sha256']['runtime.py'] = source['runtime.py']
                (code/'extra_test_dependency.py').write_text('# changed\n', encoding='utf-8')
                with self.assertRaises(RuntimeError):
                    queue.validate_cpu_receipt(root/'receipt.json')


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root/'code'
        self.directory = self.root/'results/smoke'
        self.data = self.code/'data/qasper_pilot'
        self.data.mkdir(parents=True)
        self.rows = prepared()
        (self.data/'dev.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows), encoding='utf-8')
        (self.data/'train.jsonl').write_text('{}\n', encoding='utf-8')
        adapter = self.root/'outputs/8b_j12_pub_4k/final/adapter.pt'
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b'fixture bytes, never loaded')
        self.source = {'runtime.py':'a'*64}
        self.plan = build_prefix_quality_plan(self.rows, mode='smoke')
        self.recipe = dict(model=str(self.root/'models/Qwen3-8B'), init_adapter_sha256=queue.runner.digest(adapter),
            train_sha256=queue.runner.digest(self.data/'train.jsonl'), dev_sha256=queue.runner.digest(self.data/'dev.jsonl'),
            mode='smoke', seed=42, max_new_tokens=128, j=12, rank=32, alpha=32.,
            plan_sha256=self.plan['plan_sha256'], branches=['native_cold','native_prefix'],
            decoding='independent-free-greedy-natural-eos', teacher_forced_ce=False,
            reuse_scope='same-original-ordered-pack-within-group')
        self.recipe_sha = queue.canonical_hash(self.recipe)
        self.records = records_for(self.plan)
        for row in self.records:
            row.update(max_new_tokens=128, run_id='run', recipe_sha256=self.recipe_sha, source_sha256=self.source)
        self.summary = summarize_prefix_quality(self.plan,self.records,stop_token_ids=[99,100],max_new_tokens=128)
        self.assertEqual(self.summary['status'], 'complete')
        self.metadata = dict(recipe=self.recipe, recipe_sha256=self.recipe_sha, source_sha256=self.source,
            run_id='run', lease={'lease':{'run_id':'lease-run', 'worker_script':str(self.code/'evaluate_native_prefix_quality.py')}},
            gpu='NVIDIA GeForce RTX3090',physical_gpu='0',backbone_dtype='torch.bfloat16',
            lora_master_dtype='torch.float32',autocast_dtype='torch.bfloat16',stop_token_ids=[99,100],
            formal_inference_timing=False,formal_inference_memory=False)
        self.status = dict(status='complete',run_id='run',records_completed=8,target_records=8,
                           formal_inference_timing=False,formal_inference_memory=False)
        self.save()
        for attr,value in (('ROOT',self.root),('CODE',self.code)):
            patcher=patch.object(queue,attr,value);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(queue.runner,'source_hashes',return_value=self.source)
        patcher.start();self.addCleanup(patcher.stop)

    def save(self):
        for name,value in (('plan',self.plan),('metadata',self.metadata),('status',self.status),('summary',self.summary)):
            write_json(self.directory/(name+'.json'),value)
        (self.directory/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.records),encoding='utf-8')

    def complete(self, **kwargs):
        return queue.completed_prefix_quality(self.directory,'smoke',{'source_sha256':self.source},
            worker_returncode=kwargs.get('rc',0),expected_lease_run_id=kwargs.get('lease','lease-run'))

    def test_valid_actual_raw_records_and_plan_complete(self):
        self.assertTrue(self.complete())

    def test_zero_exit_alone_partial_or_failed_never_completes(self):
        self.assertFalse(self.complete(rc=1))
        self.assertFalse(self.complete(rc=None))
        self.assertFalse(self.complete(lease='different'))
        self.status['status']='failed';self.save()
        self.assertFalse(self.complete())
        self.status['status']='complete';self.records.pop();self.save()
        self.assertFalse(self.complete())

    def test_old_source_recipe_or_summary_rejected(self):
        self.records[0]['run_id']='old';self.save();self.assertFalse(self.complete())
        self.records[0]['run_id']='run'
        self.metadata['source_sha256']={'old.py':'a'*64};self.save();self.assertFalse(self.complete())
        self.metadata['source_sha256']=self.source
        self.recipe['seed']=1;self.save();self.assertFalse(self.complete())
        self.recipe['seed']=42
        self.summary['metrics']['prefix_hits']=999;self.save();self.assertFalse(self.complete())

    def test_rebuilt_dev_pack_detects_token_changes(self):
        self.rows[0]['document_chunks'][0][0]=999
        (self.data/'dev.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows),encoding='utf-8')
        self.assertFalse(self.complete())


class QueuePolicyTests(unittest.TestCase):
    def test_modes_are_four_questions_then_all99_no_forced_length(self):
        for mode,count in (('smoke',4),('full',99)):
            plan=build_prefix_quality_plan(prepared(),mode=mode)
            self.assertEqual(len(plan['ordered_ids']),count)
            cmd=queue.worker_command('/tmp/out',mode,'/tmp/lease')
            self.assertEqual(cmd[cmd.index('--mode')+1],mode)
            self.assertEqual(cmd[cmd.index('--max-new-tokens')+1],'128')
            self.assertEqual(cmd[cmd.index('--seed')+1],'42')
            self.assertNotIn('--force-length',cmd)
        with self.assertRaises(ValueError):queue.worker_command('/tmp/out','other','/tmp/lease')

    def test_restart_preserves_preworker_failure_and_partial_out(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(queue,'completed_prefix_quality',return_value=False):
            out=Path(temp)
            self.assertEqual(queue.initial_jobs(out,{}, {})['smoke']['phase'],'queued')
            old={'jobs':{'smoke':{'phase':'failed','error':'before model'}}}
            self.assertEqual(queue.initial_jobs(out,old,{})['smoke']['phase'],'failed')
            (out/'full').mkdir()
            self.assertEqual(queue.initial_jobs(out,{}, {})['full']['phase'],'failed')
            control=out/'smoke_control';control.mkdir();(control/'worker.log').write_text('failed lease')
            self.assertEqual(queue.initial_jobs(out,{}, {})['smoke']['phase'],'failed')

    def test_original512mib_and_external_process_gate_is_reused(self):
        from remote_sparse_queue import gpu_inventory
        def xml(used,process=''):
            return '<nvidia_smi_log><gpu><product_name>RTX 3090</product_name><fb_memory_usage><used>'+str(used)+' MiB</used></fb_memory_usage><processes>'+process+'</processes></gpu></nvidia_smi_log>'
        self.assertEqual(queue.eligible_gpus(gpu_inventory(xml(511))),[0])
        self.assertEqual(queue.eligible_gpus(gpu_inventory(xml(512))),[])
        foreign='<process_info><pid>19</pid><type>C</type><process_name>python</process_name></process_info>'
        self.assertEqual(queue.eligible_gpus(gpu_inventory(xml(5,foreign))),[])

    def test_bind_or_save_exception_stops_only_owned_child(self):
        class Child:
            pid=52
            returncode=None
            def poll(self):return self.returncode
        class Leases:
            def __init__(self,*args):pass
            def add(self,*args,**kwargs):pass
            def start(self):pass
            def remove(self,*args):pass
            def stop(self):pass
            def failures(self):return {}
        for failure in ('bind','save'):
            child=Child()
            def stop(proc):
                self.assertIs(proc,child);proc.returncode=-15
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                control=Path(temp)
                with (control/'lock').open('a') as lock, \
                     patch.object(queue,'LeaseRefresher',Leases), \
                     patch.object(queue.subprocess,'Popen',return_value=child), \
                     patch.object(queue,'bind_owned_process',side_effect=RuntimeError('bind') if failure=='bind' else None), \
                     patch.object(queue,'process_identity',return_value={'pid':52}), \
                     patch.object(queue,'terminate_owned',side_effect=stop) as terminate:
                    def save(reason):raise PermissionError('save')
                    rc,error=queue.run_owned_worker([],{},0,lock,control,{'lease_run_id':'test'},save)
                    self.assertEqual(rc,-15);self.assertTrue(error);terminate.assert_called_once_with(child)


if __name__ == '__main__':
    unittest.main(verbosity=2)
