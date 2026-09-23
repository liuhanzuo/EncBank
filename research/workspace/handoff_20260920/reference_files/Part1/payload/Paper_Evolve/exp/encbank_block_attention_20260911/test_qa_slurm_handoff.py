"""Standard-library-only tests for the migration resource-priority adapter."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import qa_slurm_handoff as handoff
import remote_backend_quality_queue as backend
import remote_prefix_quality_queue as prefix


class DeclarationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.code = self.root / 'code'
        self.code.mkdir()
        self.out = self.root / 'outputs/training'
        self.out.mkdir(parents=True)
        for name in ('ROOT', 'CODE', 'TRAIN_OUT'):
            self.stack.enter_context(patch.object(handoff, name,
                {'ROOT': self.root, 'CODE': self.code, 'TRAIN_OUT': self.out}[name]))
        self.owner = dict(pid=123, start_ticks=456, argv=['training'])
        self.jobs = {stage+'/'+arm: dict(phase='queued', steps=250 if stage == 'train' else 1)
                     for stage in ('smoke', 'train') for arm in ('D0', 'A', 'B', 'D1')}
        state = dict(controller=self.owner, jobs=self.jobs)
        self.write(self.out / 'queue.json', state)
        self.migration = dict(source_controller=self.owner, source_jobs=self.jobs,
            source_queue_path=str(self.out / 'queue.json'),
            source_queue_sha256=handoff.sha(self.out / 'queue.json'), tasks=[dict(key=k) for k in self.jobs])
        self.migration_path = self.root / 'receipt.json'
        self.write(self.migration_path, self.migration)
        self.submission_path = self.root / 'submission.json'
        self.submission = dict(job_id='24577', raw_job_id='24577', job_name='encbank-sparse-1gpu',
            destination_backend='slurm-l20d', migration_sha256=handoff.sha(self.migration_path),
            command=['sbatch', '--parsable', '--chdir', handoff.DESTINATION, '--gres',
                     'gpu:nvidia_l20d:1', '--nodes', '1', '--ntasks', '1', '--job-name', 'encbank-sparse-1gpu'],
            submitted_utc='2026-09-12T07:43:30+00:00', stdin_script='actual submission')
        self.write(self.submission_path, self.submission)
        names = ('qa_slurm_handoff.py', 'remote_sparse_queue.py', 'remote_gpu_guard.py',
                 'remote_backend_quality_queue.py', 'remote_prefix_quality_queue.py', 'submit_slurm_sparse.py')
        for name in names:
            (self.code / name).write_text('fixture '+name)
        self.value = dict(schema=handoff.SCHEMA, policy=handoff.POLICY, source_root=str(self.root),
            source_host='longjing-1', destination_root=handoff.DESTINATION, source_priority_released=True,
            training_complete_asserted=False, source_sha256={n:handoff.sha(self.code/n) for n in names},
            migration_receipt=str(self.migration_path), migration_sha256=handoff.sha(self.migration_path),
            source_migration_receipt=str(self.migration_path), source_migration_sha256=handoff.sha(self.migration_path),
            tasks=self.migration['tasks'], destination_submission=str(self.submission_path),
            submission_sha256=handoff.sha(self.submission_path), destination_job_id='24577')
        self.path = self.root / 'declaration.json'
        self.write(self.path, self.value)
        self.queue = SimpleNamespace(process_identity=Mock(return_value=None))
        self.processes = self.stack.enter_context(patch.object(handoff, 'source_processes', return_value=[]))
        # The reused migration validator has its own full real-topology tests;
        # this suite targets fresh filesystem and process checks in the adapter.
        self.stack.enter_context(patch('submit_slurm_sparse.migration_receipt', side_effect=lambda v, **kw: v))

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def check(self):
        return handoff.validate_declaration(self.path, handoff.sha(self.path), self.queue)

    def test_drained_preserves_all_source_queued_jobs(self):
        before = (self.out / 'queue.json').read_bytes()
        result = self.check()
        self.assertFalse(result['training_complete_asserted'])
        self.assertEqual(before, (self.out / 'queue.json').read_bytes())
        self.assertTrue(result['source_jobs_remain_queued'])

    def test_old_controller_identity_blocks_but_recycled_pid_does_not(self):
        self.queue.process_identity.return_value = self.owner
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            self.check()
        self.queue.process_identity.return_value = dict(self.owner, start_ticks=457)
        self.check()

    def test_any_new_source_controller_or_trainer_blocks(self):
        for script in ('remote_sparse_queue.py', 'train_sparse.py'):
            self.processes.return_value = [dict(pid=999, argv=[script])]
            with self.assertRaisesRegex(RuntimeError, 'reappeared'):
                self.check()

    def test_unknown_process_information_blocks(self):
        self.processes.side_effect = PermissionError('unreadable process')
        with self.assertRaises(PermissionError):
            self.check()

    def test_queue_change_blocks(self):
        self.write(self.out / 'queue.json', dict(controller=self.owner, jobs={}))
        with self.assertRaisesRegex(ValueError, 'queue changed'):
            self.check()

    def test_mapping_change_blocks(self):
        self.value['tasks'] = []
        self.write(self.path, self.value)
        with self.assertRaisesRegex(ValueError, 'mapping changed'):
            self.check()

    def test_submission_missing_or_different_resources_blocks(self):
        for field, value in (('raw_job_id', ''), ('stdin_script', ''), ('job_id', '24023')):
            altered = dict(self.submission, **{field:value})
            self.write(self.submission_path, altered)
            self.value['submission_sha256'] = handoff.sha(self.submission_path)
            self.write(self.path, self.value)
            with self.assertRaises(ValueError):
                self.check()
        altered = copy.deepcopy(self.submission)
        altered['command'][altered['command'].index('--gres')+1] = 'gpu:nvidia_l20d:2'
        self.write(self.submission_path, altered)
        self.value['submission_sha256'] = handoff.sha(self.submission_path)
        self.write(self.path, self.value)
        with self.assertRaisesRegex(ValueError, 'resources differ'):
            self.check()

    def test_declaration_revocation_and_source_binding_block(self):
        digest = handoff.sha(self.path)
        self.value['source_priority_released'] = False
        self.write(self.path, self.value)
        with self.assertRaisesRegex(ValueError, 'revoked'):
            handoff.validate_declaration(self.path, digest, self.queue)
        with self.assertRaises(ValueError):
            self.check()
        self.value['source_priority_released'] = True
        self.write(self.path, self.value)
        (self.code / 'remote_sparse_queue.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.check()


class InstallTests(unittest.TestCase):
    def test_original_success_preserved_and_each_migration_check_is_fresh(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(handoff, 'ROOT', Path(directory).resolve()):
            log = Path(directory) / 'events.jsonl'
            original = Mock(return_value=(True, 'training_complete'))
            module = SimpleNamespace(training_priority_clear=original)
            with patch.object(handoff, 'validate_declaration', return_value={'training_complete_asserted':False}) as validate:
                function = handoff.install_priority(module, None, None, 'bound', log)
                self.assertEqual(function(), (True, 'training_complete'))
                validate.assert_not_called()
                original.return_value = False, 'training_controller_not_active'
                self.assertTrue(function()[0])
                self.assertTrue(function()[0])
                self.assertEqual(validate.call_count, 2)
                validate.side_effect = ValueError('revoked')
                self.assertFalse(function()[0])
            self.assertEqual(len(log.read_text().splitlines()), 4)

    def test_shared_module_and_prefix_backend_gate_remain(self):
        self.assertIs(prefix.backend_queue, backend)
        with tempfile.TemporaryDirectory() as directory, patch.object(handoff, 'ROOT', Path(directory).resolve()), \
                patch.object(backend, 'training_priority_clear', return_value=(False, 'training_controller_not_active')), \
                patch.object(handoff, 'validate_declaration', return_value={'training_complete_asserted':False}), \
                patch.object(prefix, 'backend_priority_clear', return_value=(False, 'backend_quality_pending_or_failed')) as dependency:
            function = handoff.install_priority(backend, None, None, 'bound', Path(directory)/'events.jsonl')
            self.assertIs(prefix.backend_queue.training_priority_clear, function)
            self.assertEqual(prefix.priorities_clear(), (False, 'backend_quality_pending_or_failed'))
            dependency.assert_called_once()


if __name__ == '__main__':
    unittest.main()
