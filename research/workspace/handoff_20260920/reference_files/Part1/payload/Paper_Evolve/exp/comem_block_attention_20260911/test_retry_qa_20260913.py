"""Bounded retry safety tests; standard library, no model or GPU imports."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import retry_qa_20260913 as retry


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.out = self.root / 'retry'
        self.code = self.root / 'code'
        self.old = {kind: self.root / kind for kind in ('backend', 'prefix')}
        for name, value in (('ROOT', self.root), ('CODE', self.code), ('OUT', self.out), ('OLD', self.old)):
            self.stack.enter_context(patch.object(retry, name, value))
        self.owner = dict(pid=3254268, start_ticks=982966640,
                          argv=['python', str(self.code / 'qa_slurm_handoff.py')])
        self.queue = SimpleNamespace(process_identity=Mock(side_effect=lambda pid: self.owner if pid == 3254268 else None))
        self.states = {
            'backend': dict(controller=dict(pid=100, start_ticks=1), reason='failed_no_promotion', config={'fixed': True},
                jobs={'smoke': dict(phase='failed', returncode=-15,
                        error='External GPU process appeared; stopped only the QA worker'),
                      'full': dict(phase='queued')}),
            'prefix': dict(controller=self.owner, reason='backend_quality_controller_not_active',
                config={'fixed': True}, jobs={'smoke': dict(phase='queued'), 'full': dict(phase='queued')})}
        for kind in self.old:
            self.old[kind].mkdir()
            for mode in ('smoke', 'full'):
                self.states[kind]['jobs'][mode]['out'] = str(self.old[kind] / mode)

    def check(self, **kwargs):
        retry.check_old_states(self.states, self.queue, **kwargs)

    def declaration(self):
        self.out.mkdir()
        for kind in self.old:
            text = json.dumps(self.states[kind])
            (self.out / (kind + '_original_queue.json')).write_text(text)
            (self.old[kind] / 'queue.json').write_text(text)
        value = dict(schema=retry.SCHEMA, bounded_attempts=1, worker_or_guard_changes=False,
            wrapper_sha256=retry.sha(retry.__file__), handoff_declaration=str(retry.HANDOFF),
            handoff_sha256=retry.HANDOFF_SHA, outputs={kind: str(self.out / kind) for kind in self.old},
            original_configs={kind: self.states[kind]['config'] for kind in self.old},
            original_snapshot_sha256={kind: retry.sha(self.out / (kind + '_original_queue.json')) for kind in self.old})
        path = self.out / 'declaration.json'
        path.write_text(json.dumps(value))
        self.queue.process_identity.side_effect = lambda pid: None
        return path, value

    def test_known_failed_backend_and_waiting_prefix_allowed(self):
        self.check()

    def test_unknown_failure_forbids_retry(self):
        self.states['backend']['jobs']['smoke']['error'] = 'OOM'
        with self.assertRaisesRegex(RuntimeError, 'known resource'):
            self.check()

    def test_backend_still_live_forbids_retry(self):
        self.queue.process_identity.side_effect = lambda pid: self.states['backend']['controller'] if pid == 100 else self.owner
        with self.assertRaisesRegex(RuntimeError, 'exited failed'):
            self.check()

    def test_reused_prefix_pid_cannot_be_stopped(self):
        self.queue.process_identity.side_effect = lambda pid: dict(self.owner, start_ticks=77) if pid == 3254268 else None
        with self.assertRaisesRegex(RuntimeError, 'liveness'):
            self.check()

    def test_prefix_running_prevents_replacement(self):
        self.states['prefix']['jobs']['smoke']['phase'] = 'running'
        with self.assertRaisesRegex(RuntimeError, 'untouched'):
            self.check()

    def test_unrecorded_prefix_artifact_prevents_replacement(self):
        (self.old['prefix'] / 'smoke').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'has output'):
            self.check()

    def test_backend_full_is_never_overwritten(self):
        (self.old['backend'] / 'full').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'has output'):
            self.check()

    def test_declared_new_outputs_preserve_original_states(self):
        path, value = self.declaration()
        before = {kind: (self.old[kind] / 'queue.json').read_bytes() for kind in self.old}
        self.assertEqual(retry.validate_retry(path, retry.sha(path), queue=self.queue), value)
        self.assertEqual(before, {kind: (self.old[kind] / 'queue.json').read_bytes() for kind in self.old})

    def test_original_queue_change_blocks_retry(self):
        path, value = self.declaration()
        (self.old['backend'] / 'queue.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'queue changed'):
            retry.validate_retry(path, retry.sha(path), queue=self.queue)

    def test_redirect_to_old_failed_output_blocks_retry(self):
        path, value = self.declaration()
        value['outputs']['backend'] = str(self.old['backend'])
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'altered retry'):
            retry.validate_retry(path, retry.sha(path), queue=self.queue)

    def test_guard_changes_cannot_be_declared(self):
        path, value = self.declaration()
        value['worker_or_guard_changes'] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'altered retry'):
            retry.validate_retry(path, retry.sha(path), queue=self.queue)

    def test_existing_retry_directory_cannot_be_reused(self):
        self.out.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            retry.execute(self.states, self.queue, None)

    def test_remote_kernel_pidfd_without_sending_an_actual_signal(self):
        fd = retry.open_pidfd(os.getpid())
        try:
            self.assertIn('Pid:\t' + str(os.getpid()), Path('/proc/self/fdinfo/' + str(fd)).read_text())
            retry.signal_pidfd(fd, 0)
        finally:
            os.close(fd)


if __name__ == '__main__':
    unittest.main()
