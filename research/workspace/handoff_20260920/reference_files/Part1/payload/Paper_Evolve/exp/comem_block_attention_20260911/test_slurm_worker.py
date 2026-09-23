"""Stdlib-only serial-worker cleanup, manifest and shim tests; no Torch import."""
from __future__ import annotations
from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import slurm_sparse_worker as worker
import train_sparse_slurm as wrapper


class WorkerTests(unittest.TestCase):
    def test_heartbeat_initial_write_failure_and_cleanup(self):
        write=Mock()
        heart=worker.LeaseHeartbeat(write,interval=.01)
        heart.start(); self.assertTrue(heart.stop_event.wait(.03) is False)
        heart.close(); self.assertGreaterEqual(write.call_count,2)
        failed=worker.LeaseHeartbeat(Mock(side_effect=OSError('disk')))
        with self.assertRaises(OSError): failed.start()
        failed.close()

    def test_wrapper_keeps_original_main_error_semantics_and_restores_module(self):
        trainer=ModuleType('train_sparse'); old=ModuleType('remote_gpu_guard')
        called=[]
        def main():
            from remote_gpu_guard import validate_worker_lease
            from slurm_gpu_guard import validate_worker_lease as expected
            called.append(validate_worker_lease is expected)
            raise ValueError('original training failure')
        trainer.main=main
        with patch.dict(sys.modules,{'train_sparse':trainer,'remote_gpu_guard':old}):
            with self.assertRaisesRegex(ValueError,'original training failure'): wrapper.main()
            self.assertIs(sys.modules['remote_gpu_guard'],old)
        self.assertEqual(called,[True])

    def test_explicit_manifest_retains_d0_retry_and_rejects_budget_or_order(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); args=Namespace(task_root=root,out=root/'outputs',steps=250,grad_accum=4)
            ids=[f'q{i}' for i in range(99)]; tasks=[]
            for stage in ('smoke','train'):
                for arm in worker.ARMS:
                    target='D0_retry1' if stage=='smoke' and arm=='D0' else arm
                    tasks.append(dict(stage=stage,arm=arm,source_job_key=f'{stage}/{arm}',
                        source_out=f'/source/{stage}/{target}',destination_relative_out=f'outputs/{stage}/{target}',
                        steps=1 if stage=='smoke' else 250,grad_accum=2 if stage=='smoke' else 4,
                        dev_ids=ids[:1] if stage=='smoke' else ids))
            jobs=worker.migrated_jobs(args,{'tasks':tasks},ids)
            self.assertTrue(jobs['smoke/D0']['out'].endswith('D0_retry1'))
            wrong=copy.deepcopy(tasks); wrong[0]['grad_accum']=1
            for changed in (wrong,list(reversed(tasks)),tasks[:-1]):
                with self.assertRaises(RuntimeError): worker.migrated_jobs(args,{'tasks':changed},ids)

    def test_cleanup_covers_on_start_and_monitor_exceptions(self):
        for callback in ('on_start','monitor'):
            proc=Mock(); proc.poll.side_effect=[None,0] if callback=='on_start' else [None,None,0]
            params=dict(cwd='.',env={},log=None,lock_fd=7,stopped=threading.Event(),
                        **{callback:Mock(side_effect=OSError('failure'))})
            with patch.object(worker.subprocess,'Popen',return_value=proc) as popen, \
                 patch.object(worker,'bind_owned_process'), patch.object(worker,'terminate_owned') as stop:
                with self.assertRaises(OSError): worker.supervise(['cpu-child'],**params)
                stop.assert_called_once_with(proc)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                self.assertEqual(popen.call_args.kwargs['pass_fds'],(7,))


@unittest.skipUnless(sys.platform=='linux','Real process group ownership requires Linux')
class LinuxSupervisionTests(unittest.TestCase):
    def test_anonymous_ticker_and_shutdown_never_touch_the_shared_filesystem(self):
        import slurm_gpu_guard as guard
        transport = guard.AnonymousHeartbeat()
        heart = worker.LeaseHeartbeat(transport.refresh, interval=.005)
        try:
            with patch.object(guard, 'publish', side_effect=OSError('shared disk unavailable')), \
                 patch.object(guard.Path, 'resolve', side_effect=OSError('shared resolve blocked')), \
                 patch.object(guard.Path, 'stat', side_effect=OSError('shared stat blocked')):
                heart.start()
                time.sleep(.03)
                self.assertIsNone(heart.error)
                self.assertGreater(transport.sequence, 4)
                started = time.monotonic()
                heart.close()
                self.assertLess(time.monotonic()-started, 1.)
                self.assertFalse(heart.thread.is_alive())
                age = guard.anonymous_heartbeat_age(transport.description, os.getpid(),
                    {'SPARSE_SLURM_HEARTBEAT_FD': str(transport.fd)})
                self.assertLess(age, 1.)
        finally:
            heart.close()
            transport.close()

    def test_supervisor_passes_anonymous_heartbeat_and_lock_to_owned_child(self):
        import fcntl
        import slurm_gpu_guard as guard
        transport = guard.AnonymousHeartbeat()
        try:
            with tempfile.TemporaryDirectory() as value:
                root = Path(value)
                with (root/'lock').open('a') as lock, (root/'child.log').open('w+') as log:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    code = ('import json,os,sys;sys.path.insert(0,sys.argv[1]);'
                            'import slurm_gpu_guard as g;'
                            'print(g.anonymous_heartbeat_age(json.loads(sys.argv[2]),os.getppid()))')
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                           'SPARSE_SLURM_HEARTBEAT_FD': str(transport.fd)}
                    result = worker.supervise(
                        [sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parent),
                         json.dumps(transport.description)], cwd=root, env=env, log=log,
                        lock_fd=lock.fileno(), stopped=threading.Event(), extra_fds=(transport.fd,))
                    log.seek(0)
                    self.assertEqual(result, 0, log.read())
        finally:
            transport.close()

    def test_owned_child_stops_after_callback_failure_other_child_survives(self):
        import fcntl
        with tempfile.TemporaryDirectory() as value:
            root=Path(value)
            with (root/'lock').open('a') as lock, (root/'child.log').open('w') as log:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                other=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'],start_new_session=True)
                own=[]
                def launched(proc):
                    own.append(proc)
                    raise OSError('simulate persist failure after Popen')
                try:
                    with self.assertRaises(OSError):
                        worker.supervise([sys.executable,'-B','-c','import time;time.sleep(30)'],
                            cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},log=log,
                            lock_fd=lock.fileno(),stopped=threading.Event(),on_start=launched)
                    self.assertIsNotNone(own[0].poll())
                    self.assertIsNone(other.poll())
                finally:
                    other.terminate(); other.wait(timeout=5)


if __name__=='__main__':
    unittest.main(verbosity=2)
