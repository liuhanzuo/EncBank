"""Stdlib tests for scheduling/lease changes; never import Torch or use a GPU."""
from __future__ import annotations
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from remote_sparse_queue import (LeaseRefresher, gpu_inventory,
                                 guard_running_jobs, retry_failed_smoke,
                                 cleanup_owned_workers, supervised_workers)


def jobs_at(out):
    return {'smoke/D0': dict(stage='smoke', arm='D0', phase='failed',
                            out=str(Path(out)/'smoke/D0'), steps=1,
                            grad_accum=2, dev_ids=['same-question'],
                            pid=123, error='original OOM', returncode=1)}


class RetryTests(unittest.TestCase):
    def test_explicit_retry_preserves_original_and_recipe(self):
        with tempfile.TemporaryDirectory() as out:
            olddir = Path(out)/'smoke/D0'
            olddir.mkdir(parents=True)
            (olddir/'status.json').write_text('original failure')
            jobs = jobs_at(out)
            original = copy.deepcopy(jobs['smoke/D0'])
            retry_failed_smoke(jobs, 'D0', out)
            item = jobs['smoke/D0']
            self.assertEqual(item['previous_attempts'], [original])
            self.assertEqual(item['phase'], 'queued')
            self.assertEqual(item['retry_count'], 1)
            self.assertEqual(item['out'], str(Path(out)/'smoke/D0_retry1'))
            self.assertEqual((olddir/'status.json').read_text(), 'original failure')
            self.assertFalse(Path(item['out']).exists())
            self.assertNotIn('pid', item)
            for key in ('stage', 'arm', 'steps', 'grad_accum', 'dev_ids'):
                self.assertEqual(item[key], original[key])

    def test_retry_is_not_repeatable_even_after_second_failure(self):
        with tempfile.TemporaryDirectory() as out:
            jobs = jobs_at(out)
            retry_failed_smoke(jobs, 'D0', out)
            jobs['smoke/D0']['phase'] = 'failed'
            with self.assertRaises(ValueError):
                retry_failed_smoke(jobs, 'D0', out)

    def test_reject_live_complete_unknown_or_existing_destination(self):
        with tempfile.TemporaryDirectory() as out:
            for phase in ('queued', 'launching', 'running', 'complete'):
                jobs = jobs_at(out)
                jobs['smoke/D0']['phase'] = phase
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    retry_failed_smoke(jobs, 'D0', out)
            with self.assertRaises(ValueError):
                retry_failed_smoke(jobs_at(out), 'A', out)
            (Path(out)/'smoke/D0_retry1').mkdir(parents=True)
            with self.assertRaises(ValueError):
                retry_failed_smoke(jobs_at(out), 'D0', out)


class HeartbeatTests(unittest.TestCase):
    def values(self):
        return dict(path='lease.json', gpu=2, lock_fd=7,
                    worker_script='train_sparse.py', run_id='one-attempt')

    def test_one_writer_refresh_and_removal_before_fd_close(self):
        writer = Mock()
        lease = LeaseRefresher(writer)
        lease.add('smoke/D0', **self.values())
        lease.refresh()
        self.assertEqual(writer.call_count, 2)
        lease.remove('smoke/D0')
        lease.refresh()
        self.assertEqual(writer.call_count, 2)

    def test_refresh_failure_is_reported_not_silently_ignored(self):
        writer = Mock()
        lease = LeaseRefresher(writer)
        lease.add('smoke/D0', **self.values())
        writer.side_effect = OSError('disk unavailable')
        lease.refresh()
        self.assertIn('disk unavailable', lease.failures()['smoke/D0'])
        lease.remove('smoke/D0')
        self.assertEqual(lease.failures(), {})

    def test_heartbeat_continues_while_main_thread_waits(self):
        observed = threading.Event()
        calls = []
        def writer(**value):
            calls.append(value)
            if len(calls) >= 4:
                observed.set()
        lease = LeaseRefresher(writer, interval=.01)
        lease.add('smoke/D0', **self.values())
        lease.start()
        try:
            self.assertTrue(observed.wait(2.))
        finally:
            lease.stop()
        self.assertFalse(lease.thread.is_alive())


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.proc = Mock(pid=123)
        self.proc.poll.return_value = None
        self.running = {'smoke/D0': (self.proc, object(), object())}
        self.jobs = {'smoke/D0': {'gpu': 2}}
        self.stop = Mock()

    def check(self, unexpected, errors=None):
        guard_running_jobs(self.running, self.jobs, [{'index': 2}], errors or {},
                           unexpected=unexpected, terminate=self.stop)

    def test_no_interference_keeps_worker(self):
        check = Mock(return_value=[])
        self.check(check)
        check.assert_called_once_with([{'index': 2}], 2, {123})
        self.stop.assert_not_called()

    def test_foreign_model_stops_owned_worker_only(self):
        foreign = {'pid': '3187282', 'type': 'G', 'process_name': 'python'}
        self.check(Mock(return_value=[foreign]))
        self.stop.assert_called_once_with(self.proc)
        self.assertEqual(self.jobs['smoke/D0']['guard_failure']['unexpected_processes'], [foreign])

    def test_unavailable_inventory_fails_closed(self):
        self.check(Mock(side_effect=RuntimeError('unknown XML')))
        self.stop.assert_called_once_with(self.proc)
        self.assertIn('unknown XML', self.jobs['smoke/D0']['guard_failure']['reason'])

    def test_heartbeat_failure_stops_without_accepting_idle_snapshot(self):
        check = Mock(return_value=[])
        self.check(check, {'smoke/D0': 'lease write failed'})
        check.assert_not_called()
        self.stop.assert_called_once_with(self.proc)

    def test_already_exited_worker_is_not_signalled(self):
        self.proc.poll.return_value = 1
        check = Mock(return_value=[{'pid': 'other'}])
        self.check(check)
        check.assert_not_called()
        self.stop.assert_not_called()


class AdmissionTests(unittest.TestCase):
    def xml(self, used='511 MiB', process='', process_text=''):
        return ('<nvidia_smi_log><gpu><product_name>NVIDIA GeForce RTX 3090</product_name>'
                f'<fb_memory_usage><used>{used}</used></fb_memory_usage>'
                f'<processes>{process_text}{process}</processes></gpu></nvidia_smi_log>')

    def test_threshold_all_process_types_and_unknown_inventory(self):
        self.assertTrue(gpu_inventory(self.xml())[0]['eligible'])
        self.assertFalse(gpu_inventory(self.xml(used='512 MiB'))[0]['eligible'])
        self.assertFalse(gpu_inventory(self.xml(process_text='N/A'))[0]['eligible'])
        for kind in ('C', 'G', 'M', 'C+G', ''):
            process = (f'<process_info><pid>321</pid><type>{kind}</type>'
                       '<process_name>python</process_name></process_info>')
            with self.subTest(kind=kind):
                self.assertFalse(gpu_inventory(self.xml(process=process))[0]['eligible'])

    def test_worker_without_lease_rejects_before_importing_torch(self):
        import train_sparse
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'live remote controller GPU lease'):
                train_sparse.run(SimpleNamespace())

    def test_source_requires_post_import_pre_cuda_check_and_preserves_recipe(self):
        tree = ast.parse((Path(__file__).parent/'train_sparse.py').read_text(encoding='utf-8'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        imports = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Import)
                   and any(alias.name == 'torch' for alias in node.names)]
        idle_checks = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == 'validate_worker_lease'
                       and any(k.arg == 'require_idle' and isinstance(k.value, ast.Constant)
                               and k.value.value is True for k in node.keywords)]
        set_device = [node.lineno for node in ast.walk(run) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_device']
        self.assertEqual(len(idle_checks), 2)
        self.assertLess(min(idle_checks), imports[0])
        self.assertLess(imports[0], max(idle_checks))
        self.assertLess(max(idle_checks), set_device[0])
        source = (Path(__file__).parent/'train_sparse.py').read_text(encoding='utf-8')
        self.assertNotIn('add_argument("--gpu-lease', source)
        self.assertIn('args.max_new_tokens', source)
        self.assertIn('torch.set_num_interop_threads(16)', source)


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        proc = Mock(pid=123)
        proc.poll.return_value = None
        log, lock, leases = Mock(), Mock(), Mock()
        running = {'smoke/D0': (proc, log, lock)}
        return proc, log, lock, leases, running

    def test_post_launch_exception_stops_before_closing_ownership(self):
        proc, log, lock, leases, running = self.fixture()
        events = []
        def stop(child):
            self.assertIs(child, proc)
            events.append('stop')
            child.poll.return_value = -15
        log.close.side_effect = lambda: events.append('log_close')
        lock.close.side_effect = lambda: events.append('lock_close')
        with self.assertRaisesRegex(RuntimeError, 'persist failed'):
            with supervised_workers(running, leases, stop):
                raise RuntimeError('persist failed')
        self.assertEqual(events, ['stop', 'log_close', 'lock_close'])
        self.assertEqual(running, {})
        leases.remove.assert_called_once_with('smoke/D0')
        leases.stop.assert_called_once()

    def test_failed_stop_does_not_release_a_live_worker_lock(self):
        proc, log, lock, leases, running = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'Unable to complete owned-worker cleanup'):
            cleanup_owned_workers(running, leases, Mock(side_effect=RuntimeError('identity unavailable')))
        self.assertIn('smoke/D0', running)
        log.close.assert_not_called()
        lock.close.assert_not_called()
        leases.remove.assert_not_called()

    def test_selective_launch_cleanup_leaves_other_worker_registered(self):
        proc, log, lock, leases, running = self.fixture()
        proc.poll.return_value = 1
        other = (Mock(), Mock(), Mock())
        running['smoke/A'] = other
        stop = Mock()
        cleanup_owned_workers(running, leases, stop, names=['smoke/D0'])
        self.assertEqual(running, {'smoke/A': other})
        stop.assert_not_called()
        lock.close.assert_called_once()

    def test_popen_is_registered_before_binding(self):
        source = (Path(__file__).parent/'remote_sparse_queue.py').read_text(encoding='utf-8')
        begin = source.index('proc = subprocess.Popen(command')
        self.assertLess(source.index('running[name] = proc, log, gpu_lock', begin),
                        source.index('bind_owned_process(proc)', begin))
        self.assertIn('with supervised_workers(running, leases, terminate_owned):', source)


@unittest.skipUnless(sys.platform == 'linux', 'Real Linux flock/lease subprocess tests')
class LinuxLeaseTests(unittest.TestCase):
    """Use private lock paths and mocked inventory, never the production GPU locks."""
    def setUp(self):
        import fcntl
        import remote_gpu_guard
        self.fcntl, self.guard = fcntl, remote_gpu_guard
        results = Path(__file__).resolve().parent/'results'
        results.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='guard-cpu-fixture-', dir=results)
        self.root = Path(self.temp.name).resolve()
        (self.root/'logs').mkdir()
        self.root_patch = patch.object(self.guard, 'ROOT', self.root)
        self.root_patch.start()
        self.lock_path = self.root/'logs/beacon_sft_gpu2.lock'
        self.lock = self.lock_path.open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.worker = self.root/'worker.py'
        self.worker.write_text('''import json, os, sys
from pathlib import Path
sys.path.insert(0, SOURCE)
import remote_gpu_guard as guard
guard.ROOT = Path(ROOT)
mode = sys.argv[1]
row = dict(index=2, name="NVIDIA GeForce RTX 3090", used_mib=4,
           eligible=True, processes=[])
if mode == "foreign":
    row.update(used_mib=1000, eligible=False, processes=[
        dict(pid=str(os.getpid()), type="C", process_name="python"),
        dict(pid="887766", type="G", process_name="python")])
guard.inventory = lambda: [row]
try:
    value = guard.validate_worker_lease(os.environ["SPARSE_GPU_LEASE_PATH"],
                                      require_idle=mode != "foreign")
except Exception as exc:
    print(json.dumps(dict(ok=False, error=str(exc), torch_imported="torch" in sys.modules)), flush=True)
    sys.exit(2)
print(json.dumps(dict(ok=True, torch_imported="torch" in sys.modules,
                      parent=value["lease"]["controller"]["pid"])), flush=True)
if mode == "hold":
    sys.stdin.readline()
'''.replace('SOURCE', repr(str(Path(__file__).resolve().parent)))
   .replace('ROOT)', repr(str(self.root)) + ')'), encoding='utf-8')
        self.lease_path = self.root/'lease.json'
        self.children = []
        self.extra_files = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
        for file in self.extra_files:
            file.close()
        self.lock.close()
        self.root_patch.stop()
        self.temp.cleanup()

    def launch(self, mode='normal', mutation=None, bad_fd=False):
        lease = self.guard.write_lease(self.lease_path, 2, self.lock.fileno(),
                                       self.worker, 'private-cpu-fixture')
        if mutation:
            mutation(lease)
            self.lease_path.write_text(json.dumps(lease), encoding='utf-8')
        fds = [self.lock.fileno()]
        worker_fd = fds[0]
        if bad_fd:
            other = (self.root/'unrelated.file').open('a')
            self.extra_files.append(other)
            worker_fd = other.fileno()
            fds.append(worker_fd)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '2',
               'SPARSE_GPU_LEASE_PATH': str(self.lease_path),
               'SPARSE_GPU_LOCK_FD': str(worker_fd),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        child = subprocess.Popen([sys.executable, str(self.worker), mode],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, pass_fds=tuple(fds))
        self.children.append(child)
        return child

    def rejected(self, *, expected, **kwargs):
        child = self.launch(**kwargs)
        stdout, stderr = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 2, stderr)
        value = json.loads(stdout)
        self.assertFalse(value['ok'])
        self.assertFalse(value['torch_imported'])
        self.assertIn(expected, value['error'])

    def test_real_inherited_lock_survives_parent_descriptor_close(self):
        child = self.launch(mode='hold')
        value = json.loads(child.stdout.readline())
        self.assertTrue(value['ok'])
        self.assertFalse(value['torch_imported'])
        self.assertEqual(value['parent'], os.getpid())
        self.lock.close()
        with self.lock_path.open('a') as probe:
            with self.assertRaises(BlockingIOError):
                self.fcntl.flock(probe, self.fcntl.LOCK_EX | self.fcntl.LOCK_NB)
            child.communicate('\n', timeout=10)
            self.assertEqual(child.returncode, 0)
            self.fcntl.flock(probe, self.fcntl.LOCK_EX | self.fcntl.LOCK_NB)

    def test_real_worker_rejects_stale_lease(self):
        self.rejected(expected='heartbeat is stale', mutation=lambda value:
                      value.update(updated_unix_s=value['updated_unix_s'] - 60))

    def test_real_worker_rejects_parent_identity_change(self):
        def changed(value):
            value['controller']['start_ticks'] += 1
        self.rejected(expected='controller has exited or changed identity', mutation=changed)

    def test_real_worker_rejects_mismatched_inherited_descriptor(self):
        self.rejected(expected='Inherited GPU lock identity differs', bad_fd=True)

    def test_real_worker_rejects_graphics_type_foreign_model(self):
        self.rejected(mode='foreign', expected='Another GPU process appeared')

    def test_terminate_owned_stops_only_bound_cpu_child(self):
        workers = []
        for _ in range(2):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)
            self.children.append(child)
            workers.append(child)
        owned, unrelated = workers
        self.guard.bind_owned_process(owned)
        self.guard.terminate_owned(owned)
        self.assertIsNotNone(owned.poll())
        self.assertIsNone(unrelated.poll())


if __name__ == '__main__':
    unittest.main()
