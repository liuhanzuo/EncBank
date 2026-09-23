"""Server CPU tests for migration-specific identity, delivery and ownership invariants."""
import base64
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/encbank_k12_server_r6_20260920'
sys.path.insert(0, str(RUN))
from runtime_identity import resolve_configs
from server_transport import Transport, save, verify_response
from live_mailbox import Mailbox, client
import host_admission


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((RUN / 'plan.json').read_text())
        self.identity = self.plan['training_model_identity']
        self.checkpoint = {'step': 4000, 'j': 21, 'model': dict(self.identity)}

    def test_relocated_model_retains_exact_training_identity(self):
        identity, runtime = resolve_configs(self.plan, self.identity, self.checkpoint)
        self.assertEqual(identity, self.checkpoint['model'])
        self.assertNotEqual(identity['path'], runtime['path'])
        self.assertEqual(runtime['path'], self.plan['model'])
        self.assertEqual(self.identity, self.plan['training_model_identity'])

    def test_wrong_checkpoint_revision_or_depth_is_rejected(self):
        for field, value in [('revision', 'different'), ('j', 12), ('path', self.plan['model'])]:
            bad = copy.deepcopy(self.checkpoint); bad['model'][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                resolve_configs(self.plan, self.identity, bad)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.run = self.home / 'run_encbank'; self.run.mkdir()
        self.box = self.home / 'rpc'; self.box.mkdir()
        self.plan = {'arm': 'encbank', 'remote_root': str(self.home)}
        self.mailbox = Mailbox(self.run / 'mailbox', lambda: {'worker_ready.json': {'ready': True}})
        endpoint = self.mailbox.endpoint(); endpoint['host'] = '127.0.0.1'
        save(self.run / 'transport_endpoint.json', endpoint)
        self.transport = Transport(self.home, self.plan, self.home / 'execution')
        self.path = self.mailbox.path()

    def tearDown(self):
        self.mailbox.server.shutdown(); self.mailbox.server.server_close()
        self.temp.cleanup()

    def request(self, rid, task):
        row = {'request_id': rid, 'task_id': task, 'task': task, 'step': 0,
               'remaining_seconds': 5, 'messages': [{'role': 'user', 'content': 'synthetic CPU transport test'}]}
        path = self.box / (rid + '.request.json'); save(path, row)
        return path

    def respond(self, ids):
        deadline = time.monotonic() + 10
        for rid, task in ids:
            while not (self.path / (rid + '.request.json')).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Synthetic request was not delivered')
                time.sleep(.01)
            (self.path / (rid + '.response.json')).write_immutable(
                {'request_id': rid, 'task': task, 'step': 0, 'status': 'ok', 'text': task, 'generated_tokens': 0})

    def test_two_concurrent_requests_have_separate_verified_replies(self):
        class Child:
            def poll(self): return None
        with ThreadPoolExecutor(max_workers=3) as pool:
            worker = pool.submit(self.respond, [('test_b', 'b'), ('test_a', 'a')])
            one = pool.submit(self.transport.handle, self.request('test_a', 'a'), Child())
            two = pool.submit(self.transport.handle, self.request('test_b', 'b'), Child())
            one.result(timeout=15); two.result(timeout=15); worker.result(timeout=15)
        for rid, task in [('test_a', 'a'), ('test_b', 'b')]:
            reply = json.loads((self.box / (rid + '.response.json')).read_text())
            self.assertEqual(reply['text'], task)
            proof = json.loads((self.box / (rid + '.broker.json')).read_text())
            self.assertEqual(proof['status'], 'delivered')
        self.assertEqual(len(list(self.path.glob('*.request.json'))), 2)

    def test_corrupt_or_cross_run_response_is_rejected(self):
        data = json.dumps({'request_id': 'r'}).encode()
        good = {'path': str(self.run / 'mailbox/r.response.json'), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'payload_base64': base64.b64encode(data).decode()}
        self.assertEqual(verify_response(good, 'r', self.home, 'encbank'), data)
        for change in [{'sha256': '0' * 64}, {'path': '/another/run/r.response.json'}, {'bytes': 1}]:
            with self.subTest(change=change), self.assertRaises(AssertionError):
                verify_response(dict(good, **change), 'r', self.home, 'encbank')

    def test_duplicate_publish_is_not_replayed(self):
        req = json.loads(self.request('once', 'task').read_text())
        self.mailbox.dispatch({'action': 'publish', 'id': 'once', 'request': dict(req)})
        with self.assertRaises(AssertionError):
            self.mailbox.dispatch({'action': 'publish', 'id': 'once', 'request': dict(req)})
        self.assertEqual(len(list(self.path.glob('*.request.json'))), 1)


class ReservationTests(unittest.TestCase):
    def test_budget_and_pid_identity_protect_reservations(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(host_admission, 'ROOT', Path(directory)), \
             patch.object(host_admission, 'memory_headroom', return_value=100000), \
             patch.dict(host_admission.PLAN, {'host_memory_budget_mb': 4096}):
            ok, _ = host_admission.acquire('owner', 'a', 3072); self.assertTrue(ok)
            ok, _ = host_admission.acquire('owner', 'b', 2048); self.assertFalse(ok)
            with self.assertRaises(AssertionError):
                host_admission.acquire('owner', 'a', 1024)
            identity = host_admission.process_identity(os.getpid())
            self.assertTrue(host_admission.identity_alive(identity))
            self.assertFalse(host_admission.identity_alive(dict(identity, start_ticks=identity['start_ticks'] + 1)))
            host_admission.release('owner', 'a'); self.assertEqual(host_admission.read(), {})


class AssembledRunTests(unittest.TestCase):
    def test_docker_client_only_success_does_not_pass_server_preflight(self):
        from server_preflight import docker_server_version
        for stdout in ['null\n', '""\n', '{}\n']:
            result = subprocess.CompletedProcess([], 0, stdout, 'permission denied')
            with self.subTest(stdout=stdout), patch('server_preflight.subprocess.run', return_value=result), self.assertRaises(AssertionError):
                docker_server_version({})
        result = subprocess.CompletedProcess([], 0, '{"Version":"26.1.3"}', '')
        with patch('server_preflight.subprocess.run', return_value=result):
            self.assertEqual(docker_server_version({}), '26.1.3')

    def test_linux_controller_completes_two_synthetic_tasks_without_ssh_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            for name in ['server_owner.py', 'server_transport.py', 'host_admission.py',
                         'control_plane_recovery.py', 'startup_wait.py', 'live_mailbox.py']:
                shutil.copy2(RUN / name, home / name)
            (home / 'bin').mkdir()
            harbor = home / 'bin/harbor'
            harbor.write_text('#!' + sys.executable + '\n' + """import json,sys,time
from pathlib import Path
config=json.loads(Path(sys.argv[-1]).read_text());h=Path.cwd();p=json.loads((h/'plan.json').read_text())
task=config['job_name'];rid='cpu_'+task;box=Path(p['rpc_root'])/'encbank';box.mkdir(parents=True,exist_ok=True)
request={'request_id':rid,'task_id':rid,'task':task,'step':0,'remaining_seconds':5,'messages':[]}
t=box/(rid+'.tmp');t.write_text(json.dumps(request));t.replace(box/(rid+'.request.json'))
end=time.monotonic()+15
while not (box/(rid+'.response.json')).exists():
 if time.monotonic()>end:raise TimeoutError('synthetic reply timeout')
 time.sleep(.02)
result=Path(config['jobs_dir'])/task/(task+'__cpu')/'result.json';result.parent.mkdir(parents=True);result.write_text(json.dumps({'synthetic_test':True}))
""")
            harbor.chmod(0o700)
            plan = {'remote_root': str(home), 'rpc_root': str(home / 'rpc'),
                    'results_root': str(home / 'results'), 'task_root': str(home / 'tasks'),
                    'arm': 'encbank', 'tasks': ['a', 'b'], 'concurrent_tasks': 2,
                    'resource_inventory': [{'task': name, 'memory_mb': 1} for name in ['a', 'b']],
                    'host_admission_root': str(home / 'reservations'), 'host_memory_budget_mb': 1024,
                    'harbor_python': str(home / 'bin/python'), 'bootstrap_timeout_seconds': 10,
                    'docker_host': 'unix:///not-used', 'container_backend': 'synthetic-test',
                    'controller_tmp': str(home), 'controller_cache': str(home), 'container_cache': str(home)}
            save(home / 'plan.json', plan); save(home / 'encbank_harbor_template.json', {})
            run = home / 'run_encbank'; run.mkdir()
            save(run / 'worker_ready.json', {'synthetic_test': True})
            mailbox = Mailbox(run / 'mailbox', lambda: {'worker_ready.json': {'synthetic_test': True}})
            endpoint = mailbox.endpoint(); endpoint['host'] = '127.0.0.1'
            save(run / 'transport_endpoint.json', endpoint)
            errors = []
            def worker():
                try:
                    box = mailbox.path(); seen = set(); deadline = time.monotonic() + 30
                    while not (box / 'stop.json').exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError('Controller failed to stop synthetic service')
                        for path in box.glob('*.request.json'):
                            if path.name in seen:
                                continue
                            request = json.loads(path.read_text()); seen.add(path.name)
                            (box / (request['request_id'] + '.response.json')).write_immutable(
                                {'request_id': request['request_id'], 'task': request['task'], 'step': 0,
                                 'status': 'ok', 'text': 'synthetic', 'generated_tokens': 0})
                        time.sleep(.01)
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=worker, daemon=True); thread.start()
            try:
                process = subprocess.run([sys.executable, str(home / 'server_owner.py')], cwd=home,
                    env=dict(os.environ, SLURM_JOB_ID='cpu-lifecycle-test', PYTHONPATH=str(home)),
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                thread.join(timeout=5); self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
                receipt = json.loads((home / 'execution/harbor_receipt.json').read_text())
                self.assertTrue(receipt['all_task_parents_waited']); self.assertEqual(receipt['tasks_closed'], 2)
                self.assertTrue((home / 'execution/owner_complete.json').exists())
                self.assertEqual(len(list((home / 'rpc/encbank').glob('*.broker.json'))), 2)
                self.assertEqual(json.loads(next((home / 'reservations').glob('*/reservations.json')).read_text()), {})
            finally:
                mailbox.server.shutdown(); mailbox.server.server_close()

    def test_all_three_runs_have_only_server_runtime_paths_and_valid_harbor_configs(self):
        from harbor.models.job.config import JobConfig
        for directory in sorted((ROOT / 'runs').iterdir()):
            plan = json.loads((directory / 'plan.json').read_text())
            self.assertTrue(plan['server_only'])
            for key in ['remote_root', 'task_root', 'rpc_root', 'results_root', 'harbor_python', 'predecessor_root']:
                self.assertTrue(plan[key].startswith('/srv/encbank/'), (directory, key, plan[key]))
                self.assertNotIn('\\', plan[key])
            for name in ['server_owner.py', 'host_admission.py', 'server_transport.py', 'tb_agent_rpc.py', 'deploy.py', 'observe.py']:
                code = (directory / name).read_text()
                for prohibited in ['F:/', '/mnt/f/', "['wsl'", 'WinDLL', 'CREATE_NO_WINDOW', 'import msvcrt']:
                    self.assertNotIn(prohibited, code, (directory.name, name, prohibited))
                compile(code, str(directory / name), 'exec')
            config = json.loads((directory / (plan['arm'] + '_harbor_template.json')).read_text())
            JobConfig.model_validate(config)
            self.assertEqual(config['retry']['max_retries'], 0)
            self.assertEqual(config['environment']['type'], 'singularity')
            self.assertEqual(config['environment']['import_path'], 'apptainer_environment:ManagedApptainerEnvironment')
            from harbor.utils.import_path import import_class
            backend=import_class(config['environment']['import_path'],label='environment')
            self.assertTrue(backend.resource_capabilities().memory_limit)
            self.assertFalse(backend.resource_capabilities().cpu_limit)
            self.assertNotIn('windows_owner', (directory / 'server.slurm').read_text())
            subprocess.run(['bash', '-n', str(directory / 'server.slurm')], check=True)


if __name__ == '__main__':
    assert sys.platform == 'linux', 'Run this test on the server'
    unittest.main(verbosity=2)
