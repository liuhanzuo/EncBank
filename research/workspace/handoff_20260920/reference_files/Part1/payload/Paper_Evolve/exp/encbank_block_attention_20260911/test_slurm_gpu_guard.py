"""Stdlib-only Slurm guard checks. NVIDIA/Slurm observations are fixtures."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import slurm_gpu_guard as guard

UUID = '9fbcbd52-d74c-723c-e8d7-3ca9d4ed2fc7'
OTHER = '11111111-2222-3333-4444-555555555555'


def row(value=UUID, used=0, processes=None):
    return dict(uuid='GPU-'+value, name='NVIDIA L20D', used_mib=used,
                processes_known=True, processes=[] if processes is None else processes)


class GuardUnitTests(unittest.TestCase):
    def test_actual_torch_uuid_and_nvml_prefix_are_same(self):
        expected = 'gpu-'+UUID
        for value in (UUID, 'GPU-'+UUID, expected, ('GPU-'+UUID).upper()):
            self.assertEqual(guard.normalize_uuid(value), expected)
        for value in ('0', 'GPU-abc', 'MIG-'+UUID, UUID[:23]+UUID[28:]):
            with self.assertRaises(RuntimeError):
                guard.normalize_uuid(value)

    def test_exactly_one_allocation_no_array(self):
        env = dict(SLURM_JOB_ID='23983', SLURM_GPUS_ON_NODE='1', SLURM_NTASKS='1', CUDA_VISIBLE_DEVICES='0')
        self.assertEqual(guard.gpu_environment(env)['allocated_gpus'], 1)
        for key, value in (('SLURM_GPUS_ON_NODE','8'), ('SLURM_NTASKS','2'),
                           ('CUDA_VISIBLE_DEVICES','0,1'), ('SLURM_ARRAY_JOB_ID','12')):
            with self.assertRaises(RuntimeError):
                guard.gpu_environment({**env, key:value})

    def test_uuid_selection_ignores_other_allocations(self):
        foreign = dict(pid='7', type='G', process_name='python')
        selected = guard.check_device([row(OTHER, 20000, [foreign]), row()], UUID, require_idle=True)
        self.assertEqual(selected['used_mib'], 0)
        for kind in ('C', 'G', 'C+G', 'M', 'unknown'):
            with self.assertRaises(RuntimeError):
                guard.check_device([row(processes=[{**foreign, 'type':kind}])], UUID, require_idle=True)
        self.assertEqual(guard.check_device([row(used=20000, processes=[foreign])], UUID,
                                            allowed_pids={7})['used_mib'], 20000)

    def test_threshold_and_unknown_inventory(self):
        guard.check_device([row(used=511)], UUID, require_idle=True)
        for sample in (row(used=512), {**row(), 'processes_known':False},
                       {**row(), 'name':'NVIDIA B300'}, {**row(), 'used_mib':None}):
            with self.assertRaises(RuntimeError):
                guard.check_device([sample], UUID, require_idle=True)
        with self.assertRaises(RuntimeError):
            guard.check_device([row(), row()], UUID, require_idle=True)

    def test_reserved_memory_is_not_used_memory(self):
        xml = ('<nvidia_smi_log><gpu><product_name>NVIDIA L20D</product_name>'
               f'<uuid>GPU-{UUID}</uuid><fb_memory_usage><used>0 MiB</used>'
               '<reserved>927 MiB</reserved></fb_memory_usage><processes/></gpu></nvidia_smi_log>')
        self.assertEqual(guard.check_device(guard.gpu_inventory(xml), UUID, require_idle=True)['used_mib'], 0)

    def test_allocation_gpu_totals_are_not_double_counted(self):
        self.assertEqual(guard.job_gpu_count('cpu=4,gres/gpu=1,gres/gpu:nvidia_l20d=1'), 1)
        for value in ('cpu=4', 'gres/gpu=2', 'gres/gpu=1,gres/gpu:nvidia_l20d=2',
                      'gres/gpu:a=1,gres/gpu:b=1'):
            with self.assertRaises(RuntimeError):
                guard.job_gpu_count(value)

    def test_site_shared_cgroup_needs_exact_slurm_membership(self):
        cgroup = '1:name=cgp:/\n0::/system.slice/slurmd.service\n'
        stepd = dict(pid=3410006, start_ticks=123, argv=['slurmstepd: [23996.batch]'])
        with patch.object(guard, '_output', return_value='PID JOBID STEPID LOCALID GLOBALID\n3410022 23996 batch 0 0'), \
             patch.object(guard, 'stepd_ancestor', return_value=stepd):
            self.assertEqual(guard.slurm_membership('23996', 3410022, cgroup)['stepd'], stepd)
        for line in ('3410022 999 batch 0 0', '3410023 23996 batch 0 0', ''):
            with patch.object(guard, '_output', return_value=line), self.assertRaises(RuntimeError):
                guard.slurm_membership('23996', 3410022, cgroup)
        self.assertTrue(guard.cgroup_job_matches('0::/slurm/job_23996/step_batch', '23996'))
        self.assertFalse(guard.cgroup_job_matches('0::/slurm/job_239960/step_batch', '23996'))


@unittest.skipUnless(sys.platform == 'linux', 'Actual inherited flock and /proc require Linux')
class LinuxLeaseTests(unittest.TestCase):
    def setUp(self):
        import fcntl
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.root = self.home/'task'; self.root.mkdir()
        self.home_patch = patch.object(guard, 'CLUSTER_HOME', self.home)
        self.home_patch.start()
        self.lock = guard.lock_path(self.root).open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.script = self.root/'cpu_child.py'
        self.script.write_text(
            'import json,os,sys\nfrom pathlib import Path\n'
            f'sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})\n'
            'import slurm_gpu_guard as g\n'
            'g.CLUSTER_HOME=Path(os.environ["TEST_HOME"])\n'
            'g.cgroup_for=lambda pid:"0::/slurm/job_123/step_batch"\n'
            'g.inventory=lambda:json.loads(os.environ["TEST_INVENTORY"])\n'
            'if os.environ.get("TEST_SLOW_SHARED_READ"):\n'
            ' import time\n'
            ' g.MAX_HEARTBEAT_AGE=.1\n'
            ' original=Path.read_text\n'
            ' def slow_read(path,*args,**kwargs):\n'
            '  if str(path)==os.environ["SPARSE_GPU_LEASE_PATH"]: time.sleep(.2)\n'
            '  return original(path,*args,**kwargs)\n'
            ' Path.read_text=slow_read\n'
            'try:\n'
            ' r=g.validate_worker_lease(os.environ["SPARSE_GPU_LEASE_PATH"],require_idle=True)\n'
            ' print(json.dumps({"passed":True,"parent":r["lease"]["controller"]["pid"]}),flush=True)\n'
            'except Exception as e:\n'
            ' print(json.dumps({"passed":False,"error":str(e)}),flush=True)\n', encoding='utf-8')
        self.lease_path = self.root/'lease.json'
        self.allocation = dict(job_id='123',cuda_visible_devices='0',allocated_gpus=1,uid=os.getuid(),
            controller_cgroup='0::/slurm/job_123/step_batch', controller_membership={'kind':'job-cgroup'})
        self.value = guard.write_lease(self.lease_path, task_root=self.root, allocation=self.allocation,
            gpu_uuid=UUID, lock_fd=self.lock.fileno(), worker_script=self.script, run_id='cpu-test')
        self.env = {**os.environ, 'TEST_HOME':str(self.home), 'TEST_INVENTORY':json.dumps([row()]),
            'SPARSE_SLURM_TASK_ROOT':str(self.root), 'SPARSE_GPU_LEASE_PATH':str(self.lease_path),
            'SPARSE_GPU_LOCK_FD':str(self.lock.fileno()), 'SPARSE_SLURM_RUN_ID':'cpu-test',
            'SLURM_JOB_ID':'123', 'SLURM_GPUS_ON_NODE':'1','SLURM_NTASKS':'1','CUDA_VISIBLE_DEVICES':'0',
            'PYTHONDONTWRITEBYTECODE':'1'}
        self.env.pop('SLURM_ARRAY_JOB_ID', None)
        self.env.pop('SPARSE_SLURM_HEARTBEAT_FD', None)

    def tearDown(self):
        self.lock.close(); self.home_patch.stop(); self.temp.cleanup()

    def child(self, env=None, extra_fds=()):
        result = subprocess.run([sys.executable,'-B',str(self.script)], env=self.env if env is None else env,
            capture_output=True,text=True,timeout=10,pass_fds=(self.lock.fileno(),*extra_fds))
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_real_inherited_lock_parent_identity_and_idle_lease(self):
        result = self.child()
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['parent'],os.getpid())

    def test_stale_or_changed_parent_is_rejected(self):
        for mutate in ('stale','parent'):
            value = copy.deepcopy(self.value)
            if mutate == 'stale': value['updated_unix_s'] -= 30
            else: value['controller']['start_ticks'] += 1
            guard.publish(self.lease_path,value)
            result=self.child()
            self.assertFalse(result['passed'],result)

    def test_wrong_descriptor_visibility_or_external_process_is_rejected(self):
        with (self.root/'other.lock').open('a') as other:
            result=self.child({**self.env,'SPARSE_GPU_LOCK_FD':str(other.fileno())},(other.fileno(),))
            self.assertFalse(result['passed'])
        self.assertFalse(self.child({**self.env,'CUDA_VISIBLE_DEVICES':'1'})['passed'])
        busy=row(processes=[dict(pid='987654',type='G',process_name='python')])
        self.assertFalse(self.child({**self.env,'TEST_INVENTORY':json.dumps([busy])})['passed'])

    def test_anonymous_heartbeat_survives_old_metadata_and_slow_shared_disk(self):
        from slurm_sparse_worker import LeaseHeartbeat
        transport = guard.AnonymousHeartbeat()
        heart = LeaseHeartbeat(transport.refresh, interval=.005)
        try:
            value = guard.write_lease(self.lease_path, task_root=self.root, allocation=self.allocation,
                gpu_uuid=UUID, lock_fd=self.lock.fileno(), worker_script=self.script, run_id='cpu-test',
                heartbeat=transport.description)
            value['updated_unix_s'] -= 3600
            guard.publish(self.lease_path, value)
            heart.start()
            result = self.child({**self.env, 'SPARSE_SLURM_HEARTBEAT_FD': str(transport.fd),
                                  'TEST_SLOW_SHARED_READ': '1'}, (transport.fd,))
            self.assertTrue(result['passed'], result)
        finally:
            heart.close()
            transport.close()

    def test_new_child_rejects_heartbeat_fd_substitution_or_metadata_downgrade(self):
        transport = guard.AnonymousHeartbeat()
        other = guard.AnonymousHeartbeat()
        try:
            value = guard.write_lease(self.lease_path, task_root=self.root, allocation=self.allocation,
                gpu_uuid=UUID, lock_fd=self.lock.fileno(), worker_script=self.script, run_id='cpu-test',
                heartbeat=transport.description)
            env = {**self.env, 'SPARSE_SLURM_HEARTBEAT_FD': str(transport.fd)}
            self.assertTrue(self.child(env, (transport.fd,))['passed'])
            self.assertFalse(self.child({**env, 'SPARSE_SLURM_HEARTBEAT_FD': str(other.fd)},
                                        (other.fd,))['passed'])
            del value['heartbeat']
            guard.publish(self.lease_path, value)
            self.assertFalse(self.child(env, (transport.fd,))['passed'])
        finally:
            transport.close()
            other.close()


@unittest.skipUnless(sys.platform == 'linux', 'Actual anonymous memfd requires Linux')
class LinuxHeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.heart = guard.AnonymousHeartbeat()
        self.env = {'SPARSE_SLURM_HEARTBEAT_FD': str(self.heart.fd)}

    def tearDown(self):
        self.heart.close()

    def age(self, description=None):
        return guard.anonymous_heartbeat_age(
            self.heart.description if description is None else description, os.getpid(), self.env)

    def test_monotonic_age_does_not_depend_on_wall_clock(self):
        with patch.object(guard.time, 'time', return_value=1e20):
            self.assertLess(self.age(), 1.)

    def test_stopped_writer_and_future_timestamp_fail_closed_at_existing_threshold(self):
        # A freshly started WSL/Linux host can have <21 seconds of uptime.
        now = 100_000_000_000
        self.assertEqual(guard.MAX_HEARTBEAT_AGE, 20.)
        for offset in (-21_000_000_000, 1_000_000_000):
            with patch.object(guard.time, 'monotonic_ns', return_value=now + offset):
                self.heart.refresh()
            with patch.object(guard.time, 'monotonic_ns', return_value=now):
                with self.assertRaisesRegex(RuntimeError, 'heartbeat is stale'):
                    self.age()

    def test_uncommitted_torn_tick_is_never_accepted(self):
        for first, last in ((3, 4), (4, 2), (0, 0)):
            guard._HEARTBEAT_TICK.pack_into(self.heart.memory, guard._HEARTBEAT_OFFSET,
                                           first, time.monotonic_ns(), last)
            with self.assertRaisesRegex(RuntimeError, 'stable committed tick'):
                self.age()

    def test_wrong_parent_token_object_or_missing_inherited_descriptor_is_rejected(self):
        for key, changed in (('parent_pid', os.getpid()+1), ('token', '00'*16),
                             ('inode', self.heart.description['inode']+1), ('fd', -1)):
            with self.assertRaises(RuntimeError):
                self.age({**self.heart.description, key: changed})
        with self.assertRaises(RuntimeError):
            guard.anonymous_heartbeat_age(self.heart.description, os.getpid(), {})

    def test_heartbeat_object_size_is_sealed(self):
        with self.assertRaises(OSError):
            os.ftruncate(self.heart.fd, guard.HEARTBEAT_BYTES * 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
