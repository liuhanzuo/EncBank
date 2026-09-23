"""Stdlib-only checks for the explicit L20D migration and submission boundary."""
from argparse import Namespace
import copy
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
import unittest
from unittest.mock import patch

from slurm_gpu_guard import gpu_environment
from submit_slurm_sparse import (check_other_jobs, logical_path, migration_receipt, submission,
    TASK_ROOT, SOURCE_ROOT, SOURCE_CODE, SOURCE_OUT, MIGRATION_SCHEMA, JOB_NAME)


def receipt_fixture(now=None):
    now = now or datetime.now(timezone.utc)
    ids = [f'q{i}' for i in range(99)]
    jobs, tasks = {}, []
    for stage in ('smoke', 'train'):
        for arm in ('D0', 'A', 'B', 'D1'):
            key = stage+'/'+arm
            leaf = 'D0_retry1' if key == 'smoke/D0' else arm
            source_out = str(SOURCE_OUT/stage/leaf)
            job = dict(stage=stage, arm=arm, phase='queued', out=source_out,
                steps=1 if stage == 'smoke' else 250, grad_accum=2 if stage == 'smoke' else 4,
                dev_ids=ids[:1] if stage == 'smoke' else ids)
            jobs[key] = job
            tasks.append(dict(stage=stage, arm=arm, source_job_key=key, source_out=source_out,
                destination_relative_out=f'outputs/sparse_comem_20260911/{stage}/{leaf}',
                steps=job['steps'], grad_accum=job['grad_accum'], dev_ids=list(job['dev_ids'])))
    return dict(schema=MIGRATION_SCHEMA, checked_utc=now.isoformat(), safe_to_submit=True,
        destination_backend='slurm-l20d', destination_task_root=str(TASK_ROOT),
        destination_resource=dict(partition='gpu', gres='gpu:nvidia_l20d:1', gpu_count=1, gpu_model='L20D'),
        source_backend='remote-3090', source_host='longjing-1', actual_host='longjing-2',
        source_controller=dict(pid=3297308, start_ticks=978365947,
            argv=[str(SOURCE_ROOT/'venv/bin/python'), '-u', str(SOURCE_CODE/'remote_sparse_queue.py'),
                  '--out', str(SOURCE_OUT), '--allow-training', '--retry-failed-smoke', 'D0']),
        controller_alive=False, live_trainers=[], source_queue_path=str(SOURCE_OUT/'queue.json'),
        source_queue_sha256='a'*64, source_jobs=jobs, tasks=tasks)


def submission_fixture():
    root = str(TASK_ROOT)
    return Namespace(task_root=root, worker=root+'/code/slurm_sparse_worker.py',
        trainer=root+'/code/train_sparse_slurm.py', model=root+'/model', init_adapter=root+'/adapter.pt',
        data_dir=root+'/data', out=root+'/outputs/sparse_comem_20260911', python='/usr/local/bin/python',
        partition='gpu', gres='gpu:nvidia_l20d:1', account=None, qos=None, cpus=4, memory='64G',
        time_limit='24:00:00', steps=250, grad_accum=4, allow_training=True,
        migration_receipt=root+'/logs/migration.json')


class SlurmAdmissionTests(unittest.TestCase):
    def test_one_gpu_only(self):
        valid = dict(SLURM_JOB_ID='123', CUDA_VISIBLE_DEVICES='GPU-abc', SLURM_GPUS_ON_NODE='1', SLURM_NTASKS='1')
        self.assertEqual(gpu_environment(valid)['allocated_gpus'], 1)
        for bad in (dict(CUDA_VISIBLE_DEVICES='0,1'), dict(SLURM_GPUS_ON_NODE='2'),
                    dict(SLURM_NTASKS='2'), dict(SLURM_JOB_ID=''),
                    dict(CUDA_VISIBLE_DEVICES=''), dict(CUDA_VISIBLE_DEVICES='all'),
                    dict(SLURM_ARRAY_JOB_ID='123')):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                gpu_environment({**valid, **bad})

    def test_no_path_escape(self):
        expected = '/srv/encbank/pilot/file'
        self.assertEqual(logical_path(expected), PurePosixPath(expected))
        for path in ('/tmp/file', '/srv/encbank', '/srv/encbank-other/f',
                     '/srv/encbank/pilot/../../other/f', 'relative/file',
                     '/srv/encbank/pilot/a\nfile'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                logical_path(path)

    def test_fresh_drained_migration_receipt(self):
        now = datetime.now(timezone.utc)
        valid = receipt_fixture(now)
        self.assertIs(migration_receipt(valid, now), valid)
        for change in (dict(checked_utc=(now-timedelta(hours=2)).isoformat()),
                       dict(checked_utc=(now+timedelta(minutes=1)).isoformat()),
                       dict(safe_to_submit=False), dict(source_jobs={}), dict(tasks=[]),
                       dict(controller_alive=True), dict(live_trainers=[123]),
                       dict(source_backend='5090'), dict(source_host='local'),
                       dict(destination_backend='slurm-b300')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                migration_receipt({**valid, **change}, now)

    def test_slurm_wait_does_not_expire_valid_drained_handoff(self):
        now=datetime.now(timezone.utc)
        value=receipt_fixture(now-timedelta(days=2))
        self.assertIs(migration_receipt(value,now,require_fresh=False),value)
        value['controller_alive']=True
        with self.assertRaises(ValueError):migration_receipt(value,now,require_fresh=False)

    def test_source_identity_and_real_training_queue_are_required(self):
        for field, value in (('source_controller',{}),('source_queue_path','/local/5090/queue.json'),
                             ('source_queue_sha256',''),('destination_task_root','/srv/encbank/other')):
            receipt=receipt_fixture();receipt[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):migration_receipt(receipt)
        for field,value in (('pid',0),('start_ticks',False),('argv',['python','local_gpu.py'])):
            receipt=receipt_fixture();receipt['source_controller'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):migration_receipt(receipt)
        old=dict(checked_utc=datetime.now(timezone.utc).isoformat(),destination_backend='slurm-l20d',
                 safe_to_submit=True,sources=[dict(backend='5090',controller_alive=False,live_trainers=[])])
        with self.assertRaises(ValueError):migration_receipt(old)

    def test_manifest_preserves_queued_source_budgets_ids_and_retry1(self):
        for mutation in (
            lambda r:r['source_jobs']['smoke/D0'].update(phase='failed'),
            lambda r:r['source_jobs']['train/B'].update(phase='complete'),
            lambda r:r['tasks'][0].update(source_out=str(SOURCE_OUT/'smoke/D0')),
            lambda r:r['tasks'][0].update(destination_relative_out='outputs/sparse_comem_20260911/smoke/D0'),
            lambda r:r['tasks'][0].update(destination_relative_out='../outside/smoke/D0_retry1'),
            lambda r:r['tasks'][0].update(steps=2),
            lambda r:r['tasks'][4].update(dev_ids=['different']),
            lambda r:r['tasks'].__setitem__(1,copy.deepcopy(r['tasks'][0]))):
            receipt=receipt_fixture();mutation(receipt)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):migration_receipt(receipt)

    def test_single_serial_script_and_localized_caches(self):
        root = str(TASK_ROOT)
        args = submission_fixture()
        cmd, script, locations = submission(args)
        self.assertEqual(cmd[cmd.index('--gres')+1], 'gpu:nvidia_l20d:1')
        self.assertEqual(cmd[cmd.index('--partition')+1], 'gpu')
        self.assertEqual(cmd[cmd.index('--ntasks')+1], '1')
        self.assertNotIn('--array', cmd)
        self.assertEqual(script.count('\nexec '), 1)
        self.assertIn('--allow-training', script)
        self.assertIn('--migration-receipt '+args.migration_receipt, script)
        self.assertIn('/usr/local/bin/python -B', script)
        self.assertNotIn('export HOME=', script)
        for location in locations.values():
            self.assertIn(PurePosixPath(root), location.parents)
        args.gres = 'gpu:nvidia_l20d:2'
        with self.assertRaises(ValueError):
            submission(args)

    def test_unrelated_same_account_gpu_jobs_do_not_block_submit(self):
        for detail in ('JobId=9 ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1',
                       'JobId=9 ReqTRES=cpu=4,gres/gpu:nvidia_l20d=1',
                       'JobId=9 AllocTRES=cpu=4,gres/gpu=1'):
            with self.subTest(detail=detail), patch('subprocess.check_output', side_effect=[
                '9|unrelated|RUNNING\n', detail+' WorkDir=/srv/encbank/other Command=/srv/encbank/other/run.sh']):
                check_other_jobs()
        with patch('subprocess.check_output', side_effect=['9|cpu-only|RUNNING\n', 'JobId=9 WorkDir=/tmp Command=/tmp/cpu.sh']):
            check_other_jobs()
        with patch('subprocess.check_output', side_effect=['9|unknown|RUNNING\n', 'JobId=9 inaccessible']):
            with self.assertRaises(RuntimeError):
                check_other_jobs()
        with patch('subprocess.check_output', return_value=''):
            check_other_jobs()

    def test_only_task_name_root_or_exact_worker_duplicates_block(self):
        with patch('subprocess.check_output',return_value='9|'+JOB_NAME+'|PENDING\n'):
            with self.assertRaisesRegex(RuntimeError,'duplicate sparse-CoMem'):check_other_jobs()
        for detail in ('JobId=9 WorkDir='+str(TASK_ROOT)+' Command=/tmp/script',
                       'JobId=9 WorkDir='+str(TASK_ROOT)+'/subdir Command=/tmp/script',
                       'JobId=9 WorkDir=/tmp Command='+str(TASK_ROOT)+'/code/slurm_sparse_worker.py'):
            with self.subTest(detail=detail),patch('subprocess.check_output',side_effect=['9|renamed|RUNNING\n',detail]):
                with self.assertRaisesRegex(RuntimeError,'duplicate sparse-CoMem'):check_other_jobs()
        with patch('subprocess.check_output',side_effect=['9|different|RUNNING\n',
                'JobId=9 WorkDir='+str(TASK_ROOT)+'_other Command=/tmp/script']):
            check_other_jobs()

    def test_render_rejects_unknown_resource_root_trainer_or_missing_receipt(self):
        for field,value in (('partition','other'),('gres','gpu:b300:1'),('task_root','/srv/encbank/other'),
                            ('trainer',str(TASK_ROOT)+'/code/train_sparse.py'),('migration_receipt',None),
                            ('allow_training',False),('steps',1)):
            args=submission_fixture();setattr(args,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):submission(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
