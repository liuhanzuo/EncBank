"""Serial sparse-Encbank training on one observed L20D Slurm allocation.

Only an explicit migrated task list is executed. Original training and 3090
queue sources remain unchanged. No formal inference timing is measured.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from remote_sparse_queue import ARMS, completed_job, process_identity, publish, train_command, utc_now
from slurm_gpu_guard import (AnonymousHeartbeat, CLUSTER_HOME, allocation_identity, bind_owned_process, check_device,
    confined, gpu_environment, inventory, lock_path, normalize_uuid, terminate_owned, write_lease)


def runtime_environment(task_root):
    runtime = task_root/'runtime'
    locations = {
        'TMPDIR': runtime/'tmp', 'TMP': runtime/'tmp', 'TEMP': runtime/'tmp',
        'XDG_CACHE_HOME': runtime/'cache', 'HF_HOME': runtime/'cache/huggingface',
        'TORCH_HOME': runtime/'cache/torch', 'TRITON_CACHE_DIR': runtime/'cache/triton',
        'CUDA_CACHE_PATH': runtime/'cache/cuda', 'MPLCONFIGDIR': runtime/'cache/matplotlib',
        'PYTHONPYCACHEPREFIX': runtime/'cache/pycache',
        'WANDB_DIR': runtime/'wandb', 'WANDB_CACHE_DIR': runtime/'cache/wandb',
    }
    for path in locations.values():
        confined(path, task_root).mkdir(parents=True, exist_ok=True)
    return {**os.environ, **{key: str(value) for key, value in locations.items()},
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'OPENBLAS_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'SPARSE_ENCBANK_QUEUE_TASK': 'sparse_encbank_20260911_slurm',
            'SPARSE_SLURM_TASK_ROOT': str(task_root)}


class LeaseHeartbeat:
    """Refresh anonymous memory independently of NVIDIA and shared-disk calls."""
    def __init__(self, writer, interval=1.):
        self.writer, self.interval = writer, interval
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='slurm-gpu-lease')

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.writer()
            except BaseException as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                return

    def start(self):
        self.writer()
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Lease writer did not stop; retain lock ownership')


def supervise(command, *, cwd, env, log, lock_fd, stopped, timeout=None,
              on_start=None, monitor=None, extra_fds=()):
    """Exceptions after Popen clean up the exact owned child before returning."""
    proc = None
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                pass_fds=(lock_fd, *extra_fds), start_new_session=True)
        bind_owned_process(proc)
        if on_start is not None:
            on_start(proc)
        while proc.poll() is None:
            if stopped.is_set():
                raise InterruptedError('Slurm worker interrupted; no promotion')
            if timeout is not None and time.monotonic()-started > timeout:
                raise TimeoutError('Owned subprocess exceeded its explicit time budget')
            if monitor is not None:
                monitor(proc)
            stopped.wait(1.)
        return proc.returncode
    finally:
        if proc is not None and proc.poll() is None:
            terminate_owned(proc)
        if proc is not None and proc.poll() is None:
            raise RuntimeError('Owned child remained live after cleanup')


def ensure_cpu_validation(args, env, allocation, lock_fd, stopped):
    code = args.trainer.parent
    names = ('validate_cpu.py', 'test_sparse_reader.py', 'sparse_reader.py', 'remote_sparse_queue.py',
             'slurm_gpu_guard.py', 'slurm_sparse_worker.py', 'train_sparse_slurm.py',
             'train_sparse.py', 'test_slurm_gpu_guard.py', 'test_slurm_worker.py')
    sources = [confined(code/name, args.task_root) for name in names]
    identity = dict(files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    job_id=allocation['job_id'], python_executable=sys.executable)
    cpu_env = {**env, 'CUDA_VISIBLE_DEVICES': ''}
    outcomes = []
    commands = ([sys.executable, '-B', str(code/'validate_cpu.py')],
                [sys.executable, '-B', '-m', 'unittest', 'test_slurm_gpu_guard', 'test_slurm_worker', '-v'])
    for index, command in enumerate(commands):
        log_path = confined(args.out/f'slurm_cpu_validation_{index}.log', args.task_root)
        with log_path.open('w', encoding='utf-8') as log:
            rc = supervise(command, cwd=code, env=cpu_env, log=log, lock_fd=lock_fd,
                           stopped=stopped, timeout=1300)
        outcomes.append(dict(command=command, returncode=rc, log=str(log_path)))
        if rc:
            break
    source = code/'results/cpu_checks.json'
    reader_receipt = json.loads(source.read_text()) if source.is_file() else {}
    passed = (len(outcomes) == 2 and all(row['returncode'] == 0 for row in outcomes)
              and reader_receipt.get('passed') is True and reader_receipt.get('cuda_visible_devices') == ''
              and reader_receipt.get('reader_sha256') == identity['files']['sparse_reader.py'])
    receipt = dict(passed=passed, identity=identity, checks=outcomes, checked_utc=utc_now(),
                   source_receipt=str(source), formal_inference_timing=False)
    publish(args.out/'slurm_cpu_validation.json', receipt)
    if not passed:
        raise RuntimeError('New Slurm environment CPU checks failed; no GPU model starts')
    return receipt


def probe_device(args, env, lock_fd, stopped):
    # Runs inside a checked allocation. UUID is reported by the visible CUDA
    # device itself; a CVD ordinal is never treated as a physical NVIDIA index.
    program = (
        'import json,sys,torch,transformers; n=torch.cuda.device_count(); '
        'assert n==1,"Exactly one CUDA device required"; p=torch.cuda.get_device_properties(0); '
        'print(json.dumps(dict(count=n,name=p.name,uuid=str(getattr(p,"uuid","")),'
        'capability=list(torch.cuda.get_device_capability(0)),bf16=torch.cuda.is_bf16_supported(),'
        'torch_version=torch.__version__,cuda_version=torch.version.cuda,'
        'transformers_version=transformers.__version__,python_version=sys.version)))')
    path = args.out/'slurm_device_probe.log'
    with path.open('w', encoding='utf-8') as log:
        rc = supervise([sys.executable, '-B', '-c', program], cwd=args.task_root, env=env,
                       log=log, lock_fd=lock_fd, stopped=stopped, timeout=90)
    if rc:
        raise RuntimeError('Allocated CUDA device probe failed')
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and 'uuid' in row:
            records.append(row)
    if len(records) != 1:
        raise RuntimeError('CUDA probe did not return one unambiguous identity')
    gpu = records[0]
    gpu['uuid'] = normalize_uuid(gpu['uuid'])
    if gpu.get('count') != 1 or 'L20D' not in gpu.get('name', '').upper() or gpu.get('bf16') is not True:
        raise RuntimeError('Require one observed L20D supporting the fixed BF16 recipe')
    gpu['idle_snapshot'] = check_device(inventory(), gpu['uuid'], require_idle=True)
    publish(args.out/'slurm_device.json', gpu)
    return gpu


def migrated_jobs(args, receipt, dev_ids):
    jobs = {}
    for task in receipt['tasks']:
        name = task['source_job_key']
        destination = confined(args.task_root/task['destination_relative_out'], args.out)
        expected_ids = dev_ids[:1] if task['stage'] == 'smoke' else dev_ids
        steps = 1 if task['stage'] == 'smoke' else args.steps
        accum = 2 if task['stage'] == 'smoke' else args.grad_accum
        if task['dev_ids'] != expected_ids or task['steps'] != steps or task['grad_accum'] != accum:
            raise RuntimeError('Migration task differs from prepared data or fixed training budget')
        if name in jobs:
            raise RuntimeError('Duplicate migration task')
        jobs[name] = {**task, 'phase': 'queued', 'out': str(destination)}
    expected = [f'{stage}/{arm}' for stage in ('smoke', 'train') for arm in ARMS]
    if list(jobs) != expected:
        raise RuntimeError('This migration requires the recorded eight pending tasks in fixed order')
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('task-root', 'trainer', 'model', 'init-adapter', 'data-dir', 'out', 'migration-receipt'):
        ap.add_argument('--'+name, required=True, type=Path)
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--allow-training', action='store_true')
    args = ap.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('This entry point runs inside a Linux Slurm allocation only')
    import fcntl
    from submit_slurm_sparse import migration_receipt
    args.task_root = confined(args.task_root)
    for field in ('trainer', 'model', 'init_adapter', 'data_dir', 'out', 'migration_receipt'):
        setattr(args, field, confined(getattr(args, field), args.task_root))
    confined(__file__, args.task_root)
    if args.trainer.name != 'train_sparse_slurm.py' or min(args.steps, args.grad_accum) < 1:
        raise ValueError('Use the explicit Slurm wrapper and positive unchanged budgets')
    for path in (args.trainer, args.trainer.parent/'train_sparse.py', args.model/'config.json',
                 args.init_adapter, args.data_dir/'train.jsonl', args.data_dir/'dev.jsonl'):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = args.migration_receipt.read_bytes()
    migration = migration_receipt(json.loads(raw), require_fresh=False)
    if migration['destination_task_root'] != str(args.task_root):
        raise RuntimeError('Migration destination differs from this worker root')
    rows = [json.loads(line) for line in (args.data_dir/'dev.jsonl').read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    dev_ids = [row['id'] for row in rows]
    if len(dev_ids) != 99 or len(set(dev_ids)) != 99:
        raise ValueError('Use the same prepared 99-example development split')
    jobs = migrated_jobs(args, migration, dev_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    env = runtime_environment(args.task_root)
    os.environ.update(env)
    os.chdir(args.task_root)
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    config = {key: str(getattr(args, key)) for key in
              ('task_root', 'trainer', 'model', 'init_adapter', 'data_dir', 'out')}
    config.update(steps=args.steps, grad_accum=args.grad_accum, migration_sha256=hashlib.sha256(raw).hexdigest(),
                  formal_inference_timing=False, scope='Slurm L20D training and accuracy only')
    old_path = args.out/'queue.json'
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    if old and old.get('config') != config:
        raise RuntimeError('Output belongs to another migration/configuration')
    for name, job in jobs.items():
        previous = old.get('jobs', {}).get(name, {})
        if previous.get('phase') == 'complete' and previous.get('returncode') == 0 and completed_job(job):
            job.update(phase='complete', returncode=0)
        elif Path(job['out']).exists():
            raise RuntimeError(f'{name} has previous artifacts; no implicit retry or overwrite')
    allocation = None

    def persist(reason):
        publish(old_path, dict(controller=process_identity(os.getpid()), host=os.uname().nodename,
            updated_utc=utc_now(), reason=reason, config=config, allocation=allocation, jobs=jobs,
            complete=all(job['phase'] == 'complete' for job in jobs.values())))

    with lock_path(args.task_root).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allocation = allocation_identity(env)
        persist('validating_new_environment_cpu')
        ensure_cpu_validation(args, env, allocation, lock.fileno(), stopped)
        gpu = probe_device(args, env, lock.fileno(), stopped)
        allocation['device'] = gpu
        persist('validated_single_gpu_allocation')
        for name, job in jobs.items():
            if stopped.is_set():
                persist('interrupted_no_promotion')
                return 1
            if job['phase'] == 'complete':
                continue
            if job['stage'] == 'train':
                if not args.allow_training:
                    persist('smokes_complete_training_not_enabled')
                    return 0
                if not all(jobs['smoke/'+arm]['phase'] == 'complete' for arm in ARMS):
                    raise RuntimeError('All four complete smokes are required before training')
            destination = Path(job['out'])
            destination.mkdir(parents=True, exist_ok=False)
            lease_path, run_id = destination/'gpu_lease.json', uuid.uuid4().hex
            lease_args = dict(path=lease_path, task_root=args.task_root, allocation=allocation,
                              gpu_uuid=gpu['uuid'], lock_fd=lock.fileno(), worker_script=args.trainer, run_id=run_id)
            transport = AnonymousHeartbeat()
            heartbeat = LeaseHeartbeat(transport.refresh)
            command = train_command(args, job['arm'], job['stage'], destination)
            worker_env = {**env, 'SPARSE_GPU_LEASE_PATH': str(lease_path),
                          'SPARSE_GPU_LOCK_FD': str(lock.fileno()), 'SPARSE_SLURM_RUN_ID': run_id,
                          'SPARSE_SLURM_HEARTBEAT_FD': str(transport.fd)}
            job.update(command=command, phase='launching', started_utc=utc_now(), run_id=run_id,
                       slurm_job_id=allocation['job_id'], gpu_uuid=gpu['uuid'])
            try:
                job['admission'] = check_device(inventory(), gpu['uuid'], require_idle=True)
                heartbeat.start()
                # Immutable metadata is published once. No Path.resolve/stat,
                # JSON write or shared-filesystem operation runs in the ticker.
                write_lease(**lease_args, heartbeat=transport.description)
                persist('worker_launching')

                def launched(proc):
                    job.update(phase='running', pid=proc.pid, process=process_identity(proc.pid))
                    persist('worker_running')

                def monitor(proc):
                    if heartbeat.error:
                        raise RuntimeError('Lease heartbeat failed: '+heartbeat.error)
                    snapshot = inventory()
                    try:
                        observed = check_device(snapshot, gpu['uuid'], allowed_pids={proc.pid})
                    except Exception:
                        job['guard_failure'] = dict(observed_utc=utc_now(), snapshot=snapshot)
                        raise
                    job['last_gpu_observation'] = observed

                with (destination/'worker.log').open('a', encoding='utf-8') as log:
                    rc = supervise(command, cwd=args.task_root, env=worker_env, log=log,
                                   lock_fd=lock.fileno(), stopped=stopped, on_start=launched, monitor=monitor,
                                   extra_fds=(transport.fd,))
                job.update(returncode=rc, finished_utc=utc_now())
                if rc != 0 or not completed_job(job):
                    raise RuntimeError('Nonzero worker exit or invalid/incomplete training artifacts')
                job['phase'] = 'complete'
            except BaseException as exc:
                job.update(phase='failed', error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
                persist('worker_failed_no_promotion')
                raise
            finally:
                heartbeat.close()
                transport.close()
            persist('worker_complete')
        persist('complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
