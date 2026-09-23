"""Remote-only GPU lease and interference checks shared by training and QA.

The public flock is advisory. These checks detect competing processes; they
cannot prevent a non-cooperating job from starting after a snapshot.
"""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from remote_sparse_queue import ROOT, DISPLAY_PROCESSES, inventory, process_identity, publish, read_json

MAX_HEARTBEAT_AGE = 20.


def _inside(path):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError('Remote GPU lease files must stay inside the experiment root')
    return path


def write_lease(path, gpu, lock_fd, worker_script, run_id):
    if sys.platform != 'linux':
        raise RuntimeError('Remote Linux only')
    path = _inside(path)
    worker_script = _inside(worker_script)
    if type(gpu) is not int or gpu not in range(4):
        raise ValueError('Use one of the four remote physical GPUs')
    lock_path = ROOT/'logs'/f'beacon_sft_gpu{gpu}.lock'
    descriptor, named = os.fstat(lock_fd), lock_path.stat()
    if (descriptor.st_dev, descriptor.st_ino) != (named.st_dev, named.st_ino):
        raise RuntimeError('GPU lock descriptor does not match the public lock')
    owner = process_identity(os.getpid())
    if not owner:
        raise RuntimeError('Cannot establish the controller identity')
    value = dict(format='remote-shared-gpu-lease-v1', controller=owner, gpu=gpu,
                 lock_path=str(lock_path), lock_fd=lock_fd, worker_script=str(worker_script),
                 run_id=run_id, updated_unix_s=time.time())
    publish(path, value)
    return value


def unexpected_processes(snapshot, gpu, allowed_pids):
    row = next((g for g in snapshot if g.get('index') == gpu), None)
    if not row or '3090' not in row.get('name', '') or row.get('used_mib') is None:
        raise RuntimeError('GPU inventory is unavailable or is not a 3090')
    # An unavailable XML process list is not an empty process list.
    if not row.get('eligible') and not row.get('processes'):
        if row.get('used_mib', 512) < 512:
            raise RuntimeError('GPU process inventory is unknown')
        raise RuntimeError('GPU memory is occupied without attributable processes')
    unexpected = []
    for process in row['processes']:
        try:
            pid = int(process.get('pid', ''))
        except (TypeError, ValueError):
            unexpected.append(process)
            continue
        if pid in allowed_pids:
            continue
        if Path(process.get('process_name') or '').name in DISPLAY_PROCESSES and process.get('type') == 'G':
            continue
        unexpected.append(process)
    return unexpected


def validate_worker_lease(path, *, require_idle):
    """Before CUDA: strict idle. After CUDA: permit only this worker + display.

    The inherited descriptor keeps the public lock held even if the controller
    dies; the worker then rejects its stale/dead parent at its next check.
    """
    if sys.platform != 'linux':
        raise RuntimeError('GPU quality/training workers must run on remote Linux')
    import fcntl
    path = _inside(path)
    lease = read_json(path)
    parent = lease.get('controller', {})
    if lease.get('format') != 'remote-shared-gpu-lease-v1' or parent.get('pid') != os.getppid():
        raise RuntimeError('Missing or unrelated controller lease')
    if process_identity(parent['pid']) != parent:
        raise RuntimeError('GPU controller has exited or changed identity')
    age = time.time() - lease.get('updated_unix_s', 0)
    if not 0 <= age <= MAX_HEARTBEAT_AGE:
        raise RuntimeError('GPU controller heartbeat is stale')
    gpu = lease.get('gpu')
    if type(gpu) is not int or gpu not in range(4) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise RuntimeError('Worker GPU visibility does not match its single-GPU lease')
    self_identity = process_identity(os.getpid())
    if not self_identity or lease.get('worker_script') not in self_identity['argv']:
        raise RuntimeError('Worker script differs from its lease')
    lock_path = ROOT/'logs'/f'beacon_sft_gpu{gpu}.lock'
    if lease.get('lock_path') != str(lock_path):
        raise RuntimeError('Lease does not name the public GPU lock')
    try:
        fd = int(os.environ['SPARSE_GPU_LOCK_FD'])
        actual, named = os.fstat(fd), lock_path.stat()
    except (KeyError, ValueError, OSError) as exc:
        raise RuntimeError('Worker did not inherit the GPU lock descriptor') from exc
    if fd != lease.get('lock_fd') or (actual.st_dev, actual.st_ino) != (named.st_dev, named.st_ino):
        raise RuntimeError('Inherited GPU lock identity differs')
    # A fresh open-file description must conflict with the inherited lock.
    with lock_path.open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Public GPU lock was not held across worker launch')
    snapshot = inventory()
    row = next((g for g in snapshot if g.get('index') == gpu), None)
    if require_idle:
        if row is None or not row.get('eligible'):
            raise RuntimeError('GPU is no longer idle before CUDA initialization')
    elif unexpected_processes(snapshot, gpu, {os.getpid()}):
        raise RuntimeError('Another GPU process appeared; this worker must stop')
    return dict(lease=lease, require_idle=require_idle, observed_unix_s=time.time(), gpu=row)


def bind_owned_process(proc):
    proc._remote_guard_identity = process_identity(proc.pid)
    if proc.poll() is None and proc._remote_guard_identity is None:
        # This is the newly spawned, unreaped Popen child. Stop it directly if
        # /proc identity acquisition fails; it must never escape launch cleanup.
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        raise RuntimeError('Cannot establish new worker identity')


def terminate_owned(proc):
    """Stop only a still-matching worker session created with start_new_session."""
    if proc.poll() is not None:
        return
    expected = getattr(proc, '_remote_guard_identity', None)
    current = process_identity(proc.pid)
    if not expected or current != expected or os.getpgid(proc.pid) != proc.pid:
        raise RuntimeError('Cannot safely identify the owned worker process group')
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # PID/start ticks must still match before escalating the same group.
        if process_identity(proc.pid) != expected:
            raise RuntimeError('Worker identity changed while stopping')
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
