"""Linux node-scoped reservations, bounded by host and cgroup memory headroom."""
import contextlib
import fcntl
import json
import os
import socket
import time
from pathlib import Path

from server_transport import save

PLAN = json.loads((Path(__file__).resolve().parent / 'plan.json').read_text())
ROOT = Path(PLAN['host_admission_root']) / socket.gethostname()


def process_identity(pid):
    # /proc/PID/stat command names may contain spaces and parentheses.
    parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(parts[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def identity_alive(identity):
    try:
        return process_identity(identity['pid']) == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


@contextlib.contextmanager
def locked():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def memory_headroom():
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    limits = [available]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            current = Path('/sys/fs/cgroup') / line.split('::', 1)[1].lstrip('/')
            while current.is_relative_to('/sys/fs/cgroup'):
                maximum, used = current / 'memory.max', current / 'memory.current'
                if maximum.exists() and used.exists():
                    value = maximum.read_text().strip()
                    if value != 'max':
                        limits.append(max(0, int(value) - int(used.read_text())))
                if current == Path('/sys/fs/cgroup'):
                    break
                current = current.parent
    return min(limits) / 2**20


def read():
    path = ROOT / 'reservations.json'
    return json.loads(path.read_text()) if path.exists() else {}


def acquire(owner, task, memory_mb):
    assert memory_mb > 0
    job_available = memory_headroom()
    available = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))) / 1024
    sampled = time.monotonic()
    with locked():
        if time.monotonic() - sampled > 5:
            return False, {'reason': 'Memory sample expired'}
        reservations = read()
        # Dead owners retain reservations for review: their containers may still exist.
        key = owner + ':' + task
        assert key not in reservations, 'Existing task reservation; no duplicate launch'
        occupied = sum(row['memory_mb'] for row in reservations.values())
        own_occupied = sum(row['memory_mb'] for row in reservations.values() if row['owner'] == owner)
        budget = PLAN['host_memory_budget_mb']
        proof = {'epoch': time.time(), 'available_mb': available, 'reserved_mb': occupied,
                 'new_mb': memory_mb, 'budget_mb': budget, 'safety_mb': 2048,
                 'job_available_mb': job_available, 'job_reserved_mb': own_occupied,
                 'hostname': socket.gethostname()}
        admitted = occupied + memory_mb <= budget and available >= occupied + memory_mb + 2048 and job_available >= own_occupied + memory_mb + 2048
        if admitted:
            reservations[key] = {'owner': owner, 'task': task, 'memory_mb': memory_mb,
                                 'identity': process_identity(os.getpid()), 'epoch': time.time()}
            save(ROOT / 'reservations.json', reservations)
        return admitted, proof


def release(owner, task):
    with locked():
        reservations = read()
        key = owner + ':' + task
        assert reservations[key]['identity'] == process_identity(os.getpid())
        del reservations[key]
        save(ROOT / 'reservations.json', reservations)
