"""Single-allocation Slurm GPU ownership checks; no Torch import or GPU work.

This is a separate backend. The existing four-3090 lease policy is unchanged.
Only the allocated GPU UUID is monitored; no foreign process is terminated.
"""
from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from remote_sparse_queue import process_identity, publish

CLUSTER_HOME = Path('/srv/encbank')
MAX_HEARTBEAT_AGE = 20.
FORMAT = 'slurm-single-gpu-lease-v1'
HEARTBEAT_FORMAT = 'anonymous-memfd-monotonic-v1'
HEARTBEAT_BYTES = 4096
_HEARTBEAT_HEADER = struct.Struct('<16sQQ')
_HEARTBEAT_TICK = struct.Struct('<QQQ')
_HEARTBEAT_OFFSET = 64


class AnonymousHeartbeat:
    """Parent-owned, same-node liveness without shared-filesystem refreshes.

    Only the immutable descriptor identity is published to the lease JSON.
    The inherited anonymous memfd has no pathname or disk-backed page faults.
    A sequence lock prevents a reader accepting a partially updated timestamp.
    """
    def __init__(self):
        if sys.platform != 'linux' or not hasattr(os, 'memfd_create'):
            raise RuntimeError('Anonymous Slurm heartbeat requires Linux memfd_create')
        import fcntl
        self.fd = os.memfd_create('encbank-slurm-heartbeat', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        self.memory = None
        self.sequence = 0
        try:
            os.ftruncate(self.fd, HEARTBEAT_BYTES)
            # Keep the object size fixed; the parent must retain write access.
            fcntl.fcntl(self.fd, fcntl.F_ADD_SEALS,
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
            self.memory = mmap.mmap(self.fd, HEARTBEAT_BYTES, access=mmap.ACCESS_WRITE)
            token = secrets.token_bytes(16)
            self.memory[:_HEARTBEAT_HEADER.size] = _HEARTBEAT_HEADER.pack(token, os.getpid(), 1)
            observed = os.fstat(self.fd)
            self.description = dict(format=HEARTBEAT_FORMAT, fd=self.fd, device=observed.st_dev,
                                    inode=observed.st_ino, size=HEARTBEAT_BYTES,
                                    token=token.hex(), clock='monotonic_ns', parent_pid=os.getpid())
            self.refresh()
        except BaseException:
            self.close()
            raise

    def refresh(self):
        if self.memory is None:
            raise RuntimeError('Anonymous heartbeat is closed')
        self.sequence += 2
        # Readers compare both sequence fields and then reread the first field.
        # All writes are aligned machine words on the supported x86_64 host.
        struct.pack_into('<Q', self.memory, _HEARTBEAT_OFFSET, self.sequence - 1)
        struct.pack_into('<QQ', self.memory, _HEARTBEAT_OFFSET + 8,
                         time.monotonic_ns(), self.sequence)
        struct.pack_into('<Q', self.memory, _HEARTBEAT_OFFSET, self.sequence)

    def close(self):
        if self.memory is not None:
            self.memory.close()
            self.memory = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def anonymous_heartbeat_age(description, parent_pid, env=None):
    """Verify the inherited object and return its monotonic age, failing closed."""
    import fcntl
    env = os.environ if env is None else env
    if (not isinstance(description, dict) or description.get('format') != HEARTBEAT_FORMAT
            or description.get('clock') != 'monotonic_ns'
            or description.get('size') != HEARTBEAT_BYTES
            or description.get('parent_pid') != parent_pid):
        raise RuntimeError('Missing or unrelated anonymous Slurm heartbeat')
    try:
        fd = int(env['SPARSE_SLURM_HEARTBEAT_FD'])
        observed = os.fstat(fd)
        token = bytes.fromhex(description['token'])
        seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
        required_seals = fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL
        if (fd != description.get('fd') or fd < 0 or len(token) != 16
                or (observed.st_dev, observed.st_ino, observed.st_size)
                != (description.get('device'), description.get('inode'), HEARTBEAT_BYTES)
                or seals & required_seals != required_seals):
            raise RuntimeError('Inherited anonymous heartbeat identity differs')
        with mmap.mmap(fd, HEARTBEAT_BYTES, access=mmap.ACCESS_READ) as memory:
            if _HEARTBEAT_HEADER.unpack_from(memory) != (token, parent_pid, 1):
                raise RuntimeError('Anonymous heartbeat token or parent differs')
            for _ in range(32):
                first, tick, last = _HEARTBEAT_TICK.unpack_from(memory, _HEARTBEAT_OFFSET)
                again = struct.unpack_from('<Q', memory, _HEARTBEAT_OFFSET)[0]
                if first > 0 and first % 2 == 0 and first == last == again:
                    age = (time.monotonic_ns() - tick) / 1e9
                    if not 0 <= age <= MAX_HEARTBEAT_AGE:
                        raise RuntimeError('Slurm controller heartbeat is stale')
                    return age
                time.sleep(0)
            raise RuntimeError('Anonymous Slurm heartbeat has no stable committed tick')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError('Worker did not inherit its anonymous Slurm heartbeat') from exc


def confined(value, boundary=None):
    boundary = Path(CLUSTER_HOME if boundary is None else boundary).resolve()
    path = Path(value).resolve()
    if path == boundary or boundary not in path.parents:
        raise ValueError(f'Path must be strictly below {boundary}: {path}')
    return path


def gpu_environment(env):
    job = env.get('SLURM_JOB_ID', '')
    visible = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if (not job.isdigit() or len(visible) != 1 or not visible[0].strip()
            or visible[0] in ('-1', 'NoDevFiles', 'all')):
        raise RuntimeError('Require a real Slurm job and exactly one visible GPU')
    if env.get('SLURM_GPUS_ON_NODE') != '1' or env.get('SLURM_NTASKS', '1') != '1':
        raise RuntimeError('Exactly one allocated GPU and one task are required')
    if env.get('SLURM_ARRAY_JOB_ID'):
        raise RuntimeError('Job arrays are not supported')
    return dict(job_id=job, cuda_visible_devices=visible[0], allocated_gpus=1)


def fields(text):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)', text))


def job_gpu_count(tres):
    values = {}
    for item in tres.split(','):
        key, sep, value = item.partition('=')
        if sep and (key == 'gres/gpu' or key.startswith('gres/gpu:')):
            if not value.isdigit():
                raise RuntimeError('Unrecognized Slurm GPU count')
            values[key] = int(value)
    if not values or any(value != 1 for value in values.values()):
        raise RuntimeError('Slurm must allocate exactly one GPU')
    typed = [key for key in values if key.startswith('gres/gpu:')]
    if len(typed) > 1:
        raise RuntimeError('Multiple GPU types in the allocation')
    return 1


def cgroup_for(pid):
    return (Path('/proc')/str(pid)/'cgroup').read_text(encoding='utf-8')


def cgroup_job_matches(text, job_id):
    # Common cgroup v1/v2 Slurm forms include job_123/step_batch and job_123.scope.
    return bool(re.search(r'(?:^|[/.-])job[_-]' + re.escape(str(job_id)) + r'(?:[/.-]|$)', text, re.M))


def slurm_membership(job_id, pid, cgroup):
    if cgroup_job_matches(cgroup, job_id):
        return dict(kind='job-cgroup', cgroup=cgroup)
    # This site currently places jobs in slurmd.service without per-job cgroups.
    # Slurm's own step PID table must explicitly attribute this process to its job.
    raw = _output(['scontrol', 'listpids', str(job_id)])
    matches = []
    for line in raw.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[0].isdigit() and columns[1].isdigit():
            if int(columns[0]) == pid and columns[1] == str(job_id):
                matches.append(line)
    if len(matches) != 1:
        raise RuntimeError('Slurm PID table does not prove current job membership')
    stepd = stepd_ancestor(pid, job_id)
    return dict(kind='slurm-listpids', matched_row=matches[0], cgroup=cgroup, stepd=stepd)


def stepd_ancestor(pid, job_id):
    seen = set()
    for _ in range(32):
        if pid <= 1 or pid in seen:
            break
        seen.add(pid)
        identity = process_identity(pid)
        if identity and re.search(r'slurmstepd:\s*\[' + re.escape(str(job_id)) + r'[.\]]',
                                  ' '.join(identity['argv'])):
            return identity
        stat = (Path('/proc')/str(pid)/'stat').read_text()
        pid = int(stat[stat.rfind(')')+2:].split()[1])
    raise RuntimeError('Cannot find this Slurm job step daemon in the process ancestry')


def _output(command):
    return subprocess.check_output(command, text=True, timeout=20).strip()


def allocation_identity(env=None):
    """Read Slurm owner/allocation and local cgroup before a CUDA probe."""
    if sys.platform != 'linux':
        raise RuntimeError('Slurm workers run on Linux only')
    env = os.environ if env is None else env
    base = gpu_environment(env)
    job = fields(_output(['scontrol', 'show', 'job', '-o', base['job_id']]))
    uid = os.getuid()
    owner = re.fullmatch(r'[^()]+\((\d+)\)', job.get('UserId', ''))
    if (job.get('JobId') != base['job_id'] or job.get('JobState') != 'RUNNING'
            or not owner or int(owner.group(1)) != uid):
        raise RuntimeError('Slurm job is not a running allocation owned by this user')
    if job.get('NumNodes') != '1' or job.get('NumTasks', '1') != '1':
        raise RuntimeError('Require one allocated node and task')
    job_gpu_count(job.get('AllocTRES', ''))
    if job.get('Partition') != 'gpu':
        raise RuntimeError('Unexpected Slurm partition')
    nodes = _output(['scontrol', 'show', 'hostnames', job.get('NodeList', '')]).splitlines()
    node = env.get('SLURMD_NODENAME') or (nodes[0] if len(nodes) == 1 else '')
    if len(nodes) != 1 or node not in nodes:
        raise RuntimeError('Current Slurm node differs from the allocation')
    node_info = fields(_output(['scontrol', 'show', 'node', '-o', node]))
    hostname = socket.gethostname().split('.')[0]
    names = {node, node_info.get('NodeHostName', '').split('.')[0], node_info.get('NodeAddr', '').split('.')[0]}
    if hostname not in names or node_info.get('NodeName') != node:
        raise RuntimeError('Operating-system hostname does not match the allocated Slurm node')
    if 'gpu:nvidia_l20d:' not in node_info.get('Gres', ''):
        raise RuntimeError('Allocated node does not declare the observed L20D resource')
    cgroup = cgroup_for(os.getpid())
    membership = slurm_membership(base['job_id'], os.getpid(), cgroup)
    return {**base, 'uid': uid, 'node': node, 'hostname': hostname,
            'partition': job['Partition'], 'alloc_tres': job['AllocTRES'],
            'node_gres': node_info['Gres'], 'controller_cgroup': cgroup,
            'controller_membership': membership}


def normalize_uuid(value):
    value = str(value).strip()
    if not value.upper().startswith('GPU-'):
        value = 'GPU-' + value
    if not re.fullmatch(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', value, re.I):
        raise RuntimeError('Require a complete physical GPU UUID; no ordinal inference')
    return value.lower()


def gpu_inventory(xml):
    rows = []
    for gpu in ET.fromstring(xml).findall('gpu'):
        raw = gpu.findtext('fb_memory_usage/used', '')
        try:
            used = int(raw.split()[0]) if raw.endswith('MiB') else None
        except (ValueError, IndexError):
            used = None
        node = gpu.find('processes')
        known = node is not None and (not node.text or not node.text.strip())
        processes = [] if node is None else [{child.tag: child.text for child in proc}
                                             for proc in node.findall('process_info')]
        rows.append(dict(uuid=gpu.findtext('uuid', ''), name=gpu.findtext('product_name', ''),
                         used_mib=used, processes_known=known, processes=processes))
    return rows


def inventory():
    return gpu_inventory(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15))


def allocated_device(snapshot, gpu_uuid):
    target = normalize_uuid(gpu_uuid)
    matches = [row for row in snapshot if str(row.get('uuid', '')).lower() == target]
    if len(matches) != 1:
        raise RuntimeError('Allocated UUID is absent or ambiguous in NVIDIA inventory')
    row = matches[0]
    if ('L20D' not in row.get('name', '').upper() or row.get('used_mib') is None
            or row.get('processes_known') is not True):
        raise RuntimeError('Allocated L20D GPU identity or memory/process inventory is unavailable')
    return row


def unexpected_processes(row, allowed_pids):
    result = []
    for process in row['processes']:
        try:
            pid = int(process.get('pid', ''))
        except (TypeError, ValueError):
            result.append(process)
            continue
        # Slurm compute allocations need no display exception. All XML types count.
        if pid not in allowed_pids:
            result.append(process)
    return result


def check_device(snapshot, gpu_uuid, *, allowed_pids=(), require_idle=False):
    row = allocated_device(snapshot, gpu_uuid)
    foreign = unexpected_processes(row, set(allowed_pids))
    if foreign or (require_idle and row['used_mib'] >= 512):
        raise RuntimeError('Allocated GPU is occupied by another process or is not below 512 MiB')
    if not require_idle and row['used_mib'] >= 512 and not row['processes']:
        raise RuntimeError('Allocated GPU memory is occupied without attributable processes')
    return row


def lock_path(task_root):
    confined(task_root)
    # One home-wide advisory lock keeps all versions of this task single-GPU.
    return confined(CLUSTER_HOME/'.codex-encbank-sparse-slurm.worker.lock')


def write_lease(path, *, task_root, allocation, gpu_uuid, lock_fd, worker_script, run_id,
                heartbeat=None):
    task_root = confined(task_root)
    path, worker_script = confined(path, task_root), confined(worker_script, task_root)
    lock = lock_path(task_root)
    actual, named = os.fstat(lock_fd), lock.stat()
    if (actual.st_dev, actual.st_ino) != (named.st_dev, named.st_ino):
        raise RuntimeError('Inherited FD does not refer to the common Slurm worker lock')
    owner = process_identity(os.getpid())
    if not owner:
        raise RuntimeError('Cannot identify this controller')
    value = dict(format=FORMAT, task_root=str(task_root), controller=owner,
                 allocation=allocation, gpu_uuid=normalize_uuid(gpu_uuid), lock_path=str(lock),
                 lock_fd=lock_fd, worker_script=str(worker_script), run_id=run_id,
                 updated_unix_s=time.time())
    if heartbeat is not None:
        anonymous_heartbeat_age(heartbeat, owner['pid'],
                                {'SPARSE_SLURM_HEARTBEAT_FD': str(heartbeat['fd'])})
        value.update(heartbeat=dict(heartbeat),
                     updated_unix_s_role='immutable_metadata_creation_not_liveness')
    publish(path, value)
    return value


def validate_worker_lease(path, *, require_idle):
    if sys.platform != 'linux':
        raise RuntimeError('Slurm GPU workers must run on Linux')
    import fcntl
    task_root = confined(os.environ.get('SPARSE_SLURM_TASK_ROOT', ''))
    path = confined(path, task_root)
    lease = json.loads(path.read_text(encoding='utf-8'))
    if lease.get('format') != FORMAT or lease.get('task_root') != str(task_root):
        raise RuntimeError('Missing or unrelated Slurm GPU lease')
    parent = lease.get('controller', {})
    if parent.get('pid') != os.getppid() or process_identity(parent['pid']) != parent:
        raise RuntimeError('Slurm controller has exited or changed identity')
    if 'heartbeat' in lease:
        age = anonymous_heartbeat_age(lease['heartbeat'], parent['pid'])
    else:
        # Old receipts remain inspectable, but a new worker cannot silently
        # downgrade its explicitly inherited anonymous heartbeat to a disk tick.
        if 'SPARSE_SLURM_HEARTBEAT_FD' in os.environ:
            raise RuntimeError('Anonymous heartbeat missing from the Slurm lease')
        age = time.time() - lease.get('updated_unix_s', 0)
        if not 0 <= age <= MAX_HEARTBEAT_AGE:
            raise RuntimeError('Slurm controller heartbeat is stale')
    allocation = lease.get('allocation', {})
    current = gpu_environment(os.environ)
    if any(allocation.get(key) != value for key, value in current.items()) or allocation.get('uid') != os.getuid():
        raise RuntimeError('Worker visibility, allocation or owner differs from its lease')
    own_cgroup, parent_cgroup = cgroup_for(os.getpid()), cgroup_for(parent['pid'])
    membership = allocation.get('controller_membership', {})
    if (membership.get('kind') not in ('job-cgroup', 'slurm-listpids')
            or own_cgroup != parent_cgroup or parent_cgroup != allocation.get('controller_cgroup')):
        raise RuntimeError('Worker and controller are not in the same leased Slurm job cgroup')
    if membership['kind'] == 'slurm-listpids':
        stepd = membership.get('stepd', {})
        if not stepd.get('pid') or process_identity(stepd['pid']) != stepd:
            raise RuntimeError('The allocated Slurm step daemon has exited or changed identity')
    identity = process_identity(os.getpid())
    if (not identity or lease.get('worker_script') not in identity['argv']
            or lease.get('run_id') != os.environ.get('SPARSE_SLURM_RUN_ID')):
        raise RuntimeError('Worker script/run identity differs from its lease')
    lock = lock_path(task_root)
    if lease.get('lock_path') != str(lock):
        raise RuntimeError('Unexpected Slurm worker lock path')
    try:
        fd = int(os.environ['SPARSE_GPU_LOCK_FD'])
        actual, named = os.fstat(fd), lock.stat()
    except (KeyError, ValueError, OSError) as exc:
        raise RuntimeError('Worker did not inherit its GPU lock') from exc
    if fd != lease.get('lock_fd') or (actual.st_dev, actual.st_ino) != (named.st_dev, named.st_ino):
        raise RuntimeError('Inherited lock FD identity differs')
    with lock.open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Common Slurm lock was not held across worker launch')
    row = check_device(inventory(), lease['gpu_uuid'], allowed_pids=() if require_idle else {os.getpid()},
                       require_idle=require_idle)
    # Check again after potentially slow filesystem/NVIDIA observations. The
    # monotonic source remains current even if those observations took >20 s.
    if 'heartbeat' in lease:
        age = anonymous_heartbeat_age(lease['heartbeat'], parent['pid'])
    return dict(lease=lease, require_idle=require_idle, observed_unix_s=time.time(), gpu=row,
                heartbeat_age_s=age)


def bind_owned_process(proc):
    proc._slurm_guard_identity = process_identity(proc.pid)
    if proc.poll() is None and proc._slurm_guard_identity is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        raise RuntimeError('Cannot identify the newly created worker')


def terminate_owned(proc):
    if proc.poll() is not None:
        return
    expected = getattr(proc, '_slurm_guard_identity', None)
    if not expected or process_identity(proc.pid) != expected or os.getpgid(proc.pid) != proc.pid:
        raise RuntimeError('Cannot safely identify the owned child process group')
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process_identity(proc.pid) != expected:
            raise RuntimeError('Worker identity changed while stopping')
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
