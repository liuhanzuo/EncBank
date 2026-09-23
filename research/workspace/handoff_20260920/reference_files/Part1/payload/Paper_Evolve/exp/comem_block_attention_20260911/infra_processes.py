"""Owned process identities, including the Windows venv Python redirector."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import psutil
from infra_protocol import read_json

MONITOR_POLICY_VERSION = "owned-tree-incremental-28g-v3-shared-read-and-failure-finalization"


def same_path(left, right):
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def identity(pid):
    process = psutil.Process(int(pid))
    return {"pid": process.pid, "create_time": process.create_time()}


def identity_alive(record):
    if not record:
        return False
    try:
        return psutil.Process(int(record["pid"])).create_time() == record["create_time"]
    except psutil.NoSuchProcess:
        return False


def _own_command(process, script, output):
    command = process.cmdline()
    if not any(same_path(value, script) for value in command if not value.startswith("-")):
        raise RuntimeError("Observed owned process does not execute this infrastructure script")
    if not any(same_path(value, output) for value in command if not value.startswith("-")):
        raise RuntimeError("Observed owned process uses a different output directory")


def validate_supervisor(supervisor_pid, heartbeat_path, script, output):
    beat = read_json(heartbeat_path)
    if not beat or beat.get("pid") != supervisor_pid or time.time() - beat.get("unix_s", 0) > 20:
        # A profiler can delay this thread after it opened an older snapshot.
        # Immediately re-read once; the same strict age/PID checks still apply.
        beat = read_json(heartbeat_path)
    if not beat or beat.get("pid") != supervisor_pid or time.time() - beat.get("unix_s", 0) > 20:
        raise RuntimeError("Live external monitor heartbeat is required")
    supervisor = psutil.Process(int(supervisor_pid))
    if supervisor.create_time() != beat.get("create_time"):
        raise RuntimeError("Supervisor process identity changed")
    worker = psutil.Process(os.getpid())
    parent = worker.parent()
    if parent is None:
        raise RuntimeError("Worker has no owning supervisor ancestor")
    _own_command(worker, script, output)
    bridge = None
    if parent.pid != supervisor_pid:
        grandparent = parent.parent()
        if grandparent is None or grandparent.pid != supervisor_pid:
            raise RuntimeError("Only a direct child or one verified venv redirector is supported")
        if not same_path(parent.exe(), sys.executable):
            raise RuntimeError("Intermediate parent is not this venv Python redirector")
        _own_command(parent, script, output)
        bridge = identity(parent.pid)
    return {"worker": identity(worker.pid), "supervisor": identity(supervisor_pid),
            "owned_root": identity(parent.pid if bridge else worker.pid), "redirector": bridge}


def validate_worker(owned_root_pid, root_create_time, supervisor_pid, lease, script, output):
    if lease.get("supervisor") != identity(supervisor_pid):
        raise RuntimeError("Worker handshake references a different supervisor")
    expected_root = {"pid": int(owned_root_pid), "create_time": root_create_time}
    if lease.get("owned_root") != expected_root or not identity_alive(expected_root):
        raise RuntimeError("Worker handshake differs from the freshly spawned owned process")
    root = psutil.Process(owned_root_pid)
    if root.parent() is None or root.parent().pid != supervisor_pid:
        raise RuntimeError("Spawned root is no longer a direct child of this supervisor")
    if not identity_alive(lease.get("worker")):
        raise RuntimeError("Worker PID was reused or exited before handshake validation")
    worker = psutil.Process(lease["worker"]["pid"])
    _own_command(root, script, output)
    _own_command(worker, script, output)
    if worker.pid != root.pid and (worker.parent() is None or worker.parent().pid != root.pid):
        raise RuntimeError("Actual worker is outside the freshly spawned redirector subtree")
    return {root.pid, worker.pid}


def terminate_owned_tree(child):
    """Stop only verified own process identities, including a still-live GPU child."""
    targets = {}
    root_record = {"pid": child.pid, "create_time": getattr(child, "_infra_root_create_time", None)}
    records = [root_record]
    lease = getattr(child, "_infra_worker_lease", None)
    if lease:
        records.append(lease["worker"])
    for record in records:
        if record["create_time"] is None or not identity_alive(record):
            continue
        process = psutil.Process(record["pid"])
        for item in [*process.children(recursive=True), process]:
            targets[item.pid] = item
    for process in targets.values():
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(targets.values()), timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=10)
    if still_alive:
        raise RuntimeError("An owned worker has not exited; refusing to advance the GPU queue")
    child.wait(timeout=10)
