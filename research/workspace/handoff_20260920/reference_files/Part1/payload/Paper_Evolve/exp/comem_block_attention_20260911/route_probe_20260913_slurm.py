"""Single-GPU diagnostic launcher using the unchanged existing Slurm guard."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import uuid
from types import SimpleNamespace
from slurm_gpu_guard import (confined, lock_path, allocation_identity, check_device,
    inventory, AnonymousHeartbeat, write_lease)
from slurm_sparse_worker import runtime_environment, probe_device, LeaseHeartbeat, supervise
from remote_sparse_queue import publish, process_identity, utc_now


def main():
    import fcntl
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    root=confined(args.task_root)
    out=confined(args.out,root)
    worker=confined(Path(__file__).with_name('route_probe_20260913.py'),root)
    if out.exists():
        raise RuntimeError('Fresh diagnostic controller output required')
    out.mkdir(parents=True)
    env=runtime_environment(root)
    os.environ.update(env)
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stopped.set())
    status=dict(complete=False,controller=process_identity(os.getpid()),started_utc=utc_now())
    with lock_path(root).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allocation=allocation_identity(env)
        status.update(allocation=allocation,phase='cpu_validation')
        publish(out/'status.json',status)
        command=[sys.executable,'-B',str(worker.with_name('test_route_probe_20260913.py'))]
        with (out/'cpu_validation.log').open('w') as log:
            rc=supervise(command,cwd=worker.parent,env={**env,'CUDA_VISIBLE_DEVICES':''},
                log=log,lock_fd=lock.fileno(),stopped=stopped,timeout=600)
        publish(out/'cpu_validation.json',dict(command=command,returncode=rc,
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [worker,worker.with_name('test_route_probe_20260913.py'),worker.with_name('sparse_reader.py')]},
            cuda_visible_devices=''))
        if rc:
            raise RuntimeError('Controlled CPU validation failed; no GPU probe')
        gpu=probe_device(SimpleNamespace(out=out,task_root=root),env,lock.fileno(),stopped)
        allocation['device']=gpu
        lease=out/'gpu_lease.json'
        run_id=uuid.uuid4().hex
        transport=AnonymousHeartbeat()
        heartbeat=LeaseHeartbeat(transport.refresh)
        command=[sys.executable,'-B','-u',str(worker),'--task-root',str(root),'--out',str(out/'data')]
        child_env={**env,'SPARSE_GPU_LEASE_PATH':str(lease),'SPARSE_GPU_LOCK_FD':str(lock.fileno()),
            'SPARSE_SLURM_RUN_ID':run_id,'SPARSE_SLURM_HEARTBEAT_FD':str(transport.fd)}
        try:
            check_device(inventory(),gpu['uuid'],require_idle=True)
            heartbeat.start()
            write_lease(lease,task_root=root,allocation=allocation,gpu_uuid=gpu['uuid'],
                lock_fd=lock.fileno(),worker_script=worker,run_id=run_id,heartbeat=transport.description)
            def started(proc):
                status.update(phase='running',command=command,worker=process_identity(proc.pid))
                publish(out/'status.json',status)
            def monitor(proc):
                if heartbeat.error:
                    raise RuntimeError('Lease heartbeat failed: '+heartbeat.error)
                check_device(inventory(),gpu['uuid'],allowed_pids={proc.pid})
            with (out/'worker.log').open('w') as log:
                rc=supervise(command,cwd=root,env=child_env,log=log,lock_fd=lock.fileno(),
                    stopped=stopped,on_start=started,monitor=monitor,extra_fds=(transport.fd,),timeout=5400)
            complete_path=out/'data/summary.json'
            complete=(rc==0 and complete_path.is_file() and json.loads(complete_path.read_text()).get('complete') is True)
            status.update(phase='complete' if complete else 'failed',returncode=rc,complete=complete,finished_utc=utc_now())
            publish(out/'status.json',status)
            if not complete:
                raise RuntimeError('Route diagnostic did not complete')
        except BaseException as exc:
            status.update(phase='failed',complete=False,error=f'{type(exc).__name__}: {exc}',finished_utc=utc_now())
            publish(out/'status.json',status)
            raise
        finally:
            heartbeat.close()
            transport.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
