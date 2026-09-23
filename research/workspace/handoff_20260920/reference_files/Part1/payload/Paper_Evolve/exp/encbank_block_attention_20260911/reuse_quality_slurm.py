"""Confined trained-reuse diagnostic controller/child on the existing Slurm lease.

The target worker lives in an isolated source pack. No canonical reader sources,
admission thresholds, ownership rules, or checkpoint parameters are changed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
import uuid
from slurm_gpu_guard import (confined, lock_path, allocation_identity, check_device,
    inventory, AnonymousHeartbeat, write_lease, validate_worker_lease)
from slurm_sparse_worker import runtime_environment, probe_device, LeaseHeartbeat, supervise
from remote_sparse_queue import publish, process_identity, utc_now


def configuration(args):
    root=confined(args.task_root)
    config_path=confined(args.config,root)
    config=json.loads(config_path.read_text())
    worker=confined(config['worker'],root)
    if worker.name!='reuse_quality_worker.py':
        raise ValueError('Use the named trained reuse quality worker')
    out=confined(config['controller_out'],root)
    tests=[confined(x,root) for x in config['cpu_test_scripts']]
    if not tests or any(not x.is_file() for x in [worker,*tests]):
        raise ValueError('Require available source pack and CPU test entry points')
    if not isinstance(config['worker_args'],list) or any(not isinstance(x,str) for x in config['worker_args']):
        raise ValueError('Worker arguments must be an exact string argv list')
    return root,config_path,config,worker,out,tests


def child(args):
    root,config_path,config,worker,out,tests=configuration(args)
    lease=os.environ['SPARSE_GPU_LEASE_PATH']
    checks=[]
    def admission_check(*,require_idle):
        observed=validate_worker_lease(lease,require_idle=require_idle)
        checks.append(observed)
        publish(out/'worker_admission.json',dict(checks=checks))
        return observed
    admission_check(require_idle=True)
    spec=importlib.util.spec_from_file_location('isolated_reuse_quality_target',worker)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    worker_args=module.parser().parse_args(config['worker_args'])
    for key in ('model','checkpoint_map','data','out'):
        confined(getattr(worker_args,key),root)
    if Path(worker_args.out).resolve()!=out/'data':
        raise ValueError('Worker output must be a fresh data subdirectory of controller output')
    receipt=Path(worker_args.checkpoint_map)
    mapping=json.loads(receipt.read_text())
    mapping=mapping.get('checkpoints',mapping)
    for arm in ('D0','A','B','D1'):
        entry=mapping[arm]
        path=entry['path'] if isinstance(entry,dict) else entry
        confined(path,root)
    return module.main(worker_args,admission_check=admission_check)


def controller(args):
    import fcntl
    root,config_path,config,worker,out,tests=configuration(args)
    if out.exists():
        raise RuntimeError('Fresh controller output is required; retain failed attempts')
    out.mkdir(parents=True)
    env=runtime_environment(root)
    os.environ.update(env)
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stopped.set())
    status=dict(complete=False,controller=process_identity(os.getpid()),started_utc=utc_now(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),formal_inference_timing=False,
        formal_inference_memory=False,worker=str(worker))
    with lock_path(root).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allocation=allocation_identity(env)
        status.update(allocation=allocation,phase='cpu_validation')
        publish(out/'status.json',status)
        cpu=[]
        for index,test in enumerate(tests):
            command=[sys.executable,'-B',str(test)]
            with (out/f'cpu_validation_{index}.log').open('w') as log:
                rc=supervise(command,cwd=worker.parent,env={**env,'CUDA_VISIBLE_DEVICES':''},
                    log=log,lock_fd=lock.fileno(),stopped=stopped,timeout=900)
            cpu.append(dict(command=command,returncode=rc,source_sha256=hashlib.sha256(test.read_bytes()).hexdigest()))
            publish(out/'cpu_validation.json',dict(checks=cpu,cuda_visible_devices=''))
            if rc:
                status.update(phase='failed_cpu_validation',finished_utc=utc_now(),returncode=rc)
                publish(out/'status.json',status)
                raise RuntimeError('CPU reuse semantics failed; no model starts')
        gpu=probe_device(SimpleNamespace(out=out,task_root=root),env,lock.fileno(),stopped)
        allocation['device']=gpu
        lease,run_id=out/'gpu_lease.json',uuid.uuid4().hex
        transport=AnonymousHeartbeat()
        heartbeat=LeaseHeartbeat(transport.refresh)
        launcher=confined(Path(__file__),root)
        command=[sys.executable,'-B','-u',str(launcher),'--child','--task-root',str(root),'--config',str(config_path)]
        child_env={**env,'SPARSE_GPU_LEASE_PATH':str(lease),'SPARSE_GPU_LOCK_FD':str(lock.fileno()),
            'SPARSE_SLURM_RUN_ID':run_id,'SPARSE_SLURM_HEARTBEAT_FD':str(transport.fd)}
        try:
            check_device(inventory(),gpu['uuid'],require_idle=True)
            heartbeat.start()
            write_lease(lease,task_root=root,allocation=allocation,gpu_uuid=gpu['uuid'],lock_fd=lock.fileno(),
                worker_script=launcher,run_id=run_id,heartbeat=transport.description)
            def started(proc):
                status.update(phase='running',command=command,child=process_identity(proc.pid))
                publish(out/'status.json',status)
            def monitor(proc):
                if heartbeat.error:
                    raise RuntimeError('Lease heartbeat failed: '+heartbeat.error)
                check_device(inventory(),gpu['uuid'],allowed_pids={proc.pid})
            with (out/'worker.log').open('w') as log:
                rc=supervise(command,cwd=root,env=child_env,log=log,lock_fd=lock.fileno(),stopped=stopped,
                    on_start=started,monitor=monitor,extra_fds=(transport.fd,),timeout=5400)
            result_path=out/'data/result.json'
            result=json.loads(result_path.read_text()) if result_path.is_file() else {}
            expected=set(config.get('expected_arms',['D0','A','B','D1','FULL']))
            complete=(rc==0 and result.get('status')=='complete' and set(result.get('arms',{}))==expected)
            passed=complete and all(row.get('passed') is True for row in result['arms'].values())
            status.update(phase='complete' if complete else 'failed',returncode=rc,complete=complete,
                quality_passed=passed,finished_utc=utc_now())
            publish(out/'status.json',status)
            if not complete:
                raise RuntimeError('Reuse diagnostic execution did not complete')
            # A completed failed-quality experiment stays visible and does not
            # receive a passing quality receipt or become eligible for timing.
        except BaseException as exc:
            status.update(phase='failed',complete=False,error=f'{type(exc).__name__}: {exc}',finished_utc=utc_now())
            publish(out/'status.json',status)
            raise
        finally:
            heartbeat.close()
            transport.close()
    return 0


def submit(args):
    import fcntl
    from submit_slurm_sparse import check_other_jobs
    root,config_path,config,worker,out,tests=configuration(args)
    launcher=confined(Path(__file__),root)
    target=config_path.with_name('quality_submission.json')
    with confined('/srv/encbank/.codex-encbank-sparse-l20d.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_other_jobs(root,launcher)
        previous=subprocess.run(['sacct','-j','27079','--noheader','--parsable2','--format','JobID,State,ExitCode'],
            text=True,capture_output=True,timeout=30,check=True)
        if '27079|COMPLETED|0:0' not in previous.stdout.splitlines():
            raise RuntimeError('Require successful completed route diagnostic before allocating the next GPU')
        if target.exists() or out.exists():
            raise RuntimeError('Quality task already submitted or has artifacts; inspect before retry')
        retained=subprocess.run(['scontrol','show','job','-o','27079'],text=True,capture_output=True,timeout=30)
        command=['sbatch','--parsable','--job-name','encbank-reuse-quality','--partition','gpu',
            '--gres','gpu:nvidia_l20d:1','--nodes','1','--ntasks','1','--cpus-per-task','4',
            '--mem','64G','--time','01:45:00','--chdir',str(root),
            '--output',str(root/'logs/reuse-quality-%j.out'),'--error',str(root/'logs/reuse-quality-%j.err'),
            '--signal','B:TERM@120']
        # Slurm may purge completed jobs from its live controller before sacct.
        # Completion is already required above; a dependency is only useful if
        # the controller still retains the completed predecessor identity.
        if retained.returncode==0:
            command+=['--dependency','afterok:27079']
        argv=[sys.executable,'-B','-u',str(launcher),'--task-root',str(root),'--config',str(config_path)]
        script='#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexec '+shlex.join(argv)+'\n'
        observed=subprocess.run(command,input=script,text=True,capture_output=True,timeout=60)
        if observed.returncode:
            failure=dict(command=command,stdin_script=script,returncode=observed.returncode,
                stdout=observed.stdout,stderr=observed.stderr,observed_utc=utc_now())
            publish(config_path.with_name('quality_submission_failure.json'),failure)
            raise RuntimeError('sbatch failed: '+observed.stderr)
        raw=observed.stdout.strip()
        if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',raw):
            raise RuntimeError('Ambiguous sbatch response; do not retry: '+raw)
        receipt=dict(job_id=raw.split(';')[0],command=command,stdin_script=script,submitted_utc=utc_now(),
            previous_accounting=previous.stdout,stdout=observed.stdout,stderr=observed.stderr,
            predecessor_controller_retained=retained.returncode==0,
            predecessor_controller_observation=retained.stdout+retained.stderr,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),gpu_count=1,
            formal_inference_timing=False,formal_inference_memory=False)
        publish(target,receipt)
        print(json.dumps(receipt,indent=2))
    return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-root',required=True,type=Path)
    ap.add_argument('--config',required=True,type=Path)
    ap.add_argument('--child',action='store_true')
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args()
    if args.child and args.submit:
        raise ValueError('Cannot submit from child mode')
    if args.submit:
        return submit(args)
    return child(args) if args.child else controller(args)


if __name__=='__main__':
    raise SystemExit(main())
