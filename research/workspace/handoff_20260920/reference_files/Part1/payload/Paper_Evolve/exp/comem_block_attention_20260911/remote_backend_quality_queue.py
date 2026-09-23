"""Training-priority remote queue: fixed eight-QA smoke, then all 99 pairs.

No local GPU work or timing results. Both jobs keep the original shared GPU
flock throughout the child's lifetime. Failures stop this bounded queue.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from remote_sparse_queue import (ROOT, CODE, inventory, eligible_gpus, process_identity,
                                 read_json, publish, completed_job, live_trainers, utc_now)
from remote_gpu_guard import write_lease, unexpected_processes, bind_owned_process, terminate_owned

TRAIN_OUT = ROOT/'outputs/sparse_comem_20260911'
TRAIN_KEYS = {stage+'/'+arm for stage in ('smoke','train') for arm in ('D0','A','B','D1')}


def training_priority_clear():
    state = read_json(TRAIN_OUT/'queue.json')
    jobs = state.get('jobs', {})
    if set(jobs) != TRAIN_KEYS:
        return False, 'training_queue_unknown'
    if all(j.get('phase') == 'complete' for j in jobs.values()):
        return (True, 'training_complete') if all(completed_job(j) for j in jobs.values()) else (False, 'training_completion_invalid')
    owner = state.get('controller', {})
    if not owner or process_identity(owner.get('pid')) != owner:
        return False, 'training_controller_not_active'
    if any(j.get('phase') not in ('complete', 'running') for j in jobs.values()):
        return False, 'training_queued_or_needs_attention'
    if any(not j.get('process') or process_identity(j['process'].get('pid')) != j['process']
           for j in jobs.values() if j['phase'] == 'running'):
        return False, 'training_worker_state_stale'
    return True, 'training_workers_already_assigned'


def validate_cpu_receipt(path):
    from evaluate_backend_quality import source_hashes, digest
    receipt = read_json(path)
    if (receipt.get('passed') is not True or receipt.get('device') != 'cpu'
            or receipt.get('cuda_visible_devices') != '' or not receipt.get('tests_run')
            or any(receipt.get(k, 0) for k in ('errors', 'failures', 'skipped'))):
        raise RuntimeError('Current QA code requires a passed remote CPU validation')
    required = source_hashes()
    bound = receipt.get('source_sha256', {})
    if not required.keys() <= bound.keys() or any(bound[k] != v for k, v in required.items()):
        raise RuntimeError('QA CPU receipt does not match current runtime sources')
    for name, expected in bound.items():
        if name in required:
            continue
        local = (CODE/name).resolve()
        if ROOT not in local.parents or not local.is_file() or digest(local) != expected:
            raise RuntimeError('CPU validation dependency changed: ' + name)
    return receipt


def completed_quality(directory, mode, cpu_receipt):
    """Recompute paired completeness; never infer success from exit code alone."""
    try:
        from evaluate_backend_quality import source_hashes, select_rows, digest
        from backend_quality_results import validate_complete_result
        directory = Path(directory)
        source = source_hashes()
        if any(cpu_receipt.get('source_sha256', {}).get(k) != v for k,v in source.items()):
            return False
        data = CODE/'data/qasper_pilot'
        rows = [json.loads(line) for line in (data/'dev.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        ids = [r['id'] for r in select_rows(rows, mode, 42)]
        expected = dict(mode=mode, ordered_ids=ids, seed=42, max_new_tokens=128, j=12, m=16,
                        rank=32, alpha=32., probe_mode='dense', retain_ratio=1., teacher_forced_ce=False,
                        model=str(ROOT/'models/Qwen3-8B'),
                        init_adapter_sha256=digest(ROOT/'outputs/8b_j12_pub_4k/final/adapter.pt'),
                        train_sha256=digest(data/'train.jsonl'), dev_sha256=digest(data/'dev.jsonl'))
        return validate_complete_result(directory, expected, expected_source_sha256=source)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/backend_quality_20260912')
    parser.add_argument('--cpu-receipt', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or not 5 <= args.poll_seconds <= 60:
        raise RuntimeError('Use remote Linux and poll-seconds in [5,60]')
    import fcntl
    args.out, args.cpu_receipt = args.out.resolve(), args.cpu_receipt.resolve()
    if any(ROOT not in p.parents for p in (args.out, args.cpu_receipt)):
        raise ValueError('All files must stay inside the named remote experiment root')
    receipt = validate_cpu_receipt(args.cpu_receipt)
    from evaluate_backend_quality import source_hashes
    sources = source_hashes()
    config = dict(source_sha256=sources, cpu_receipt=str(args.cpu_receipt),
                  cpu_receipt_sha256=hashlib.sha256(args.cpu_receipt.read_bytes()).hexdigest(),
                  training_priority=str(TRAIN_OUT), modes=['smoke', 'full'])
    args.out.mkdir(parents=True, exist_ok=True)
    singleton = (ROOT/'logs/backend_quality_20260912.queue.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read_json(args.out/'queue.json')
    if old and old.get('config') != config:
        raise RuntimeError('Existing QA output belongs to a different code/configuration')
    if live_trainers(CODE/'evaluate_backend_quality.py', ROOT/'outputs'):
        raise RuntimeError('A QA worker is still active; do not duplicate')
    jobs = {mode:dict(mode=mode, phase='queued', out=str(args.out/mode)) for mode in ('smoke', 'full')}
    for mode, job in jobs.items():
        if completed_quality(job['out'], mode, receipt):
            job['phase'] = 'complete'
        elif (Path(job['out']).exists() or (args.out/(mode+'_control')/'worker.log').exists()
              or old.get('jobs', {}).get(mode, {}).get('phase') in ('failed', 'running', 'launching', 'complete')):
            job.update(phase='failed', error='Existing partial QA output retained; no automatic retry')
    state = dict(config=config, jobs=jobs, controller=process_identity(os.getpid()),
                 pid=os.getpid(), formal_inference_timing=False)
    def save(reason):
        state.update(reason=reason, updated_utc=utc_now())
        publish(args.out/'queue.json', state)
    save('ready' if args.run else 'dry_plan')
    if not args.run:
        return
    while True:
        if any(j['phase'] == 'failed' for j in jobs.values()):
            save('failed_no_promotion')
            return
        if all(j['phase'] == 'complete' for j in jobs.values()):
            save('complete')
            return
        clear, reason = training_priority_clear()
        if not clear:
            save(reason)
            time.sleep(args.poll_seconds)
            continue
        try:
            before = inventory()
            candidates = eligible_gpus(before)
        except Exception as exc:
            state['inventory_error'] = str(exc)
            save('inventory_unavailable')
            time.sleep(args.poll_seconds)
            continue
        if not candidates:
            save('waiting_for_free_3090')
            time.sleep(args.poll_seconds)
            continue
        mode = next(m for m in ('smoke','full') if jobs[m]['phase'] == 'queued')
        job = jobs[mode]
        launched = False
        for gpu in candidates:
            lock = (ROOT/'logs'/f'beacon_sft_gpu{gpu}.lock').open('a')
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                clear, reason = training_priority_clear()
                locked = inventory()
                if not clear or gpu not in eligible_gpus(locked):
                    continue
                # Recheck source binding immediately before each independent job.
                validate_cpu_receipt(args.cpu_receipt)
                control = args.out/(mode+'_control')
                control.mkdir(exist_ok=True)
                lease = control/'lease.json'
                run_id = uuid.uuid4().hex
                worker = CODE/'evaluate_backend_quality.py'
                write_lease(lease, gpu, lock.fileno(), worker, run_id)
                command = [sys.executable,'-u',str(worker),'--model',str(ROOT/'models/Qwen3-8B'),
                    '--init-adapter',str(ROOT/'outputs/8b_j12_pub_4k/final/adapter.pt'),
                    '--train',str(CODE/'data/qasper_pilot/train.jsonl'),'--dev',str(CODE/'data/qasper_pilot/dev.jsonl'),
                    '--out',job['out'],'--mode',mode,'--lease',str(lease)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), SPARSE_GPU_LOCK_FD=str(lock.fileno()),
                    SPARSE_GPU_LEASE_PATH=str(lease), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                job.update(phase='launching', gpu=gpu, admission_before_lock=before, admission_inside_lock=locked,
                           command=command, lease=str(lease), started_utc=utc_now())
                save('launching')
                with (control/'worker.log').open('x',encoding='utf-8') as log:
                    proc = None
                    failure = None
                    try:
                        proc = subprocess.Popen(command,cwd=ROOT/'workspace',env=env,stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                        bind_owned_process(proc)
                        job.update(phase='running', process=process_identity(proc.pid))
                        save('running')
                        with (control/'observations.jsonl').open('x',encoding='utf-8',buffering=1) as obs:
                            while proc.poll() is None:
                                write_lease(lease,gpu,lock.fileno(),worker,run_id)
                                observed = inventory()
                                outsiders = unexpected_processes(observed,gpu,{proc.pid})
                                obs.write(json.dumps(dict(utc=utc_now(),gpu=observed[gpu],outsiders=outsiders))+'\n')
                                if outsiders:
                                    failure = 'External GPU process appeared; stopped only the QA worker'
                                    terminate_owned(proc)
                                    break
                                save('running')
                                time.sleep(1)
                    except BaseException as exc:
                        failure = type(exc).__name__+': '+str(exc)
                        if proc is not None:
                            terminate_owned(proc)
                    finally:
                        if proc is not None and proc.poll() is None:
                            terminate_owned(proc)
                    returncode = None if proc is None else proc.returncode
                    ok = failure is None and returncode == 0 and completed_quality(job['out'],mode,receipt)
                    job.update(phase='complete' if ok else 'failed', returncode=returncode,
                        error=failure if failure else (None if ok else 'Worker failure or invalid paired completion'),
                        finished_utc=utc_now())
                    save('worker_finished')
                launched = True
                break
            finally:
                # Never LOCK_UN the inherited description before worker exit.
                lock.close()
        if not launched:
            save('waiting_for_training_priority_or_gpu_lock')
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
