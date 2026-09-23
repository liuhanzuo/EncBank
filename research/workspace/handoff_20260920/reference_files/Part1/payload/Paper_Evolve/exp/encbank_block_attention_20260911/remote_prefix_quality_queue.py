"""Independent remote prefix QA queue, after training and backend QA priority.

Two fixed jobs: four-question natural-pack smoke, then all 99 prepared questions.
This module only defines a controller; importing it never loads Torch or a GPU.
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

import remote_backend_quality_queue as backend_queue
import evaluate_native_prefix_quality as runner
from backend_quality_results import canonical_hash
from prefix_quality_protocol import build_prefix_quality_plan, summarize_prefix_quality
from remote_sparse_queue import (ROOT, CODE, inventory, eligible_gpus, process_identity,
    read_json, publish, live_trainers, utc_now, LeaseRefresher)
from remote_gpu_guard import write_lease, unexpected_processes, bind_owned_process, terminate_owned

OUT = ROOT / 'outputs/native_prefix_quality_20260912'
BACKEND_OUT = ROOT / 'outputs/backend_quality_20260912'
MODES = ('smoke', 'full')
VALIDATION_DEPENDENCIES = ('remote_prefix_quality_queue.py', 'test_remote_prefix_quality_queue.py',
                         'test_native_prefix_quality.py', 'test_prefix_quality_protocol.py',
                         'validate_native_prefix_quality_cpu.py')


def validate_cpu_receipt(path):
    """Pin runner, controller and validation dependencies before admission."""
    receipt = read_json(path)
    if (receipt.get('passed') is not True or receipt.get('device') != 'cpu'
            or receipt.get('cuda_visible_devices') != ''
            or receipt.get('cuda_initialized') is not False
            or receipt.get('cpu_threads') != 2 or receipt.get('cpu_interop_threads') != 16
            or type(receipt.get('tests_run')) is not int or receipt['tests_run'] <= 0
            or any(receipt.get(key, 0) for key in ('errors', 'failures', 'skipped'))):
        raise RuntimeError('Prefix QA requires a passed CPU receipt without skips')
    required = runner.source_hashes()
    required.update({name: runner.digest(CODE / name) for name in VALIDATION_DEPENDENCIES})
    bound = receipt.get('source_sha256', {})
    if not required.keys() <= bound.keys() or any(bound[name] != digest for name, digest in required.items()):
        raise RuntimeError('Prefix QA CPU receipt differs from current runtime/test sources')
    for name, expected in bound.items():
        if name in required:
            continue
        path = (CODE / name).resolve()
        if ROOT not in path.parents or not path.is_file() or runner.digest(path) != expected:
            raise RuntimeError('Prefix QA validation dependency changed: ' + name)
    return receipt


def backend_priority_clear():
    """Completed backend jobs may have a naturally exited controller."""
    try:
        state = read_json(BACKEND_OUT / 'queue.json')
        jobs = state.get('jobs', {})
        if set(jobs) != set(MODES):
            return False, 'backend_quality_queue_unknown'
        if not all(job.get('phase') == 'complete' for job in jobs.values()):
            owner = state.get('controller', {})
            if not owner or process_identity(owner.get('pid')) != owner:
                return False, 'backend_quality_controller_not_active'
            return False, 'backend_quality_pending_or_failed'
        config = state.get('config', {})
        receipt_path = Path(config['cpu_receipt']).resolve()
        if ROOT not in receipt_path.parents or runner.digest(receipt_path) != config['cpu_receipt_sha256']:
            return False, 'backend_quality_cpu_receipt_changed'
        receipt = backend_queue.validate_cpu_receipt(receipt_path)
        from evaluate_backend_quality import source_hashes
        if config.get('source_sha256') != source_hashes():
            return False, 'backend_quality_source_changed'
        for mode, job in jobs.items():
            if (Path(job.get('out', '')).resolve() != (BACKEND_OUT / mode).resolve()
                    or type(job.get('returncode')) is not int or job['returncode'] != 0
                    or not backend_queue.completed_quality(job['out'], mode, receipt)):
                return False, 'backend_quality_completion_invalid'
        return True, 'backend_quality_complete'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        return False, 'backend_quality_validation_unavailable:' + type(exc).__name__


def priorities_clear():
    try:
        clear, reason = backend_queue.training_priority_clear()
        return backend_priority_clear() if clear else (False, reason)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        return False, 'training_priority_unavailable:' + type(exc).__name__


def _dev_rows():
    path = CODE / 'data/qasper_pilot/dev.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def completed_prefix_quality(directory, mode, cpu_receipt, *, worker_returncode, expected_lease_run_id):
    """Require our child's zero exit plus freshly recomputed data/quality receipts."""
    try:
        if type(worker_returncode) is not int or worker_returncode != 0 or not expected_lease_run_id:
            return False
        directory = Path(directory)
        status, metadata, saved, recorded_plan = (read_json(directory / name) for name in
            ('status.json', 'metadata.json', 'summary.json', 'plan.json'))
        sources = runner.source_hashes()
        if (metadata.get('source_sha256') != sources or
                any(cpu_receipt.get('source_sha256', {}).get(k) != v for k, v in sources.items())):
            return False
        plan = build_prefix_quality_plan(_dev_rows(), mode=mode, seed=42)
        count = 4 if mode == 'smoke' else 99 if mode == 'full' else 0
        if count == 0 or len(plan['ordered_ids']) != count or recorded_plan != plan:
            return False
        data = CODE / 'data/qasper_pilot'
        expected = dict(model=str(ROOT / 'models/Qwen3-8B'),
            init_adapter_sha256=runner.digest(ROOT / 'outputs/8b_j12_pub_4k/final/adapter.pt'),
            train_sha256=runner.digest(data / 'train.jsonl'), dev_sha256=runner.digest(data / 'dev.jsonl'),
            mode=mode, seed=42, max_new_tokens=128, j=12, rank=32, alpha=32.,
            plan_sha256=plan['plan_sha256'], branches=['native_cold', 'native_prefix'],
            decoding='independent-free-greedy-natural-eos', teacher_forced_ce=False,
            reuse_scope='same-original-ordered-pack-within-group')
        recipe = metadata['recipe']
        if any(recipe.get(key) != value for key, value in expected.items()):
            return False
        recipe_sha, run_id = metadata['recipe_sha256'], metadata['run_id']
        if (not isinstance(run_id, str) or not run_id or canonical_hash(recipe) != recipe_sha
                or status.get('status') != 'complete' or status.get('run_id') != run_id
                or status.get('records_completed') != 2 * count or status.get('target_records') != 2 * count):
            return False
        lease = metadata.get('lease', {}).get('lease', {})
        if (lease.get('run_id') != expected_lease_run_id
                or lease.get('worker_script') != str(CODE / 'evaluate_native_prefix_quality.py')):
            return False
        if ('3090' not in metadata.get('gpu', '') or metadata.get('physical_gpu') not in ('0', '1', '2', '3')
                or metadata.get('backbone_dtype') != 'torch.bfloat16'
                or metadata.get('lora_master_dtype') != 'torch.float32'
                or metadata.get('autocast_dtype') != 'torch.bfloat16'):
            return False
        for item in (metadata, status, saved):
            if item.get('formal_inference_timing') is not False or item.get('formal_inference_memory') is not False:
                return False
        records = [json.loads(line) for line in (directory / 'records.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip()]
        if any(row.get('run_id') != run_id or row.get('recipe_sha256') != recipe_sha or
               row.get('source_sha256') != sources for row in records):
            return False
        recomputed = summarize_prefix_quality(plan, records,
            stop_token_ids=metadata['stop_token_ids'], max_new_tokens=128)
        return (recomputed.get('status') == 'complete' and recomputed.get('complete') is True
                and recomputed.get('metrics') is not None
                and all(saved.get(key) == value for key, value in recomputed.items()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, OverflowError):
        return False


def initial_jobs(out, old, receipt):
    """Retain failed/partial evidence; never automatically retry an old attempt."""
    jobs = {}
    for mode in MODES:
        previous = old.get('jobs', {}).get(mode, {})
        job = dict(mode=mode, phase='queued', out=str(out / mode), expected_questions=4 if mode == 'smoke' else 99)
        if completed_prefix_quality(job['out'], mode, receipt, worker_returncode=previous.get('returncode'),
                                    expected_lease_run_id=previous.get('lease_run_id')):
            job.update(previous, phase='complete')
        elif ((out / mode).exists() or (out / (mode + '_control') / 'worker.log').exists()
                or previous.get('phase') in ('failed', 'running', 'launching', 'complete')):
            job.update(phase='failed', error='Existing prefix QA attempt retained; no automatic retry',
                       previous_attempt=previous)
        jobs[mode] = job
    return jobs


def worker_command(out, mode, lease):
    if mode not in MODES:
        raise ValueError('Use fixed smoke4 or full99 mode')
    return [sys.executable, '-u', str(CODE / 'evaluate_native_prefix_quality.py'),
        '--model', str(ROOT / 'models/Qwen3-8B'), '--init-adapter', str(ROOT / 'outputs/8b_j12_pub_4k/final/adapter.pt'),
        '--train', str(CODE / 'data/qasper_pilot/train.jsonl'), '--dev', str(CODE / 'data/qasper_pilot/dev.jsonl'),
        '--out', str(out), '--mode', mode, '--lease', str(lease), '--seed', '42', '--max-new-tokens', '128']


def run_owned_worker(command, env, gpu, lock, control, job, save):
    """One owner, one lease writer, fail-closed monitoring and confirmed cleanup."""
    worker = CODE / 'evaluate_native_prefix_quality.py'
    leases = LeaseRefresher(write_lease)
    proc, failure = None, None
    lease = control / 'lease.json'
    leases.add('worker', path=lease, gpu=gpu, lock_fd=lock.fileno(),
               worker_script=worker, run_id=job['lease_run_id'])
    leases.start()
    try:
        with (control / 'worker.log').open('x', encoding='utf-8') as log:
            try:
                proc = subprocess.Popen(command, cwd=ROOT / 'workspace', env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                bind_owned_process(proc)
                job.update(phase='running', process=process_identity(proc.pid))
                save('running')
                with (control / 'observations.jsonl').open('x', encoding='utf-8', buffering=1) as observations:
                    while proc.poll() is None:
                        heartbeat_error = leases.failures().get('worker')
                        if heartbeat_error:
                            raise RuntimeError('GPU lease heartbeat failed: ' + heartbeat_error)
                        snapshot = inventory()
                        outsiders = unexpected_processes(snapshot, gpu, {proc.pid})
                        row = next(item for item in snapshot if item['index'] == gpu)
                        observations.write(json.dumps(dict(utc=utc_now(), gpu=row, outsiders=outsiders)) + '\n')
                        if outsiders:
                            raise RuntimeError('External GPU process appeared; stop only the prefix QA worker')
                        save('running')
                        time.sleep(1)
            except BaseException as exc:
                failure = type(exc).__name__ + ': ' + str(exc)
            finally:
                if proc is not None and proc.poll() is None:
                    terminate_owned(proc)
                if proc is not None and proc.poll() is None:
                    raise RuntimeError('Owned prefix worker remains alive; stop failed')
        return (None if proc is None else proc.returncode), failure
    finally:
        # Never remove a live child's lease or release the public lock early.
        if proc is None or proc.poll() is not None:
            leases.remove('worker')
        leases.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--cpu-receipt', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or not 5 <= args.poll_seconds <= 60:
        raise RuntimeError('Remote Linux only; poll-seconds must be in [5,60]')
    import fcntl
    args.out, args.cpu_receipt = args.out.resolve(), args.cpu_receipt.resolve()
    if any(ROOT not in path.parents for path in (args.out, args.cpu_receipt)):
        raise ValueError('Prefix queue files must stay inside the named experiment root')
    receipt = validate_cpu_receipt(args.cpu_receipt)
    config = dict(source_sha256=runner.source_hashes(), controller_sha256=runner.digest(Path(__file__)),
        cpu_receipt=str(args.cpu_receipt), cpu_receipt_sha256=runner.digest(args.cpu_receipt),
        training_priority=str(backend_queue.TRAIN_OUT), backend_quality_priority=str(BACKEND_OUT),
        modes=list(MODES), questions={'smoke': 4, 'full': 99}, max_new_tokens=128)
    args.out.mkdir(parents=True, exist_ok=True)
    singleton = (ROOT / 'logs/native_prefix_quality_20260912.queue.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read_json(args.out / 'queue.json')
    if old and old.get('config') != config:
        raise RuntimeError('Existing prefix queue belongs to another code/configuration')
    if live_trainers(CODE / 'evaluate_native_prefix_quality.py', ROOT / 'outputs'):
        raise RuntimeError('A prefix QA worker is already active; do not duplicate')
    jobs = initial_jobs(args.out, old, receipt)
    state = dict(config=config, jobs=jobs, controller=process_identity(os.getpid()), pid=os.getpid(),
                 formal_inference_timing=False, formal_inference_memory=False)
    def save(reason):
        state.update(reason=reason, updated_utc=utc_now())
        publish(args.out / 'queue.json', state)
    save('ready' if args.run else 'dry_plan')
    if not args.run:
        return
    while True:
        if any(job['phase'] == 'failed' for job in jobs.values()):
            save('failed_no_promotion')
            return
        if all(job['phase'] == 'complete' for job in jobs.values()):
            save('complete')
            return
        clear, reason = priorities_clear()
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
        mode = next(mode for mode in MODES if jobs[mode]['phase'] == 'queued')
        job, launched = jobs[mode], False
        for gpu in candidates:
            lock = (ROOT / 'logs' / f'beacon_sft_gpu{gpu}.lock').open('a')
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                clear, reason = priorities_clear()
                locked = inventory()
                if not clear or gpu not in eligible_gpus(locked):
                    continue
                receipt = validate_cpu_receipt(args.cpu_receipt)
                if runner.source_hashes() != config['source_sha256'] or runner.digest(args.cpu_receipt) != config['cpu_receipt_sha256']:
                    raise RuntimeError('Pinned prefix runtime/CPU receipt changed before dispatch')
                control = args.out / (mode + '_control')
                control.mkdir(exist_ok=True)
                lease = control / 'lease.json'
                command = worker_command(job['out'], mode, lease)
                job.update(phase='launching', gpu=gpu, admission_before_lock=before,
                    admission_inside_lock=locked, command=command, lease=str(lease),
                    lease_run_id=uuid.uuid4().hex, started_utc=utc_now())
                save('launching')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), SPARSE_GPU_LOCK_FD=str(lock.fileno()),
                    SPARSE_GPU_LEASE_PATH=str(lease), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                try:
                    rc, error = run_owned_worker(command, env, gpu, lock, control, job, save)
                    ok = error is None and completed_prefix_quality(job['out'], mode, receipt,
                        worker_returncode=rc, expected_lease_run_id=job['lease_run_id'])
                    job.update(phase='complete' if ok else 'failed', returncode=rc,
                        error=error if error else None if ok else 'Nonzero exit or invalid prefix QA completion',
                        finished_utc=utc_now())
                    save('worker_finished')
                except BaseException as exc:
                    job.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc), finished_utc=utc_now())
                    save('controller_failed_no_promotion')
                    raise
                launched = True
                break
            finally:
                # No LOCK_UN: inherited child descriptor retains lock ownership.
                lock.close()
        if not launched:
            save('waiting_for_priority_or_gpu_lock')
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
