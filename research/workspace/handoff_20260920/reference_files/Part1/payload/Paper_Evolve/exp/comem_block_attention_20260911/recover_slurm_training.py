"""One explicit recovery of job 24023: retain completed work, retry only A at step 0.

The failed A directory and original queue/source evidence remain available. This
does not change the training recipe, shorten runs, or retry arbitrary failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path('/srv/encbank/comem_sparse_slurm_20260912')
CODE = ROOT/'workspace/exp/comem_block_attention_20260911'
OUT = ROOT/'outputs/sparse_comem_20260911'
ARCHIVE = ROOT/'logs/recovery-24023'
OLD_QUEUE_SHA = 'c54737496e12439bd797df4e2f1884cb504721e2c3aca1a3ac4b663585d331b2'
RETRY_OUT = OUT/'train/A_retry1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_failed_a(state, files):
    if (state.get('phase') != 'failed' or state.get('complete') is not False
            or state.get('arm') != 'A' or state.get('step') != 0
            or state.get('cursor') != 0 or state.get('raw_tokens') != 0
            or state.get('training_seconds') != 0 or state.get('target_steps') != 250
            or state.get('error') != 'Slurm controller heartbeat is stale'):
        raise RuntimeError('Recovery is limited to the observed A heartbeat failure before training')
    if any(name.endswith('.pt') or name == 'train.jsonl' for name in files):
        raise RuntimeError('A has training/checkpoint artifacts; do not restart it from step zero')


def old_failure():
    result = subprocess.run(['sacct', '-j', '24023', '-n', '-P',
                             '--format=JobID,JobName,State,ExitCode'],
                            capture_output=True, text=True, timeout=30, check=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    if not any(row[:4] == ['24023', 'comem-sparse-1gpu', 'FAILED', '1:0'] for row in rows):
        raise RuntimeError('Accounting does not confirm original job 24023 terminated with failure')
    return result.stdout


def checked_inputs():
    from remote_sparse_queue import completed_job
    from slurm_gpu_guard import confined
    confined(__file__, ROOT)
    queue_bytes = (OUT/'queue.json').read_bytes()
    if digest(queue_bytes) != OLD_QUEUE_SHA:
        raise RuntimeError('Original failed queue changed; do not duplicate a recovery')
    queue = json.loads(queue_bytes)
    if (queue.get('reason') != 'worker_failed_no_promotion'
            or queue.get('allocation', {}).get('job_id') != '24023'):
        raise RuntimeError('Unexpected source queue')
    expected = [f'{stage}/{arm}' for stage in ('smoke', 'train') for arm in ('D0','A','B','D1')]
    if list(queue['jobs']) != expected:
        raise RuntimeError('Unexpected task list')
    completed = expected[:5]
    for key in completed:
        job = queue['jobs'][key]
        if job.get('phase') != 'complete' or job.get('returncode') != 0 or not completed_job(job):
            raise RuntimeError('Completed artifacts no longer validate: '+key)
    failed = OUT/'train/A'
    check_failed_a(json.loads((failed/'status.json').read_text()), [p.name for p in failed.iterdir()])
    if RETRY_OUT.exists():
        raise RuntimeError('A_retry1 already exists; do not overwrite or retry again')
    for key in ('train/B','train/D1'):
        job = queue['jobs'][key]
        if job.get('phase') != 'queued' or Path(job['out']).exists():
            raise RuntimeError('Unstarted remaining arm changed: '+key)
    return queue, queue_bytes


def prepare_or_submit(submit):
    import fcntl
    from slurm_gpu_guard import lock_path
    from submit_slurm_sparse import check_other_jobs, submission
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (ROOT.parent/'.codex-comem-sparse-l20d.submit.lock').open('a') as submission_lock:
        fcntl.flock(submission_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_other_jobs(ROOT, Path(__file__).resolve())
        with lock_path(ROOT).open('a') as worker_lock:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue, raw = checked_inputs()
            accounting = old_failure()
            config = queue['config']
            args = SimpleNamespace(**{key: config[key] for key in
                ('task_root','trainer','model','init_adapter','data_dir','out','steps','grad_accum')},
                worker=str(CODE/'slurm_sparse_worker.py'),
                python='/srv/encbank/Paper_Evolve/.venv/bin/python',
                migration_receipt=str(ROOT/'logs/migration_receipt.json'),
                partition='gpu', gres='gpu:nvidia_l20d:1', cpus=4, memory='64G',
                time_limit='24:00:00', allow_training=True, account=None, qos=None)
            command, script, locations = submission(args)
            receipt_path = ARCHIVE/'recovery.json'
            original_worker = str(CODE/'slurm_sparse_worker.py')
            recovery_worker = str(Path(__file__).resolve())
            if script.count(original_worker) != 1:
                raise RuntimeError('Unexpected worker script rendering')
            script = script.replace(original_worker, recovery_worker)
            script = script.rstrip()+' --run --recovery-receipt '+shlex.quote(str(receipt_path))+'\n'
            tracked = ['slurm_gpu_guard.py','slurm_sparse_worker.py','train_sparse_slurm.py',
                       'train_sparse.py','sparse_reader.py','remote_sparse_queue.py',
                       'recover_slurm_training.py','test_slurm_gpu_guard.py',
                       'test_slurm_worker.py','test_slurm_recovery.py']
            receipt = dict(schema='sparse-slurm-recovery-24023-v1', original_job='24023',
                original_queue_sha256=OLD_QUEUE_SHA, original_queue=str(ARCHIVE/'queue-before.json'),
                retry_key='train/A', retry_out=str(RETRY_OUT),
                retained_completed=list(queue['jobs'])[:5], remaining=['train/A','train/B','train/D1'],
                recipe_unchanged=True, accounting=accounting,
                source_sha256={name:digest((CODE/name).read_bytes()) for name in tracked},
                observed_utc=datetime.now(timezone.utc).isoformat())
            plan = dict(submitted=False, command=command, stdin_script=script, receipt=receipt)
            if not submit:
                print(json.dumps(plan, indent=2))
                return
            if receipt_path.exists():
                raise RuntimeError('A recovery receipt already exists; inspect it before another submission')
            (ARCHIVE/'queue-before.json').write_bytes(raw)
            # Retain the failed task's small artifacts, not model/checkpoint copies.
            failed_dir = ARCHIVE/'failed-A'
            failed_dir.mkdir(exist_ok=False)
            for path in (OUT/'train/A').iterdir():
                if path.is_file():
                    (failed_dir/path.name).write_bytes(path.read_bytes())
            receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
            for path in locations.values():
                Path(path).mkdir(parents=True, exist_ok=True)
            # An immediately scheduled batch must acquire this lock itself.
            # The submit lock and one-use receipt still prevent another submitter.
            fcntl.flock(worker_lock, fcntl.LOCK_UN)
            result = subprocess.run(command, input=script, capture_output=True, text=True, timeout=60)
            (ARCHIVE/'sbatch-result.json').write_text(json.dumps(dict(returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)), encoding='utf-8')
            result.check_returncode()
            if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', result.stdout.strip()):
                raise RuntimeError('Ambiguous sbatch result; inspect squeue, do not submit again')
            job_id=result.stdout.strip().split(';')[0]
            record=dict(job_id=job_id, raw_job_id=result.stdout.strip(), job_name='comem-sparse-1gpu',
                command=command, stdin_script=script, submitted_utc=datetime.now(timezone.utc).isoformat(),
                migration_receipt=str(ROOT/'logs/migration_receipt.json'),
                migration_sha256=config['migration_sha256'], recovery_receipt=str(receipt_path),
                original_job='24023', destination_backend='slurm-l20d', stderr=result.stderr)
            (ROOT/f'logs/submission-{job_id}.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(record, indent=2))


def run_worker(receipt_path, worker_args):
    import slurm_sparse_worker as worker
    from slurm_gpu_guard import confined
    receipt_path = confined(receipt_path, ROOT)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('schema') != 'sparse-slurm-recovery-24023-v1'
            or receipt.get('original_queue_sha256') != OLD_QUEUE_SHA
            or receipt.get('retry_out') != str(RETRY_OUT)):
        raise RuntimeError('Unexpected explicit recovery receipt')
    for name, expected in receipt['source_sha256'].items():
        if digest(confined(CODE/name, CODE).read_bytes()) != expected:
            raise RuntimeError('Recovery source changed after submission: '+name)
    original = worker.migrated_jobs

    def recovered_jobs(args, migration, dev_ids):
        queue, raw = checked_inputs()
        jobs = original(args, migration, dev_ids)
        jobs['train/A']['out'] = str(RETRY_OUT)
        jobs['train/A']['destination_relative_out'] = str(RETRY_OUT.relative_to(ROOT))
        jobs['train/A']['retry_of'] = str(OUT/'train/A')
        jobs['train/A']['recovery_receipt'] = str(receipt_path)
        return jobs

    worker.migrated_jobs = recovered_jobs
    sys.argv = [sys.argv[0], *worker_args]
    try:
        return worker.main()
    finally:
        worker.migrated_jobs = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--recovery-receipt', type=Path)
    args, rest = parser.parse_known_args()
    if sys.platform != 'linux':
        raise RuntimeError('Use the cluster Linux host')
    if args.run:
        if args.submit or not args.recovery_receipt:
            raise ValueError('Run requires the recorded recovery receipt')
        return run_worker(args.recovery_receipt, rest)
    if rest or args.recovery_receipt:
        raise ValueError('Unexpected submission arguments')
    prepare_or_submit(args.submit)


if __name__ == '__main__':
    raise SystemExit(main())
