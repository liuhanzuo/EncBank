"""Render, or explicitly submit, one serial single-L20D sparse-Encbank Slurm job.

No partition/account/GRES value is guessed. Default mode only prints a reviewable
submission request; --submit requires a fresh migration receipt and a real Linux
cluster connection. The helper must itself be deployed below the authorized home.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

HOME_BOUNDARY = PurePosixPath('/srv/encbank')
TASK_ROOT = HOME_BOUNDARY / 'encbank_sparse_slurm_20260912'
SOURCE_ROOT = PurePosixPath('/data/liuhanzuo/encbank_v2_20260908')
SOURCE_CODE = SOURCE_ROOT / 'workspace/exp/encbank_block_attention_20260911'
SOURCE_OUT = SOURCE_ROOT / 'outputs/sparse_encbank_20260911'
JOB_NAME = 'encbank-sparse-1gpu'
MIGRATION_SCHEMA = 'encbank-sparse-slurm-migration-v1'
TASK_KEYS = tuple(stage+'/'+arm for stage in ('smoke', 'train') for arm in ('D0', 'A', 'B', 'D1'))


def logical_path(value, boundary=HOME_BOUNDARY):
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or path == boundary or boundary not in path.parents:
        raise ValueError(f'Path must be strictly inside {boundary}: {value}')
    if '\n' in str(path) or '\r' in str(path) or '\x00' in str(path):
        raise ValueError('Control characters are not allowed in paths')
    return path


def plain_value(value, field):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+,:-]*', value):
        raise ValueError(f'Invalid {field}')
    return value


def migration_receipt(value, now=None, *, require_fresh=True):
    """Validate the explicit drained-3090 handoff; workers may wait in Slurm.

    Freshness applies at submission. The allocation worker still checks all
    identity, source-job and destination fields after a longer Slurm wait.
    """
    now = now or datetime.now(timezone.utc)
    if not isinstance(value, dict):
        raise ValueError('Migration receipt must be an object')
    try:
        stamp = datetime.fromisoformat(value.get('checked_utc', '').replace('Z', '+00:00'))
    except (TypeError, AttributeError) as exc:
        raise ValueError('Invalid migration observation time') from exc
    if (stamp.tzinfo is None or now.tzinfo is None or (now-stamp).total_seconds() < 0
            or require_fresh and (now-stamp).total_seconds() > 3600):
        raise ValueError('Require a migration receipt checked within the last hour at submission')
    expected_resource = dict(partition='gpu', gres='gpu:nvidia_l20d:1', gpu_count=1, gpu_model='L20D')
    resource = value.get('destination_resource', {})
    if (value.get('schema') != MIGRATION_SCHEMA or value.get('destination_backend') != 'slurm-l20d'
            or value.get('destination_task_root') != str(TASK_ROOT) or value.get('safe_to_submit') is not True
            or not isinstance(resource, dict) or any(resource.get(k) != v for k,v in expected_resource.items())
            or type(resource.get('gpu_count')) is not int):
        raise ValueError('Require the named single-L20D task root and migration schema')
    if (value.get('source_backend') != 'remote-3090' or value.get('source_host') != 'longjing-1'
            or value.get('controller_alive') is not False or value.get('live_trainers') != []
            or value.get('source_queue_path') != str(SOURCE_OUT / 'queue.json')
            or not re.fullmatch(r'[0-9a-f]{64}', str(value.get('source_queue_sha256', '')))):
        raise ValueError('Require the real drained longjing-1 sparse training queue; a 5090 receipt is insufficient')
    controller = value.get('source_controller', {})
    if (not isinstance(controller, dict) or type(controller.get('pid')) is not int or controller['pid'] <= 0
            or type(controller.get('start_ticks')) is not int or controller['start_ticks'] <= 0):
        raise ValueError('Source controller requires actual PID/start_ticks identity')
    argv = controller.get('argv')
    if (not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv)
            or str(SOURCE_CODE / 'remote_sparse_queue.py') not in argv
            or '--out' not in argv or argv.count('--out') != 1
            or argv.index('--out')+1 >= len(argv) or argv[argv.index('--out')+1] != str(SOURCE_OUT)
            or '--allow-training' not in argv):
        raise ValueError('Source controller argv must identify this 3090 training queue')
    jobs, tasks = value.get('source_jobs'), value.get('tasks')
    if not isinstance(jobs, dict) or set(jobs) != set(TASK_KEYS) or not isinstance(tasks, list) or len(tasks) != 8:
        raise ValueError('Require all eight actual source jobs and their explicit migration tasks')
    seen, destinations, train_ids, smoke_ids = set(), set(), [], []
    for task in tasks:
        if not isinstance(task, dict) or task.get('source_job_key') not in jobs:
            raise ValueError('Invalid source task key')
        key = task['source_job_key']
        if key in seen:
            raise ValueError('Duplicate migrated source job')
        seen.add(key)
        stage, arm = key.split('/')
        source = jobs[key]
        if (not isinstance(source, dict) or source.get('phase') != 'queued'
                or source.get('stage') != stage or source.get('arm') != arm
                or task.get('stage') != stage or task.get('arm') != arm):
            raise ValueError('Only actual queued source jobs may migrate; failed/complete jobs are not restarted')
        leaf = 'D0_retry1' if key == 'smoke/D0' else arm
        if source.get('out') != str(SOURCE_OUT / stage / leaf) or task.get('source_out') != source['out']:
            raise ValueError('Source output mapping differs, including the authorized D0_retry1')
        for field in ('steps', 'grad_accum', 'dev_ids'):
            if task.get(field) != source.get(field):
                raise ValueError('Migration changes source '+field)
        steps, accumulation, ids = task.get('steps'), task.get('grad_accum'), task.get('dev_ids')
        if (type(steps) is not int or type(accumulation) is not int
                or steps != (1 if stage == 'smoke' else 250) or accumulation != (2 if stage == 'smoke' else 4)
                or not isinstance(ids, list) or len(ids) != (1 if stage == 'smoke' else 99)
                or any(not isinstance(ident, str) or not ident for ident in ids) or len(ids) != len(set(ids))):
            raise ValueError('Source budgets/dev IDs differ from the matched sparse pilot')
        (smoke_ids if stage == 'smoke' else train_ids).append(ids)
        relative = task.get('destination_relative_out')
        if not isinstance(relative, str) or '\\' in relative or any(c in relative for c in ('\n','\r','\x00')):
            raise ValueError('Invalid destination-relative task output')
        target = PurePosixPath(relative)
        if (target.is_absolute() or target.as_posix() != relative or '..' in target.parts
                or len(target.parts) < 3 or target.parts[-2:] != (stage, leaf) or relative in destinations):
            raise ValueError('Destination must preserve each stage/arm and D0_retry1 inside the task root')
        logical_path(str(TASK_ROOT / target), TASK_ROOT)
        destinations.add(relative)
    if any(ids != train_ids[0] for ids in train_ids) or any(ids != smoke_ids[0] for ids in smoke_ids) or smoke_ids[0][0] not in train_ids[0]:
        raise ValueError('Matched source arms must preserve the same development IDs')
    return value


def submission(args):
    root = logical_path(str(args.task_root))
    if root != TASK_ROOT:
        raise ValueError('Use the explicitly authorized L20D task root')
    args.task_root = str(root)
    for field in ('worker', 'trainer', 'model', 'init_adapter', 'data_dir', 'out'):
        setattr(args, field, str(logical_path(str(getattr(args, field)), root)))
    executable = PurePosixPath(str(args.python))
    if (not executable.is_absolute() or '..' in executable.parts
            or any(c in str(executable) for c in ('\n', '\r', '\x00'))):
        raise ValueError('Python must be an absolute executable path without traversal or control characters')
    args.python = str(executable)
    if not getattr(args, 'migration_receipt', None):
        raise ValueError('The rendered worker must receive --migration-receipt')
    args.migration_receipt = str(logical_path(str(args.migration_receipt), root))
    for field in ('partition', 'gres', 'memory', 'time_limit'):
        plain_value(getattr(args, field), field)
    if args.partition != 'gpu' or args.gres != 'gpu:nvidia_l20d:1':
        raise ValueError('Use the site-confirmed partition gpu and exactly gpu:nvidia_l20d:1')
    if PurePosixPath(args.worker).name != 'slurm_sparse_worker.py' or PurePosixPath(args.trainer).name != 'train_sparse_slurm.py':
        raise ValueError('Use the dedicated Slurm worker and trainer adapter')
    for field in ('account', 'qos'):
        if getattr(args, field):
            plain_value(getattr(args, field), field)
    if args.cpus <= 0 or args.steps != 250 or args.grad_accum != 4 or args.allow_training is not True:
        raise ValueError('Keep the matched 250-step x4 budget and explicitly enable serial training')
    cmd = ['sbatch', '--parsable', '--job-name', JOB_NAME, '--partition', args.partition,
           '--gres', args.gres, '--nodes', '1', '--ntasks', '1',
           '--cpus-per-task', str(args.cpus), '--mem', args.memory,
           '--time', args.time_limit, '--chdir', args.task_root,
           '--output', str(root/'logs/slurm-%j.out'),
           '--error', str(root/'logs/slurm-%j.err'), '--signal', 'B:TERM@120']
    for field in ('account', 'qos'):
        if getattr(args, field):
            cmd += ['--'+field, getattr(args, field)]
    worker_cmd = [args.python, '-B', '-u', args.worker]
    for field in ('task_root', 'trainer', 'model', 'init_adapter', 'data_dir', 'out', 'steps', 'grad_accum', 'migration_receipt'):
        worker_cmd += ['--'+field.replace('_', '-'), str(getattr(args, field))]
    if args.allow_training:
        worker_cmd += ['--allow-training']
    runtime = root/'runtime'
    locations = {
        'TMPDIR': runtime/'tmp', 'TMP': runtime/'tmp', 'TEMP': runtime/'tmp',
        'HF_HOME': runtime/'cache/huggingface', 'XDG_CACHE_HOME': runtime/'cache',
        'TORCH_HOME': runtime/'cache/torch', 'TRITON_CACHE_DIR': runtime/'cache/triton',
        'CUDA_CACHE_PATH': runtime/'cache/cuda', 'PYTHONPYCACHEPREFIX': runtime/'cache/pycache',
        'MPLCONFIGDIR': runtime/'cache/matplotlib', 'WANDB_DIR': runtime/'wandb',
        'WANDB_CACHE_DIR': runtime/'cache/wandb', 'PIP_CACHE_DIR': runtime/'cache/pip'}
    # sbatch reads this script from stdin; no local or remote temp script is used.
    script = '#!/bin/bash\nset -euo pipefail\n'
    script += '\n'.join('export '+k+'='+shlex.quote(str(v)) for k, v in locations.items())+'\n'
    script += 'export PYTHONDONTWRITEBYTECODE=1\n'
    script += 'exec '+shlex.join(worker_cmd)+'\n'
    return cmd, script, locations


def check_other_jobs(task_root=TASK_ROOT, worker=None):
    """Reject only this task's duplicate allocation, allowing unrelated GPU jobs."""
    task_root = str(logical_path(str(task_root)))
    worker = str(worker) if worker is not None else None
    text = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                   '--format', '%i|%j|%T'], text=True, timeout=30)
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 3 or not re.fullmatch(r'[0-9_\[\],%+.-]+', fields[0]):
            raise RuntimeError('Unable to safely interpret existing Slurm jobs')
        job_id, name, state = fields
        if name == JOB_NAME:
            jobs.append(dict(job_id=job_id, name=name, state=state))
            continue
        detail = subprocess.check_output(['scontrol', 'show', 'job', '-o', job_id], text=True, timeout=30)
        fields_found = dict(re.findall(r'(?:^|\s)(WorkDir|Command)=(\S+)', detail))
        if not fields_found:
            raise RuntimeError(f'Cannot establish task identity for existing job {job_id}')
        same_root = any(value.rstrip('/') == task_root or value.startswith(task_root+'/')
                        for value in fields_found.values())
        same_worker = worker is not None and fields_found.get('Command') == worker
        if same_root or same_worker:
            jobs.append(dict(job_id=job_id, name=name, state=state))
    if jobs:
        raise RuntimeError(f'Existing duplicate sparse-Encbank task prevents another allocation: {jobs}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ('task-root', 'worker', 'trainer', 'model', 'init-adapter', 'data-dir', 'out', 'python', 'partition', 'gres'):
        ap.add_argument('--'+field, required=True)
    ap.add_argument('--account')
    ap.add_argument('--qos')
    ap.add_argument('--cpus', type=int, default=4)
    ap.add_argument('--memory', default='64G')
    ap.add_argument('--time-limit', default='24:00:00')
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--allow-training', action='store_true')
    ap.add_argument('--migration-receipt', required=True)
    ap.add_argument('--submit', action='store_true')
    args = ap.parse_args()
    command, script, locations = submission(args)
    if not args.submit:
        print(json.dumps(dict(mode='dry-run', command=command, stdin_script=script,
                              submitted=False, scope='Confirmed partition gpu, single L20D; no job submitted'), indent=2))
        return
    if sys.platform != 'linux' or not args.migration_receipt:
        raise RuntimeError('--submit requires Linux and a fresh --migration-receipt')
    import fcntl
    from slurm_gpu_guard import confined
    root = confined(args.task_root)
    confined(__file__, root)
    for field in ('worker', 'trainer', 'model', 'init_adapter', 'data_dir', 'out'):
        confined(getattr(args, field), root)
    receipt_path = confined(args.migration_receipt, root)
    migration_bytes = receipt_path.read_bytes()
    migration_sha256 = hashlib.sha256(migration_bytes).hexdigest()
    migration = migration_receipt(json.loads(migration_bytes))
    expected_out = PurePosixPath(args.out)
    for task in migration['tasks']:
        if (TASK_ROOT / task['destination_relative_out']).parent.parent != expected_out:
            raise ValueError('Migration task outputs differ from the requested --out directory')
    if not root.is_dir():
        raise FileNotFoundError(root)
    for field in ('worker', 'trainer', 'init_adapter', 'python'):
        if not Path(getattr(args, field)).is_file():
            raise FileNotFoundError(getattr(args, field))
    if not os.access(args.python, os.X_OK):
        raise ValueError('Python path is not an executable file')
    for path in (Path(args.model)/'config.json', Path(args.data_dir)/'train.jsonl', Path(args.data_dir)/'dev.jsonl'):
        if not path.is_file():
            raise FileNotFoundError(path)
    with confined(Path(HOME_BOUNDARY)/'.codex-encbank-sparse-l20d.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_other_jobs(root, args.worker)
        logs = confined(root/'logs', root)
        logs.mkdir(exist_ok=True)
        for path in locations.values():
            confined(path, root).mkdir(parents=True, exist_ok=True)
        current_migration_bytes = receipt_path.read_bytes()
        if hashlib.sha256(current_migration_bytes).hexdigest() != migration_sha256:
            raise RuntimeError('Migration receipt changed while preparing submission')
        migration_receipt(json.loads(current_migration_bytes))
        result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=60, check=True)
        raw_id = result.stdout.strip()
        if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', raw_id):
            raise RuntimeError(f'Ambiguous sbatch result; inspect squeue before retrying: {raw_id!r}')
        receipt = dict(job_id=raw_id.split(';')[0], raw_job_id=raw_id, job_name=JOB_NAME,
                       command=command, stdin_script=script, submitted_utc=datetime.now(timezone.utc).isoformat(),
                       migration_receipt=str(receipt_path),
                       migration_sha256=migration_sha256,
                       destination_backend='slurm-l20d', stderr=result.stderr)
        destination = confined(logs/f'submission-{receipt["job_id"]}.json', root)
        destination.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
