"""Run an unchanged QA queue after an explicitly drained training migration.

Only the imported backend queue's training-priority function is replaced. The
original GPU admission, shared locks, workers, completion checks and prefix
dependency remain in force. This does not mark any training job complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path('/data/liuhanzuo/encbank_v2_20260908')
CODE = ROOT / 'workspace/exp/encbank_block_attention_20260911'
TRAIN_OUT = ROOT / 'outputs/sparse_encbank_20260911'
DESTINATION = '/srv/encbank/encbank_sparse_slurm_20260912'
SCHEMA = 'encbank-qa-slurm-resource-handoff-v1'
POLICY = 'source-training-resources-migrated-not-training-complete'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(value):
    path = Path(value).resolve()
    if ROOT not in path.parents:
        raise ValueError('Handoff paths must stay within the named source root')
    return path


def same_identity(current, expected):
    return bool(current and current.get('pid') == expected['pid']
                and current.get('start_ticks') == expected['start_ticks'])


def source_processes(queue):
    """Strict same-UID scan plus the original script/--out worker detection."""
    found = queue.live_trainers(CODE / 'train_sparse.py', ROOT / 'outputs')
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            identity = queue.process_identity(int(directory.name))
            if identity is None:
                # Processes may disappear during enumeration; unreadable live
                # same-user processes cannot prove this source is drained.
                if (directory / 'cmdline').exists():
                    raise RuntimeError('Cannot inspect a live same-user process')
                continue
            args = identity['argv']
            relevant = any(str(CODE / name) in args for name in
                           ('remote_sparse_queue.py', 'train_sparse.py'))
            if not relevant and any(Path(a).name in ('remote_sparse_queue.py', 'train_sparse.py') for a in args):
                cwd = (directory / 'cwd').resolve(strict=True)
                relevant = any((cwd / a).resolve() in
                               (CODE / 'remote_sparse_queue.py', CODE / 'train_sparse.py') for a in args)
            if relevant and identity not in found:
                found.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def validate_declaration(path, expected_sha, queue):
    """Reload every time, including after the unchanged controller's GPU lock."""
    path = confined(path)
    if sha(path) != expected_sha:
        raise ValueError('Migration declaration changed or was revoked')
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema') != SCHEMA or value.get('policy') != POLICY
            or value.get('source_root') != str(ROOT) or value.get('source_host') != 'longjing-1'
            or value.get('destination_root') != DESTINATION
            or value.get('source_priority_released') is not True
            or value.get('training_complete_asserted') is not False):
        raise ValueError('Unknown or non-resource-only handoff')
    for name, digest in value['source_sha256'].items():
        if Path(name).name != name or sha(CODE / name) != digest:
            raise ValueError('Handoff source changed: ' + name)
    required = {'qa_slurm_handoff.py', 'remote_sparse_queue.py', 'remote_gpu_guard.py',
                'remote_backend_quality_queue.py', 'remote_prefix_quality_queue.py', 'submit_slurm_sparse.py'}
    if set(value['source_sha256']) != required:
        raise ValueError('Incomplete handoff source binding')
    migration_path = confined(value['migration_receipt'])
    if sha(migration_path) != value['migration_sha256']:
        raise ValueError('Migration receipt changed')
    from submit_slurm_sparse import migration_receipt
    migration = migration_receipt(json.loads(migration_path.read_text()), require_fresh=False)
    source_receipt_path = confined(value['source_migration_receipt'])
    if (sha(source_receipt_path) != value['source_migration_sha256']
            or json.loads(source_receipt_path.read_text()) != migration):
        raise ValueError('Original source receipt differs from accepted migration')
    state_path = confined(migration['source_queue_path'])
    if state_path != TRAIN_OUT / 'queue.json' or sha(state_path) != migration['source_queue_sha256']:
        raise ValueError('Source training queue changed')
    state = json.loads(state_path.read_text())
    if (state['controller'] != migration['source_controller']
            or state['jobs'] != migration['source_jobs']
            or value['tasks'] != migration['tasks']):
        raise ValueError('Source controller, jobs or destination mapping changed')
    submission_path = confined(value['destination_submission'])
    if sha(submission_path) != value['submission_sha256']:
        raise ValueError('Actual Slurm submission evidence changed')
    submission = json.loads(submission_path.read_text())
    job = str(submission.get('job_id', ''))
    command = submission.get('command', [])
    if (not re.fullmatch(r'[1-9][0-9]*', job) or job != str(value['destination_job_id'])
            or submission.get('raw_job_id', '').split(';')[0] != job
            or submission.get('job_name') != 'encbank-sparse-1gpu'
            or submission.get('destination_backend') != 'slurm-l20d'
            or submission.get('migration_sha256') != value['migration_sha256']
            or not command or command[0] != 'sbatch' or '--parsable' not in command
            or not submission.get('submitted_utc') or not submission.get('stdin_script')):
        raise ValueError('Require actual accepted single-GPU destination submission')
    for key, expected in (('--chdir', DESTINATION), ('--gres', 'gpu:nvidia_l20d:1'),
                          ('--nodes', '1'), ('--ntasks', '1'), ('--job-name', 'encbank-sparse-1gpu')):
        if command.count(key) != 1 or command[command.index(key) + 1] != expected:
            raise ValueError('Slurm submission destination/resources differ')
    owner = migration['source_controller']
    if same_identity(queue.process_identity(owner['pid']), owner):
        raise RuntimeError('Original source training controller is still alive')
    processes = source_processes(queue)
    if processes:
        raise RuntimeError('Source training controller/worker reappeared: ' +
                           ','.join(str(p['pid']) for p in processes))
    return dict(destination_job_id=job, source_queue_sha256=migration['source_queue_sha256'],
                source_jobs_remain_queued=True, source_controller_absent=True,
                source_training_processes=processes, training_complete_asserted=False)


def install_priority(backend, queue, declaration, declaration_sha, log):
    original = backend.training_priority_clear
    def priority():
        evidence = None
        try:
            clear, reason = original()
            if not clear:
                evidence = validate_declaration(declaration, declaration_sha, queue)
                clear, reason = True, 'training_migrated_to_slurm_source_drained'
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, IndexError) as exc:
            clear, reason = False, 'training_migration_unavailable:' + type(exc).__name__ + ':' + str(exc)
        event = dict(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), policy=POLICY,
                     wrapper_sha256=sha(__file__), declaration_sha256=declaration_sha,
                     clear=clear, reason=reason, evidence=evidence)
        # A failed provenance write must also prevent GPU admission.
        with confined(log).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, sort_keys=True) + '\n')
        return clear, reason
    backend.training_priority_clear = priority
    return priority


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-kind', choices=('backend', 'prefix'), required=True)
    parser.add_argument('--migration-declaration', type=Path, required=True)
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--handoff-log', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('queue_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('This handoff runs only on the source Linux server')
    import remote_sparse_queue as queue
    import remote_backend_quality_queue as backend
    priority = install_priority(backend, queue, args.migration_declaration,
                                args.declaration_sha256, args.handoff_log)
    selected = backend if args.queue_kind == 'backend' else importlib.import_module('remote_prefix_quality_queue')
    if args.queue_kind == 'prefix' and selected.backend_queue is not backend:
        raise RuntimeError('Prefix must use the same imported backend module')
    if args.check_only:
        training = priority()
        effective = training if args.queue_kind == 'backend' else selected.priorities_clear()
        print(json.dumps(dict(training_priority=training, effective_priority=effective,
                              training_complete_asserted=False)))
        return
    forwarded = args.queue_arguments
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    sys.argv = [selected.__file__] + forwarded
    selected.main()


if __name__ == '__main__':
    main()
