"""Resume four inspected 17:22 I/O failures after shared storage is writable."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = '/srv/encbank/Paper_Evolve/.venv/bin/python'
FAILED = {'train-r128-m1': '104746', 'quant-full-m0-s2': '104747',
          'quant-full-m1-s0': '104748', 'quant-full-m1-s1': '104749'}
PREDICTIONS = {'quant-full-m0-s2': 2713, 'quant-full-m1-s0': 1851,
               'quant-full-m1-s1': 1576}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def replace_bytes(path, data):
    tmp = path.with_name(path.name + '.recovery.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def dump(path, value):
    replace_bytes(path, (json.dumps(value, indent=2) + '\n').encode())


def preserve(path, target):
    if not path.exists():
        return
    if target.exists():
        assert path.read_bytes() == target.read_bytes(), (path, target)
        path.unlink()
    else:
        path.rename(target)


def scheduler(args):
    return subprocess.run(args, text=True, capture_output=True, check=True, timeout=30).stdout


def main():
    assert str(ROOT) == '/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
    lock = (ROOT / 'coordinator.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue = scheduler(['squeue', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T'])
    assert not any('qcm-q18-' in row for row in queue.splitlines()), queue
    accounting = scheduler(['sacct', '-j', ','.join(FAILED.values()), '-n', '-P',
                            '-o', 'JobIDRaw,State,ExitCode'])
    assert all(any(row.startswith(job + '|FAILED|1:0') for row in accounting.splitlines())
               for job in FAILED.values()), accounting
    # Write/read/fsync/rename must succeed before touching any task receipts.
    probe = ROOT / 'control/storage-health.recovery-probe'
    replace_bytes(probe, b'read-write-rename-ok\n')
    assert probe.read_bytes() == b'read-write-rename-ok\n'
    probe.unlink()
    for name, digest in json.loads((ROOT / 'package_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    train = ROOT / 'training/Qwen3.8-27B'
    # A read-only torch load already verified these optimizer/RNG values and
    # exited 0 at 18:43. No owner or GPU writer has run since. Reuse that result.
    checkpoint = json.loads((ROOT / 'control/checkpoint_verified_1843.json').read_text())
    step = checkpoint['step']
    assert step == 5500 and checkpoint['rank'] == checkpoint['alpha'] == 128 and checkpoint['j'] == 21
    assert checkpoint['target_steps'] == 8000 and checkpoint['cursor'] == (step + 8) * 4096
    assert checkpoint['optimizer_steps'] == [step] and checkpoint['rng_restorable']
    assert checkpoint['actual_read_only_process_exit'] == 0
    checkpoint_stat = (train / 'last.pt').stat()
    assert checkpoint_stat.st_size == checkpoint['checkpoint_bytes']
    verified_at = datetime.datetime.fromisoformat(checkpoint['at_shanghai']).timestamp()
    assert checkpoint_stat.st_mtime <= verified_at
    checkpoint_identity = dict(step=step, bytes=checkpoint_stat.st_size,
                               mtime_ns=checkpoint_stat.st_mtime_ns)
    plan = json.loads((ROOT / 'effective_plan.json').read_text())
    history = ROOT / 'maintenance_history/storage-recovery-20260918-1722'
    history.mkdir(parents=True, exist_ok=True)
    if (history / 'recovery.json').exists():
        raise RuntimeError('Recovery already committed; inspect owner status before a separate launch.')
    if not (history / 'effective_plan_before.json').exists():
        replace_bytes(history / 'effective_plan_before.json', (ROOT / 'effective_plan.json').read_bytes())
    replace_bytes(history / 'sacct_original.txt', accounting.encode())
    observed, worker_locks = {}, []
    for task in plan['tasks']:
        ident = task['id']
        if ident not in FAILED:
            continue
        out = ROOT / 'runs' / ident
        archive = out / 'attempt_history' / ('failed-' + FAILED[ident])
        receipt_path = out / 'submission.json'
        if not receipt_path.exists():
            receipt_path = archive / 'submission.json'
        assert json.loads(receipt_path.read_text())['job'] == FAILED[ident]
        assert not (ROOT / task['complete']).exists()
        worker_lock = (out / 'worker.lock').open('a+b')
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker_locks.append(worker_lock)
        if ident in PREDICTIONS:
            pred = (ROOT / task['complete']).parent / 'predictions.jsonl'
            data = pred.read_bytes()
            rows = [json.loads(line) for line in data.splitlines()]
            assert len(rows) == PREDICTIONS[ident]
            assert len({(r['id'], r['arm']) for r in rows}) == len(rows)
            observed[ident] = dict(saved_predictions=len(rows),
                                  unaltered_sha256=hashlib.sha256(data).hexdigest())
        archive.mkdir(parents=True, exist_ok=True)
    # Preserve all pre-resume training rows, including steps beyond checkpoint.
    old_log = history / 'training_preoutage.jsonl'
    log = train / 'training.jsonl'
    if not old_log.exists():
        replace_bytes(old_log, log.read_bytes())
    lines = old_log.read_bytes().splitlines(keepends=True)
    parsed = [json.loads(line) for line in lines]
    assert [r['step'] for r in parsed] == list(range(1, parsed[-1]['step'] + 1))
    assert step <= parsed[-1]['step'] < step + 250
    canonical = b''.join(line for line, row in zip(lines, parsed) if row['step'] <= step)
    assert log.read_bytes() in (old_log.read_bytes(), canonical)
    replace_bytes(log, canonical)
    progress_backup = history / 'training_progress_preoutage.json'
    if not progress_backup.exists() and (train / 'progress.json').exists():
        replace_bytes(progress_backup, (train / 'progress.json').read_bytes())
    dump(train / 'progress.json', dict(phase='awaiting_resume', step=step, target_steps=8000,
         elapsed_s=0, resume_from_step=step, discarded_uncheckpointed_steps=parsed[-1]['step'] - step))
    for task in plan['tasks']:
        ident = task['id']
        if ident in FAILED:
            out = ROOT / 'runs' / ident
            archive = out / 'attempt_history' / ('failed-' + FAILED[ident])
            for path in list(out.iterdir()):
                if path.is_file() and path.name != 'worker.lock':
                    preserve(path, archive / path.name)
            task['recovery_from_job'] = FAILED[ident]
            if task['kind'] == 'training':
                assert '--resume' in task['command']
                task['resume_step'] = step
                task['resume_checkpoint'] = checkpoint_identity
                task['prestart_duration_hours'] = [5, 7]
        if not (ROOT / task['complete']).exists():
            task.update(launcher='control/recovery.sh', io_wrapper=True)
    for name in ('coordinator_launch.json', 'status.json', 'coordinator_failure.json'):
        preserve(ROOT / name, history / name)
    dump(ROOT / 'effective_plan.json', plan)
    dump(history / 'recovery.json', dict(at=now(), reason='Inspected shared-storage Errno121 at 17:22',
         original_failures=FAILED, checkpoint=checkpoint, checkpoint_identity=checkpoint_identity, predictions=observed,
         original_log_last_step=parsed[-1]['step'], scientific_code_unchanged=True,
         io_wrapper='Transient atomic-write retries; progress snapshot optional. JSONL append never retried.',
         actual_failed_parent_receipts_preserved=True))
    for worker_lock in worker_locks:
        worker_lock.close()
    lock.close()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / 'coordinator.stdout.log').open('ab') as out, (ROOT / 'coordinator.stderr.log').open('ab') as err:
        child = subprocess.Popen([PYTHON, '-B', '-u', str(ROOT / 'control/coordinator.py')],
              cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    launch = dict(pid=child.pid, at=now(), remote_root=str(ROOT),
                  script=str(ROOT / 'control/coordinator.py'), maximum_gpu_requests=4,
                  recovery_evidence=str(history / 'recovery.json'))
    dump(ROOT / 'coordinator_launch.json', launch)
    print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    main()
