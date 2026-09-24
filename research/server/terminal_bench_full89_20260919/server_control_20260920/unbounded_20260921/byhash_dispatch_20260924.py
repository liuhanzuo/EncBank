"""Admit prepared by-hash shards one time each as drained GPU jobs exit."""
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

S = Path('/cluster/home/USER/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
R = U / 'byhash_recovery_20260924'
REG = U / 'scale4_20260921' / 'registry.json'
MAX_MAIN_GPU_REQUESTS = 9
STATUS = Path('/tmp/encbank_byhash_dispatch_20260924.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def count_main():
    text = subprocess.check_output(['squeue', '-r', '-u', 'USER', '-h',
                                    '-o', '%i|%j|%T|%b'], text=True)
    rows = [line for line in text.splitlines()
            if ('encbank-tb-' in line or 'qencbank-tb-' in line)
            and 'gpu' in line.split('|')[-1].lower()]
    return rows


def submit_one(run):
    root = Path(run['root'])
    assert not (root / 'submission.json').exists()
    assert sha(root / 'server_source_manifest.json') == run['source_manifest_sha256']
    assert all(sha(root / name) == digest
               for name, digest in read(root / 'server_source_manifest.json').items())
    record = {'status': 'intent', 'epoch': time.time(), 'root': str(root),
              'arm': run['arm'], 'tasks': len(run['tasks']),
              'parent_job': run['parent_job'],
              'source_manifest_sha256': run['source_manifest_sha256'],
              'automatic_scientific_retries': 0}
    save(root / 'submission.json', record)
    child = subprocess.run(['sbatch', '--parsable', str(root / 'server.slurm')],
                           capture_output=True, text=True)
    record.update(status='submitted' if child.returncode == 0 else 'submission_failed',
                  exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr,
                  actual_parent_wait=True)
    if child.returncode == 0:
        record['job_id'] = child.stdout.strip().split(';')[0]
    save(root / 'submission.json', record)
    assert child.returncode == 0, record
    print(json.dumps({'root': str(root), 'job': record['job_id']}), flush=True)
    return record['job_id']


def main():
    assert read(REG)['version'] == 5
    assert sha(REG) == read(R / 'activated.json')['registry_sha256']
    rows = read(R / 'prepared.json')['runs']
    order = sorted(rows, key=lambda x: (x['index'], x['arm'] != 'k12'))
    assert len(order) == 8
    lock_path = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
    with lock_path.open('a') as admission:
        while order:
            fcntl.flock(admission, fcntl.LOCK_EX)
            try:
                queued = count_main()
                assert len(queued) <= MAX_MAIN_GPU_REQUESTS, queued
                if len(queued) < MAX_MAIN_GPU_REQUESTS:
                    run = order.pop(0)
                    submit_one(run)
                    queued = count_main()
                receipt = {'state': 'ADMITTING', 'epoch': time.time(),
                           'main_running_plus_pending': len(queued), 'main_queue': queued,
                           'remaining_roots': [r['root'] for r in order]}
                save(STATUS, receipt)
                save(R / 'dispatcher_status.json', receipt)
            finally:
                fcntl.flock(admission, fcntl.LOCK_UN)
            if order:
                time.sleep(30)
    done = {'state': 'SUBMITTED_ALL', 'epoch': time.time(), 'roots': [r['root'] for r in rows]}
    save(STATUS, done)
    save(R / 'dispatcher_complete.json', done)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(STATUS, {'state': 'FAILED', 'epoch': time.time(), 'error': repr(exc)})
        raise
