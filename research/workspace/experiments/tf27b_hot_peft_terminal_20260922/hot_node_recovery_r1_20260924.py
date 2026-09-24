"""Move first pre-model Hot89 shard away from observed BeeGFS-stalled lj-gpu1."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S = Path('/cluster/home/USER/qencbank_align_codex_20260911/tf27b_hot_live_20260921')
R = S / 'hot24_peft_node_recovery_r1_20260924'
PY = '/cluster/home/USER/qencbank_runtime_20260911/python312/bin/python'


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


def state(job):
    lines = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=State'], text=True).splitlines()
    return lines[0].split('|')[0].split()[0] if lines else None


def prepare():
    assert not R.exists()
    R.mkdir()
    rows = []
    for index, old_job in [(1, 140658)]:
        old = S / f'hot24_peft_acc_r{index}_20260924'
        assert state(old_job) == 'CANCELLED'
        assert read(old / 'submission.json')['job_id'] == str(old_job)
        assert not list((old / 'jobs').glob('launch-*--hot.json'))
        assert not list((old / 'results').glob('*--hot/actual_result.json'))
        assert not list((old / 'mailbox').rglob('*.request.json'))
        old_plan = read(old / 'plan.json')
        name = f'hot24_peft_acc_proxy_r{index}_20260924'
        new = S / name
        assert not new.exists()
        new.mkdir()
        manifest = read(old / 'source_manifest.json')
        assert all(sha(old / x) == digest for x, digest in manifest.items())
        for source in list(manifest) + ['preflight.py', 'accuracy_selection.json']:
            (new / source).write_bytes((old / source).read_bytes())
        plan = read(new / 'plan.json')
        assert plan['tasks'] == old_plan['tasks'] and plan['task_concurrency'] == 12
        plan.update(remote_root=str(new), evidence_id=plan['evidence_id'] + '-NODE-RECOVERY',
                    node_policy='Any qualified node except observed BeeGFS-stalled lj-gpu1',
                    node_recovery_parent=str(old), node_recovery_parent_job=old_job,
                    node_recovery_reason='No benchmark trial launched or model request; worker stuck in BeeGFS revalidateIntent.')
        save(new / 'plan.json', plan)
        for directory in ('jobs', 'logs', 'worker', 'results', 'pairs', 'mailbox', 'qualification', 'tmp', 'cache', 'compile_cache'):
            (new / directory).mkdir(exist_ok=True)
        for source in manifest:
            if source.endswith('.py'):
                compile((new / source).read_text(), str(new / source), 'exec')
        save(new / 'source_manifest.json', {source: sha(new / source) for source in manifest})
        rows.append({'index': index, 'root': str(new), 'tasks': plan['tasks'],
                     'source_manifest_sha256': sha(new / 'source_manifest.json'),
                     'predecessor_job': old_job})
    assert len(rows) == 1 and len(set(t for row in rows for t in row['tasks'])) == 28
    save(R / 'prepared.json', {'epoch': time.time(), 'runs': rows})
    print(json.dumps({'prepared': [(row['index'], len(row['tasks'])) for row in rows]}), flush=True)


def submit():
    rows = read(R / 'prepared.json')['runs']
    assert len(rows) == 1
    with (S / 'hot_accuracy_recovery_20260924.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for row in rows:
            root = Path(row['root'])
            assert state(row['predecessor_job']) == 'CANCELLED'
            assert not (root / 'submission.json').exists()
            assert sha(root / 'source_manifest.json') == row['source_manifest_sha256']
            assert all(sha(root / x) == digest for x, digest in read(root / 'source_manifest.json').items())
            argv = ['sbatch', '--parsable', '--job-name=encbank-hot-px-r' + str(row['index']),
                    '--partition=gpu', '--time=UNLIMITED', '--no-requeue', '--cpus-per-task=32',
                    '--mem=160G', '--gres=gpu:nvidia_l20d:1', '--exclude=lj-gpu1',
                    '--chdir=' + str(root), '--output=' + str(root / 'logs/slurm-%j.out'),
                    '--error=' + str(root / 'logs/slurm-%j.err'), '--wrap',
                    PY + ' -B ' + str(root / 'launch_trial_group.py')]
            receipt = {'status': 'intent', 'epoch': time.time(), 'root': str(root),
                       'tasks': len(row['tasks']), 'task_concurrency': 12,
                       'predecessor_job': row['predecessor_job'], 'argv': argv,
                       'source_manifest_sha256': row['source_manifest_sha256'],
                       'automatic_scientific_retries': 0}
            save(root / 'submission.json', receipt)
            env = os.environ.copy()
            env.update(read(root / 'submission_environment.json')['proxy_environment'])
            child = subprocess.run(argv, capture_output=True, text=True, env=env)
            receipt.update(status='submitted' if child.returncode == 0 else 'submission_failed',
                           exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr,
                           actual_parent_wait=True)
            if child.returncode == 0:
                receipt['job_id'] = child.stdout.strip().split(';')[0]
            save(root / 'submission.json', receipt)
            assert child.returncode == 0, receipt
            print(json.dumps({'index': row['index'], 'job': receipt['job_id']}), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in {'prepare', 'submit'}
    {'prepare': prepare, 'submit': submit}[sys.argv[1]]()
