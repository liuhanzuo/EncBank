"""Fresh Hot89 source roots after three pre-model startup failures on GPU7.

Retains the 4 scored predecessor tasks and the 85-task partition.  Skips only
the install-only Harbor qualification; actual task worlds and model checks run.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

S = Path('/cluster/home/USER/qencbank_align_codex_20260911/tf27b_hot_live_20260921')
R = S / 'hot24_peft_score_only_recovery_20260924'
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
    runs = []
    for index, job in [(1, 141091), (2, 141077), (3, 141078)]:
        prior = S / f'hot24_peft_acc_proxy_r{index}_20260924'
        origin = S / f'hot24_peft_acc_r{index}_20260924'
        assert state(job) == 'FAILED' and read(prior / 'submission.json')['job_id'] == str(job)
        assert not list((prior / 'jobs').glob('launch-*--hot.json'))
        assert not list((prior / 'results').glob('*--hot/actual_result.json'))
        assert not list((prior / 'mailbox').rglob('*.request.json'))
        manifest = read(prior / 'source_manifest.json')
        assert all(sha(prior / x) == digest for x, digest in manifest.items())
        name = f'hot24_peft_score_only_r{index}_20260924'
        root = S / name
        assert not root.exists()
        root.mkdir()
        for source in list(manifest) + ['preflight.py', 'accuracy_selection.json']:
            (root / source).write_bytes((prior / source).read_bytes())
        preflight = read(origin / 'peft_preflight.json')
        assert preflight['passed'] and preflight['model_calls'] == 0
        (root / 'peft_preflight.json').write_bytes((origin / 'peft_preflight.json').read_bytes())
        plan = read(root / 'plan.json')
        plan.update(remote_root=str(root), evidence_id=plan['evidence_id'] + '-SCORE-ONLY',
                    accuracy_only_skip_qualification=True,
                    startup_preflight_source=str(origin / 'peft_preflight.json'),
                    score_only_parent=str(prior), score_only_parent_job=job,
                    score_only_reason='No formal trials or model calls; user waived prior install-only qualification.')
        save(root / 'plan.json', plan)
        launcher = root / 'launch_trial_group.py'
        code = launcher.read_text()
        before = ("    qual=start('environment_qualification',[P['harbor_python'],'-B',str(ROOT/'parallel_trials.py'),'qualify'],env,task_cpus)\n"
                  "    assert wait('environment_qualification',qual)==0\n"
                  "    receipt=json.loads((ROOT/'qualification'/(P['tasks'][0]+'--qualify')/'execution_receipt.json').read_text())\n"
                  "    assert receipt['closure']['cgroup_empty'] and receipt['model_calls']==0 and not receipt['exception']\n")
        after = ("    if not P.get('accuracy_only_skip_qualification'):\n"
                 "        qual=start('environment_qualification',[P['harbor_python'],'-B',str(ROOT/'parallel_trials.py'),'qualify'],env,task_cpus)\n"
                 "        assert wait('environment_qualification',qual)==0\n"
                 "        receipt=json.loads((ROOT/'qualification'/(P['tasks'][0]+'--qualify')/'execution_receipt.json').read_text())\n"
                 "        assert receipt['closure']['cgroup_empty'] and receipt['model_calls']==0 and not receipt['exception']\n"
                 "    else:\n"
                 "        save(OUT/'qualification_skipped.json',dict(scope='install_only',user_authorized=True,model_calls=0,epoch=time.time()))\n")
        assert code.count(before) == 1
        launcher.write_text(code.replace(before, after))
        for directory in ('jobs', 'logs', 'worker', 'results', 'pairs', 'mailbox', 'qualification', 'tmp', 'cache', 'compile_cache'):
            (root / directory).mkdir(exist_ok=True)
        names = list(manifest) + ['peft_preflight.json']
        for source in names:
            if source.endswith('.py'):
                compile((root / source).read_text(), str(root / source), 'exec')
        save(root / 'source_manifest.json', {source: sha(root / source) for source in names})
        runs.append({'index': index, 'root': str(root), 'tasks': plan['tasks'],
                     'source_manifest_sha256': sha(root / 'source_manifest.json'),
                     'predecessor_job': job})
    assert len(runs) == 3 and len(set(task for run in runs for task in run['tasks'])) == 85
    save(R / 'prepared.json', {'epoch': time.time(), 'runs': runs})
    print(json.dumps({'prepared': [(r['index'], len(r['tasks'])) for r in runs]}), flush=True)


def submit():
    runs = read(R / 'prepared.json')['runs']
    assert len(runs) == 3
    with (S / 'hot_accuracy_recovery_20260924.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for run in runs:
            root = Path(run['root'])
            assert state(run['predecessor_job']) == 'FAILED'
            assert not (root / 'submission.json').exists()
            assert sha(root / 'source_manifest.json') == run['source_manifest_sha256']
            assert all(sha(root / x) == digest for x, digest in read(root / 'source_manifest.json').items())
            argv = ['sbatch', '--parsable', '--job-name=encbank-hot-score-r' + str(run['index']),
                    '--partition=gpu', '--reservation=SITE_GPU_RESERVATION', '--time=UNLIMITED',
                    '--no-requeue', '--cpus-per-task=32', '--mem=160G', '--gres=gpu:nvidia_l20d:1',
                    '--exclude=lj-gpu1', '--chdir=' + str(root),
                    '--output=' + str(root / 'logs/slurm-%j.out'),
                    '--error=' + str(root / 'logs/slurm-%j.err'), '--wrap',
                    PY + ' -B ' + str(root / 'launch_trial_group.py')]
            receipt = {'status': 'intent', 'epoch': time.time(), 'root': str(root),
                       'tasks': len(run['tasks']), 'task_concurrency': 12,
                       'predecessor_job': run['predecessor_job'], 'argv': argv,
                       'source_manifest_sha256': run['source_manifest_sha256'],
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
            print(json.dumps({'index': run['index'], 'job': receipt['job_id']}), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in {'prepare', 'submit'}
    {'prepare': prepare, 'submit': submit}[sys.argv[1]]()
