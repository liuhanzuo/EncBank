"""One-shot three-GPU continuation of the interrupted Hot24+PEFT 89-task run.

The four scored predecessor tasks are retained; only unscored tasks run again.
Preparation and submission are separate so an ambiguous sbatch is never retried.
"""
import fcntl
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SERIES = Path('/cluster/home/USER/qencbank_align_codex_20260911/tf27b_hot_live_20260921')
SOURCE = SERIES / 'hot24_peft_adaptive_full89_20260924'
REC = SERIES / 'hot24_peft_accuracy_recovery_20260924'
PY = '/cluster/home/USER/qencbank_runtime_20260911/python312/bin/python'
OLD_JOB = '128897'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def job_state(job):
    lines = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=State'], text=True).splitlines()
    return lines[0].split('|')[0] if lines else None


def scored_tasks():
    rows = {}
    for path in sorted((SOURCE / 'results').glob('*--hot/actual_result.json')):
        task = path.parent.name[:-5]
        receipt = read(path.parent / 'execution_receipt.json')
        result = read(path)
        parent = read(SOURCE / 'jobs' / ('wait-' + task + '--hot.json'))
        assert result['task_name'] == 'terminal-bench/' + task and result['verifier_result'] is not None
        assert not result.get('exception_info') and not receipt['exception']
        assert receipt['closure']['cgroup_empty'] and parent['actual_parent_wait'] and parent['exit_code'] == 0
        rows[task] = {'reward': result['verifier_result']['rewards']['reward'],
                      'result_sha256': sha(path),
                      'receipt_sha256': sha(path.parent / 'execution_receipt.json'),
                      'parent_wait_sha256': sha(SOURCE / 'jobs' / ('wait-' + task + '--hot.json'))}
    assert len(rows) == 4, rows
    return rows


def prepare():
    assert read(SOURCE / 'submission.json')['job_id'] == OLD_JOB
    plan = read(SOURCE / 'plan.json')
    completed = scored_tasks()
    remaining = [task for task in plan['tasks'] if task not in completed]
    assert len(plan['tasks']) == 89 and len(remaining) == 85
    assert len(set(remaining)) == 85
    assert not REC.exists()
    REC.mkdir()
    save(REC / 'selection.json', {'epoch': time.time(), 'predecessor_job': OLD_JOB,
        'predecessor_root': str(SOURCE), 'predecessor_plan_sha256': sha(SOURCE / 'plan.json'),
        'scored': completed, 'remaining': remaining, 'automatic_scientific_retries': 0,
        'reason': 'Predecessor model worker exited on shared-storage Remote I/O error; only four tasks scored.'})
    resources = {row['task']: row for row in plan['resource_inventory']}
    shards = [[], [], []]
    loads = [0, 0, 0]
    for task in sorted(remaining, key=lambda t: (-resources[t]['memory_mb'], t)):
        idx = min(range(3), key=lambda i: (loads[i], len(shards[i]), i))
        shards[idx].append(task)
        loads[idx] += resources[task]['memory_mb']
    manifest = read(SOURCE / 'source_manifest.json')
    for name, digest in manifest.items():
        assert sha(SOURCE / name) == digest, name
    runs = []
    for idx, tasks in enumerate(shards, 1):
        name = 'hot24_peft_acc_r' + str(idx) + '_20260924'
        root = SERIES / name
        assert not root.exists()
        root.mkdir()
        for source_name in manifest:
            (root / source_name).write_bytes((SOURCE / source_name).read_bytes())
        new_plan = read(root / 'plan.json')
        new_plan.update(remote_root=str(root), tasks=tasks,
            resource_inventory=[resources[task] for task in tasks],
            evidence_id=plan['evidence_id'] + '-ACC-R' + str(idx),
            authorization='2026-09-24 user: resume accuracy measurement, increase concurrency; infrastructure match no longer required',
            task_selection='Only unscored tasks from interrupted Hot89 job 128897',
            benchmark_accounting='Independent Hot89 aggregate: four prior scored tasks plus these disjoint continuations',
            task_concurrency=12, max_live_sessions=12, decode_batch_size=12,
            accuracy_recovery_parent=str(SOURCE), accuracy_recovery_predecessor_job=OLD_JOB,
            global_task_order=plan['tasks'], automatic_scientific_retries=0,
            node_policy='Any eligible GPU node')
        save(root / 'plan.json', new_plan)
        # The controller's former six-task pool was the actual task admission cap.
        controller = root / 'controller.py'
        code = controller.read_text()
        needle = 'ThreadPoolExecutor(max_workers=6)'
        assert code.count(needle) == 1
        controller.write_text(code.replace(needle, 'ThreadPoolExecutor(max_workers=12)'))
        for directory in ('jobs','logs','worker','results','pairs','mailbox','qualification','tmp','cache','compile_cache'):
            (root / directory).mkdir(exist_ok=True)
        save(root / 'accuracy_selection.json', {'tasks': tasks, 'predecessor_selection_sha256': sha(REC / 'selection.json'),
                                                'scored_predecessor_tasks': sorted(completed), 'arm': 'hot'})
        source_names = [n for n in manifest if n not in ('submit_full89.py', 'preflight.py', 'peft_preflight.json')]
        for source_name in source_names:
            if source_name.endswith('.py'):
                compile((root / source_name).read_text(), str(root / source_name), 'exec')
        save(root / 'source_manifest.json', {n: sha(root / n) for n in source_names})
        runs.append({'name': name, 'root': str(root), 'tasks': tasks, 'source_manifest_sha256': sha(root / 'source_manifest.json')})
    assert sum(len(x['tasks']) for x in runs) == 85
    save(REC / 'prepared.json', {'epoch': time.time(), 'selection_sha256': sha(REC / 'selection.json'),
                                 'runs': runs, 'scored_count': 4, 'remaining_count': 85})
    print(json.dumps({'prepared': [(x['name'],len(x['tasks'])) for x in runs]}), flush=True)


def submit():
    prepared = read(REC / 'prepared.json')
    assert prepared['selection_sha256'] == sha(REC / 'selection.json')
    assert scored_tasks().keys() == read(REC / 'selection.json')['scored'].keys()
    assert len(prepared['runs']) == 3
    with (SERIES / 'hot_accuracy_recovery_20260924.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = job_state(OLD_JOB)
        if old == 'RUNNING':
            # The model worker is already dead; close its controller/trial allocation
            # only after the scored-task evidence and fresh source roots are frozen.
            record = {'epoch':time.time(), 'old_job':OLD_JOB, 'old_state':old,
                      'scored_tasks':list(read(REC / 'selection.json')['scored'])}
            save(REC / 'predecessor_close_intent.json', record)
            child = subprocess.run(['scancel', OLD_JOB], capture_output=True, text=True)
            record.update(exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr)
            save(REC / 'predecessor_close_request.json', record)
            assert child.returncode == 0
            deadline = time.monotonic() + 120
            while job_state(OLD_JOB) == 'RUNNING' and time.monotonic() < deadline:
                time.sleep(2)
        assert job_state(OLD_JOB) and job_state(OLD_JOB).split(' ')[0] in ('CANCELLED','FAILED','COMPLETED'), job_state(OLD_JOB)
        for run in prepared['runs']:
            root = Path(run['root'])
            assert not (root / 'submission.json').exists()
            assert sha(root / 'source_manifest.json') == run['source_manifest_sha256']
            for name, digest in read(root / 'source_manifest.json').items():
                assert sha(root / name) == digest, (root,name)
            new_plan = read(root / 'plan.json')
            assert len(new_plan['tasks']) == len(run['tasks']) and new_plan['task_concurrency'] == 12
            job_name = 'encbank-hot-acc-r' + run['name'].split('_r')[1].split('_')[0]
            command = ['sbatch','--parsable','--job-name='+job_name,'--partition=gpu',
                       '--time=UNLIMITED','--no-requeue','--cpus-per-task=32','--mem=160G',
                       '--gres=gpu:nvidia_l20d:1','--chdir='+str(root),
                       '--output='+str(root/'logs/slurm-%j.out'),
                       '--error='+str(root/'logs/slurm-%j.err'),
                       '--wrap',PY+' -B '+str(root/'launch_trial_group.py')]
            record = {'status':'intent','epoch':time.time(),'root':str(root),
                      'tasks':len(run['tasks']),'task_concurrency':12,'decode_batch_size':12,
                      'source_manifest_sha256':run['source_manifest_sha256'],
                      'selection_sha256':prepared['selection_sha256'],
                      'predecessor_job':OLD_JOB,'argv':command,'automatic_scientific_retries':0}
            save(root/'submission.json',record)
            env = os.environ.copy()
            env.update(read(root/'submission_environment.json')['proxy_environment'])
            child = subprocess.run(command,capture_output=True,text=True,env=env)
            record.update(status='submitted' if child.returncode==0 else 'submission_failed',
                          exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,
                          actual_parent_wait=True)
            if child.returncode==0:
                record['job_id']=child.stdout.strip().split(';')[0]
            save(root/'submission.json',record)
            assert child.returncode==0,record
            print(json.dumps({'name':run['name'],'job_id':record['job_id'],
                              'tasks':len(run['tasks'])}),flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','submit')
    {'prepare':prepare,'submit':submit}[sys.argv[1]]()
