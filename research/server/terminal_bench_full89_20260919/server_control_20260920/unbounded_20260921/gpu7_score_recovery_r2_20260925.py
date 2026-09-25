"""Recover only the 44 zero-call tasks from four canceled score-only shards.

Each source shard becomes two immutable children. One child per source may use
gpu7; the other excludes gpu7. No task result or model artifact is copied.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


S = Path('/cluster/home/liuhanzuo/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
R = U / 'gpu7_score_recovery_r2_20260925'
REG = U / 'scale4_20260921' / 'registry.json'
LOCK = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
SOURCES = [
    ('k12', 2, '141571'), ('k12', 4, '141573'),
    ('k48', 1, '141576'), ('k48', 3, '141578'),
]


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


def accounting(job_id):
    out = subprocess.check_output(
        ['sacct', '-n', '-P', '-X', '-j', job_id,
         '--format=JobIDRaw,State,ExitCode,NodeList'], text=True, timeout=20)
    rows = [line.split('|') for line in out.splitlines() if line.strip()]
    assert len(rows) == 1 and rows[0][0] == job_id, rows
    return dict(job_id=job_id, state=rows[0][1], exit_code=rows[0][2], node=rows[0][3])


def source_evidence(arm, index, job_id):
    root = S / f'{arm}_score_resume_{index}_20260925'
    plan = read(root / 'plan.json')
    receipt = read(root / 'submission.json')
    manifest = read(root / 'server_source_manifest.json')
    assert receipt['status'] == 'submitted' and str(receipt['job_id']) == job_id
    assert all(sha(root / file) == digest for file, digest in manifest.items())
    acct = accounting(job_id)
    assert acct['state'].startswith('CANCELLED') and acct['node'] == 'lj-gpu7', acct
    closure = read(root / 'execution' / 'harbor_receipt.json')
    owner = read(root / 'execution' / 'owner_complete.json')
    assert closure['all_task_parents_waited'] and closure['tasks_closed'] == 0
    assert set(closure['pending_not_launched']) == set(plan['tasks'])
    assert owner['all_task_launches_accounted'] and set(owner['remaining_tasks']) == set(plan['tasks'])
    launches = root / 'execution' / 'launches'
    assert not launches.exists() or not list(launches.glob('*.json'))
    box = Path(plan['rpc_root']) / plan['arm']
    assert not box.exists() or not list(box.glob('*.request.json'))
    results = Path(plan['results_root'])
    assert not results.exists() or not list(results.rglob('result.json'))
    events = root / 'run_comem' / 'events.jsonl'
    event_text = events.read_text() if events.exists() else ''
    assert 'request_start' not in event_text and 'request_complete' not in event_text
    assert len(plan['tasks']) == len(set(plan['tasks']))
    return dict(root=str(root), arm=arm, index=index, old_job_id=job_id,
                old_plan_sha256=sha(root / 'plan.json'),
                old_manifest_sha256=sha(root / 'server_source_manifest.json'),
                old_slurm=acct, zero_requests=True, zero_results=True,
                tasks=list(plan['tasks']))


def halves(plan):
    inventory = {row['task']: row for row in plan['resource_inventory']}
    tasks = list(plan['tasks'])
    a, b = [], []
    load = [0, 0]
    for name in sorted(tasks, key=lambda t: (-inventory[t]['memory_mb'], t)):
        side = min(range(2), key=lambda i: (load[i], len((a, b)[i]), i))
        (a, b)[side].append(name)
        load[side] += inventory[name]['memory_mb']
    assert a and b and set(a) | set(b) == set(tasks) and not set(a) & set(b)
    return (a, b), inventory


def prepare():
    assert read(REG)['version'] == 6
    assert not R.exists(), R
    evidence = [source_evidence(*row) for row in SOURCES]
    assert sum(len(x['tasks']) for x in evidence) == 44
    R.mkdir()
    save(R / 'source_evidence.json', evidence)
    runs = []
    for old in evidence:
        source = Path(old['root'])
        plan = read(source / 'plan.json')
        manifest = read(source / 'server_source_manifest.json')
        source_runtime = str(Path(plan['rpc_root']).parent)
        split, inventory = halves(plan)
        for side, tasks in zip(('a', 'b'), split):
            name = f"{old['arm']}_score_resume_{old['index']}_r2{side}_20260925"
            target = S / name
            assert not target.exists(), target
            target.mkdir()
            runtime = str(Path(source_runtime).parent / name)
            ipc = f"/cluster/home/liuhanzuo/qcomem_runtime_20260911/t89r2{old['arm']}{old['index']}{side}"
            job_name = f"encbank-tb-{old['arm']}-sr2-{old['index']}{side}"
            swaps = ((str(source), str(target)), (source_runtime, runtime),
                     (plan['ipc_root'], ipc), (plan['job_name'], job_name))
            for file in manifest:
                content = (source / file).read_bytes()
                for before, after in swaps:
                    content = content.replace(before.encode(), after.encode())
                (target / file).write_bytes(content)
            child = read(target / 'plan.json')
            child.update(tasks=tasks, resource_inventory=[inventory[t] for t in tasks],
                         remote_root=str(target), ipc_root=ipc, job_name=job_name,
                         evidence_id=child['evidence_id'] + f'-ZERO-CALL-R2-{side}',
                         zero_call_recovery_parent=str(source),
                         zero_call_recovery_source_evidence_sha256=sha(R / 'source_evidence.json'),
                         allocation_policy='Any eligible node; at most four benchmark GPUs on lj-gpu7; Reader excluded.')
            assert child['max_new_tokens'] is None
            save(target / 'plan.json', child)
            template = read(target / 'comem_harbor_template.json')
            template['job_name'] = name
            template['jobs_dir'] = child['results_root']
            template['tasks'] = [{'path': str(Path(child['task_root']) / task)} for task in tasks]
            save(target / 'comem_harbor_template.json', template)
            slurm = target / 'server.slurm'
            script = slurm.read_text()
            assert '#SBATCH --time=0' in script and '#SBATCH --gres=gpu:nvidia_l20d:1' in script
            assert '#SBATCH --exclude=lj-gpu7' not in script
            if side == 'b':
                script = script.replace('#SBATCH --partition=gpu\n',
                                        '#SBATCH --partition=gpu\n#SBATCH --exclude=lj-gpu7\n')
            slurm.write_text(script)
            for file in manifest:
                if file.endswith('.py'):
                    compile((target / file).read_text(), str(target / file), 'exec')
            save(target / 'server_source_manifest.json',
                 {file: sha(target / file) for file in manifest})
            save(target / 'score_resume_selection.json',
                 dict(arm=old['arm'], tasks=tasks, old_root=str(source),
                      old_job_id=old['old_job_id'], zero_model_calls=True,
                      source_evidence_sha256=sha(R / 'source_evidence.json')))
            runs.append(dict(arm=old['arm'], side=side, root=str(target), tasks=tasks,
                             gpu7_eligible=(side == 'a'),
                             source_manifest_sha256=sha(target / 'server_source_manifest.json')))
    assert len(runs) == 8 and sum(row['gpu7_eligible'] for row in runs) == 4
    assert len({(row['arm'], task) for row in runs for task in row['tasks']}) == 44
    save(R / 'prepared.json', dict(epoch=time.time(), previous_registry_sha256=sha(REG),
                                  source_evidence_sha256=sha(R / 'source_evidence.json'), runs=runs))
    print(json.dumps({'prepared': [(Path(x['root']).name, len(x['tasks']), x['gpu7_eligible']) for x in runs]}))


def activate():
    prepared = read(R / 'prepared.json')
    assert not (R / 'activated.json').exists()
    assert sha(REG) == prepared['previous_registry_sha256']
    assert sha(R / 'source_evidence.json') == prepared['source_evidence_sha256']
    registry = read(REG)
    assert registry['version'] == 6
    previous = {arm: [task for row in registry['runs'] if row['arm'] == arm for task in row['tasks']]
                for arm in ('dense', 'k12', 'k48')}
    for row in prepared['runs']:
        source = next(old for old in registry['runs'] if old['root'] == read(S / Path(row['root']).name / 'score_resume_selection.json')['old_root'])
        transfer = set(row['tasks'])
        assert transfer <= set(source['tasks'])
        source['tasks'] = [task for task in source['tasks'] if task not in transfer]
        source.setdefault('zero_call_recovery_transfers', []).append(row['root'])
        registry['runs'].append(dict(id=Path(row['root']).name, arm=row['arm'],
                                     root=row['root'], tasks=row['tasks'],
                                     role='zero_call_recovery_20260925',
                                     source_manifest_sha256=row['source_manifest_sha256']))
    for arm, prior in previous.items():
        current = [task for row in registry['runs'] if row['arm'] == arm for task in row['tasks']]
        assert len(current) == len(set(current)) and set(current) == set(prior)
    registry.update(version=7, gpu7_benchmark_cap=4,
                    gpu7_reader_excluded_from_benchmark_cap=True,
                    zero_call_recovery_path=str(R / 'prepared.json'),
                    zero_call_recovery_epoch=time.time())
    save(R / 'registry_v6_before_recovery.json', read(REG))
    save(REG, registry)
    save(R / 'activated.json', dict(epoch=time.time(), registry_sha256=sha(REG),
                                  previous_sha256=prepared['previous_registry_sha256']))
    print(json.dumps({'activated': True, 'registry_sha256': sha(REG)}))


def submit():
    prepared = read(R / 'prepared.json')
    assert sha(REG) == read(R / 'activated.json')['registry_sha256']
    assert sha(R / 'source_evidence.json') == prepared['source_evidence_sha256']
    health = read(R / 'gpu7_storage_health.json')
    assert health['passed'] and time.time() - health['epoch'] < 3600
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = subprocess.check_output(
            ['squeue', '-h', '-u', 'liuhanzuo', '-o', '%i|%j|%T|%N|%b'], text=True)
        main_gpu7 = [line for line in output.splitlines() if '|encbank-tb-' in line
                     and '|lj-gpu7|' in line and 'gpu:' in line]
        assert len(main_gpu7) + sum(row['gpu7_eligible'] for row in prepared['runs']) <= 4, main_gpu7
        for row in prepared['runs']:
            root = Path(row['root'])
            assert not (root / 'submission.json').exists(), root
            assert sha(root / 'server_source_manifest.json') == row['source_manifest_sha256']
            assert all(sha(root / file) == digest
                       for file, digest in read(root / 'server_source_manifest.json').items())
            if not row['gpu7_eligible']:
                assert '#SBATCH --exclude=lj-gpu7\n' in (root / 'server.slurm').read_text()
            receipt = dict(status='intent', epoch=time.time(), root=str(root),
                           arm=row['arm'], tasks=len(row['tasks']),
                           zero_call_recovery_source_evidence_sha256=prepared['source_evidence_sha256'],
                           source_manifest_sha256=row['source_manifest_sha256'],
                           gpu7_eligible=row['gpu7_eligible'],
                           gpu7_cap=4, automatic_scientific_retries=0)
            save(root / 'submission.json', receipt)
            child = subprocess.run(['sbatch', '--parsable', str(root / 'server.slurm')],
                                   capture_output=True, text=True, timeout=30)
            receipt.update(status='submitted' if child.returncode == 0 else 'submission_failed',
                           exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr,
                           actual_parent_wait=True)
            if child.returncode == 0:
                receipt['job_id'] = child.stdout.strip().split(';')[0]
            save(root / 'submission.json', receipt)
            assert child.returncode == 0, receipt
            print(json.dumps({'root': root.name, 'job_id': receipt['job_id'],
                              'tasks': len(row['tasks']), 'gpu7_eligible': row['gpu7_eligible']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'activate', 'submit'))
    globals()[parser.parse_args().phase]()
