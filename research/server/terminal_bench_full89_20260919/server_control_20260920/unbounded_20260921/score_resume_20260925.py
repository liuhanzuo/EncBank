"""Prepare and submit immutable score-only k12/k48 shards after storage recovery."""

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
R = U / 'score_resume_20260925'
REG = U / 'scale4_20260921' / 'registry.json'
SOURCE = {arm: S / f'{arm}_byhash_recovery_2_20260924' for arm in ('k12', 'k48')}
ORIGINAL = {arm: S / f'{arm}_unbounded_20260921' for arm in ('k12', 'k48')}
SHARDS = {'k12': 6, 'k48': 3}
CONCURRENCY = {'k12': 4, 'k48': 6}
LOCK = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def main_queue():
    output = subprocess.check_output(
        ['squeue', '-r', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%b'], text=True)
    return [line for line in output.splitlines()
            if ('encbank-tb-' in line or 'qcomem-tb-' in line)
            and 'gpu' in line.split('|')[-1].lower()]


def validate_selection():
    selection = read(R / 'selection.json')
    historical = read(U / 'accuracy_recovery_20260924' / 'selection.json')
    assert selection['mode'] == 'score_only_unverified_20260925'
    assert read(REG)['version'] == 5
    assert selection['arms']['k12']['verified_count'] == 31
    assert selection['arms']['k48']['verified_count'] == 52
    proof = selection['new_score_proof']
    assert proof['task'] == 'portfolio-optimization' and proof['reward'] == 1
    assert proof['requests'] == proof['responses'] == 8
    assert proof['parent_wait'] and proof['reply_proofs_ok']
    assert len(proof['result_sha256']) == 64
    for arm, expected in (('k12', 58), ('k48', 37)):
        row = selection['arms'][arm]
        verified, pending = set(row['verified']), set(row['pending'])
        old_all = set(historical['verified'][arm]) | set(historical['pending'][arm])
        assert len(verified) == row['verified_count']
        assert len(pending) == row['pending_count'] == expected
        assert not verified & pending and verified | pending == old_all
        assert len(old_all) == 89
        if arm == 'k12':
            assert pending == set(historical['pending'][arm]) - {'portfolio-optimization'}
        else:
            assert pending == set(historical['pending'][arm])
    return selection


def partition(arm, names):
    original = read(ORIGINAL[arm] / 'plan.json')
    inventory = {row['task']: row for row in original['resource_inventory']}
    assert set(names).issubset(inventory)
    shards = [[] for _ in range(SHARDS[arm])]
    loads = [0] * SHARDS[arm]
    for name in sorted(names, key=lambda x: (-inventory[x]['memory_mb'], x)):
        i = min(range(len(shards)), key=lambda j: (loads[j], len(shards[j]), j))
        shards[i].append(name)
        loads[i] += inventory[name]['memory_mb']
    assert all(shards) and sum(map(len, shards)) == len(names)
    return shards, inventory


def prepare():
    selection = validate_selection()
    assert not (R / 'prepared.json').exists()
    registry_sha = sha(REG)
    selection_sha = sha(R / 'selection.json')
    runs = []
    for arm in ('k12', 'k48'):
        source = SOURCE[arm]
        old_plan = read(source / 'plan.json')
        old_manifest = read(source / 'server_source_manifest.json')
        assert all(sha(source / name) == digest for name, digest in old_manifest.items())
        assert old_plan['apt_by_hash_force'] and old_plan['accuracy_only_environment_check']
        source_runtime = str(Path(old_plan['rpc_root']).parent)
        shards, inventory = partition(arm, selection['arms'][arm]['pending'])
        for i, tasks in enumerate(shards, 1):
            name = f'{arm}_score_resume_{i}_20260925'
            target = S / name
            assert not target.exists(), target
            target.mkdir()
            runtime = str(Path(source_runtime).parent / name)
            ipc = f'/cluster/home/liuhanzuo/qcomem_runtime_20260911/t89sr{arm}{i}'
            job_name = f'encbank-tb-{arm}-sr25-{i}'
            swaps = ((str(source), str(target)), (source_runtime, runtime),
                     (old_plan['ipc_root'], ipc), (old_plan['job_name'], job_name))
            for filename in old_manifest:
                content = (source / filename).read_bytes()
                for before, after in swaps:
                    content = content.replace(before.encode(), after.encode())
                (target / filename).write_bytes(content)
            plan = read(target / 'plan.json')
            plan.update(tasks=tasks, resource_inventory=[inventory[t] for t in tasks],
                        remote_root=str(target), ipc_root=ipc, job_name=job_name,
                        evidence_id=plan['evidence_id'] + f'-SCORE-RESUME-20260925-{i}',
                        concurrent_tasks=CONCURRENCY[arm], max_live_sessions=CONCURRENCY[arm],
                        decode_batch_size=CONCURRENCY[arm],
                        score_resume_parent=str(source), score_resume_selection_sha256=selection_sha,
                        score_resume_reason='Only tasks with no verified score; infrastructure attempts remain archived.',
                        allocation_policy='Nine one-GPU shards, any normally schedulable GPU node; k12 four and k48 six concurrent tasks per GPU.')
            assert set(plan['tasks']) == set(tasks) and plan['max_new_tokens'] is None
            save(target / 'plan.json', plan)
            template = read(target / 'comem_harbor_template.json')
            template['job_name'] = name
            template['jobs_dir'] = plan['results_root']
            template['tasks'] = [{'path': str(Path(plan['task_root']) / task)} for task in tasks]
            save(target / 'comem_harbor_template.json', template)
            slurm = target / 'server.slurm'
            script = slurm.read_text()
            assert '#SBATCH --reservation=gaomingju_gpu7\n' in script
            assert '#SBATCH --exclude=lj-gpu1\n' in script
            script = script.replace('#SBATCH --reservation=gaomingju_gpu7\n', '')
            script = script.replace('#SBATCH --exclude=lj-gpu1\n', '')
            assert '#SBATCH --time=0' in script and '#SBATCH --gres=gpu:nvidia_l20d:1' in script
            slurm.write_text(script)
            for filename in old_manifest:
                if filename.endswith('.py'):
                    compile((target / filename).read_text(), str(target / filename), 'exec')
            save(target / 'server_source_manifest.json',
                 {filename: sha(target / filename) for filename in old_manifest})
            save(target / 'score_resume_selection.json', dict(
                arm=arm, tasks=tasks, selection_sha256=selection_sha,
                source_root=str(source), source_manifest_sha256=sha(source / 'server_source_manifest.json'),
                previous_verified_count=selection['arms'][arm]['verified_count']))
            runs.append(dict(arm=arm, index=i, root=str(target), tasks=tasks,
                             concurrency=CONCURRENCY[arm],
                             source_manifest_sha256=sha(target / 'server_source_manifest.json')))
    assert len(runs) == 9
    assert len({(r['arm'], t) for r in runs for t in r['tasks']}) == 95
    save(R / 'prepared.json', dict(epoch=time.time(), registry_sha256=registry_sha,
                                 selection_sha256=selection_sha, runs=runs))
    print(json.dumps({'prepared': [(r['arm'], r['index'], len(r['tasks'])) for r in runs]}))


def activate():
    prepared = read(R / 'prepared.json')
    assert not (R / 'activated.json').exists()
    assert sha(REG) == prepared['registry_sha256']
    assert sha(R / 'selection.json') == prepared['selection_sha256']
    validate_selection()
    registry = read(REG)
    original = {arm: [t for row in registry['runs'] if row['arm'] == arm for t in row['tasks']]
                for arm in ('dense', 'k12', 'k48')}
    for row in prepared['runs']:
        transfer = set(row['tasks'])
        assert len(transfer) == len(row['tasks'])
        owners = [old for old in registry['runs']
                  if old['arm'] == row['arm'] and transfer & set(old['tasks'])]
        assert {t for old in owners for t in transfer & set(old['tasks'])} == transfer
        for old in owners:
            old['tasks'] = [t for t in old['tasks'] if t not in transfer]
            old.setdefault('score_resume_transfers', []).append(row['root'])
        registry['runs'].append(dict(id=Path(row['root']).name, arm=row['arm'],
                                     root=row['root'], tasks=row['tasks'],
                                     role='score_only_unverified_20260925',
                                     source_manifest_sha256=row['source_manifest_sha256']))
    for arm, previous in original.items():
        now = [t for row in registry['runs'] if row['arm'] == arm for t in row['tasks']]
        assert len(now) == len(set(now)) and set(now) == set(previous)
    registry.update(version=6, gpu_limit=9, score_resume_path=str(R / 'prepared.json'),
                    score_resume_epoch=time.time())
    save(R / 'registry_v5_before_score_resume.json', read(REG))
    save(REG, registry)
    save(R / 'activated.json', dict(epoch=time.time(), registry_sha256=sha(REG),
                                  previous_sha256=prepared['registry_sha256']))
    print(json.dumps({'activated': True, 'registry_sha256': sha(REG)}))


def submit():
    prepared = read(R / 'prepared.json')
    assert sha(REG) == read(R / 'activated.json')['registry_sha256']
    assert sha(R / 'selection.json') == prepared['selection_sha256']
    health = read(R / 'storage_health.json')
    assert health['passed'] and time.time() - health['epoch'] < 3600
    assert set(health['nodes']) >= {'lj-gpu1', 'lj-gpu6'}
    assert all(x['passed'] and x['rounds'] == 20 for x in health['nodes'].values())
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = main_queue()
        assert len(queue) + len(prepared['runs']) <= 9, queue
        for row in prepared['runs']:
            root = Path(row['root'])
            assert not (root / 'submission.json').exists(), root
            assert sha(root / 'server_source_manifest.json') == row['source_manifest_sha256']
            assert all(sha(root / name) == digest
                       for name, digest in read(root / 'server_source_manifest.json').items())
            assert len(main_queue()) < 9
            receipt = dict(status='intent', epoch=time.time(), root=str(root),
                           arm=row['arm'], tasks=len(row['tasks']),
                           source_manifest_sha256=row['source_manifest_sha256'],
                           selection_sha256=prepared['selection_sha256'],
                           health_sha256=sha(R / 'storage_health.json'),
                           automatic_scientific_retries=0)
            save(root / 'submission.json', receipt)
            child = subprocess.run(['sbatch', '--parsable', str(root / 'server.slurm')],
                                   capture_output=True, text=True)
            receipt.update(status='submitted' if child.returncode == 0 else 'submission_failed',
                           exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr,
                           actual_parent_wait=True)
            if child.returncode == 0:
                receipt['job_id'] = child.stdout.strip().split(';')[0]
            save(root / 'submission.json', receipt)
            assert child.returncode == 0, receipt
            print(json.dumps({'root': root.name, 'job_id': receipt['job_id'],
                              'tasks': len(row['tasks'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'activate', 'submit'))
    globals()[parser.parse_args().phase]()
