"""Prepare fresh, score-only EncBank continuations after the 2026-09-24 storage incident.

The source roots and task selection are immutable inputs.  Preparation never submits
jobs; submission is a separate, one-shot phase with per-root intent records.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

S = Path('/cluster/home/USER/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
REC = U / 'accuracy_recovery_20260924'
REGISTRY = U / 'scale4_20260921' / 'registry.json'
SOURCE = {'k12': S / 'k12_scale6_a_20260924', 'k48': S / 'k48_scale6_b_20260924'}
ORIGINAL = {'k12': S / 'k12_unbounded_20260921', 'k48': S / 'k48_unbounded_20260921'}
SHARDS = {'k12': 5, 'k48': 3}
CONCURRENCY = {'k12': 4, 'k48': 6}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def status(job):
    lines = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=State'], text=True).splitlines()
    return lines[0].split('|')[0] if lines else None


def validate_selection(selection):
    registry = read(REGISTRY)
    assert registry['version'] == 2
    assert selection['registry_sha256'] == digest(REGISTRY)
    assert selection['mode'] == 'accuracy_only_unverified_tasks'
    assert selection['verified_counts'] == {'dense': 89, 'k12': 30, 'k48': 52}
    expected = {'k12': 59, 'k48': 37}
    for arm in expected:
        pending = selection['pending'][arm]
        verified = selection['verified'][arm]
        assert len(pending) == expected[arm] and len(verified) + len(pending) == 89
        assert not set(pending) & set(verified)
        assert len(pending) == len(set(pending))
        original = read(ORIGINAL[arm] / 'plan.json')
        assert set(pending).issubset(set(original['tasks']))
        prior = read(U / 'selection.json')['arms'][arm]
        assert set(verified + pending) == set(prior['tasks']) | set(prior['retained'])
    for job in (116712, 116713, 117352, 117353, 120001, 120002, 137900, 137901, 137902):
        assert status(job) in {'FAILED', 'COMPLETED', 'CANCELLED'}, (job, status(job))
    return selection


def partition(arm, selection):
    original = read(ORIGINAL[arm] / 'plan.json')
    resources = {row['task']: row for row in original['resource_inventory']}
    tasks = sorted(selection['pending'][arm], key=lambda name: (-resources[name]['memory_mb'], name))
    shards = [[] for _ in range(SHARDS[arm])]
    loads = [0] * SHARDS[arm]
    for name in tasks:
        index = min(range(len(shards)), key=lambda i: (loads[i], len(shards[i]), i))
        shards[index].append(name)
        loads[index] += resources[name]['memory_mb']
    assert sum(map(len, shards)) == len(tasks)
    return shards, resources


def prepare():
    selection = validate_selection(read(REC / 'selection.json'))
    assert not (REC / 'prepared.json').exists()
    runs = []
    for arm in ('k12', 'k48'):
        shards, resources = partition(arm, selection)
        source = SOURCE[arm]
        old_plan = read(source / 'plan.json')
        old_manifest = read(source / 'server_source_manifest.json')
        for name, sha in old_manifest.items():
            assert digest(source / name) == sha, (source, name)
        source_runtime = str(Path(old_plan['rpc_root']).parent)
        for index, tasks in enumerate(shards, 1):
            name = f'{arm}_acc_recovery_{index}_20260924'
            target = S / name
            assert not target.exists(), target
            target.mkdir()
            runtime = str(Path(source_runtime).parent / name)
            ipc = '/cluster/home/USER/qencbank_runtime_20260911/t89acc' + arm + str(index)
            job_name = 'encbank-tb-' + arm + '-acc-r' + str(index)
            replacements = ((str(source), str(target)), (source_runtime, runtime),
                            (old_plan['ipc_root'], ipc), (old_plan['job_name'], job_name))
            for source_name in old_manifest:
                data = (source / source_name).read_bytes()
                for before, after in replacements:
                    data = data.replace(before.encode(), after.encode())
                (target / source_name).write_bytes(data)
            plan = read(target / 'plan.json')
            plan.update(tasks=tasks, resource_inventory=[resources[task] for task in tasks],
                        remote_root=str(target), ipc_root=ipc, job_name=job_name,
                        evidence_id=plan['evidence_id'] + '-ACC-RECOVERY-' + str(index),
                        concurrent_tasks=CONCURRENCY[arm], max_live_sessions=CONCURRENCY[arm],
                        decode_batch_size=CONCURRENCY[arm],
                        accuracy_only_environment_check=True,
                        accuracy_recovery_parent=str(source),
                        accuracy_recovery_reason='Only tasks without a valid score; prior infrastructure interruption remains separately archived.',
                        allocation_policy='Eight independent one-GPU shards; k12 four and k48 six live sessions per card; no fixed node.')
            save(target / 'plan.json', plan)
            template = read(target / 'encbank_harbor_template.json')
            template['job_name'] = name
            template['jobs_dir'] = plan['results_root']
            template['tasks'] = [{'path': str(Path(plan['task_root']) / task)} for task in tasks]
            save(target / 'encbank_harbor_template.json', template)
            # All 89 task sources were hashed previously. Accuracy-only work still
            # checks those source bytes, model identity, and the actual Harbor run.
            # It skips the frozen per-node install-only qualification matrix.
            preflight = target / 'server_preflight.py'
            code = preflight.read_text()
            needle = '    if check_container:\n'
            assert code.count(needle) == 1
            bypass = ('    if check_container and plan.get("accuracy_only_environment_check"):\n'
                      '        return {"status":"PASS", "epoch":time.time(), "server_only":True, '
                      '"model_calls":0, "source_manifest_sha256":source_sha, '
                      '"tasks_checked":checked_tasks, "model_path":json.loads(model.stdout), '
                      '"checkpoint_identity":checkpoint_result, "harbor_version":harbor.stdout.strip(), '
                      '"container":{"checked":False,"scope":"accuracy_only"}}\n')
            preflight.write_text(code.replace(needle, bypass + needle))
            save(target / 'accuracy_selection.json', {'arm': arm, 'tasks': tasks,
                'selection_sha256': digest(REC / 'selection.json'), 'source_root': str(source),
                'source_manifest_sha256': digest(source / 'server_source_manifest.json'),
                'prior_verified_count': len(selection['verified'][arm]),
                'replays_only_unscored_tasks': True})
            # Retain the original compatibility matrix as historical provenance;
            # the accuracy-only preflight does not treat it as a current gate.
            for source_name in old_manifest:
                if source_name.endswith('.py'):
                    compile((target / source_name).read_text(), str(target / source_name), 'exec')
            save(target / 'server_source_manifest.json',
                 {source_name: digest(target / source_name) for source_name in old_manifest})
            runs.append({'arm': arm, 'name': name, 'root': str(target), 'tasks': tasks,
                         'concurrency': CONCURRENCY[arm],
                         'source_manifest_sha256': digest(target / 'server_source_manifest.json')})
    assert len(runs) == 8 and sum(len(r['tasks']) for r in runs) == 96
    save(REC / 'prepared.json', {'epoch': time.time(), 'selection_sha256': digest(REC / 'selection.json'),
                                'registry_sha256': digest(REGISTRY), 'runs': runs})
    print(json.dumps({'prepared': [(r['name'], len(r['tasks'])) for r in runs]}), flush=True)


def activate():
    prepared = read(REC / 'prepared.json')
    selection = validate_selection(read(REC / 'selection.json'))
    assert digest(REC / 'selection.json') == prepared['selection_sha256']
    assert not (REC / 'registry_activated.json').exists()
    registry = read(REGISTRY)
    assert digest(REGISTRY) == prepared['registry_sha256']
    original_by_arm = {
        arm: {task for run in registry['runs'] if run['arm'] == arm for task in run['tasks']}
        for arm in ('dense', 'k12', 'k48')
    }
    for run in prepared['runs']:
        transfer = set(run['tasks'])
        owners = [row for row in registry['runs'] if row['arm'] == run['arm'] and transfer & set(row['tasks'])]
        assert len({task for row in owners for task in transfer & set(row['tasks'])}) == len(transfer)
        for old in owners:
            old['tasks'] = [task for task in old['tasks'] if task not in transfer]
            old.setdefault('accuracy_recovery_transfers', []).append(run['name'])
        registry['runs'].append({'id': run['name'], 'arm': run['arm'], 'root': run['root'],
                                 'tasks': run['tasks'], 'role': 'accuracy_only_unscored_continuation',
                                 'source_manifest_sha256': run['source_manifest_sha256']})
    for arm, original in original_by_arm.items():
        current = [task for run in registry['runs'] if run['arm'] == arm for task in run['tasks']]
        assert len(current) == len(set(current)) and set(current) == original
    assert len(selection['pending']['k12']) + len(selection['pending']['k48']) == 96
    registry.update(version=3, gpu_limit=8, maximum_total_concurrency=38,
                    accuracy_recovery_path=str(REC / 'prepared.json'),
                    accuracy_recovery_selection_sha256=prepared['selection_sha256'],
                    accuracy_recovery_epoch=time.time())
    save(REC / 'registry_v2_before_acc_recovery.json', read(REGISTRY))
    save(REGISTRY, registry)
    save(REC / 'registry_activated.json', {'epoch': time.time(), 'registry_sha256': digest(REGISTRY),
                                           'old_registry_sha256': prepared['registry_sha256']})
    print(json.dumps({'activated': True, 'registry_sha256': digest(REGISTRY)}), flush=True)


def submit():
    prepared = read(REC / 'prepared.json')
    assert digest(REC / 'selection.json') == prepared['selection_sha256']
    activated = read(REC / 'registry_activated.json')
    assert activated['old_registry_sha256'] == prepared['registry_sha256']
    assert digest(REGISTRY) == activated['registry_sha256']
    assert read(REGISTRY)['version'] == 3
    for job in (116712, 116713, 117352, 117353, 120001, 120002, 137900, 137901, 137902):
        assert status(job) in {'FAILED', 'COMPLETED', 'CANCELLED'}
    lock_path = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
    with lock_path.open('a') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        for run in prepared['runs']:
            root = Path(run['root'])
            assert not (root / 'submission.json').exists(), root
            assert digest(root / 'server_source_manifest.json') == run['source_manifest_sha256']
            for name, sha in read(root / 'server_source_manifest.json').items():
                assert digest(root / name) == sha, (root, name)
            queue = subprocess.check_output(['squeue', '-r', '-u', os.environ['USER'], '-h', '-o', '%i|%j|%T|%b'], text=True)
            owned = [line for line in queue.splitlines() if ('encbank-tb-' in line or 'qencbank-tb-' in line)
                     and 'gpu' in line.split('|')[-1].lower()]
            assert len(owned) < 8, (len(owned), owned)
            record = {'status': 'intent', 'epoch': time.time(), 'root': str(root),
                      'arm': run['arm'], 'tasks': len(run['tasks']),
                      'concurrency': run['concurrency'], 'gpu_requests': 1,
                      'source_manifest_sha256': run['source_manifest_sha256'],
                      'selection_sha256': prepared['selection_sha256'],
                      'prior_owned_gpu_queue': owned, 'automatic_scientific_retries': 0}
            save(root / 'submission.json', record)
            child = subprocess.run(['sbatch', '--parsable', str(root / 'server.slurm')], capture_output=True, text=True)
            record.update(status='submitted' if child.returncode == 0 else 'submission_failed',
                          exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr,
                          actual_parent_wait=True)
            if child.returncode == 0:
                record['job_id'] = child.stdout.strip().split(';')[0]
            save(root / 'submission.json', record)
            assert child.returncode == 0, record
            print(json.dumps({'name': run['name'], 'job_id': record['job_id'],
                              'tasks': len(run['tasks'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'activate', 'submit'))
    phase = parser.parse_args().phase
    {'prepare': prepare, 'activate': activate, 'submit': submit}[phase]()
