"""Prepare and submit disjoint, never-started scale-out tasks on the server.

Run only on the server. Existing run sources and results remain immutable. The
three phases are separate so an uncertain submission is never retried blindly.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

S = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
OLD = U / 'scale4_20260921' / 'registry.json'
D = U / 'scale6_20260924'
SPECS = (
    ('k12_a', 'k12_scale6_a_20260924', 'k12', 15, 't89s6k12a'),
    ('k12_b', 'k12_scale6_b_20260924', 'k12', 32, 't89s6k12b'),
    ('k48_b', 'k48_scale6_b_20260924', 'k48', 13, 't89s6k48b'),
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def old_state(spec):
    root = Path(spec['root'])
    plan = read(root / 'plan.json')
    submission = read(root / 'submission.json')
    job = submission['job_id']
    state = subprocess.check_output(
        ['sacct', '-X', '-n', '-P', '-j', str(job), '--format=State'], text=True
    ).splitlines()[0].split('|')[0]
    assert state == 'FAILED', (spec['id'], job, state)
    box = Path(plan['rpc_root']) / plan['arm']
    requested = set()
    for path in box.glob('*.request.json'):
        requested.add(read(path)['task'])
    launched = {path.stem for path in (root / 'execution' / 'launches').glob('*.json')}
    outcomes = {path.stem for path in (root / 'execution' / 'task_outcomes').glob('*.json')}
    untouched = [task for task in spec['tasks'] if task not in requested | launched | outcomes]
    for task in untouched:
        assert not (root / 'execution' / 'interrupted_parents' / (task + '.json')).exists()
    return root, plan, submission, untouched, requested, launched, outcomes


def prepare():
    registry = read(OLD)
    assert registry['gpu_limit'] == 4
    runs = {item['id']: item for item in registry['runs']}
    D.mkdir(exist_ok=False)
    summary = dict(epoch=time.time(), authorization='2026-09-24 user requested more main-experiment GPUs',
                   old_registry_sha256=sha(OLD), intended_gpu_limit=6,
                   new_per_gpu_concurrency=4, new_runs=[])
    for old_id, new_name, arm, expected_count, ipc_name in SPECS:
        source = runs[old_id]
        root, plan, submission, tasks, requested, launched, outcomes = old_state(source)
        job = submission['job_id']
        assert len(tasks) == expected_count, (old_id, len(tasks))
        assert set(tasks).issubset(source['tasks'])
        old_manifest = read(root / 'server_source_manifest.json')
        for name, digest in old_manifest.items():
            assert sha(root / name) == digest, (root, name)
        target = S / new_name
        target.mkdir(exist_ok=False)
        old_runtime = str(Path(plan['rpc_root']).parent)
        new_runtime = str(Path(old_runtime).parent / new_name)
        new_ipc = '/srv/encbank/qencbank_runtime_20260911/' + ipc_name
        new_job_name = 'qencbank-tb-' + arm + '-scale6-' + old_id[-1]
        replacements = ((str(root), str(target)), (old_runtime, new_runtime),
                        (plan['ipc_root'], new_ipc), (plan['job_name'], new_job_name))
        for name in old_manifest:
            data = (root / name).read_bytes()
            for before, after in replacements:
                data = data.replace(before.encode(), after.encode())
            (target / name).write_bytes(data)
        new_plan = read(target / 'plan.json')
        new_plan.update(
            tasks=tasks,
            resource_inventory=[row for row in plan['resource_inventory'] if row['task'] in tasks],
            remote_root=str(target), ipc_root=new_ipc, job_name=new_job_name,
            evidence_id=plan['evidence_id'] + '-SCALE6-' + old_id.upper(),
            concurrent_tasks=4, max_live_sessions=4, decode_batch_size=4,
            scale6_parent_root=str(root), scale6_parent_job_id=job,
            scale6_transfer_reason='Only never-started tasks from failed predecessor; no normal or interrupted attempt replay.',
            allocation_policy='At most 6 running+pending main-benchmark GPUs; new shards at most 4 tasks each; any eligible node.',
        )
        assert len(new_plan['resource_inventory']) == len(tasks)
        save(target / 'plan.json', new_plan)
        template = read(target / 'encbank_harbor_template.json')
        template['job_name'] = new_name
        template['tasks'] = [dict(path=str(Path(plan['task_root']) / task)) for task in tasks]
        save(target / 'encbank_harbor_template.json', template)
        qualification = read(root / 'container_qualification.json')
        for name, digest in qualification['backend_hashes'].items():
            assert sha(target / name) == digest
        qualification.update(
            tasks=tasks,
            checks=[row for row in qualification['checks'] if row['task'] in tasks],
            task_concurrency=4,
            plan_sha256=sha(target / 'plan.json'),
            scale6_transfer=dict(source=str(root / 'container_qualification.json'),
                                 sha256=sha(root / 'container_qualification.json'),
                                 transferred_only_never_started=True,
                                 allocated_host_smoke_required=True),
        )
        assert len(qualification['checks']) == len(tasks)
        assert all(row['status'] == 'PASS' for row in qualification['checks'])
        save(target / 'container_qualification.json', qualification)
        save(target / 'selection_provenance.json', dict(
            selection_sha256=registry['selection_sha256'], predecessor_run=old_id,
            predecessor_root=str(root), predecessor_job_id=job,
            predecessor_plan_sha256=sha(root / 'plan.json'),
            tasks=tasks, zero_prior_launches=True, zero_prior_model_calls=True,
            interrupted_predecessor_tasks_excluded=sorted(set(source['tasks']) & launched - outcomes),
        ))
        # The predecessor's 60-second dependency-import check failed before any
        # model call; it is an environment guard, not a task/request deadline.
        preflight = target / 'server_preflight.py'
        text = preflight.read_text()
        needle = "capture_output=True, text=True, timeout=60)"
        assert text.count(needle) == 1
        preflight.write_text(text.replace(needle, "capture_output=True, text=True, timeout=None)"))
        for name in old_manifest:
            if name.endswith('.py'):
                compile((target / name).read_text(), str(target / name), 'exec')
        save(target / 'server_source_manifest.json', {name: sha(target / name) for name in old_manifest})
        summary['new_runs'].append(dict(
            id=old_id.replace('_', '_scale6_', 1), arm=arm, source_run=old_id,
            source_root=str(root), source_job_id=job, root=str(target), tasks=tasks,
            source_manifest_sha256=sha(target / 'server_source_manifest.json'),
            source_plan_sha256=sha(root / 'plan.json'),
        ))
    assert sum(len(run['tasks']) for run in summary['new_runs']) == 60
    assert len({(run['arm'], task) for run in summary['new_runs'] for task in run['tasks']}) == 60
    save(D / 'transfer.json', summary)
    print(json.dumps(dict(status='PREPARED', runs=[(r['id'], len(r['tasks'])) for r in summary['new_runs']])))


def verify_prepared():
    transfer = read(D / 'transfer.json')
    assert transfer['old_registry_sha256'] == sha(OLD)
    old = read(OLD)
    old_runs = {item['id']: item for item in old['runs']}
    for run in transfer['new_runs']:
        target = Path(run['root'])
        assert not (target / 'submission.json').exists()
        assert not (target / 'run_encbank').exists()
        assert sha(target / 'server_source_manifest.json') == run['source_manifest_sha256']
        for name, digest in read(target / 'server_source_manifest.json').items():
            assert sha(target / name) == digest, (target, name)
        preflight = read(target / 'server_cpu_preflight.json')
        assert preflight['status'] == 'PASS' and preflight['tasks_checked'] == len(run['tasks'])
        assert preflight['model_calls'] == 0
        source = old_runs[run['source_run']]
        _, _, _, tasks, _, _, _ = old_state(source)
        assert set(run['tasks']).issubset(tasks)
    return transfer, old


def activate():
    transfer, registry = verify_prepared()
    assert not (D / 'registry_activated.json').exists()
    by_id = {item['id']: item for item in registry['runs']}
    for run in transfer['new_runs']:
        source = by_id[run['source_run']]
        assert set(run['tasks']).issubset(source['tasks'])
        source['tasks'] = [task for task in source['tasks'] if task not in set(run['tasks'])]
        source['transferred_never_started_to'] = run['id']
        registry['runs'].append(dict(id=run['id'], arm=run['arm'], root=run['root'],
                                     tasks=run['tasks'], role='scale6_never_started',
                                     source_manifest_sha256=run['source_manifest_sha256']))
    selection = read(U / 'selection.json')
    for arm in ('dense', 'k12', 'k48'):
        tasks = [task for run in registry['runs'] if run['arm'] == arm for task in run['tasks']]
        assert len(tasks) == len(set(tasks))
        assert set(tasks) == set(selection['arms'][arm]['tasks'])
    registry.update(version=2, gpu_limit=6, maximum_total_concurrency=36,
                    scale6_transfer_path=str(D / 'transfer.json'),
                    scale6_transfer_sha256=sha(D / 'transfer.json'),
                    scale6_new_per_gpu_concurrency=4,
                    scale6_epoch=time.time())
    save(D / 'registry_before_scale6.json', read(OLD))
    save(OLD, registry)
    save(D / 'registry_activated.json', dict(epoch=time.time(), registry_sha256=sha(OLD),
                                            new_runs=[run['id'] for run in transfer['new_runs']]))
    print(json.dumps(dict(status='ACTIVATED', registry_sha256=sha(OLD))))


def submit():
    transfer = read(D / 'transfer.json')
    activated = read(D / 'registry_activated.json')
    assert sha(OLD) == activated['registry_sha256']
    registry = read(OLD)
    lock_path = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
    with lock_path.open('a') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        for run in transfer['new_runs']:
            target = Path(run['root'])
            assert not (target / 'submission.json').exists(), run['id']
            queue = subprocess.check_output(['squeue', '-r', '-u', 'encbank-user', '-h',
                                             '-o', '%i|%j|%T|%b'], text=True)
            owned = [line for line in queue.splitlines()
                     if any(name in line for name in ('qencbank-tb-', 'qencbank-agentmem-'))
                     and 'gpu' in line.split('|')[-1].lower()]
            assert len(owned) < registry['gpu_limit'], (len(owned), owned)
            plan = read(target / 'plan.json')
            assert not any(plan['job_name'] == line.split('|')[1] for line in queue.splitlines())
            assert read(target / 'server_cpu_preflight.json')['status'] == 'PASS'
            assert sha(target / 'server_source_manifest.json') == run['source_manifest_sha256']
            record = dict(status='intent', epoch=time.time(), root=str(target),
                          run_id=run['id'], arm=run['arm'], tasks=len(run['tasks']),
                          task_concurrency=4, gpu_requests=1, requested_nodes=None,
                          requested_time_limit='UNLIMITED', registry_sha256=sha(OLD),
                          source_manifest_sha256=run['source_manifest_sha256'],
                          prior_owned_gpu_queue=owned, automatic_scientific_retries=0)
            save(target / 'submission.json', record)
            child = subprocess.run(['sbatch', '--parsable', str(target / 'server.slurm')],
                                   capture_output=True, text=True)
            record.update(status='submitted' if child.returncode == 0 else 'submission_failed',
                          exit_code=child.returncode, stdout=child.stdout, stderr=child.stderr,
                          actual_parent_wait=True)
            if child.returncode == 0:
                record['job_id'] = child.stdout.strip().split(';')[0]
            save(target / 'submission.json', record)
            assert child.returncode == 0, record
            print(json.dumps(dict(run_id=run['id'], job_id=record['job_id'],
                                  tasks=len(run['tasks']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'activate', 'submit'))
    phase = parser.parse_args().phase
    {'prepare': prepare, 'activate': activate, 'submit': submit}[phase]()
