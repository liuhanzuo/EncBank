"""One-shot, immutable proxy recovery of zero-score Terminal-Bench tasks.

The previous eight allocations are never restarted.  The single active qemu
trial remains owned by its drained predecessor until it reaches a real result.
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
R = U / 'proxy_recovery_20260924'
REG = U / 'scale4_20260921' / 'registry.json'
PROXY = 'http://PROXY_HOST:8888'
NO_PROXY = 'localhost,127.0.0.1,::1,10.0.0.213,10.0.0.214,10.0.0.215,10.0.0.216,10.0.0.217,10.0.0.218,10.0.0.219,10.0.0.220,PROXY_HOST,.local,.cluster.local'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def state(job):
    lines = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=State'], text=True).splitlines()
    return lines[0].split('|')[0].split()[0] if lines else None


def make():
    assert read(REG)['version'] == 3
    assert not (R / 'prepared.json').exists()
    predecessor_ids = {'k12': [140587, 140588, 140589, 140590, 140591],
                       'k48': [140592, 140593, 140594]}
    rows = []
    for arm, jobs in predecessor_ids.items():
        for index, job in enumerate(jobs, 1):
            old = S / f'{arm}_acc_recovery_{index}_20260924'
            plan = read(old / 'plan.json')
            source_manifest = read(old / 'server_source_manifest.json')
            assert all(sha(old / name) == expected for name, expected in source_manifest.items())
            requests = list((Path(plan['rpc_root']) / plan['arm']).glob('*.request.json'))
            outcomes = [read(p) for p in (old / 'execution' / 'task_outcomes').glob('*.json')]
            if index == 1 and arm == 'k12':
                assert job == 140587 and state(job) in {'RUNNING', 'COMPLETED'}
                assert {x['task'] for x in outcomes} == set(plan['tasks']) - {'qemu-alpine-ssh'}
                assert all(x['model_requests'] == 0 and not x['valid_result'] for x in outcomes)
                assert all(read(p)['task'] == 'qemu-alpine-ssh' for p in requests)
                tasks = [x for x in plan['tasks'] if x != 'qemu-alpine-ssh']
            else:
                assert state(job) == 'CANCELLED', (job, state(job))
                assert not requests and all(x.get('model_requests') == 0 and not x.get('valid_result')
                                            for x in outcomes), (job, len(requests), outcomes)
                assert {x['task'] for x in outcomes}.issubset(set(plan['tasks']))
                tasks = plan['tasks']
            new = S / f'{arm}_proxy_recovery_r2_{index}_20260924'
            assert not new.exists(), new
            new.mkdir()
            old_runtime = str(Path(plan['rpc_root']).parent)
            new_runtime = str(Path(old_runtime).parent / new.name)
            new_ipc = f'/cluster/home/USER/qencbank_runtime_20260911/t89px{arm}{index}'
            new_job_name = f'encbank-tb-{arm}-px-r{index}'
            swaps = ((str(old), str(new)), (old_runtime, new_runtime),
                     (plan['ipc_root'], new_ipc), (plan['job_name'], new_job_name))
            for name in source_manifest:
                data = (old / name).read_bytes()
                for before, after in swaps:
                    data = data.replace(before.encode(), after.encode())
                (new / name).write_bytes(data)
            new_plan = read(new / 'plan.json')
            new_plan.update(tasks=tasks,
                            resource_inventory=[x for x in plan['resource_inventory'] if x['task'] in tasks],
                            remote_root=str(new), ipc_root=new_ipc, job_name=new_job_name,
                            evidence_id=new_plan['evidence_id'] + '-PROXY-RECOVERY',
                            proxy_recovery_parent=str(old),
                            proxy_recovery_reason='Only unscored tasks; original environment failure preserved.',
                            excluded_observed_stalled_node='lj-gpu1')
            save(new / 'plan.json', new_plan)
            template = read(new / 'encbank_harbor_template.json')
            template['job_name'] = new.name
            template['jobs_dir'] = new_plan['results_root']
            template['tasks'] = [{'path': str(Path(new_plan['task_root']) / task)} for task in tasks]
            save(new / 'encbank_harbor_template.json', template)
            slurm = (new / 'server.slurm').read_text()
            assert '#SBATCH --exclude=' not in slurm
            slurm = slurm.replace('#SBATCH --no-requeue', '#SBATCH --exclude=lj-gpu1\n#SBATCH --no-requeue')
            anchor = 'set -euo pipefail\n'
            assert slurm.count(anchor) == 1
            proxy_exports = (f'export http_proxy={PROXY} https_proxy={PROXY} all_proxy={PROXY}\n'
                             f'export HTTP_PROXY={PROXY} HTTPS_PROXY={PROXY} ALL_PROXY={PROXY}\n'
                             f'export no_proxy={NO_PROXY} NO_PROXY={NO_PROXY}\n')
            (new / 'server.slurm').write_text(slurm.replace(anchor, anchor + proxy_exports))
            for name in source_manifest:
                if name.endswith('.py'):
                    compile((new / name).read_text(), str(new / name), 'exec')
            save(new / 'server_source_manifest.json', {name: sha(new / name) for name in source_manifest})
            rows.append({'arm': arm, 'index': index, 'root': str(new), 'parent': str(old),
                         'parent_job': job, 'tasks': tasks,
                         'source_manifest_sha256': sha(new / 'server_source_manifest.json')})
    assert len(rows) == 8 and sum(len(x['tasks']) for x in rows) == 95
    assert len(set((x['arm'], t) for x in rows for t in x['tasks'])) == 95
    save(R / 'prepared.json', {'epoch': time.time(), 'registry_sha256': sha(REG), 'runs': rows})
    print(json.dumps({'prepared': [(x['arm'], x['index'], len(x['tasks'])) for x in rows]}), flush=True)


def activate():
    data = read(R / 'prepared.json')
    assert not (R / 'activated.json').exists()
    assert read(REG)['version'] == 3 and sha(REG) == data['registry_sha256']
    registry = read(REG)
    before = {arm: [t for x in registry['runs'] if x['arm'] == arm for t in x['tasks']]
              for arm in ('dense', 'k12', 'k48')}
    for row in data['runs']:
        matches = [x for x in registry['runs'] if x['root'] == row['parent']]
        assert len(matches) == 1 and matches[0]['arm'] == row['arm']
        old = matches[0]
        assert set(row['tasks']).issubset(set(old['tasks']))
        old['tasks'] = [t for t in old['tasks'] if t not in row['tasks']]
        old.setdefault('transferred_to_proxy_recovery', []).append(row['root'])
        registry['runs'].append({'id': Path(row['root']).name, 'arm': row['arm'],
                                 'root': row['root'], 'tasks': row['tasks'],
                                 'role': 'proxy_recovery_unscored',
                                 'source_manifest_sha256': row['source_manifest_sha256']})
    for arm, prior in before.items():
        now = [t for x in registry['runs'] if x['arm'] == arm for t in x['tasks']]
        assert len(now) == len(set(now)) and set(now) == set(prior), arm
    registry.update(version=4, gpu_limit=9, maximum_total_concurrency=42,
                    proxy_recovery_path=str(R / 'prepared.json'),
                    proxy_recovery_epoch=time.time())
    save(R / 'registry_v3_before_proxy_recovery.json', read(REG))
    save(REG, registry)
    save(R / 'activated.json', {'epoch': time.time(), 'registry_sha256': sha(REG)})
    print(json.dumps({'activated': True, 'registry_sha256': sha(REG)}), flush=True)


def submit():
    data = read(R / 'prepared.json')
    assert read(REG)['version'] == 4 and sha(REG) == read(R / 'activated.json')['registry_sha256']
    lock_path = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for row in data['runs']:
            root = Path(row['root'])
            assert not (root / 'submission.json').exists()
            assert sha(root / 'server_source_manifest.json') == row['source_manifest_sha256']
            assert all(sha(root / name) == expected
                       for name, expected in read(root / 'server_source_manifest.json').items())
            queue = subprocess.check_output(['squeue', '-r', '-u', 'USER', '-h', '-o', '%i|%j|%T|%b'], text=True)
            owned = [line for line in queue.splitlines() if ('encbank-tb-' in line or 'qencbank-tb-' in line)
                     and 'gpu' in line.split('|')[-1].lower()]
            assert len(owned) < 9, (len(owned), owned)
            receipt = {'status': 'intent', 'epoch': time.time(), 'root': str(root),
                       'arm': row['arm'], 'tasks': len(row['tasks']),
                       'source_manifest_sha256': row['source_manifest_sha256'],
                       'predecessor_job': row['parent_job'], 'automatic_scientific_retries': 0}
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
            print(json.dumps({'arm': row['arm'], 'index': row['index'], 'job': receipt['job_id']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('make', 'activate', 'submit'))
    globals()[p.parse_args().phase]()
