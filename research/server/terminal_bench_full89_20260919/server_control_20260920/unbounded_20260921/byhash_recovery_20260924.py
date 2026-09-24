"""One-shot by-hash apt recovery for unlaunched and zero-call failed tasks.

Running trials remain with drained parents.  Each new immutable shard inherits
the model and task sources, changes only container apt index retrieval, and
keeps every prior attempt and its evidence intact.
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
R = U / 'byhash_recovery_20260924'
REG = U / 'scale4_20260921' / 'registry.json'
APT_CONF = 'Acquire::By-Hash "force";\n'
JOBS = {'k12': [141032, 141033, 141034, 141035, 141036],
        'k48': [141037, 141038, 141039]}


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


def mirror_failure(row):
    if row.get('model_requests') != 0 or row.get('valid_result'):
        return False
    messages = [str(x.get('exception_message') or '') for x in row.get('results', [])]
    return bool(messages) and all(any(term in message for term in
        ('Failed to fetch', 'Could not connect', 'File has unexpected size',
         'Mirror sync in progress', '404  Not Found')) for message in messages)


def prepare():
    assert read(REG)['version'] == 4
    assert not (R / 'prepared.json').exists()
    runs = []
    for arm, jobs in JOBS.items():
        for index, job in enumerate(jobs, 1):
            old = S / f'{arm}_proxy_recovery_r2_{index}_20260924'
            plan = read(old / 'plan.json')
            manifest = read(old / 'server_source_manifest.json')
            assert all(sha(old / name) == digest for name, digest in manifest.items())
            launch_dir = old / 'execution' / 'launches'
            launched = {p.stem for p in launch_dir.glob('*.json')}
            outcome_rows = [read(p) for p in (old / 'execution' / 'task_outcomes').glob('*.json')]
            replay = {row['task'] for row in outcome_rows if mirror_failure(row)}
            assert replay.issubset(launched)
            assert not any(row['task'] in replay and row.get('model_requests') for row in outcome_rows)
            if state(job) == 'RUNNING':
                assert (old / 'execution' / 'drain_requested.json').exists(), job
                assert (old / 'execution' / 'pause_new_tasks.json').exists(), job
            else:
                assert state(job) in {'CANCELLED', 'COMPLETED'}, (job, state(job))
            tasks = [task for task in plan['tasks'] if task not in launched or task in replay]
            assert tasks and len(tasks) == len(set(tasks))
            # A pending or completed scored trial is never replayed by this batch.
            assert not any(row['task'] in tasks and row.get('valid_result') for row in outcome_rows)
            name = f'{arm}_byhash_recovery_{index}_20260924'
            new = S / name
            assert not new.exists(), new
            new.mkdir()
            old_runtime = str(Path(plan['rpc_root']).parent)
            new_runtime = str(Path(old_runtime).parent / name)
            new_ipc = f'/cluster/home/USER/qencbank_runtime_20260911/t89bh{arm}{index}'
            job_name = f'encbank-tb-{arm}-bh-r{index}'
            swaps = ((str(old), str(new)), (old_runtime, new_runtime),
                     (plan['ipc_root'], new_ipc), (plan['job_name'], job_name))
            for source in manifest:
                data = (old / source).read_bytes()
                for before, after in swaps:
                    data = data.replace(before.encode(), after.encode())
                (new / source).write_bytes(data)
            new_plan = read(new / 'plan.json')
            new_plan.update(tasks=tasks,
                            resource_inventory=[row for row in plan['resource_inventory'] if row['task'] in tasks],
                            remote_root=str(new), ipc_root=new_ipc, job_name=job_name,
                            evidence_id=new_plan['evidence_id'] + '-BYHASH-RECOVERY',
                            byhash_recovery_parent=str(old), byhash_recovery_parent_job=job,
                            byhash_recovery_boundary='Only never-launched or completed zero-model apt mirror failures; active trials remain with drained parent.',
                            apt_by_hash_force=True)
            save(new / 'plan.json', new_plan)
            template = read(new / 'encbank_harbor_template.json')
            template['job_name'] = name
            template['jobs_dir'] = new_plan['results_root']
            template['tasks'] = [{'path': str(Path(new_plan['task_root']) / task)} for task in tasks]
            save(new / 'encbank_harbor_template.json', template)
            (new / 'apt_by_hash.conf').write_text(APT_CONF)
            service = new / 'apptainer_service.py'
            code = service.read_text()
            needle = "        cmd.extend(['-B',str(root/'resolv.conf')+':/etc/resolv.conf:ro'])\n"
            assert code.count(needle) == 1
            insertion = ("        apt_conf=Path(__file__).with_name('apt_by_hash.conf')\n"
                         "        assert apt_conf.read_text() == 'Acquire::By-Hash \\\"force\\\";\\n'\n"
                         "        cmd.extend(['-B',str(apt_conf)+':/etc/apt/apt.conf.d/99encbank-byhash:ro'])\n")
            service.write_text(code.replace(needle, needle + insertion))
            slurm = new / 'server.slurm'
            script = slurm.read_text()
            assert '#SBATCH --cpus-per-task=32' in script
            script = script.replace('#SBATCH --cpus-per-task=32', '#SBATCH --cpus-per-task=24')
            script = script.replace('#SBATCH --partition=gpu',
                                    '#SBATCH --partition=gpu\n#SBATCH --reservation=SITE_GPU_RESERVATION')
            slurm.write_text(script)
            names = list(manifest) + ['apt_by_hash.conf']
            for source in names:
                if source.endswith('.py'):
                    compile((new / source).read_text(), str(new / source), 'exec')
            save(new / 'server_source_manifest.json', {source: sha(new / source) for source in names})
            save(new / 'byhash_selection.json', {'epoch': time.time(), 'parent_root': str(old),
                 'parent_job': job, 'tasks': tasks, 'unlaunched': sorted(set(tasks) - replay),
                 'zero_call_mirror_failure': sorted(replay),
                 'launched_but_not_selected': sorted(launched - replay),
                 'parent_manifest_sha256': sha(old / 'server_source_manifest.json'),
                 'source_manifest_sha256': sha(new / 'server_source_manifest.json')})
            runs.append({'arm': arm, 'index': index, 'root': str(new), 'parent': str(old),
                         'parent_job': job, 'tasks': tasks,
                         'source_manifest_sha256': sha(new / 'server_source_manifest.json')})
    assert len(runs) == 8
    assert len(set((r['arm'], task) for r in runs for task in r['tasks'])) == sum(len(r['tasks']) for r in runs)
    save(R / 'prepared.json', {'epoch': time.time(), 'registry_sha256': sha(REG), 'runs': runs})
    print(json.dumps({'prepared': [(r['arm'], r['index'], len(r['tasks'])) for r in runs]}), flush=True)


def activate():
    data = read(R / 'prepared.json')
    assert read(REG)['version'] == 4 and sha(REG) == data['registry_sha256']
    assert not (R / 'activated.json').exists()
    registry = read(REG)
    original = {arm: [task for run in registry['runs'] if run['arm'] == arm for task in run['tasks']]
                for arm in ('dense', 'k12', 'k48')}
    for row in data['runs']:
        matches = [run for run in registry['runs'] if run['root'] == row['parent']]
        assert len(matches) == 1 and matches[0]['arm'] == row['arm']
        old = matches[0]
        assert set(row['tasks']).issubset(set(old['tasks']))
        old['tasks'] = [task for task in old['tasks'] if task not in row['tasks']]
        old.setdefault('byhash_transfers', []).append(row['root'])
        registry['runs'].append({'id': Path(row['root']).name, 'arm': row['arm'],
                                'root': row['root'], 'tasks': row['tasks'],
                                'role': 'byhash_recovery_unscored',
                                'source_manifest_sha256': row['source_manifest_sha256']})
    for arm, before in original.items():
        after = [task for run in registry['runs'] if run['arm'] == arm for task in run['tasks']]
        assert len(after) == len(set(after)) and set(after) == set(before)
    registry.update(version=5, gpu_limit=9, byhash_recovery_path=str(R / 'prepared.json'),
                    byhash_recovery_epoch=time.time())
    save(R / 'registry_v4_before_byhash.json', read(REG))
    save(REG, registry)
    save(R / 'activated.json', {'epoch': time.time(), 'registry_sha256': sha(REG)})
    print(json.dumps({'activated': True, 'registry_sha256': sha(REG)}), flush=True)


def submit():
    data = read(R / 'prepared.json')
    assert read(REG)['version'] == 5 and sha(REG) == read(R / 'activated.json')['registry_sha256']
    lock_path = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for row in data['runs']:
            root = Path(row['root'])
            assert not (root / 'submission.json').exists()
            assert sha(root / 'server_source_manifest.json') == row['source_manifest_sha256']
            assert all(sha(root / name) == digest
                       for name, digest in read(root / 'server_source_manifest.json').items())
            receipt = {'status': 'intent', 'epoch': time.time(), 'root': str(root),
                       'arm': row['arm'], 'tasks': len(row['tasks']),
                       'parent_job': row['parent_job'], 'source_manifest_sha256': row['source_manifest_sha256'],
                       'automatic_scientific_retries': 0}
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
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'activate', 'submit'))
    globals()[parser.parse_args().phase]()
