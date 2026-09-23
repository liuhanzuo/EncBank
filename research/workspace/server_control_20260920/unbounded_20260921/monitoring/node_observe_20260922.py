"""Read-only host audit; run through SSH on each allocated node, stdout JSON only."""
import json
import os
import socket
import subprocess
import time
from pathlib import Path

S = Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
M = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920/managed_instances')

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def proc(pid):
    try:
        p = Path('/proc') / str(pid)
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        env = (p / 'environ').read_bytes().split(b'\0')
        job = next((v.split(b'=', 1)[1].decode() for v in env if v.startswith(b'SLURM_JOB_ID=')), None)
        return dict(alive=stat[0] != 'Z', start_ticks=int(stat[19]), job_id=job)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return dict(alive=False)

report = dict(epoch=time.time(), host=socket.gethostname(), runs={}, errors=[])
registry = read(U / 'scale4_20260921/registry.json')
queue = subprocess.check_output(['squeue', '-r', '-h', '-u', 'liuhanzuo', '-o', '%i|%T|%N'], text=True)
active_ids = {line.split('|')[0] for line in queue.splitlines()}
gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
gpus = {}
for line in gpu_rows.splitlines():
    uuid, used, total, util = [v.strip() for v in line.split(',')]
    gpus[uuid.removeprefix('GPU-')] = dict(memory_mib=int(used), total_mib=int(total), utilization_percent=int(util))
processes = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid == os.getuid():
            argv = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'server_owner.py' in argv:
                processes.append((int(p.name), argv))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
specs = [(p.parent, read(p)) for p in M.glob('*/spec.json')]
for spec in registry['runs']:
    h = Path(spec['root'])
    submission = read(h / 'submission.json') or {}
    job = str(submission.get('job_id'))
    owner = read(h / 'execution/owner_registration.json') or {}
    if job not in active_ids or owner.get('hostname') != socket.gethostname():
        continue
    plan = read(h / 'plan.json')
    run = h / ('run_' + plan['arm'])
    worker = read(run / 'worker_ready.json') or {}
    state = read(h / 'execution/status.json') or {}
    op = proc(owner['identity']['pid'])
    wp = proc(worker.get('pid'))
    op['identity_matches'] = (op.get('start_ticks') == owner['identity']['start_ticks']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == owner['identity']['boot_id'])
    candidates = [pid for pid, argv in processes if str(h / 'server_owner.py') in argv]
    row = dict(job_id=job, root=str(h), owner=op, worker=wp, owner_pids=candidates,
        unique_owner=candidates == [owner['identity']['pid']], state=state.get('state'),
        status_age_seconds=time.time()-state.get('epoch', 0), active=list(state.get('active', {})),
        gpu_uuid=worker.get('uuid'), gpu=gpus.get(worker.get('uuid', '').removeprefix('GPU-')),
        containers=[])
    transport = read(h / 'execution/transport_snapshot.json') or {}
    row['transport'] = {k: transport.get(k) for k in ['event_cursor', 'response_counts', 'generated_tokens', 'disk_event_errors']}
    for root, cs in specs:
        if Path(cs.get('startup_lock', '/missing')).parent != h:
            continue
        logs = next((source for source, target in cs.get('binds', []) if target == '/logs/agent'), None)
        if not logs:
            continue
        task = Path(logs).parent.parent.name
        qualification = 'node_runtime_qualification' in Path(logs).parts
        if task not in row['active'] or qualification:
            continue
        service = read(root / 'service.json') or {}
        cp = proc(cs.get('owner_pid'))
        cg = Path('/sys/fs/cgroup') / service.get('cgroup', '/missing').lstrip('/')
        evidence = dict(name=cs['name'], task=task, role='benchmark', service_state=service.get('state'),
            owner=cp, owner_ticks_match=str(cp.get('start_ticks')) == str(cs.get('owner_ticks')))
        for filename in ['cgroup.events', 'memory.events', 'cgroup.procs']:
            evidence[filename] = (cg / filename).read_text() if (cg / filename).exists() else None
        row['containers'].append(evidence)
    if not op.get('alive') or not op.get('identity_matches') or op.get('job_id') != job or not row['unique_owner']:
        report['errors'].append(spec['id'] + ': owner identity')
    if not wp.get('alive') or wp.get('job_id') != job:
        report['errors'].append(spec['id'] + ': worker identity')
    if sorted(c['task'] for c in row['containers']) != sorted(row['active']):
        report['errors'].append(spec['id'] + ': active/container mismatch')
    for c in row['containers']:
        if not c['owner'].get('alive') or not c['owner_ticks_match'] or 'populated 1' not in (c['cgroup.events'] or ''):
            report['errors'].append(spec['id'] + ': container ' + c['task'])
    report['runs'][spec['id']] = row
report['dispatcher'] = {name: read(U / 'scale4_20260921' / name) for name in
    ['dispatcher_status.json', 'dispatcher_complete.json', 'dispatcher_failure.json']}
report['dispatcher_accounting'] = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '117351',
    '--format=JobID,State,ExitCode,Elapsed,Timelimit,NodeList'], text=True)
print(json.dumps(report))
