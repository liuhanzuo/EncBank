"""Read-only closed-task audit on its allocated host. Emit compact evidence, no model artifacts."""
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

S = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
M = Path('/srv/encbank/qencbank_runtime_20260911/server_control_20260920/managed_instances')
root = (S / sys.argv[1]).resolve()
assert root.is_relative_to(S)

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

plan = read(root / 'plan.json')
registration = read(root / 'execution/owner_registration.json')
assert registration['hostname'] == socket.gethostname(), 'Use the actual allocated host'
specs = [(p.parent, read(p)) for p in M.glob('*/spec.json')]
report = dict(epoch=time.time(), hostname=socket.gethostname(), root=str(root), tasks={})
for task in sys.argv[2:]:
    assert task in plan['tasks']
    execution = root / 'execution'
    outcome = read(execution / 'task_outcomes' / (task + '.json'))
    row = dict(task=task, outcome=outcome)
    report['tasks'][task] = row
    if not outcome:
        continue
    receipt = read(execution / 'receipts' / (task + '.json'))
    launch = read(execution / 'launches' / (task + '.json'))
    assert len(outcome['results']) == 1
    result_path = Path(outcome['results'][0]['path'])
    raw_result = result_path.read_bytes()
    result = json.loads(raw_result)
    trial = result_path.parent.name
    task_id = 'tb_' + hashlib.sha256(trial.encode()).hexdigest()[:24]
    box = Path(plan['rpc_root']) / plan['arm']
    proofs = []
    for path in sorted(box.glob(task_id + '*.request.json')):
        raw_request = path.read_bytes()
        request = json.loads(raw_request)
        assert request['task'] == task and request['trial'] == trial
        response_path = path.with_name(path.name.replace('.request.', '.response.'))
        broker_path = path.with_name(path.name.replace('.request.', '.broker.'))
        raw = response_path.read_bytes()
        response = json.loads(raw)
        broker = read(broker_path)['proof']
        sha = hashlib.sha256(raw).hexdigest()
        proofs.append(dict(request_id=request['request_id'], request_sha256=hashlib.sha256(raw_request).hexdigest(),
            response_sha256=sha, sha_matches=sha == broker['sha256'], bytes_match=len(raw) == broker['bytes'],
            identity_matches=response['request_id'] == request['request_id'], status=response['status'],
            H_unchanged=response.get('state_hashes_unchanged'), generated_tokens=response.get('generated_tokens'),
            request_seconds=response.get('request_seconds'), hit_generation_cap=response.get('hit_generation_cap'),
            hit_context_capacity=response.get('hit_context_capacity')))
    releases = [json.loads(line) for line in (root / ('run_' + plan['arm']) / 'events.jsonl').read_text().splitlines()
                if 'session_release' in line and task_id in line]
    closures = []
    for directory, spec in specs:
        if Path(spec.get('startup_lock', '/missing')).parent != root:
            continue
        if not any(trial in source for source, target in spec.get('binds', []) if target == '/logs/agent'):
            continue
        service = read(directory / 'service.json') or {}
        cgroup = Path('/sys/fs/cgroup') / service.get('cgroup', '/missing').lstrip('/')
        closures.append(dict(name=spec['name'], closure=read(directory / 'closure.json'),
            cgroup_exists_on_actual_host=cgroup.exists(),
            cgroup_procs=(cgroup / 'cgroup.procs').read_text() if (cgroup / 'cgroup.procs').exists() else None))
    trajectory = read(result_path.parent / 'agent/trajectory.json') or {}
    warnings, errors = [], []
    for step in trajectory.get('steps', []):
        for observation in (step.get('observation') or {}).get('results', []):
            content = str(observation.get('content', ''))
            item = dict(step=step['step_id'], message=content.split('New Terminal Output:', 1)[0][:500])
            if content.startswith('Previous response had warnings:'):
                warnings.append(item)
            if content.startswith(('Previous response had parsing errors:', 'Previous response had errors:')):
                errors.append(item)
    verifier = {}
    for filename in ['test-stdout.txt', 'test-stderr.txt', 'reward.txt']:
        path = result_path.parent / 'verifier' / filename
        if not path.exists():
            continue
        raw = path.read_bytes()
        lines = raw.decode(errors='replace').splitlines()
        verifier[filename] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            result_lines=[line[:600] for line in lines if any(s in line.lower() for s in
                ['failed', 'passed', 'error', 'assert', 'timeout'])][-15:])
        if filename == 'reward.txt':
            verifier[filename]['value'] = raw.decode(errors='replace')[:300]
    row.update(receipt=receipt, launch=launch, result_sha256=hashlib.sha256(raw_result).hexdigest(),
        result={k: result.get(k) for k in ['started_at', 'finished_at', 'environment_setup', 'agent_setup',
            'agent_execution', 'verifier', 'exception_info', 'verifier_result']},
        responses=proofs, session_release=releases, containers=closures,
        trajectory_agent_steps=sum(s.get('source') == 'agent' for s in trajectory.get('steps', [])),
        parser_warnings=warnings, parser_errors=errors, verifier_files=verifier,
        parent_pid_present_on_actual_host=Path('/proc/' + str(receipt['pid'])).exists() if receipt else None)
print(json.dumps(report))
