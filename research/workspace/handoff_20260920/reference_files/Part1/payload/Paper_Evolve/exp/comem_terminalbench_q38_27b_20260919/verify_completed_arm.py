"""Check completed trial artifacts without touching the active controller/report."""
import collections, datetime, json, math, sys
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf8'))
plan = read(root / 'plan.json')
arm = sys.argv[1]
assert arm in plan['arms']
config = read(root / 'harbor_configs' / (arm + '.json'))
receipt = read(root / 'receipts' / (arm + '.json'))
assert receipt['actual_child_wait'] and receipt['exit_code'] == 0
released = read(root / 'receipts' / (arm + '.released.json'))
trials = [read(p) for p in (root / plan['results_subdir'] / config['job_name']).glob('*/result.json')]
assert len(trials) == len(plan['tasks'])
assert {r['task_name'].rsplit('/', 1)[-1] for r in trials} == set(plan['tasks'])
assert all(r.get('finished_at') for r in trials)
rpc = root / plan['rpc_subdir']
requests = [read(p) for p in rpc.glob('*.request.json')]
requests = [q for q in requests if q['arm'] == arm and q['request_id'] not in plan.get('excluded_request_ids', [])]
assert len(requests) == receipt['requests']
assert len({q['request_id'] for q in requests}) == len(requests)
assert {q['task_id'] for q in requests} == set(released)
assert any(not r['remaining_sessions'] for r in released.values())
responses = []
for q in requests:
    r = read(rpc / (q['request_id'] + '.response.json'))
    assert r['request_id'] == q['request_id'] and r['arm'] == arm
    assert r['status'] in ('ok', 'deadline', 'cancelled', 'context_limit')
    responses.append(r)
rows = []
for r in trials:
    rewards = (r.get('verifier_result') or {}).get('rewards')
    score = None if rewards is None else rewards['reward']
    assert score is None or (isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1)
    rows.append(dict(task=r['task_name'], reward=score, exception=(r.get('exception_info') or {}).get('exception_type')))
print(json.dumps(dict(status='PASS', checked_at=datetime.datetime.now().astimezone().isoformat(),
    arm=arm, job_name=config['job_name'], actual_child_wait=True, exit_code=0,
    trials=rows, request_count=len(requests), response_statuses=dict(collections.Counter(r['status'] for r in responses)),
    released_sessions=len(released), empty_bank_ack=True, connection_retries=receipt['connection_retries']), indent=2))
