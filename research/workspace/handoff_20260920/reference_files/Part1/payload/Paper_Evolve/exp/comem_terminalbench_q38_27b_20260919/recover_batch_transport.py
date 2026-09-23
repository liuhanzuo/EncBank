"""Archive the objectively broken controller attempt; keep the loaded GPU."""
import ast,json,shlex,subprocess,time
from pathlib import Path
from io_utils import save
import owner_batch as owner

ROOT=owner.ROOT
for name in ['owner_batch.py','exchange.py','test_exchange.py']:
    ast.parse((ROOT/name).read_text())
requests=[json.loads(p.read_text()) for p in (ROOT/'rpc').glob('*.request.json')]
assert requests and {q['arm'] for q in requests}=={'iter_k12'}
ids=[q['request_id'] for q in requests];sessions=sorted({q['task_id'] for q in requests})
packet=dict(poll=ids,cancel=ids,release=sessions)
cmd=owner.PY+' '+shlex.quote(owner.REMOTE+'/exchange.py')
started=time.monotonic();checks=[]
for _ in range(40):
    response=owner.retry_remote(cmd,packet)
    assert not response['state'],response['state']
    checks.append(dict(at=time.time(),responses=len(response['responses']),released=len(response['released'])))
    if len(response['released'])==len(sessions):break
    time.sleep(1)
else:raise RuntimeError('Old H not released')
archive=ROOT/'infrastructure_attempts'/'20260920_second_arm_transport'
archive.mkdir(exist_ok=False)
save(archive/'remote_exchange.json',response)
save(archive/'release_validation.json',dict(checks=checks,seconds=time.monotonic()-started,passed=True))
code="import json;from pathlib import Path;r=Path("+repr(owner.REMOTE)+");print(json.dumps(dict(mailbox={p.name:json.loads(p.read_text()) for p in (r/'mailbox').glob('*.json')},events=(r/'events.jsonl').read_text() if (r/'events.jsonl').exists() else None)))"
save(archive/'remote_records.json',owner.retry_remote(owner.PY+' -c '+shlex.quote(code)))
save(ROOT/'batch_recovery_ready.json',dict(archive=str(archive),ids=ids,sessions=sessions,at=owner.now(),
    classification='Entire second six-task controller attempt invalidated due one fatal SSH connection error at 00:16 that stopped dispatch for every task. No task outcome selection. Owner stopped before GPU finally; own Harbor SIGINT. Scientific configuration unchanged.',
    fixes='Single sequential batched mailbox transaction; idempotent request publication and response retrieval; bounded connection retry, original absolute agent deadlines preserved; GPU retained.'))
print(json.dumps(dict(released=len(sessions),responses=len(response['responses']),archive=str(archive))))
