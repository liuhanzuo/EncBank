"""Reuse the existing verified SSH/SCP mailbox route, with task-specific deadlines."""
from pathlib import Path
import hashlib,json,sys,time
from persistence import dump
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
action,arm=sys.argv[1:3];assert arm=='comem';R=H/'run_comem';B=R/'mailbox'
if action=='status':
    print(json.dumps({n:json.loads((R/n).read_text()) for n in ['worker_ready.json','worker_failure.json','process_receipt.json','memory_cap_failure.json'] if (R/n).exists()}))
elif action=='stop':
    assert B.exists();dump(B/'stop.json',dict(reason='local Harbor parent closed'));print('{"stop_requested":true}')
elif action=='release':
    sid=sys.argv[3];assert sid.replace('_','').isalnum()
    dump(B/(sid+'.release.json'),dict(task_id=sid,reason='owned Harbor task parent wait closed'))
    print('{"release_requested":true}')
else:
    rid=sys.argv[3];assert rid.replace('_','').isalnum()
    req=B/(rid+'.request.json');resp=B/(rid+'.response.json');incoming=H/'incoming'/(rid+'.json')
    if action=='publish':
        d=json.loads(incoming.read_text());assert d['request_id']==rid and not req.exists()
        assert (R/'worker_ready.json').exists() and not (R/'worker_failure.json').exists() and not (R/'process_receipt.json').exists()
        remaining=d.pop('remaining_seconds');assert 0<=remaining<=12000
        d['published_epoch']=time.time();d['deadline_epoch']=d['published_epoch']+remaining;dump(req,d);print('{"published":true}')
    elif action=='cancel':dump(B/(rid+'.cancel.json'),dict(cancelled=True));print('{"cancel_requested":true}')
    elif action=='wait':
        d=json.loads(req.read_text())
        while not resp.exists():
            if (R/'worker_failure.json').exists():raise RuntimeError((R/'worker_failure.json').read_text())
            if (R/'process_receipt.json').exists():raise RuntimeError('Worker exited before response')
            if time.time()>d['deadline_epoch']+90:raise TimeoutError('Worker did not close expired generation')
            time.sleep(.2)
        raw=resp.read_bytes();print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),path=str(resp))))
    else:raise ValueError(action)
