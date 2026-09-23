"""Reuse the existing verified SSH/SCP mailbox route, with task-specific deadlines."""
from pathlib import Path
import hashlib,json,sys,time
from persistence import dump
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
action,arm=sys.argv[1:3];assert arm=='dense';R=H/'run_dense';B=R/'mailbox'
if action=='status':
    print(json.dumps({n:json.loads((R/n).read_text()) for n in ['worker_ready.json','worker_failure.json','process_receipt.json','memory_cap_failure.json'] if (R/n).exists()}))
elif action=='stop':
    assert B.exists()
    if (H/'service_transfer.json').exists() and (len(sys.argv)<4 or sys.argv[3]!='successor'):
        print(json.dumps(dict(stop_deferred_to_successor=True,transfer=json.loads((H/'service_transfer.json').read_text()))))
    else:
        dump(B/'stop.json',dict(reason='local Harbor parent closed'));print('{"stop_requested":true}')
else:
    rid=sys.argv[3];assert rid.replace('_','').isalnum()
    req=B/(rid+'.request.json');resp=B/(rid+'.response.json');incoming=H/'incoming'/(rid+'.json')
    if action=='publish':
        d=json.loads(incoming.read_text());assert d['request_id']==rid and not req.exists()
        assert (R/'worker_ready.json').exists() and not (R/'worker_failure.json').exists() and not (R/'process_receipt.json').exists()
        remaining=d.pop('remaining_seconds');assert 0<=remaining<=12000
        d['deadline_epoch']=time.time()+remaining;dump(req,d);print('{"published":true}')
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
