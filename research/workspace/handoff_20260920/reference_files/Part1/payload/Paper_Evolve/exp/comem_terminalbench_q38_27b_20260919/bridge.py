"""Node-local request publish/wait/control. No remote shared-filesystem writes."""
import json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BOX=ROOT/'mailbox';BOX.mkdir(exist_ok=True)
action=sys.argv[1]
if action=='status':
    names=['ready.json','environment.json','smoke.json','failure.json','status.json','complete.json','parent_exit.json']
    print(json.dumps({n:json.loads((ROOT/n).read_text()) for n in names if (ROOT/n).exists()}))
elif action=='stop':
    (BOX/'stop.json').write_text('{}');print('{}')
elif action=='release':
    sid=sys.argv[2];assert re.fullmatch('[a-f0-9]{24}',sid)
    (BOX/(sid+'.release.json')).write_text(json.dumps(dict(task_id=sid)));print('{}')
elif action in ['publish','cancel','wait','poll']:
    rid=sys.argv[2];assert re.fullmatch('[a-f0-9]{24}_[0-9]{5}',rid)
    if action=='publish':
        q=ROOT/'incoming'/(rid+'.json');d=json.loads(q.read_text())
        assert d['request_id']==rid and not (BOX/(rid+'.request.json')).exists()
        # Account for file transfer and controller overhead before GPU execution.
        d['deadline_epoch']=d['client_created_epoch']+d['remaining_seconds'];d['published_epoch']=time.time()
        tmp=BOX/(rid+'.publish');tmp.write_text(json.dumps(d));tmp.replace(BOX/(rid+'.request.json'));q.unlink();print('{}')
    elif action=='cancel':(BOX/(rid+'.cancel.json')).write_text('{}');print('{}')
    elif action=='poll':
        q=BOX/(rid+'.response.json')
        if q.exists():print(json.dumps(dict(ready=True,response=json.loads(q.read_text()))))
        elif (ROOT/'failure.json').exists():raise RuntimeError((ROOT/'failure.json').read_text())
        elif (ROOT/'parent_exit.json').exists():raise RuntimeError('Worker exited before reply')
        else:print(json.dumps(dict(ready=False)))
    else:
        began=time.monotonic();q=BOX/(rid+'.response.json')
        while not q.exists():
            if (ROOT/'failure.json').exists():raise RuntimeError((ROOT/'failure.json').read_text())
            if (ROOT/'parent_exit.json').exists():raise RuntimeError('Worker exited before reply')
            if time.monotonic()-began>15000:raise TimeoutError(rid)
            time.sleep(.15)
        print(q.read_text())
else:raise ValueError(action)
