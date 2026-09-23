"""No GPU calls: validate retries cannot create duplicate or altered generations."""
import json,tempfile
from pathlib import Path
from exchange import exchange

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);name='a'*24+'_00000';session='a'*24
    q=dict(request_id=name,task_id=session,step=0,remaining_seconds=900,client_created_epoch=100,messages=[dict(role='user',content='probe')])
    first=exchange(root,dict(requests=[q],poll=[name]))
    path=root/'mailbox'/(name+'.request.json');before=path.read_bytes()
    again=exchange(root,dict(requests=[q],poll=[name]))
    assert first['accepted']==again['accepted']==[name] and path.read_bytes()==before
    assert json.loads(before)['deadline_epoch']==1000
    try:exchange(root,dict(requests=[dict(q,step=1)]))
    except AssertionError:pass
    else:raise AssertionError('Conflicting retry accepted')
    answer=dict(request_id=name,status='ok',text='cached response')
    (root/'mailbox'/(name+'.response.json')).write_text(json.dumps(answer))
    assert exchange(root,dict(poll=[name]))['responses'][name]==answer
    assert exchange(root,dict(poll=[name]))['responses'][name]==answer
    exchange(root,dict(cancel=[name],release=[session]))
    assert (root/'mailbox'/(name+'.cancel.json')).exists()
    assert (root/'mailbox'/(session+'.release.json')).exists()
    print('PASS: idempotent publication/reply, conflict rejection, unchanged deadline, cancellation/release; zero model calls')
