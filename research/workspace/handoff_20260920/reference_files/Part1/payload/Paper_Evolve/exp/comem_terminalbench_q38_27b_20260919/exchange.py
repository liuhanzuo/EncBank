"""Idempotent node-local mailbox transaction over one bounded SSH call."""
import json,re,sys,time
from pathlib import Path

def exchange(root,packet):
    box=root/'mailbox';box.mkdir(exist_ok=True)
    def rid(value):
        assert re.fullmatch('[a-f0-9]{24}_[0-9]{5}',value),value
        return value
    def sid(value):
        assert re.fullmatch('[a-f0-9]{24}',value),value
        return value
    accepted=[]
    for q in packet.get('requests',[]):
        name=rid(q['request_id']);path=box/(name+'.request.json')
        if path.exists():
            old=json.loads(path.read_text())
            assert all(old.get(k)==v for k,v in q.items()),'Request ID reused with different contents'
        else:
            d=dict(q,deadline_epoch=q['client_created_epoch']+q['remaining_seconds'],published_epoch=time.time())
            tmp=path.with_suffix('.exchange.tmp');tmp.write_text(json.dumps(d));tmp.replace(path)
        accepted.append(name)
    for name in packet.get('cancel',[]):
        (box/(rid(name)+'.cancel.json')).write_text('{}')
    for name in packet.get('release',[]):
        (box/(sid(name)+'.release.json')).write_text(json.dumps(dict(task_id=name)))
    responses={}
    for name in packet.get('poll',[]):
        path=box/(rid(name)+'.response.json')
        if path.exists():responses[name]=json.loads(path.read_text())
    released={}
    for name in packet.get('release',[]):
        path=box/(sid(name)+'.released.json')
        if path.exists():released[name]=json.loads(path.read_text())
    state={}
    for name in ['failure.json','parent_exit.json']:
        if (root/name).exists():state[name]=json.loads((root/name).read_text())
    return dict(accepted=accepted,responses=responses,released=released,state=state,at=time.time())

if __name__=='__main__':
    print(json.dumps(exchange(Path(__file__).resolve().parent,json.load(sys.stdin))))
