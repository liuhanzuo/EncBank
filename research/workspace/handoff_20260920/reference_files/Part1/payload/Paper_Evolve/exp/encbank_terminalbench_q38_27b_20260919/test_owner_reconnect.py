"""A lost SSH reply after publication must not regenerate or cancel the task."""
import json,tempfile,time
from pathlib import Path
import owner_batch as owner
from exchange import exchange

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);owner.ROOT=root/'host';owner.ROOT.mkdir();owner.RPC=owner.ROOT/'rpc';owner.RPC.mkdir()
    remote=root/'remote';remote.mkdir();sid='b'*24;rid=sid+'_00000'
    q=dict(request_id=rid,task_id=sid,arm='iter_k12',step=0,client_created_epoch=time.time(),remaining_seconds=900,messages=[])
    owner.save(owner.RPC/(rid+'.request.json'),q)
    calls=[];published=[]
    def emulate(packet):
        calls.append(packet);result=exchange(remote,packet)
        request=remote/'mailbox'/(rid+'.request.json')
        if request.exists():published.append(request.read_bytes())
        if len(calls)==1:raise owner.TransientTransportError('Injected connection drop after remote publication')
        if packet.get('requests') or packet.get('poll'):
            path=remote/'mailbox'/(rid+'.response.json')
            if not path.exists():owner.save(path,dict(request_id=rid,status='ok',generated_ids=[42]))
        for s in packet.get('release',[]):owner.save(remote/'mailbox'/(s+'.released.json'),dict(task_id=s))
        return exchange(remote,packet)
    class Child:
        def poll(self):return 0 if (owner.RPC/(rid+'.response.json')).exists() else None
        def wait(self):return 0
    owner.exchange=emulate
    owner.retry_remote=lambda command,packet=None,**kwargs:emulate(packet or {})
    receipt=owner.serve_arm(Child(),'iter_k12')
    assert receipt['connection_retries']==1 and receipt['requests']==1
    assert len(set(published))==1 and len(list((remote/'mailbox').glob('*.request.json')))==1
    assert not (remote/'mailbox'/(rid+'.cancel.json')).exists()
    assert not list(owner.RPC.glob('*.error.json'))
    assert json.loads((owner.RPC/(rid+'.response.json')).read_text())['generated_ids']==[42]
    print('PASS: connection lost after publish; same ID/deadline retained, one generation, reply delivered, release acknowledged; no GPU calls')
