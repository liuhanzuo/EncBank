"""Verified reply transport through the existing Windows OpenSSH identity."""
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path
from server_transport import save, verify_response

SSH='/mnt/c/Windows/System32/OpenSSH/ssh.exe'
PY='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'


class Transport:
    def __init__(self,home,plan,output):
        self.home,self.plan,self.output=Path(home),plan,Path(output)
        self.cursor=0

    def argv(self,action,value=None):
        assert action in {'status','publish','wait','cancel','release','stop'}
        if value is not None:assert all(c.isalnum() or c in '_-' for c in str(value))
        remote=f"{PY} {self.plan['remote_root']}/file_bridge.py {action} {self.plan['arm']}"
        if value is not None:remote+=' '+str(value)
        return [SSH,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gpu-node1',remote]

    def ctl(self,action,rid=None):
        value=self.cursor if action=='status' else rid
        result=subprocess.run(self.argv(action,value),capture_output=True,text=True,timeout=65)
        if result.returncode:raise RuntimeError(result.stderr[-3000:])
        data=json.loads(result.stdout)
        if 'transport' in data:
            save(self.output/('transport_status_'+str(time.time_ns())+'.json'),data['transport'])
            save(self.output/'transport_snapshot.json',data['transport'])
            self.cursor=data['transport']['event_cursor']
        return data

    def handle(self,request_path,child):
        path=Path(request_path);q=json.loads(path.read_text());rid=q['request_id'];box=path.parent
        started=time.monotonic();cancelled=False;wait=None
        try:
            # An uncertain publish acknowledgement is never retried.
            result=subprocess.run(self.argv('publish',rid),input=path.read_bytes(),capture_output=True,timeout=65)
            if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
            json.loads(result.stdout)
            transfer=box/(rid+'.transfer')
            with transfer.open('wb') as output, (box/(rid+'.ssh-error.log')).open('wb') as error:
                wait=subprocess.Popen(self.argv('wait',rid),stdin=subprocess.DEVNULL,stdout=output,stderr=error)
                while wait.poll() is None:
                    if not cancelled and ((box/(rid+'.cancel.json')).exists() or child.poll() is not None):
                        self.ctl('cancel',rid);cancelled=True
                    if time.monotonic()-started>q['remaining_seconds']+240:
                        raise TimeoutError('Reply transport deadline; do not replay generation')
                    time.sleep(.1)
                code=wait.wait()
            assert code==0, 'SSH reply failed; see request ssh-error log'
            proof=json.loads(transfer.read_bytes())
            data=verify_response(proof,rid,self.plan['remote_root'],self.plan['arm'])
            response=json.loads(data)
            assert response['task']==q['task'] and response['step']==q['step']
            save(box/(rid+'.response.json'),response)
            proof.pop('payload_base64')
            save(box/(rid+'.broker.json'),dict(status='delivered',proof=proof,seconds=time.monotonic()-started,
                 transport='Windows OpenSSH to server-owned GPU mailbox',cancel_sent=cancelled))
        except BaseException as exc:
            if wait is not None and wait.poll() is None:
                wait.terminate();wait.wait(timeout=30)
            try:self.ctl('cancel',rid)
            except Exception:pass
            save(box/(rid+'.error.json'),dict(error=repr(exc),no_automatic_request_retry=True,classification='INFRASTRUCTURE_FAILURE'))
            raise
