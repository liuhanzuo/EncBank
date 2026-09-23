"""One persistent local owner follows three serial server model holders."""
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/srv/encbank/client/comem_local_20260921')


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    lock=(ROOT/'batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'owner_start.json').exists(),'Existing owner must be reconciled, never silently restarted'
    save(ROOT/'owner_start.json',dict(pid=os.getpid(),epoch=time.time()))
    rows=[]
    for family in ['dense','k12','k48']:
        run=ROOT/'runs'/family
        save(ROOT/'status.json',dict(state='running',family=family,completed=rows,epoch=time.time()))
        with (run/'owner.stdout.log').open('xb') as out,(run/'owner.stderr.log').open('xb') as err:
            child=subprocess.run(['/srv/encbank/client/comem_harbor_023/bin/python','-u',str(run/'hybrid_owner.py')],cwd=run,stdout=out,stderr=err,
                 env=dict(os.environ,PYTHONPATH=str(run),LITELLM_LOCAL_MODEL_COST_MAP='True'))
        row=dict(family=family,owner_exit=child.returncode,actual_parent_wait=True,epoch=time.time())
        rows.append(row);save(run/'local_owner_receipt.json',row)
        if child.returncode!=0:
            save(ROOT/'status.json',dict(state='attention_required',completed=rows,epoch=time.time()))
            raise RuntimeError('Controller failed; no automatic restart')
    save(ROOT/'status.json',dict(state='all_controllers_closed',completed=rows,epoch=time.time(),scores_require_result_audit=True))


if __name__=='__main__':main()
