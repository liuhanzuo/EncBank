"""Reserve one of this task's four GPUs for the local 5090, under the existing admission lock."""
import argparse,fcntl,json,subprocess
from pathlib import Path
ROOT=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['reserve','release','status']);p.add_argument('--token',required=True);p.add_argument('--pid',type=int)
    a=p.parse_args();lock=(ROOT.parent/'qcomem_gpu_admission.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX)
    path=ROOT/'local_gpu_reservation.json';old=json.loads(path.read_text()) if path.exists() else None
    if a.action=='reserve':
        assert old is None or old['token']==a.token,'Different owner has local reservation'
        value=dict(token=a.token,local_pid=a.pid,gpus=1,reason='Same-BGE matched-TTFT on local RTX5090; keep total own GPUs <=4')
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.replace(path);old=value
    elif a.action=='release':
        if old:
            assert old['token']==a.token;path.unlink();old=None
    text=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],universal_newlines=True,timeout=30)
    plan=json.loads((ROOT/'effective_plan.json').read_text());rs=plan['resources'];jobs=[]
    for line in text.splitlines():
        ident,name,state,gres=line.split('|')
        if ident in rs['shared_preexisting_jobs'] or any(name.startswith(x) for x in rs['counted_name_prefixes']):
            jobs.append(dict(id=ident,name=name,state=state,gpus=int(gres.split(':')[-1])))
    print(json.dumps(dict(reservation=old,remote_requests=sum(r['gpus'] for r in jobs),jobs=jobs,
        local_may_start=old is not None and sum(r['gpus'] for r in jobs)<=3)))
if __name__=='__main__':main()
