"""One-shot: reserve local capacity and defer ONLY this owner's still-pending job."""
import datetime,fcntl,json,subprocess,time
from pathlib import Path
ROOT=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
TOKEN='comem-bge-matched-20260919'
def run(cmd):return subprocess.check_output(cmd,universal_newlines=True,timeout=30).strip()
def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
    lock=(ROOT.parent/'qcomem_gpu_admission.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX)
    plan=json.loads((ROOT/'effective_plan.json').read_text());rs=plan['resources']
    def queue():
        rows=[]
        for line in run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b']).splitlines():
            ident,name,state,gres=line.split('|')
            if ident in rs['shared_preexisting_jobs'] or any(name.startswith(x) for x in rs['counted_name_prefixes']):
                rows.append(dict(id=ident,name=name,state=state,gpus=int(gres.split(':')[-1])))
        return rows
    hist=ROOT/'maintenance_history'/'local-bge-start-20260919';hist.mkdir(exist_ok=True)
    assert not (hist/'complete.json').exists(),'One-shot already executed; inspect receipt'
    p=ROOT/'local_gpu_reservation.json';old=json.loads(p.read_text()) if p.exists() else None
    assert old is None or old['token']==TOKEN
    dump(p,dict(token=TOKEN,local_pid=None,gpus=1,reason='User authorized immediate local same-BGE run; controller will adopt token'))
    before=queue();dump(hist/'before.json',before)
    deferred=[]
    if sum(j['gpus'] for j in before)>3:
        candidates=[j for j in before if j['id']=='105748' and j['name']=='qcm-q18-large-final-m0-s2' and j['state']=='PENDING']
        if candidates:
            job=candidates[0];task=job['name'][len('qcm-q18-'):]
            receipt=ROOT/'runs'/task/'submission.json';rec=json.loads(receipt.read_text());assert rec['job']==job['id']
            assert not (ROOT/'runs'/task/'parent_exit.json').exists()
            dump(hist/'deferred_submission.json',rec)
            # Scheduler-side PENDING filter avoids a race that could kill a newly running job.
            r=subprocess.run(['scancel','--state=PENDING',job['id']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
            dump(hist/'cancel_receipt.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
            assert r.returncode==0
            for _ in range(10):
                current=queue()
                if not any(j['id']==job['id'] for j in current):break
                if any(j['id']==job['id'] and j['state']=='RUNNING' for j in current):break
                time.sleep(1)
            accounting=run(['sacct','-j',job['id'],'-n','-P','-o','JobIDRaw,State,ExitCode,Start'])
            dump(hist/'accounting.json',dict(raw=accounting))
            if not any(j['id']==job['id'] for j in current):
                parent=[l.split('|') for l in accounting.splitlines() if l.split('|')[0]==job['id']]
                assert len(parent)==1 and parent[0][1].startswith('CANCELLED') and parent[0][3] in ('Unknown','None',''),parent
                assert json.loads(receipt.read_text())['job']==job['id']
                receipt.rename(hist/'original_submission.json')
                deferred.append(dict(job=job['id'],task=task,reason='Pending only; requeue automatically when local reservation releases'))
    after=queue();result=dict(at=datetime.datetime.utcnow().isoformat()+'Z',before=before,after=after,deferred=deferred,
        local_may_start=sum(j['gpus'] for j in after)<=3)
    dump(hist/'complete.json',result);print(json.dumps(result))
if __name__=='__main__':main()
