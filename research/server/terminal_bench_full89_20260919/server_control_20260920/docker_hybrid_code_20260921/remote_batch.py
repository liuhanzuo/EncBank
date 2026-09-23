"""One server GPU allocation; local task parents drive each model arm to closure."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
B=ROOT.parents[1]


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def verify(family):
    r=ROOT/family
    for name,digest in json.loads((r/'source_manifest.json').read_text()).items():
        assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name


def submit():
    assert json.loads((ROOT/'local_preflight.json').read_text())['status']=='PASS'
    for family in ['dense','k12','k48']:
        verify(family);assert not (ROOT/family/('run_dense' if family=='dense' else 'run_comem')).exists()
        p=json.loads((ROOT/family/'plan.json').read_text())
        result=subprocess.run([p['engine_python'],str(ROOT/family/'model_path_preflight.py')],capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stderr
    with (B.parent/'terminal_bench_20260918/admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (ROOT/'submission.json').exists()
        queue=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
        own=[x for x in queue.splitlines() if any(s in x for s in ['qcomem-tb-','qcomem-agentmem-']) and 'gpu' in x.split('|')[-1].lower()]
        known={'112400':'comem_k12_no_task_deadline_r6_20260920','112403':'comem_k48_no_task_deadline_r6_20260920','113514':'dense_no_task_deadline_r6_20260921'}
        assert len(own)<4
        proofs=[]
        for line in own:
            job=line.split('|')[0];assert job in known, 'Unknown controller: '+line
            f=B/known[job]/'plan.json';p=json.loads(f.read_text());assert not set(p['tasks'])&{'regex-chess','vulnerable-secret'}
            proofs.append(dict(job_id=job,plan=str(f),plan_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),tasks=p['tasks']))
        record=dict(status='intent',epoch=time.time(),queue=own,ownership=proofs,model_remote=True,task_environments='local Docker')
        save(ROOT/'submission.json',record)
        result=subprocess.run(['sbatch','--parsable',str(ROOT/'batch.slurm')],capture_output=True,text=True,timeout=30)
        record.update(status='submitted' if result.returncode==0 else 'failed',exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,actual_parent_wait=True)
        if result.returncode==0:record['job_id']=result.stdout.strip().split(';')[0]
        save(ROOT/'submission.json',record);print(json.dumps(record))
        assert result.returncode==0,result.stderr


def run():
    assert os.environ.get('SLURM_JOB_ID')
    rows=[]
    for family in ['dense','k12','k48']:
        verify(family);r=ROOT/family
        save(ROOT/'status.json',dict(state='running',family=family,completed=rows,job_id=os.environ['SLURM_JOB_ID'],epoch=time.time()))
        with (r/'allocation.stdout.log').open('xb') as out,(r/'allocation.stderr.log').open('xb') as err:
            proc=subprocess.run(['bash',str(r/'gpu.sh')],cwd=r,stdout=out,stderr=err)
        row=dict(family=family,exit_code=proc.returncode,actual_parent_wait=True,epoch=time.time());rows.append(row)
        save(r/'allocation_receipt.json',row)
        assert proc.returncode==0,'GPU holder failed; no automatic replay'
        receipt=json.loads((r/('run_dense' if family=='dense' else 'run_comem')/'process_receipt.json').read_text())
        assert receipt['actual_parent_wait']
    save(ROOT/'status.json',dict(state='all_holders_closed',completed=rows,job_id=os.environ['SLURM_JOB_ID'],epoch=time.time(),scores_require_local_result_audit=True))


if __name__=='__main__':
    if sys.argv[1]=='submit':submit()
    elif sys.argv[1]=='verify':verify(sys.argv[2])
    elif sys.argv[1]=='run':run()
    else:raise ValueError('Unknown command')
