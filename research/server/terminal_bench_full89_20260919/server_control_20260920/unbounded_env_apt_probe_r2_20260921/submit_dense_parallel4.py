"""Submit the unique four-way Dense continuation after qualification."""
import fcntl
import hashlib
import json
import pathlib
import subprocess
import time

H=pathlib.Path(__file__).resolve().parent
B=H.parents[1]


def main():
    q=json.loads((H/'container_qualification.json').read_text());assert q['status']=='PASS'
    assert q['plan_sha256']==hashlib.sha256((H/'plan.json').read_bytes()).hexdigest()
    assert not (H/'submission.json').exists()
    with (B.parent/'terminal_bench_20260918/admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (H/'submission.json').exists()
        from server_preflight import check_source
        source_sha=check_source()
        queue=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
        own=[r for r in queue.splitlines() if ('qencbank-tb-' in r or 'qencbank-agentmem-' in r) and 'gpu' in r.split('|')[-1].lower()]
        assert len(own)<4,own
        assert all(r.split('|')[0] in {'112400','112403','114684'} for r in own),own
        predecessor=subprocess.check_output(['sacct','-X','-j','113514','--format=State','-n','-P'],text=True).strip()
        assert predecessor.startswith('CANCELLED'),predecessor
        record=dict(epoch=time.time(),existing_gpu_jobs=own,source_sha256=source_sha,status='intent',task_concurrency=4,task_timeout=None)
        (H/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        result=subprocess.run(['sbatch','--parsable',str(H/'server.slurm')],capture_output=True,text=True)
        record.update(status='submitted' if result.returncode==0 else 'failed',exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode==0:record['job_id']=result.stdout.strip().split(';')[0]
        (H/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record));assert result.returncode==0,result.stderr


if __name__=='__main__':main()
