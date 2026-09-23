"""Pause admission after the second verified outage while preserving active GPU jobs."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import datetime,fcntl,json,os,subprocess
from pathlib import Path
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
failed={'large-final-m1-s0':'106607','large-final-m1-s1':'106608','large-final-m1-s2':'106609'}
out=dict(at=datetime.datetime.utcnow().isoformat()+'Z',failed=failed,errors={},pause_saved=False)
for task,job in failed.items():
    d=r/'runs'/task
    assert json.loads((d/'submission.json').read_text())['job']==job
    err=(d/'child.stderr.log').read_text();assert '[Errno 121]' in err
    p=d/'parent_exit.json'
    out['errors'][task]=dict(stderr_tail=err[-4500:],parent_exit=json.loads(p.read_text()) if p.exists() else None)
err=(r/'judge_gpt6_astra/watch.stderr.log').read_text();assert '[Errno 121]' in err
out['judge_error']=err
queue=subprocess.run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%R'],universal_newlines=True,stdout=subprocess.PIPE,check=True,timeout=20).stdout
out['own_queue']=[x for x in queue.splitlines() if 'qcm-q18-' in x]
out['accounting']=subprocess.run(['sacct','-j',','.join(failed.values()),'-n','-P','-o','JobIDRaw,State,ExitCode,End'],universal_newlines=True,stdout=subprocess.PIPE,check=True,timeout=20).stdout
for job in failed.values():assert any(x.startswith(job+'|FAILED|1:0|') for x in out['accounting'].splitlines())
admission=(r.parent/'qcomem_gpu_admission.lock').open('a+b');fcntl.flock(admission,fcntl.LOCK_EX)
p=r/'PAUSE_SUBMISSIONS.json';t=p.with_suffix('.recurrence.tmp')
try:
    if p.exists():out['existing_pause']=json.loads(p.read_text())
    else:
        record=dict(at=out['at'],reason='Recurring shared-storage Errno121 after 03:36 recovery',
                    failed_jobs=failed,active_jobs_preserved=True,
                    resume_condition='Require sustained successful storage checks and inspect current jobs; do not rerun prior one-shot recovery.')
        with t.open('w') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
        t.replace(p);assert json.loads(p.read_text())==record
    out['pause_saved']=True
except OSError as e:out['pause_error']=dict(errno=e.errno,error=str(e))
admission.close()
print(json.dumps(out))
'''
def main():
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
        'timeout -k 5s 45s /usr/bin/python3 -c '+shlex.quote(CODE)],capture_output=True,text=True,timeout=55)
    assert p.returncode==0,p.stderr
    data=json.loads(p.stdout);out=ROOT/'delivery/storage_failure_20260919_recurrence';out.mkdir(exist_ok=True)
    (out/'inspection_and_pause.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k not in ['errors','judge_error']}))
if __name__=='__main__':main()
