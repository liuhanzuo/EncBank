"""One serialized three-cell admission; existing AppWorld owns at most one GPU."""
from pathlib import Path
import fcntl,hashlib,json,subprocess,time
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
with (H/'admission.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (H/'submissions.json').exists(),'Already submitted; reconcile, never repeat'
    for name,sha in json.loads((H/'manifest.json').read_text()).items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==sha,name
    results=[]
    for arm in ['dense','raw_shared','comem']:
        queue=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
        own=[l for l in queue.splitlines() if any(p in l for p in ['qcomem-agentmem-','qcomem-tb-'])]
        assert all('gpu:nvidia_l20d:1' in l for l in own),own
        assert len(own)<4 and sum('qcomem-agentmem-' in l for l in own)<=1,own
        assert not any('qcomem-tb-'+arm+'-' in l for l in own)
        argv=['sbatch','--parsable',str(H/(arm+'.slurm'))]
        record=dict(arm=arm,at_epoch=time.time(),prior_own_queue=own,argv=argv,status='intent');results.append(record);save(H/'submissions.json',results)
        p=subprocess.run(argv,capture_output=True,text=True,timeout=30);record.update(exit_code=p.returncode,actual_parent_wait=True,stdout=p.stdout,stderr=p.stderr);save(H/'submissions.json',results)
        assert p.returncode==0,p.stderr
        jid=p.stdout.strip().split(';')[0];assert jid.isdigit();record.update(job_id=jid,status='submitted');save(H/'submissions.json',results)
    print(json.dumps(results))
