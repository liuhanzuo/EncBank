"""One Slurm allocation, isolated sequential children, true parent waits."""
import hashlib,json,os,socket,subprocess,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
def save(name,value):
    with (H/name).open('x') as f:json.dump(value,f,indent=2)
def main():
    manifest=json.loads((H/'source_manifest.json').read_text())
    for n,d in manifest.items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==d,n
    protocol=json.loads((H/'protocol.json').read_text())
    save('started.json',dict(epoch=time.time(),job_id=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),pid=os.getpid(),benchmark_attempts=0))
    waits=[]
    for case in protocol['runs']:
        started=time.time();rid=case['id']
        with (H/f'{rid}.stdout.log').open('x') as out,(H/f'{rid}.stderr.log').open('x') as err:
            child=subprocess.Popen([PY,'-u',str(H/'hot_speed_probe.py'),rid],stdout=out,stderr=err,cwd=H)
            (H/'active_case.json').write_text(json.dumps(dict(case=case,pid=child.pid,started_epoch=started))+'\n')
            rc=child.wait()
        receipt=dict(run_id=rid,pid=child.pid,started_epoch=started,ended_epoch=time.time(),exit_code=rc,actual_parent_wait=True)
        save(f'{rid}.parent.json',receipt);waits.append(receipt)
        print(json.dumps(receipt),flush=True)
        assert rc==0,'Child failed; preserve evidence and stop, no retry: '+rid
    rc=subprocess.call([PY,'-u',str(H/'compare_results.py')],cwd=H)
    assert rc==0,'CPU comparison failed'
    save('complete.json',dict(epoch=time.time(),job_id=os.environ['SLURM_JOB_ID'],case_parent_waits=waits,benchmark_attempts=0))
if __name__=='__main__':
    try:main()
    except BaseException:
        save('failure.json',dict(epoch=time.time(),error=traceback.format_exc()));raise
