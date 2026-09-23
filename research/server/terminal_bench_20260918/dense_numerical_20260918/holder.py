"""Actual parent wait plus owned-process device memory telemetry."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
from persistence import dump
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve());arm=sys.argv[1]
assert arm in ['dense','raw_shared','comem'];R=H/('run_'+arm);assert not R.exists();(R/'mailbox').mkdir(parents=True)
def save(n,d):dump(R/n,d)
args=['/srv/encbank/Paper_Evolve/.venv/bin/python','-u',str(H/'diagnose.py'),'--run',str(R)]
with (R/'worker.stdout.log').open('wb') as out,(R/'worker.stderr.log').open('wb') as err:
    p=subprocess.Popen(args,stdout=out,stderr=err,cwd=H)
    save('process_start.json',dict(pid=p.pid,parent_pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],argv=args,started_epoch=time.time()))
    with (R/'owned_nvml.jsonl').open('w') as mon:
        while p.poll() is None:
            r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
            rows=[x for x in r.stdout.splitlines() if x.split(',')[0].strip()==str(p.pid)]
            mon.write(json.dumps(dict(epoch=time.time(),pid=p.pid,rows=rows,query_exit=r.returncode))+'\n');mon.flush();time.sleep(2)
    code=p.wait();receipt=dict(exit_code=code,actual_parent_wait=True,pid=p.pid,ended_epoch=time.time());print(json.dumps(dict(event='actual_worker_wait',**receipt)),flush=True);save('process_receipt.json',receipt)
sys.exit(code)
