"""Reserve one of the user's four GPU slots without interrupting running work."""
import datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
receipt=root/'submission.json'
assert not receipt.exists() and not (root/'launch.json').exists(),'Inspect existing submission before retry'
assert json.loads((root/'cpu_checks.json').read_text())['passed']
q=subprocess.run(['squeue','-u','liuhanzuo','-n','midcache-cacheblend-lora','-h','-o','%i'],text=True,capture_output=True,check=True)
assert not q.stdout.strip(),q.stdout
for array in ('45014','59046'):
    q=subprocess.run(['squeue','-j',array,'-h','-o','%T'],text=True,capture_output=True)
    if q.stdout.strip():subprocess.run(['scontrol','update','JobId='+array,'ArrayTaskThrottle=3'],check=True)
q=subprocess.run(['squeue','-r','-j','45014,59046','-t','RUNNING,COMPLETING','-h','-o','%i'],text=True,capture_output=True,check=True)
running=q.stdout.split();assert len(running)<=4,running
cmd=['sbatch','--parsable']
if len(running)==4:
    # Any current task finishing frees the reserved fourth slot. All remaining
    # evaluation tasks stay capped at three while this one-GPU job is active.
    cmd.append('--dependency='+'?'.join('afterany:'+job for job in running))
cmd.append('pipeline.slurm')
record=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,
    preexisting_active_tasks=running,total_gpu_cap=4,evaluation_throttle_during_cacheblend=3,
    cacheblend_gpu_count=1,automatic_restore_after_terminal_job=True)
receipt.write_text(json.dumps(record,indent=2)+'\n')
r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
receipt.write_text(json.dumps(record,indent=2)+'\n')
assert r.returncode==0,r.stderr
job=r.stdout.strip().split(';')[0];assert job.isdigit()
launch=dict(pipeline_job=job,maximum_total_gpus=4,status='SUBMITTED',training_steps=4000,evaluation_samples=500,evaluation_records=2000)
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
cmd=['sbatch','--parsable','--dependency=afterany:'+job,'release.slurm']
r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
(root/'release_submission.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,r.stderr
launch['release_job']=r.stdout.strip().split(';')[0]
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
