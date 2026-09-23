"""Low-cost server coordinator; all task work remains in owned Slurm allocations."""
import fcntl,json,subprocess,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.with_name('hidden_reader_terminal_r4_20260921')
P=json.loads((R/'plan.json').read_text());jobs=json.loads((R/'jobs/submissions_run.json').read_text())
lock=(A/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (A/'dispatch_started.json').exists()
(A/'dispatch_started.json').write_text(json.dumps(dict(epoch=time.time(),pid=__import__('os').getpid())))
waiting={j['task']:j for j in jobs if j['kind']=='worker'};live={};done=[]
while waiting or live:
    for task,j in list(waiting.items()):
        out=R/'pairs'/task
        if (out/'parent_exit.json').exists():
            done.append(dict(task=task,job=j['job'],error='worker already closed'));del waiting[task];continue
        if not (out/'ready.json').exists():continue
        state=subprocess.check_output(['squeue','-j',j['job'],'-h','-o','%T'],text=True).strip()
        if state!='RUNNING':continue
        cmd=['srun','--jobid='+j['job'],'--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=4','--cpu-bind=none',
            P['engine_python'],'-B',str(A/'step_parent.py'),str(A/'controls'/task),task,j['job']]
        stdout=(out/'slurm_step.stdout.log').open('wb');stderr=(out/'slurm_step.stderr.log').open('wb')
        child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr);live[task]=(child,stdout,stderr,j,time.time());del waiting[task]
        (out/'slurm_step_launch.json').write_text(json.dumps(dict(pid=child.pid,argv=cmd,epoch=time.time()),indent=2))
    for task,(child,stdout,stderr,j,start) in list(live.items()):
        code=child.poll()
        if code is None:continue
        code=child.wait();stdout.close();stderr.close()
        receipt=dict(task=task,job=j['job'],exit_code=code,actual_parent_wait=True,start_epoch=start,end_epoch=time.time())
        (R/'pairs'/task/'slurm_step_parent_exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
        done.append(receipt);del live[task]
    (A/'dispatch_status.json').write_text(json.dumps(dict(waiting=list(waiting),running=list(live),done=done),indent=2)+'\n')
    if waiting or live:time.sleep(5)
(A/'dispatch_complete.json').write_text(json.dumps(dict(done=done,all_steps_zero=all(r.get('exit_code')==0 for r in done)),indent=2)+'\n')
