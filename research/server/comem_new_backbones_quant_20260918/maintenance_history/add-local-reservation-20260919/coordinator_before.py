"""CPU queue owner: <=4 requests for this agent, independent of other agents."""
import datetime, fcntl, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
PYTHON='/srv/encbank/Paper_Evolve/.venv/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def explicitly_rejected(rec):
    return (rec.get('returncode')==1 and not rec.get('job') and not rec.get('stdout','').strip()
            and rec.get('stderr','').strip()=='sbatch: error: Batch job submission failed: I/O error writing script/environment to file')
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def run(cmd):
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
    if r.returncode:raise RuntimeError(f'{cmd[0]} failed: '+r.stderr[:700])
    return r.stdout
def owned_jobs(plan):
    jobs=[]
    for line in run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R']).splitlines():
        ident,name,state,gres,reason=line.split('|',4)
        if ident in plan['resources']['shared_preexisting_jobs'] or any(name.startswith(p) for p in plan['resources']['counted_name_prefixes']):
            assert 'gpu' in gres.lower(),(ident,name,gres)
            n=int(gres.split(':')[-1]);assert n>0
            jobs.append(dict(job=ident,name=name,state=state,gpus=n,reason=reason))
    return jobs
def completed(receipts):
    ids=[r['job'] for r in receipts.values() if r.get('job')]
    if not ids:return {}
    out={}
    raw=run(['sacct','-j',','.join(ids),'-n','-P','-o','JobIDRaw,State,ExitCode'])
    for line in raw.splitlines():
        ident,state,exitcode,*_=line.split('|')
        if ident in ids:out[ident]=dict(state=state,exitcode=exitcode)
    return out
def eta(task,receipt,slurm):
    if task['kind']!='training':return None
    if slurm.get('state') not in ('RUNNING','COMPLETED'):
        return dict(phase=slurm.get('state','PENDING'),eta_shanghai=None,reason='Awaiting this attempt start; prior ETA is invalid after interruption')
    p=ROOT/'training'/task['model']/'progress.json'
    if not p.exists():return dict(phase=slurm.get('state','PENDING'),eta_shanghai=None,
         duration_after_start_hours=task['prestart_duration_hours'],reason='Queue start time unknown')
    r=json.loads(p.read_text());step=r.get('step',0)
    if r.get('phase')=='complete':return dict(phase='complete',step=step)
    elapsed=r.get('elapsed_s',0)
    when=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc)
    start_step=task.get('resume_step',0)
    if step>start_step and elapsed>0:
        seconds_per_step=elapsed/(step-start_step)
        end=when+datetime.timedelta(seconds=seconds_per_step*(8000-step))
        return dict(phase='training',step=step,resume_step=start_step,seconds_per_step=seconds_per_step,
                    eta_shanghai=end.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    job=receipt.get('job'),method='observed average, excludes queue delay and future checkpoint overhead')
    return dict(phase='starting',step=step)

def main():
    lock=(ROOT/'coordinator.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'effective_plan.json').read_text());assert str(ROOT.resolve())==plan['remote_root']
    (ROOT/'runs').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    # Preflight is CPU-only and runs before admission. Its real exit is retained.
    if not (ROOT/'verified_inputs.json').exists():
        dump(ROOT/'status.json',dict(phase='VERIFYING_INPUTS',pid=os.getpid(),at=now()))
        with (ROOT/'preflight.stdout.log').open('ab') as out,(ROOT/'preflight.stderr.log').open('ab') as err:
            code=subprocess.Popen([PYTHON,'-u','-B',str(ROOT/'preflight_remote.py')],cwd=ROOT,stdout=out,stderr=err).wait()
        dump(ROOT/'preflight_parent_exit.json',dict(returncode=code,actual_wait=True,at=now()))
        assert code==0,'Preflight failed; inspect logs. No GPU submission.'
    while not (ROOT/'STOP_COORDINATOR').exists():
        receipts={}
        for task in plan['tasks']:
            p=ROOT/'runs'/task['id']/'submission.json'
            if p.exists():receipts[task['id']]=json.loads(p.read_text())
        accounting=completed(receipts);states={};done=set();failed=set()
        for task in plan['tasks']:
            rec=receipts.get(task['id']);state='WAITING_DEPENDENCY' if task['depends'] else 'READY'
            details={}
            if rec:
                if not rec.get('job'):
                    state='SUBMISSION_REJECTED' if explicitly_rejected(rec) else 'SUBMISSION_UNCERTAIN'
                    if state=='SUBMISSION_UNCERTAIN':failed.add(task['id'])
                else:
                    details=accounting.get(rec['job'],{});state=details.get('state','AWAITING_ACCOUNTING')
                    exitfile=ROOT/'runs'/task['id']/'parent_exit.json'
                    if state=='COMPLETED':
                        valid=(details['exitcode']=='0:0' and exitfile.exists() and (ROOT/task['complete']).exists())
                        if valid:
                            wait=json.loads(exitfile.read_text());valid=wait['actual_wait'] and wait['returncode']==0 and wait['completion_exists']
                        if valid:done.add(task['id'])
                        else:state='INVALID_COMPLETION';failed.add(task['id'])
                    elif any(state.startswith(s) for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL')):failed.add(task['id'])
            states[task['id']]=dict(state=state,job=rec.get('job') if rec else None,accounting=details,
                                    eta=eta(task,rec or {},details))
        # Shared serialized admission lock and fresh queue inspection before EACH sbatch.
        with (ROOT.parent/'qcomem_gpu_admission.lock').open('a+b') as admission:
            fcntl.flock(admission,fcntl.LOCK_EX)
            for task in plan['tasks']:
                if (ROOT/'PAUSE_SUBMISSIONS.json').exists():break
                if task['id'] in receipts or not set(task['depends']).issubset(done):continue
                jobs=owned_jobs(plan);used=sum(j['gpus'] for j in jobs)
                if used>=4:break
                out=ROOT/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
                name='qcm-q18-'+task['id']
                assert not any(j['name']==name for j in jobs),'Duplicate queue owner detected'
                cmd=['sbatch','--parsable','--job-name='+name,'--partition=gpu','--gres=gpu:nvidia_l20d:1',
                     '--cpus-per-task=4','--mem=128G','--time=72:00:00',
                     '--output='+str(ROOT/'logs'/(task['id']+'-%j.out')),
                     '--error='+str(ROOT/'logs'/(task['id']+'-%j.err')),
                     '--chdir='+str(ROOT),str(ROOT/task.get('launcher','environment.sh')),task['id']]
                receipt=dict(at=now(),command=cmd,pre_submit_owned_queue=jobs,pre_submit_gpu_requests=used)
                dump(out/'submission.json',receipt)
                result=subprocess.run(cmd,text=True,capture_output=True,timeout=45)
                receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                if result.returncode==0:
                    job=result.stdout.strip().split(';')[0];assert job.isdigit();receipt['job']=job
                dump(out/'submission.json',receipt)
                assert result.returncode==0,'Submission failed. No automatic resubmit.'
                receipts[task['id']]=receipt;states[task['id']]=dict(state='SUBMITTED',job=receipt['job'])
                print(json.dumps(dict(submitted=task['id'],job=receipt['job'],at=now())),flush=True)
        jobs=owned_jobs(plan)
        for task in plan['tasks']:
            if set(task['depends'])&failed:states[task['id']]['state']='BLOCKED_FAILED_DEPENDENCY'
            elif task['id'] not in receipts and set(task['depends']).issubset(done):states[task['id']]['state']='WAITING_GPU_SLOT'
        all_terminal=all(t['id'] in done or t['id'] in failed or states[t['id']]['state']=='BLOCKED_FAILED_DEPENDENCY' for t in plan['tasks'])
        phase='GENERATION_COMPLETE' if len(done)==len(plan['tasks']) else ('NEEDS_ATTENTION' if all_terminal else 'ACTIVE' if any(t['id'] in receipts for t in plan['tasks']) else 'WAITING_SHARED_GPU_SLOTS')
        paused=(ROOT/'PAUSE_SUBMISSIONS.json').exists()
        if paused and len(done)<len(plan['tasks']):phase='SUBMISSIONS_PAUSED'
        dump(ROOT/'status.json',dict(phase=phase,at=now(),pid=os.getpid(),owned_queue=jobs,
             total_owned_gpu_requests=sum(j['gpus'] for j in jobs),maximum=4,completed=len(done),failed=len(failed),
             task_count=len(plan['tasks']),tasks=states,submissions_paused=paused,
             interpretation='Queued/prepared is not running or measured; Judge completion is separate.'))
        if all_terminal:break
        time.sleep(60)

if __name__=='__main__':
    for retry in range(4):
        try:
            main();break
        except BlockingIOError:raise SystemExit('Existing queue owner holds lock.')
        except OSError as exc:
            record=dict(at=now(),error=traceback.format_exc(),pid=os.getpid(),retry=retry)
            print(json.dumps(record),file=sys.stderr,flush=True)
            try:dump(ROOT/'coordinator_failure.json',record)
            except OSError:pass
            if exc.errno not in (5,70,110,121) or retry==3:raise
            # Retry only the CPU controller. Existing submission receipts and
            # scheduler identities remain authoritative; never retry sbatch blindly.
            time.sleep(60)
        except BaseException:
            dump(ROOT/'coordinator_failure.json',dict(at=now(),error=traceback.format_exc(),pid=os.getpid()))
            raise
