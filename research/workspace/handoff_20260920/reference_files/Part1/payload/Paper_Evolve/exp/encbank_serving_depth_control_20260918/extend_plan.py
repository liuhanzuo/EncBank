"""One-shot CPU owner handover, preserving all accepted GPU jobs and original plan."""
import datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path

ROOT=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
NEW=Path('/srv/encbank/encbank_serving_depth_control_20260918')
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'

def dump(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)

def main():
    history=ROOT/'maintenance_history/add-serving-depth-20260918'
    assert not history.exists(),'One-shot extension already attempted; inspect history before taking any action.'
    assert json.loads((NEW/'data/complete.json').read_text())['quality_rows']==800
    assert not (ROOT/'PAUSE_SUBMISSIONS.json').exists()
    plan=json.loads((ROOT/'effective_plan.json').read_text())
    assert not any(t['id'].startswith('ctrl-') for t in plan['tasks'])
    new=[]
    def add(name,cmd,complete,depends,kind):
        task=dict(id='ctrl-'+name,kind=kind,command=[str(NEW/cmd[0]),*cmd[1:]],
            complete=str(NEW/complete),depends=depends,launcher='control/recovery.sh',io_wrapper=False)
        new.append(task);return task['id']
    smoke=add('smoke',['smoke.py'],'smoke/complete.json',[],'controlled_smoke')
    # A first serving process becomes ready immediately after smoke; interleave depth and old tasks.
    serving=[]
    for rep in range(3):
        for length in [32768,131072]:
            serving.append(add('serve-r%d-n%d'%(rep,length),['serving.py','--rep',str(rep),'--length',str(length)],
                'serving/rep%d/%d/complete.json'%(rep,length),[smoke],'controlled_serving'))
    training=[];follow=[]
    for seed in [42,43,44]:
        for j in [6,9,12,15,18,24]:
            name='j%d-s%d'%(j,seed);folder='j%d_s%d'%(j,seed)
            tr=add('train-'+name,['train.py','--j',str(j),'--seed',str(seed)],
                'training/'+folder+'/complete.json',[smoke],'controlled_training')
            training.append(new[-1])
            add('eval-'+name,['evaluate.py','--j',str(j),'--seed',str(seed)],
                'quality/'+folder+'/complete.json',[tr],'controlled_quality');follow.append(new[-1])
    # All depth timing uses a SINGLE GPU allocation; independent model processes in randomized order.
    add('latency-all',['latency_all.py'],'latency_depth/complete.json',
        [t['id'] for t in training],'controlled_latency');follow.append(new[-1])
    lookup={t['id']:t for t in new}
    # Prioritize correctness, then mix old work with serving and paired depth runs.
    old=plan['tasks'];priority=[lookup[smoke],*follow];new_ready=[]
    for i,t in enumerate(training):
        if i<len(serving):new_ready.append(lookup[serving[i]])
        new_ready.append(t)
    old_active=[t for t in old if (ROOT/'runs'/t['id']/'submission.json').exists()]
    old_wait=[t for t in old if not (ROOT/'runs'/t['id']/'submission.json').exists()]
    for i in range(max(len(old_wait),len(new_ready))):
        if i<len(new_ready):priority.append(new_ready[i])
        if i<len(old_wait):priority.append(old_wait[i])
    plan['tasks']=old_active+priority
    assert len(plan['tasks'])==len(old)+44 and len({t['id'] for t in plan['tasks']})==len(plan['tasks'])
    # Freeze admission under its lock before changing owner; running GPU/Judge processes remain untouched.
    admission=(ROOT.parent/'qencbank_gpu_admission.lock').open('a+b');fcntl.flock(admission,fcntl.LOCK_EX)
    launch=json.loads((ROOT/'coordinator_launch.json').read_text());pid=launch['pid']
    proc=Path('/proc')/str(pid)/'cmdline'
    assert proc.exists() and b'control/coordinator.py' in proc.read_bytes()
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],universal_newlines=True,timeout=30)
    ours=[l for l in queue.splitlines() if '|qcm-q18-' in l];assert len(ours)<=4
    history.mkdir(parents=True)
    for f in ['effective_plan.json','coordinator_launch.json','status.json']:(history/f).write_bytes((ROOT/f).read_bytes())
    dump(history/'handover.json',dict(previous_owner=launch,queue=queue,new_tasks=len(new),running_gpu_jobs_preserved=True))
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or b'control/coordinator.py' not in proc.read_bytes():break
        time.sleep(.1)
    assert not proc.exists() or b'control/coordinator.py' not in proc.read_bytes()
    lock=(ROOT/'coordinator.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(ROOT/'effective_plan.json',plan);dump(NEW/'submitted_plan.json',dict(tasks=new,total_new_tasks=len(new)))
    lock.close();admission.close()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (ROOT/'coordinator.stdout.log').open('ab') as out,(ROOT/'coordinator.stderr.log').open('ab') as err:
        child=subprocess.Popen([PY,'-B','-u',str(ROOT/'control/coordinator.py')],cwd=str(ROOT),env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    fresh=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_root=str(ROOT),
        maximum_gpu_requests=4,script=str(ROOT/'control/coordinator.py'),handover=str(history),task_count=len(plan['tasks']))
    dump(ROOT/'coordinator_launch.json',fresh);dump(history/'new_launch.json',fresh)
    print(json.dumps(fresh))

if __name__=='__main__':main()
