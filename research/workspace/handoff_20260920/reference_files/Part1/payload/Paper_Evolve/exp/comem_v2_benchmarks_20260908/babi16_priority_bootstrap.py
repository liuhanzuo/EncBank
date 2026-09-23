"""Six canonical BABI16 baseline shards on idle GPU1; leave the main queue intact."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/data/liuhanzuo/comem_v2_20260908');B=R/'workspace/exp/comem_v2_benchmarks_20260908'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def alive(pid):
    if not pid:return False
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def verify_scope(plan,main_plan):
    originals={j['id']:j for j in main_plan['jobs']}
    if len(plan['jobs'])!=6 or sum(j['expected_n'] for j in plan['jobs'])!=600:raise RuntimeError('Not the six authorized BABI16 cells')
    actual=set()
    for job in plan['jobs']:
        if job!=originals.get(job['id']):raise RuntimeError('Canonical native job changed')
        if job['benchmark']!='babilong' or job['arm'] not in ('pub','pub_sink','cbos') or len(job['cells'])!=1:raise RuntimeError('Unexpected native scope')
        cell=job['cells'][0]
        if cell['task'] not in ('qa2','qa3') or cell['length']!='16k' or cell['indices']!=list(range(100)) or job['expected_n']!=100:raise RuntimeError('Unexpected native dataset cell')
        actual.add((job['arm'],cell['task']))
    if actual!={(a,t) for a in ('pub','pub_sink','cbos') for t in ('qa2','qa3')}:raise RuntimeError('Incomplete native scope')
def launch_allowed(job,main_plan,main_state,completed):
    entry=main_state['jobs'][job['id']]
    if entry['status']=='running':return False,'main_already_running',0
    if completed(job):return True,'canonical_complete_reuse_before_model_load',0
    if entry['status']!='pending':return False,'main_nonpending_requires_inspection',0
    idx=next(i for i,j in enumerate(main_plan['jobs']) if j['id']==job['id'])
    gap=sum(main_state['jobs'][j['id']]['status'] in ('pending','running') and not completed(j) for j in main_plan['jobs'][:idx])
    return gap>=40,'far_from_main_frontier' if gap>=40 else 'near_frontier_wait_for_main',gap
def main():
    from remote_queue import atomic_json,gpu_idle
    out=R/'outputs/bootstrap_babi16_priority';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'bootstrap.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'status.json';previous=read(status_path)
    if previous.get('status')=='completed':return 0
    if alive(previous.get('queue_pid')):raise RuntimeError('Existing auxiliary queue still alive')
    plan=read(B/'babi16_priority_full_plan.json');main_plan=read(B/'full_plan.json');verify_scope(plan,main_plan)
    state={'protocol':'babi16-canonical-priority-v1','pid':os.getpid(),'status':'waiting','gpu':1,'accuracy_only':True,'plan':str(B/'babi16_priority_full_plan.json'),'canonical_outputs_shared_with_main':True,'jobs':{},'target_jobs':6,'target_predictions':600,'after_completion':'exit; next GPU1 priority reserved for root-coordinated nonQwen smoke if ready'}
    def save():state['updated_at']=time.time();atomic_json(status_path,state)
    from native_priority_receipt import completed
    try:
        # The immediately preceding finite MFQA queue must also be fully finished.
        prior=read(R/'outputs/bootstrap_mfqa_priority/status.json')
        if prior.get('status')!='completed' or prior.get('completed_jobs')!=6 or prior.get('completed_predictions')!=450 or len(prior.get('jobs',{}))!=6:raise RuntimeError('MFQA predecessor incomplete')
        if any(alive(p) for p in [prior.get('pid'),prior.get('queue_pid')]+[j.get('model_pid') for j in prior['jobs'].values()]):raise RuntimeError('MFQA predecessor still alive')
        for item in prior['jobs'].values():
            qs=read(Path(item['queue_state']))
            if alive(qs.get('queue_pid')):raise RuntimeError('MFQA predecessor one-job queue still alive')
        # The immediately preceding 2Wiki finite queue must also be complete/exited.
        prior2=read(R/'outputs/bootstrap_twowiki_priority/status.json')
        if prior2.get('status')!='completed' or prior2.get('completed_jobs')!=6 or prior2.get('completed_predictions')!=600 or len(prior2.get('jobs',{}))!=6:raise RuntimeError('2Wiki predecessor incomplete')
        if any(alive(p) for p in [prior2.get('pid'),prior2.get('queue_pid')]+[j.get('model_pid') for j in prior2['jobs'].values()]):raise RuntimeError('2Wiki predecessor still alive')
        for item in prior2['jobs'].values():
            qs=read(Path(item['queue_state']))
            if alive(qs.get('queue_pid')):raise RuntimeError('2Wiki predecessor one-job queue still alive')
        # Narrative priority is the immediately preceding GPU1 owner.
        prior3=read(R/'outputs/bootstrap_narrative_priority/status.json')
        if prior3.get('status')!='completed' or prior3.get('completed_jobs')!=6 or prior3.get('completed_predictions')!=600 or len(prior3.get('jobs',{}))!=6:raise RuntimeError('Narrative predecessor incomplete')
        if any(alive(p) for p in [prior3.get('pid'),prior3.get('queue_pid')]+[j.get('model_pid') for j in prior3['jobs'].values()]):raise RuntimeError('Narrative predecessor still alive')
        for item in prior3['jobs'].values():
            qs=read(Path(item['queue_state']))
            if alive(qs.get('queue_pid')):raise RuntimeError('Narrative predecessor one-job queue still alive')
        # BABI32 priority is the immediately preceding GPU1 owner.
        prior4=read(R/'outputs/bootstrap_babi32_priority/status.json')
        if prior4.get('status')!='completed' or prior4.get('completed_jobs')!=6 or prior4.get('completed_predictions')!=600 or len(prior4.get('jobs',{}))!=6:raise RuntimeError('BABI32 predecessor incomplete')
        if any(alive(p) for p in [prior4.get('pid'),prior4.get('queue_pid')]+[j.get('model_pid') for j in prior4['jobs'].values()]):raise RuntimeError('BABI32 predecessor still alive')
        for item in prior4['jobs'].values():
            qs=read(Path(item['queue_state']))
            if alive(qs.get('queue_pid')):raise RuntimeError('BABI32 predecessor one-job queue still alive')
        # Old priority and oracle must have naturally completed and released every PID.
        for name,expected in [('visibility_priority',4),('oracle_support',40)]:
            pred=read(R/f'outputs/bootstrap_{name}/status.json');queue=read(R/f'outputs/queue_{name}/full/state.json')
            if pred.get('status')!='completed' or len(queue.get('jobs',{}))!=expected or any(j.get('status')!='completed' or j.get('exit_code')!=0 for j in queue['jobs'].values()):raise RuntimeError(f'{name} predecessor not complete')
            if any(alive(p) for p in [pred.get('pid'),pred.get('queue_pid'),queue.get('queue_pid')]+[j.get('pid') for j in queue['jobs'].values()]):raise RuntimeError(f'{name} predecessor still alive')
        state['predecessors_complete_and_exited']=True;save()
        for number,job in enumerate(plan['jobs']):
            while True:
                main_state=read(R/'outputs/queue_v2/full/state.json')
                if not alive(main_state.get('queue_pid')):raise RuntimeError('Main queue disappeared; inspect before launching')
                allowed,reason,gap=launch_allowed(job,main_plan,main_state,completed)
                idle,reading=gpu_idle(1,512);state.update(status='waiting',active_job=job['id'],reason=reason,main_unfinished_canonical_jobs_before_target=gap,gpu_check=reading);save()
                if allowed and idle:break
                time.sleep(10)
            # Independent one-job state preserves every dependency; all belong to the
            # completed original BABILong V2/j0 stage, and no shared main state is modified.
            originals={j['id']:j for j in main_plan['jobs']};deps={}
            for dep in job.get('depends_on',[]):
                entry=main_state['jobs'][dep]
                if entry.get('status')!='completed' or entry.get('exit_code')!=0 or not completed(originals[dep]):raise RuntimeError('Original dependency lacks complete canonical output')
                deps[dep]={'status':'completed','exit_code':0,'external_dependency':True,'source_state':str(R/'outputs/queue_v2/full/state.json')}
            qdir=R/'outputs/queue_babi16_priority/full'/job['id'];qdir.mkdir(parents=True,exist_ok=True)
            qstate=read(qdir/'state.json')
            if any(alive(j.get('pid')) for j in qstate.get('jobs',{}).values() if not j.get('external_dependency')):raise RuntimeError('Recorded own model still alive')
            if not qstate:atomic_json(qdir/'state.json',{'jobs':deps})
            one={**plan,'jobs':[job],'state_dir':str(qdir),'counts':{'jobs':1,'new_predictions':job['expected_n'],'reusable_predictions':0,'total_predictions':job['expected_n']}}
            one_path=out/f'job_{number:02d}_plan.json';atomic_json(one_path,one)
            state.update(status='running',phase='full',reason=reason,active_job=job['id'],active_queue_state=str(qdir/'state.json'))
            with (out/f'job_{number:02d}_queue.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(B/'remote_queue.py'),'--plan',str(one_path),'--gpus','1','--max-idle-mib','512'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['queue_pid']=child.pid;save();rc=child.wait()
            result=read(qdir/'state.json').get('jobs',{}).get(job['id'],{})
            if rc or result.get('status')!='completed' or result.get('exit_code')!=0 or not completed(job):raise RuntimeError('Auxiliary canonical output incomplete; preserve logs/cache')
            state['jobs'][job['id']]={'status':'complete','output':job['output'],'queue_state':str(qdir/'state.json'),'expected_n':job['expected_n'],'model_pid':result.get('pid'),'exit_code':0}
            state.update(queue_pid=None,completed_jobs=len(state['jobs']),completed_predictions=sum(j['expected_n'] for j in state['jobs'].values()));save()
        state.update(status='completed',active_job=None,finished_at=time.time());save();return 0
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':raise SystemExit(main())
