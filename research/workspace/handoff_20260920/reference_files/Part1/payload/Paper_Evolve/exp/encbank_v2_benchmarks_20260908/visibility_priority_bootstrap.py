"""Run four already-planned natural controls on GPU1 after oracle exits naturally."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('/data/liuhanzuo/encbank_v2_20260908')
B=R/'workspace/exp/encbank_v2_benchmarks_20260908'

def read(p):return json.loads(p.read_text()) if p.exists() else {}
def alive(pid):
    if not pid:return False
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def ready(status,summary,queue,is_alive=alive):
    if status.get('status')=='failed':raise RuntimeError('Oracle predecessor failed')
    if status.get('status')!='completed':return False
    if not (summary.get('planned_jobs')==summary.get('completed_jobs')==40 and summary.get('observed_predictions')==3104 and summary.get('all_complete') is True):
        raise RuntimeError('Oracle completion lacks 40 jobs / 3104 raw outputs')
    jobs=queue.get('jobs',{})
    if len(jobs)!=40 or any(j.get('status')!='completed' or j.get('exit_code')!=0 for j in jobs.values()):
        raise RuntimeError('Oracle queue completion is inconsistent')
    pids=[status.get('pid'),status.get('queue_pid'),queue.get('queue_pid')]+[j.get('pid') for j in jobs.values()]
    return not any(is_alive(pid) for pid in pids if pid)

def main():
    from remote_queue import gpu_idle,atomic_json
    out=R/'outputs/bootstrap_visibility_priority';out.mkdir(parents=True,exist_ok=True)
    guard=(out/'bootstrap.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'status.json';previous=read(status_path)
    if previous.get('status')=='completed':return 0
    if previous.get('queue_pid') and alive(previous['queue_pid']):raise RuntimeError('Existing priority queue is alive')
    state={'pid':os.getpid(),'status':'waiting','phase':'after_oracle','gpu':1,'accuracy_only':True,
           'plan':str(B/'visibility_priority_full_plan.json'),'canonical_outputs_shared_with_main':True}
    def save():state['updated_at']=time.time();atomic_json(status_path,state)
    try:
        while True:
            predecessor=read(R/'outputs/bootstrap_oracle_support/status.json')
            summary=read(R/'outputs/bootstrap_oracle_support/full_summary.json')
            queue=read(R/'outputs/queue_oracle_support/full/state.json')
            if ready(predecessor,summary,queue):
                idle,reading=gpu_idle(1,512);state['gpu_check']=reading
                if idle:break
                state['reason']='waiting_for_empty_gpu1'
            else:state['reason']='waiting_for_oracle_completion_and_all_process_exits'
            save();time.sleep(10)
        plan=read(B/'visibility_priority_full_plan.json');main_plan=read(B/'full_plan.json')
        main_jobs={j['id']:j for j in main_plan['jobs']};main_state=read(R/'outputs/queue_v2/full/state.json')
        if len(plan['jobs'])!=4 or sum(j['expected_n'] for j in plan['jobs'])!=400:raise RuntimeError('Wrong control scope')
        for job in plan['jobs']:
            if job!=main_jobs.get(job['id']):raise RuntimeError('Control differs from original main-plan job')
            if main_state['jobs'][job['id']]['status']=='running':raise RuntimeError('Main queue already runs this control')
            if job['arm']!='fix_none' or job['benchmark']!='longbench':raise RuntimeError('Unexpected control arm/task')
        state.update(status='running',phase='full',reason='oracle_complete_gpu1_empty')
        with (out/'queue.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(B/'remote_queue.py'),'--plan',str(B/'visibility_priority_full_plan.json'),'--gpus','1','--max-idle-mib','512'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['queue_pid']=child.pid;save();rc=child.wait()
        summary_path=out/'full_summary.json'
        sr=subprocess.call([sys.executable,str(B/'summarize_runs.py'),'--plan',str(B/'visibility_priority_full_plan.json'),'--out',str(summary_path)])
        result=read(summary_path)
        if rc or sr or not (result.get('completed_jobs')==result.get('planned_jobs')==4 and result.get('observed_predictions')==400 and result.get('all_complete') is True):
            raise RuntimeError('Priority control queue/results incomplete; retain all evidence')
        state.update(status='completed',phase='full',queue_pid=None,completed_jobs=4,observed_predictions=400,finished_at=time.time());save();return 0
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise

if __name__=='__main__':raise SystemExit(main())
