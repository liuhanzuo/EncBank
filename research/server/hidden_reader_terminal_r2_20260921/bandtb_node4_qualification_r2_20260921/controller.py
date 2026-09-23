"""Server CPU controller: actual child waits, shared host RAM admission, unique trials."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,PLAN as P,save,verify_sources,sha
import host_admission
QUALIFY='--qualify' in sys.argv
def trial(task,arm):
    name=task+'--'+arm;directory=ROOT/('qualification' if QUALIFY else 'results')/name
    assert not directory.exists(),'Refuse duplicate trial'
    memory=next(x['memory_mb'] for x in P['resource_inventory'] if x['task']==task)
    while True:
        admitted,proof=host_admission.acquire(ROOT.name,name,memory)
        if admitted:break
        save(ROOT/'jobs'/('admission-'+name+'.json'),proof);time.sleep(5)
    out=(ROOT/'logs'/(name+'.stdout.log')).open('wb');err=(ROOT/'logs'/(name+'.stderr.log')).open('wb')
    try:
        start=time.time();env=os.environ.copy();env.update(PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor'),PYTHONUNBUFFERED='1',LITELLM_LOCAL_MODEL_COST_MAP='True',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
        for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:env.pop(key,None)
        process=subprocess.Popen([P['harbor_python'],str(ROOT/'run_trial.py'),task,arm],cwd=ROOT,env=env,stdout=out,stderr=err)
        save(ROOT/'jobs'/('launch-'+name+'.json'),dict(pid=process.pid,identity=host_admission.process_identity(process.pid),epoch=start,admission=proof))
        code=process.wait();save(ROOT/'jobs'/('wait-'+name+'.json'),dict(exit_code=code,actual_parent_wait=True,epoch=time.time(),elapsed_seconds=time.time()-start))
        return code
    finally:
        out.close();err.close();host_admission.release(ROOT.name,name)
def pair(task,index):
    if QUALIFY:
        code=trial(task,'qualify');return dict(task=task,qualification_exit=code)
    pair_root=ROOT/'pairs'/task
    while not (pair_root/'ready.json').exists():
        if (pair_root/'worker_failure.json').exists() or (pair_root/'parent_exit.json').exists():
            return dict(task=task,error='Worker failed before ready')
        time.sleep(1)
    results=[]
    for arm in (P['arms'] if index%2==0 else list(reversed(P['arms']))):
        if (pair_root/'worker_failure.json').exists():break
        code=trial(task,arm);results.append(dict(arm=arm,exit_code=code))
    save(ROOT/'mailbox'/task/'stop.json',dict(live_trials_closed=True,epoch=time.time()))
    while not (pair_root/'parent_exit.json').exists():time.sleep(1)
    result=dict(task=task,trials=results,worker=json.loads((pair_root/'parent_exit.json').read_text()))
    save(ROOT/'jobs'/('pair-'+task+'.json'),result);return result
def main():
    verify_sources();assert os.environ.get('SLURM_JOB_ID')
    name='qualification' if QUALIFY else 'controller'
    lock=(ROOT/'jobs'/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'jobs'/(name+'_start.json')).exists()
    save(ROOT/'jobs'/(name+'_start.json'),dict(pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],epoch=time.time()))
    if QUALIFY:
        for path,expected in json.loads((ROOT/'task_manifest.json').read_text()).items():assert sha(path)==expected,path
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(pair,t,i) for i,t in enumerate(P['tasks'])]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());save(ROOT/'jobs'/(name+'_progress.json'),results)
    passed=all(r.get('qualification_exit',0)==0 and not r.get('error') and r.get('worker',{}).get('exit_code',0)==0 and all(t['exit_code']==0 for t in r.get('trials',[])) for r in results)
    save(ROOT/'jobs'/(name+'_complete.json'),dict(passed=passed,rows=results,epoch=time.time()))
    if not passed:raise RuntimeError('At least one actual child failed; preserve result and diagnose')
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'jobs'/('qualification_failure.json' if QUALIFY else 'controller_failure.json'),dict(error=traceback.format_exc()));raise
