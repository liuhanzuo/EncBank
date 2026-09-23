"""One GPU model and an independently admitted set of real Harbor task worlds."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,PLAN as P,save,verify_sources
verify_sources();OUT=ROOT/'worker';OUT.mkdir(exist_ok=True)
allowed=sorted(os.sched_getaffinity(0));cores={}
for c in allowed:
    top=Path('/sys/devices/system/cpu')/('cpu'+str(c))/'topology'
    key=((top/'physical_package_id').read_text().strip(),(top/'core_id').read_text().strip())
    cores.setdefault(key,[]).append(c)
groups=list(cores.values());model_cpus=set(sum(groups[:4],[]));task_cpus=set(allowed)-model_cpus
assert len(task_cpus)>=P['task_concurrency'],(len(task_cpus),P['task_concurrency'])
save(OUT/'cpu_partition.json',dict(allocated=allowed,model=sorted(model_cpus),tasks=sorted(task_cpus)))
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
    XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(os.getuid())+'/bus',
    LITELLM_LOCAL_MODEL_COST_MAP='True',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:env.pop(k,None)
def start(name,cmd,environ,cpus):
    out=(OUT/(name+'.stdout.log')).open('wb');err=(OUT/(name+'.stderr.log')).open('wb')
    p=subprocess.Popen(cmd,cwd=ROOT,env=environ,stdout=out,stderr=err,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
    save(OUT/(name+'_launch.json'),dict(pid=p.pid,argv=cmd,job=os.environ['SLURM_JOB_ID'],epoch=time.time()))
    return p,out,err,time.time()
def wait(name,owned):
    p,out,err,begin=owned;code=p.wait();out.close();err.close()
    receipt=dict(exit_code=code,actual_parent_wait=True,pid=p.pid,start_epoch=begin,end_epoch=time.time())
    save(OUT/(name+'_parent_exit.json'),receipt)
    if name=='model':
        for task in P['tasks']:save(ROOT/'pairs'/task/'parent_exit.json',receipt)
    return code
model=None;code=1
try:
    bus=subprocess.run(['loginctl','enable-linger',str(os.getuid())],capture_output=True,text=True)
    deadline=time.monotonic()+30
    while not Path(env['XDG_RUNTIME_DIR']+'/bus').exists() and time.monotonic()<deadline:time.sleep(.2)
    save(OUT/'bus.json',dict(code=bus.returncode,stdout=bus.stdout,stderr=bus.stderr))
    assert bus.returncode==0 and Path(env['XDG_RUNTIME_DIR']+'/bus').exists()
    qual=start('environment_qualification',[P['harbor_python'],'-B',str(ROOT/'parallel_trials.py'),'qualify'],env,task_cpus)
    assert wait('environment_qualification',qual)==0
    receipt=json.loads((ROOT/'qualification'/(P['tasks'][0]+'--qualify')/'execution_receipt.json').read_text())
    assert receipt['closure']['cgroup_empty'] and receipt['model_calls']==0 and not receipt['exception']
    wen=env.copy();wen.update(PYTHONPATH=str(ROOT)+':'+P['peft_dependency_path'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CPATH='/srv/encbank/.cache/python/include/python3.12',
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'compile_cache'),TOKENIZERS_PARALLELISM='false')
    model=start('model',[P['engine_python'],'-B',str(ROOT/'worker.py')],wen,model_cpus)
    while not (OUT/'ready.json').exists():
        assert model[0].poll() is None,'Model failed before ready'
        time.sleep(.2)
    trials=start('trials',[P['harbor_python'],'-B',str(ROOT/'parallel_trials.py'),'run'],env,task_cpus)
    code=wait('trials',trials)
except BaseException:save(OUT/'integrated_failure.json',dict(error=traceback.format_exc(),epoch=time.time()))
finally:
    if model is not None:
        save(OUT/'stop.json',dict(epoch=time.time(),controller_closed=True));mcode=wait('model',model)
        if mcode:code=mcode
    save(ROOT/'jobs'/'integrated_complete.json',dict(exit_code=code,epoch=time.time(),job=os.environ['SLURM_JOB_ID']))
sys.exit(code)
