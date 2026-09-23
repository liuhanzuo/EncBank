"""One Slurm allocation owns qualification, model, both trials and every real wait."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,PLAN as P,save,verify_sources
TASK=P['tasks'][0];OUT=ROOT/'pairs'/TASK;BOX=ROOT/'mailbox'/TASK
verify_sources();assert os.environ.get('SLURM_JOB_ID')
allowed=sorted(os.sched_getaffinity(0));cores={}
for c in allowed:
    top=Path('/sys/devices/system/cpu')/('cpu'+str(c))/'topology'
    key=((top/'physical_package_id').read_text().strip(),(top/'core_id').read_text().strip())
    cores.setdefault(key,[]).append(c)
assert len(cores)>=2,(allowed,cores)
groups=list(cores.values());task_cpus=set(groups[-1]);model_cpus=set(allowed)-task_cpus
save(OUT/'cpu_partition.json',dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,allocated_cpus=allowed,
    model_cpus=sorted(model_cpus),controller_and_task_cpus=sorted(task_cpus),disjoint_physical_cores=True))
env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor'),
    XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(os.getuid())+'/bus',
    LITELLM_LOCAL_MODEL_COST_MAP='True',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:env.pop(k,None)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
def start(name,cmd,environ,cpus):
    out=(OUT/(name+'.stdout.log')).open('wb');err=(OUT/(name+'.stderr.log')).open('wb')
    child=subprocess.Popen(cmd,cwd=ROOT,env=environ,stdout=out,stderr=err,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
    save(OUT/(name+'_launch.json'),dict(pid=child.pid,argv=cmd,job=os.environ['SLURM_JOB_ID'],epoch=time.time()))
    return child,out,err,time.time()
def wait(name,owned,path=None):
    child,out,err,began=owned;code=child.wait();out.close();err.close()
    save(path or OUT/(name+'_parent_exit.json'),dict(exit_code=code,actual_parent_wait=True,pid=child.pid,job=os.environ['SLURM_JOB_ID'],start_epoch=began,end_epoch=time.time()))
    return code
worker=None;code=1
try:
    linger=subprocess.run(['loginctl','enable-linger',str(os.getuid())],capture_output=True,text=True)
    deadline=time.monotonic()+30
    while not Path(env['XDG_RUNTIME_DIR']+'/bus').exists() and time.monotonic()<deadline:time.sleep(.2)
    save(OUT/'user_bus_setup.json',dict(exit_code=linger.returncode,stdout=linger.stdout,stderr=linger.stderr,bus_exists=Path(env['XDG_RUNTIME_DIR']+'/bus').exists()))
    assert linger.returncode==0 and Path(env['XDG_RUNTIME_DIR']+'/bus').exists()
    qualification=start('qualification',[P['harbor_python'],'-B',str(ROOT/'phase_child.py'),'qualify'],env,task_cpus)
    assert wait('qualification',qualification)==0,'Actual-node environment qualification failed'
    receipt=json.loads((ROOT/'qualification'/(TASK+'--qualify')/'execution_receipt.json').read_text())
    assert receipt['closure']['cgroup_empty'] and receipt['model_calls']==0 and not receipt['exception']
    wen=env.copy();wen.update(PYTHONPATH=env['PYTHONPATH']+':/srv/encbank/comem_infra_recheck_20260912/deps',
        PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        CPATH='/srv/encbank/.cache/python/include/python3.12',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(ROOT/'cache'),TORCHINDUCTOR_CACHE_DIR=str(OUT/'compile_cache'))
    worker=start('worker',[P['engine_python'],'-B',str(ROOT/'worker.py'),TASK],wen,model_cpus)
    while not (OUT/'ready.json').exists():
        assert worker[0].poll() is None,'Model worker failed before ready'
        time.sleep(.2)
    live=start('task_controller',[P['harbor_python'],'-B',str(ROOT/'phase_child.py'),'run'],env,task_cpus)
    code=wait('task_controller',live)
except BaseException:save(OUT/'integrated_failure.json',dict(error=traceback.format_exc(),epoch=time.time()))
finally:
    if worker is not None:
        save(BOX/'stop.json',dict(live_controller_closed=True,epoch=time.time()))
        wcode=wait('worker',worker,OUT/'parent_exit.json')
        if wcode:code=wcode
    save(ROOT/'jobs'/'integrated_complete.json',dict(exit_code=code,job=os.environ['SLURM_JOB_ID'],epoch=time.time()))
sys.exit(code)
