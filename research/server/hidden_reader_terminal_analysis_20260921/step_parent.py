"""Pin model and task CPUs disjointly inside this research's own allocation."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
C=Path(sys.argv[1]);task=sys.argv[2];job=sys.argv[3]
sys.path.insert(0,str(C))
from common import ROOT,PLAN as P,save,verify_sources
R=Path(P['worker_root']);OUT=R/'pairs'/task
def main():
    verify_sources();assert os.environ['SLURM_JOB_ID']==job
    assert not list((R/'mailbox'/task).glob('*.request.json')),'Cannot change CPU assignment after live requests'
    candidates=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            if str(R/'worker.py').encode() in args and task.encode() in args:candidates.append(int(p.name))
        except (OSError,PermissionError):pass
    assert len(candidates)==1,candidates
    pid=candidates[0];cores=sorted(os.sched_getaffinity(pid))
    assert len(cores)==4 and set(cores)<=os.sched_getaffinity(0),(cores,sorted(os.sched_getaffinity(0)))
    modelcores=set(cores[:3]);controlcore=cores[3];before={}
    for path in (Path('/proc')/str(pid)/'task').iterdir():
        tid=int(path.name)
        try:before[tid]=sorted(os.sched_getaffinity(tid));os.sched_setaffinity(tid,modelcores)
        except ProcessLookupError:pass
    os.sched_setaffinity(0,{controlcore})
    save(OUT/'cpu_partition.json',dict(job=job,node=os.uname().nodename,worker_pid=pid,allocated_cpus=cores,
        model_cpus=sorted(modelcores),controller_and_task_cpus=[controlcore],original_thread_affinities=before,
        note='Disjoint CPUs inside own four-CPU allocation; same assignment for both arms.'))
    env=os.environ.copy();env.update(PYTHONPATH=str(C),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()),DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(os.getuid())+'/bus')
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:env.pop(key,None)
    linger=subprocess.run(['loginctl','enable-linger',str(os.getuid())],capture_output=True,text=True)
    save(OUT/'user_bus_setup.json',dict(exit_code=linger.returncode,stdout=linger.stdout,stderr=linger.stderr,
        bus_exists=Path(env['XDG_RUNTIME_DIR']+'/bus').exists()))
    assert linger.returncode==0 and Path(env['XDG_RUNTIME_DIR']+'/bus').exists()
    start=time.time()
    with (OUT/'task_controller.stdout.log').open('wb') as out,(OUT/'task_controller.stderr.log').open('wb') as err:
        child=subprocess.Popen([P['harbor_python'],str(C/'controller_child.py'),task],cwd=C,env=env,stdout=out,stderr=err)
        code=child.wait()
    save(OUT/'task_controller_parent_exit.json',dict(exit_code=code,actual_parent_wait=True,pid=child.pid,job=job,start_epoch=start,end_epoch=time.time()))
    return code
code=1
try:code=main()
except BaseException:save(OUT/'task_controller_failure.json',dict(error=traceback.format_exc(),epoch=time.time()))
finally:save(R/'mailbox'/task/'stop.json',dict(task_controller_closed=True,epoch=time.time()))
sys.exit(code)
