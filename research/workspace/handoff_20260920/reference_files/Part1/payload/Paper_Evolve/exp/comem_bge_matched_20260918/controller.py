"""One local campaign owner, bounded GPU admission, sequential calibration then frozen QA."""
import datetime,json,os,subprocess,sys,time,uuid
import config
from authorized_admission import local_ready

TOKEN='comem-bge-matched-20260919'
ACTIVE_CHILD=None
REMOTE=config.REMOTE_OWNER+'/control/bge_reservation.py'
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2),encoding='utf-8');t.replace(p)
def status(phase,**kw):dump(config.ROOT/'status.json',dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now().astimezone().isoformat(),**kw))
def reservation(action):
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3',REMOTE,action,'--token',TOKEN,'--pid',str(os.getpid())]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr[-1000:]
    return json.loads(r.stdout)
def run_script(script,args,path):
    global ACTIVE_CHILD
    path.mkdir(parents=True,exist_ok=True)
    if (path/'parent_exit.json').exists():
        old=json.loads((path/'parent_exit.json').read_text());assert old['returncode']==0 and old['actual_wait'];return
    cmd=[sys.executable,'-X','utf8','-B','-u',str(config.ROOT/script),*args]
    with (path/'stdout.log').open('ab') as out,(path/'stderr.log').open('ab') as err:
        child=subprocess.Popen(cmd,cwd=str(config.ROOT),stdout=out,stderr=err)
        ACTIVE_CHILD=child
        status('RUNNING',command=cmd,child_pid=child.pid,output=str(path));code=child.wait()
        ACTIVE_CHILD=None
    dump(path/'parent_exit.json',dict(returncode=code,actual_wait=True,pid=child.pid,command=cmd));assert code==0,cmd
def main():
    import msvcrt
    lock=(config.ROOT/'controller.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((config.ROOT/'data/complete.json').read_text())['complete']
    reserved=False
    try:
        # Do not reduce remote throughput while the 5090 is occupied by desktop/another GPU job.
        deadline=time.monotonic()+6*3600
        while True:
            ready,obs=local_ready()
            if ready:break
            status('WAITING_LOCAL_GPU',observation=obs,remote_slots_reserved=0)
            if time.monotonic()>deadline:raise TimeoutError('5090 admission did not become available within 6 hours')
            time.sleep(30)
        reserved=True;snapshot=reservation('reserve');status('WAITING_OWN_GPU_QUOTA',reservation=snapshot)
        while not snapshot['local_may_start']:
            if time.monotonic()>deadline:raise TimeoutError('Four-GPU budget did not admit local task within 6 hours')
            time.sleep(30);snapshot=reservation('status');status('WAITING_OWN_GPU_QUOTA',reservation=snapshot)
        for rep in range(3):run_script('worker.py',['--phase','calibration','--process',str(rep)],config.ROOT/'calibration'/f'process_{rep:02d}')
        if not (config.ROOT/'budget_decision.json').exists() and not (config.ROOT/'calibration_decision.json').exists():
            run_script('select_budget.py',[],config.ROOT/'selection_primary')
        if not (config.ROOT/'budget_decision.json').exists():
            assert json.loads((config.ROOT/'calibration_decision.json').read_text())['needs_extension']
            for rep in range(3):run_script('worker.py',['--phase','extended','--process',str(rep)],config.ROOT/'extended'/f'process_{rep:02d}')
            run_script('select_budget.py',[],config.ROOT/'selection_extended')
        run_script('worker.py',['--phase','quality','--process','0'],config.ROOT/'quality/process_00')
        run_script('report.py',[],config.ROOT/'report_run')
        status('COMPLETE',report=str(config.ROOT/'RESULTS_zh.md'))
    except BaseException as exc:
        status('FAILED',error=repr(exc));raise
    finally:
        # Every GPU child was actually waited for above; release the quota only after it has exited.
        if reserved and (ACTIVE_CHILD is None or ACTIVE_CHILD.poll() is not None):
            r=reservation('release');dump(config.ROOT/'reservation_release.json',r)
        elif reserved:
            dump(config.ROOT/'reservation_still_held.json',dict(child_pid=ACTIVE_CHILD.pid,reason='Child still running; do not release local GPU quota yet.'))

if __name__=='__main__':main()
