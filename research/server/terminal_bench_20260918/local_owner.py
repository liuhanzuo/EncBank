"""Persistent local owner: official Harbor jobs, three arms together, no retry."""
from pathlib import Path
import asyncio,datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from tb_agent import exchange
H=Path(__file__).resolve().parent;E=H.parent;R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918')
O=E/'execution';O.mkdir(exist_ok=True)
lock=(O/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
async def arm_run(arm):
    started=time.monotonic();status=O/(arm+'_status.json');launched=False
    try:
        while True:
            r=await exchange(arm,dict(action='status'))
            if 'worker_failure.json' in r:raise RuntimeError(r['worker_failure.json'])
            if 'worker_ready.json' in r:break
            save(status,dict(state='waiting_gpu_worker',at=datetime.datetime.now().astimezone().isoformat()))
            if time.monotonic()-started>7200:raise TimeoutError('GPU did not become ready in bounded two-hour queue wait; no resubmit')
            await asyncio.sleep(20)
        save(O/(arm+'_worker_ready.json'),r['worker_ready.json'])
        argv=[str(R/'harbor_env/bin/harbor'),'run','-c',str(H/(arm+'_harbor.json'))]
        env=dict(os.environ,PYTHONPATH=str(H),HARBOR_CACHE_DIR=str(R/'harbor_cache'))
        with (O/(arm+'.stdout.log')).open('wb') as out,(O/(arm+'.stderr.log')).open('wb') as err:
            child=await asyncio.create_subprocess_exec(*argv,stdout=out,stderr=err,env=env,cwd=R)
            launched=True;save(status,dict(state='harbor_running',pid=child.pid,argv=argv,started_at=datetime.datetime.now().astimezone().isoformat()))
            code=await child.wait()
        save(O/(arm+'_harbor_receipt.json'),dict(exit_code=code,actual_parent_wait=True,ended_at=datetime.datetime.now().astimezone().isoformat()))
        result=R/'results'/arm/'result.json'
        save(status,dict(state='harbor_exited_results_require_validation',exit_code=code,result_path=str(result),result_exists=result.exists(),ended_at=datetime.datetime.now().astimezone().isoformat()))
    except BaseException:
        save(status,dict(state='infrastructure_or_controller_failure_no_retry',error=traceback.format_exc(),launched=launched));raise
    finally:
        if launched:await exchange(arm,dict(action='stop',reason='local Harbor parent exited or controller failed'))
async def main():
    assert not (O/'owner_registration.json').exists()
    save(O/'owner_registration.json',dict(pid=os.getpid(),started_at=datetime.datetime.now().astimezone().isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms=['dense','raw_shared','encbank']))
    outcomes=await asyncio.gather(*(arm_run(a) for a in ['dense','raw_shared','encbank']),return_exceptions=True)
    save(O/'owner_complete.json',dict(ended_at=datetime.datetime.now().astimezone().isoformat(),outcomes=[repr(o) for o in outcomes]))
    if any(isinstance(o,BaseException) for o in outcomes):raise RuntimeError('A controller arm failed; inspect saved status, no retry')
asyncio.run(main())
