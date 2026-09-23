"""Single Windows controller: official Harbor -> local RPC -> node-local GPU worker."""
import concurrent.futures,datetime,json,msvcrt,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
from io_utils import save
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text());S=json.loads((ROOT/'submission.json').read_text())
REMOTE=S['stage'];RPC=ROOT/'rpc';PY='/usr/bin/python3 -I -B'

def now():return datetime.datetime.now().astimezone().isoformat()
def run(argv,timeout=50):
    r=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,encoding='utf8',errors='replace',timeout=timeout)
    if r.returncode:raise RuntimeError(dict(argv=argv,exit=r.returncode,stderr=r.stderr[-2500:]))
    return r.stdout
def ctl(action,rid=None):
    command=PY+' '+shlex.quote(REMOTE+'/bridge.py')+' '+action+(' '+rid if rid else '')
    return json.loads(run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',command]))
def snapshot():
    code="import json,subprocess;from pathlib import Path;r=Path("+repr(REMOTE)+");names=['ready.json','failure.json','status.json','smoke.json','environment.json','parent_exit.json','worker_complete.json','probe_correctness.json'];print(json.dumps(dict(files={n:json.loads((r/n).read_text()) for n in names if (r/n).exists()},queue=subprocess.check_output(['squeue','-j',"+repr(S['job'])+",'-h','-o','%T|%R'],universal_newlines=True),sacct=subprocess.check_output(['sacct','-j',"+repr(S['job'])+",'-n','-P','-o','JobIDRaw,State,ExitCode'],universal_newlines=True))))"
    state=json.loads(run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',PY+' -c '+shlex.quote(code)]))
    save(ROOT/'monitor.json',state);return state

def handle(path,child):
    q=json.loads(path.read_text());rid=q['request_id'];start=time.monotonic();polls=0;cancelled=False
    try:
        run(['scp','-q','-o','BatchMode=yes',str(path),'gpu-node1:'+REMOTE+'/incoming/'+rid+'.json'])
        ctl('publish',rid)
        while True:
            # Bounded short connections: avoid the observed long-wait SSH reply stall.
            packet=ctl('poll',rid);polls+=1
            if packet['ready']:
                save(RPC/(rid+'.response.json'),packet['response'])
                save(RPC/(rid+'.transport.json'),dict(seconds=time.monotonic()-start,cancelled=cancelled,polls=polls,transport='bounded SSH polling',poll_interval_seconds=1.0))
                return
            if not cancelled and ((RPC/(rid+'.cancel.json')).exists() or child.poll() is not None):
                ctl('cancel',rid);cancelled=True
            if time.monotonic()-start>q['remaining_seconds']+30:
                raise TimeoutError('Remote response deadline')
            time.sleep(1.0)
    except BaseException:
        try:ctl('cancel',rid)
        except Exception:pass
        save(RPC/(rid+'.error.json'),dict(error=traceback.format_exc()));raise


def main():
    lock=(ROOT/'owner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (ROOT/'owner_registration.json').exists(),'Do not restart an owner without inspecting existing trials'
    save(ROOT/'owner_registration.json',dict(pid=os.getpid(),at=now(),job=S['job']))
    began=time.monotonic();child=None;ready=False
    try:
        while True:
            state=snapshot();files=state['files']
            if files.get('failure.json') or files.get('parent_exit.json'):raise RuntimeError(state)
            if any(word in state['sacct'] for word in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY']):raise RuntimeError(state)
            if 'DependencyNeverSatisfied' in state['queue']:raise RuntimeError(state)
            if files.get('ready.json'):ready=True;break
            save(ROOT/'owner_status.json',dict(phase='WAITING_GPU',job=S['job'],queue=state['queue'],at=now()))
            if time.monotonic()-began>48*3600:raise TimeoutError('Queue bound')
            time.sleep(60)
        for name,value in files.items():save(ROOT/'remote_records'/name,value)
        for arm in P['arms']:
            engine_id=run(['wsl','-d','Ubuntu','--exec','docker','-H','unix:///var/run/docker.sock','info','--format','{{.ID}}']).strip()
            assert engine_id==P['docker_engine_id'],'Docker engine changed; stop before starting a trial'
            admission_began=time.monotonic()
            while True:
                mem=run(['wsl','-d','Ubuntu','--exec','cat','/proc/meminfo'])
                available=int(next(l for l in mem.splitlines() if l.startswith('MemAvailable:')).split()[1])*1024
                required=(sum(x['memory_mb'] for x in P['resource_inventory'])+2048)*2**20
                if available>=required:break
                save(ROOT/'owner_status.json',dict(phase='WAITING_HOST_MEMORY',available_bytes=available,required_bytes=required,at=now()))
                if time.monotonic()-admission_began>3600:raise TimeoutError('Host memory admission unavailable for one hour')
                time.sleep(20)
            cfg=P['local_wsl_root']+'/harbor_configs/'+arm+'.json'
            command='cd '+shlex.quote(P['local_wsl_root'])+' && env DOCKER_HOST=unix:///var/run/docker.sock LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(P['local_wsl_root'])+' '+shlex.quote(P['host_runtime']+'/bin/harbor')+' run -c '+shlex.quote(cfg)
            argv=['wsl','-d','Ubuntu','--exec','/bin/bash','--noprofile','--norc','-c',command]
            with (ROOT/(arm+'.stdout.log')).open('wb') as out,(ROOT/(arm+'.stderr.log')).open('wb') as err,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
                save(ROOT/'owner_status.json',dict(phase='HARBOR_RUNNING',arm=arm,pid=child.pid,at=now()))
                seen=set();futures=[]
                while child.poll() is None:
                    for path in sorted(RPC.glob('*.request.json')):
                        if path.name in seen or path.with_name(path.name.replace('.request.json','.response.json')).exists() or path.with_name(path.name.replace('.request.json','.error.json')).exists():continue
                        q=json.loads(path.read_text())
                        if q['arm']!=arm:continue
                        seen.add(path.name);futures.append(pool.submit(handle,path,child))
                    for f in futures:
                        if f.done() and f.exception():raise f.exception()
                    time.sleep(.15)
                code=child.wait()
            errors=[str(f.exception()) for f in futures if f.exception()]
            save(ROOT/'receipts'/(arm+'.json'),dict(exit_code=code,actual_child_wait=True,requests=len(seen),transport_errors=errors,at=now()))
            if code or errors:raise RuntimeError(dict(arm=arm,exit=code,errors=errors))
            task_ids={json.loads(p.read_text())['task_id'] for p in RPC.glob('*.request.json') if json.loads(p.read_text())['arm']==arm}
            for sid in sorted(task_ids):ctl('release',sid)
            run([sys.executable,'-X','utf8','-B',str(ROOT/'report.py')])
        save(ROOT/'owner_status.json',dict(phase='WAITING_WORKER_EXIT',at=now()))
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=20)
        if ready:save(ROOT/'stop_receipt.json',ctl('stop'))
    for _ in range(30):
        state=snapshot();wait=state['files'].get('parent_exit.json')
        if wait and wait['actual_wait'] and wait['returncode']==0 and S['job']+'|COMPLETED|0:0' in state['sacct']:break
        if state['files'].get('failure.json'):raise RuntimeError(state)
        time.sleep(20)
    else:raise RuntimeError('No verified scientific-child and Slurm completion')
    for name,value in state['files'].items():save(ROOT/'remote_records'/name,value)
    run([sys.executable,'-X','utf8','-B',str(ROOT/'report.py')])
    save(ROOT/'owner_status.json',dict(phase='COMPLETE',at=now()))
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'owner_failure.json',dict(error=traceback.format_exc(),at=now()));raise
