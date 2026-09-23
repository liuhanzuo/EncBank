"""One controller, one bounded SSH mailbox exchange at a time; one GPU service."""
import datetime,json,msvcrt,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
from io_utils import save
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'plan.json').read_text());S=json.loads((ROOT/'submission.json').read_text())
REMOTE=S['stage'];RPC=ROOT/P.get('rpc_subdir','rpc');PY='/usr/bin/python3 -I -B'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gpu-node1']

class TransientTransportError(RuntimeError):pass
def now():return datetime.datetime.now().astimezone().isoformat()
def run(argv,timeout=50):
    r=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,encoding='utf8',errors='replace',timeout=timeout)
    if r.returncode:raise RuntimeError(dict(argv=argv,exit=r.returncode,stderr=r.stderr[-2500:]))
    return r.stdout
def remote(command,packet=None):
    try:
        r=subprocess.run(SSH+[command],input=json.dumps(packet) if packet is not None else '',capture_output=True,encoding='utf8',errors='replace',timeout=35)
    except subprocess.TimeoutExpired as e:raise TransientTransportError('SSH exchange exceeded 35 seconds') from e
    if r.returncode:
        error=dict(exit=r.returncode,stderr=r.stderr[-2500:])
        if r.returncode==255:raise TransientTransportError(error)
        raise RuntimeError(error)
    return json.loads(r.stdout)
def log_retry(error,operation,streak):
    with (ROOT/'transport_retry.jsonl').open('a',encoding='utf8') as stream:
        stream.write(json.dumps(dict(at=now(),operation=operation,streak=streak,error=str(error)))+'\n')
def retry_remote(command,packet=None,limit_seconds=180):
    started=time.monotonic();streak=0
    while True:
        try:return remote(command,packet)
        except TransientTransportError as error:
            streak+=1;log_retry(error,'control',streak)
            if time.monotonic()-started>limit_seconds:raise
            time.sleep(min(10,streak))
def ctl(action,rid=None):
    return retry_remote(PY+' '+shlex.quote(REMOTE+'/bridge.py')+' '+action+(' '+rid if rid else ''))
def exchange(packet):return remote(PY+' '+shlex.quote(REMOTE+'/exchange.py'),packet)
def snapshot():
    code="import json,subprocess;from pathlib import Path;r=Path("+repr(REMOTE)+");names=['ready.json','failure.json','status.json','smoke.json','environment.json','parent_exit.json','worker_complete.json','probe_correctness.json'];print(json.dumps(dict(files={n:json.loads((r/n).read_text()) for n in names if (r/n).exists()},queue=subprocess.check_output(['squeue','-j',"+repr(S['job'])+",'-h','-o','%T|%R'],universal_newlines=True),sacct=subprocess.check_output(['sacct','-j',"+repr(S['job'])+",'-n','-P','-o','JobIDRaw,State,ExitCode'],universal_newlines=True))))"
    state=retry_remote(PY+' -c '+shlex.quote(code));save(ROOT/'monitor.json',state);return state

def serve_arm(child,arm):
    known={};accepted=set();polls={};retries=0;streak=0;last_ok=time.monotonic();last_progress=0
    while child.poll() is None:
        for path in sorted(RPC.glob('*.request.json')):
            if path.stem.removesuffix('.request') in known:continue
            q=json.loads(path.read_text())
            if q['arm']==arm and q['request_id'] not in P.get('excluded_request_ids',[]):known[q['request_id']]=q
        pending={rid:q for rid,q in known.items() if not (RPC/(rid+'.response.json')).exists() and not (RPC/(rid+'.error.json')).exists()}
        if pending:
            packet=dict(requests=[q for rid,q in pending.items() if rid not in accepted],poll=list(pending),
                cancel=[rid for rid,q in pending.items() if (RPC/(rid+'.cancel.json')).exists() or time.time()>q['client_created_epoch']+q['remaining_seconds']])
            try:
                result=exchange(packet)
            except TransientTransportError as error:
                retries+=1;streak+=1;log_retry(error,arm,streak)
                save(ROOT/'transport_status.json',dict(phase='RETRYING_SAME_IDS',arm=arm,at=now(),streak=streak,error=str(error)))
                if time.monotonic()-last_ok>180:raise RuntimeError('SSH unavailable for 180s; preserve GPU and classify affected trials as infrastructure failures') from error
                time.sleep(min(10,streak));continue
            accepted.update(result['accepted']);streak=0;last_ok=time.monotonic()
            if result['state']:raise RuntimeError(result['state'])
            for rid in pending:polls[rid]=polls.get(rid,0)+1
            for rid,response in result['responses'].items():
                q=known[rid]
                save(RPC/(rid+'.response.json'),response)
                save(RPC/(rid+'.transport.json'),dict(seconds=time.time()-q['client_created_epoch'],polls=polls[rid],
                    transport='single sequential batched SSH exchange',poll_interval_seconds=1.0,arm_connection_retries_so_far=retries))
        else:last_ok=time.monotonic()
        if time.monotonic()-last_progress>10:
            save(ROOT/'transport_status.json',dict(phase='RUNNING',arm=arm,at=now(),known_requests=len(known),
                pending_requests=len(pending),published_requests=len(accepted),connection_retries=retries))
            last_progress=time.monotonic()
        time.sleep(1.0 if pending else .15)
    code=child.wait()
    packet=dict(poll=list(known),cancel=[rid for rid in known if not (RPC/(rid+'.response.json')).exists()])
    result=retry_remote(PY+' '+shlex.quote(REMOTE+'/exchange.py'),packet)
    for rid,response in result['responses'].items():
        if not (RPC/(rid+'.response.json')).exists():save(RPC/(rid+'.response.json'),response)
    receipt=dict(exit_code=code,actual_child_wait=True,requests=len(known),connection_retries=retries,at=now())
    save(ROOT/'receipts'/(arm+'.json'),receipt)
    if code:raise RuntimeError(receipt)
    ids=sorted({q['task_id'] for q in known.values()});began=time.monotonic()
    while ids:
        result=retry_remote(PY+' '+shlex.quote(REMOTE+'/exchange.py'),dict(release=ids))
        if result['state']:raise RuntimeError(result['state'])
        if len(result['released'])==len(ids):
            save(ROOT/'receipts'/(arm+'.released.json'),result['released']);break
        if time.monotonic()-began>90:raise RuntimeError('Session releases not acknowledged; refuse next arm')
        time.sleep(1)
    return receipt

def main():
    lock=(ROOT/'owner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (ROOT/'owner_registration.json').exists(),'Inspect and archive previous controller before restart'
    save(ROOT/'owner_registration.json',dict(pid=os.getpid(),at=now(),job=S['job'],transport='single sequential batched SSH exchange'))
    child=None;began=time.monotonic()
    try:
        while True:
            state=snapshot();files=state['files']
            if files.get('failure.json') or files.get('parent_exit.json'):raise RuntimeError(state)
            if any(word in state['sacct'] for word in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY']):raise RuntimeError(state)
            if 'DependencyNeverSatisfied' in state['queue']:raise RuntimeError(state)
            if files.get('ready.json'):break
            save(ROOT/'owner_status.json',dict(phase='WAITING_GPU',job=S['job'],queue=state['queue'],at=now()))
            if time.monotonic()-began>48*3600:raise TimeoutError('Queue bound')
            time.sleep(30)
        for name,value in files.items():save(ROOT/'remote_records'/name,value)
        for arm in P['arms']:
            receipt_path=ROOT/'receipts'/(arm+'.json')
            if receipt_path.exists():
                prior=json.loads(receipt_path.read_text())
                assert prior['exit_code']==0 and prior['actual_child_wait']
                assert (ROOT/'receipts'/(arm+'.released.json')).exists()
                continue
            engine_id=run(['wsl','-d','Ubuntu','--exec','docker','-H','unix:///var/run/docker.sock','info','--format','{{.ID}}']).strip()
            assert engine_id==P['docker_engine_id'],'Docker engine changed; stop before a trial'
            admission_began=time.monotonic()
            while True:
                mem=run(['wsl','-d','Ubuntu','--exec','cat','/proc/meminfo'])
                available=int(next(l for l in mem.splitlines() if l.startswith('MemAvailable:')).split()[1])*1024
                required=(sum(x['memory_mb'] for x in P['resource_inventory'])+2048)*2**20
                if available>=required:break
                save(ROOT/'owner_status.json',dict(phase='WAITING_HOST_MEMORY',available_bytes=available,required_bytes=required,at=now()))
                if time.monotonic()-admission_began>3600:raise TimeoutError('Host RAM unavailable for one hour')
                time.sleep(20)
            cfg=P['local_wsl_root']+'/harbor_configs/'+arm+'.json'
            command='cd '+shlex.quote(P['local_wsl_root'])+' && env DOCKER_HOST=unix:///var/run/docker.sock LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(P['local_wsl_root'])+' '+shlex.quote(P['host_runtime']+'/bin/harbor')+' run -c '+shlex.quote(cfg)
            argv=['wsl','-d','Ubuntu','--exec','/bin/bash','--noprofile','--norc','-c',command]
            with (ROOT/(arm+'.stdout.log')).open('wb') as out,(ROOT/(arm+'.stderr.log')).open('wb') as err:
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
                save(ROOT/'owner_status.json',dict(phase='HARBOR_RUNNING',arm=arm,pid=child.pid,at=now()))
                serve_arm(child,arm)
            run([sys.executable,'-X','utf8','-B',str(ROOT/'report.py')])
        save(ROOT/'owner_status.json',dict(phase='WAITING_WORKER_EXIT',at=now()))
        save(ROOT/'stop_receipt.json',ctl('stop'))
        for _ in range(60):
            state=snapshot();wait=state['files'].get('parent_exit.json')
            if wait and wait['actual_wait'] and wait['returncode']==0 and S['job']+'|COMPLETED|0:0' in state['sacct']:break
            if state['files'].get('failure.json'):raise RuntimeError(state)
            time.sleep(10)
        else:raise RuntimeError('Scientific child/Slurm completion not verified')
        for name,value in state['files'].items():save(ROOT/'remote_records'/name,value)
        run([sys.executable,'-X','utf8','-B',str(ROOT/'report.py')])
        save(ROOT/'owner_status.json',dict(phase='COMPLETE',at=now()))
    except BaseException:
        # A control-plane error must not terminate the loaded GPU or create replacement scientific trials.
        save(ROOT/'owner_failure.json',dict(error=traceback.format_exc(),at=now(),gpu_preserved=True,harbor_pid=child.pid if child else None))
        save(ROOT/'owner_status.json',dict(phase='FAILED_GPU_PRESERVED',at=now()))
        raise

if __name__=='__main__':main()
