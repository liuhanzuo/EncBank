"""One Windows owner, one Harbor job, four simultaneous isolated RPC calls."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,subprocess,sys,time,traceback
H=Path(__file__).resolve().parent;R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919')
O=H/'execution';O.mkdir(exist_ok=True);BOX=R/'local_rpc_toolchain_fix/dense';BOX.mkdir(parents=True,exist_ok=True)
P=json.loads((H/'plan.json').read_text());REMOTE=P['remote_root'];PY='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');t.replace(p)
def run(argv,timeout=60):
    p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if p.returncode:raise RuntimeError(json.dumps(dict(argv=argv,exit=p.returncode,stdout=p.stdout[-2000:],stderr=p.stderr[-4000:])))
    return p.stdout
def ctl(action,rid=None):
    cmd=f'{PY} {REMOTE}/file_bridge.py {action} dense'+(' '+rid if rid else '')
    return json.loads(run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',cmd]))
def handle(req,child):
    d=json.loads(req.read_text());rid=d['request_id'];started=time.monotonic();target=BOX/(rid+'.response.json')
    try:
        run(['scp','-o','BatchMode=yes',str(req),'gpu-node1:'+REMOTE+'/incoming/'+rid+'.json'])
        ctl('publish',rid)
        cmd=f'{PY} {REMOTE}/file_bridge.py wait dense {rid}'
        wait=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',cmd],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',errors='replace')
        cancel_sent=False
        while wait.poll() is None:
            if not cancel_sent and ((BOX/(rid+'.cancel.json')).exists() or child.poll() is not None):ctl('cancel',rid);cancel_sent=True
            if time.monotonic()-started>d['remaining_seconds']+180:wait.terminate();wait.wait();raise TimeoutError('Remote response deadline exceeded')
            time.sleep(.2)
        out,err=wait.communicate()
        if wait.returncode:raise RuntimeError(f'SSH wait exit {wait.returncode}: {err[-2000:]}')
        proof=json.loads(out);assert proof['path']==REMOTE+'/run_dense/mailbox/'+rid+'.response.json'
        incoming=target.with_suffix('.download')
        run(['scp','-o','BatchMode=yes','gpu-node1:'+proof['path'],str(incoming)])
        raw=incoming.read_bytes();assert len(raw)==proof['bytes'] and hashlib.sha256(raw).hexdigest()==proof['sha256'];incoming.replace(target)
        save(BOX/(rid+'.broker.json'),dict(status='delivered',seconds=time.monotonic()-started,proof=proof,cancel_sent=cancel_sent))
    except BaseException:
        # An uncertain publish is never retried; cancel only this owned request.
        try:ctl('cancel',rid)
        except Exception:pass
        save(BOX/(rid+'.error.json'),dict(error=traceback.format_exc(),at=now(),no_automatic_request_retry=True))
        raise
def main():
    assert not (O/'owner_registration.json').exists(),'Owner already registered'
    save(O/'owner_registration.json',dict(pid=os.getpid(),at=now(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    child=None;ready_seen=False
    try:
        began=time.monotonic()
        while True:
            s=ctl('status')
            if any(x in s for x in ['worker_failure.json','process_receipt.json','memory_cap_failure.json']):raise RuntimeError(json.dumps(s))
            if 'worker_ready.json' in s:break
            save(O/'status.json',dict(state='waiting_gpu_worker',at=now()))
            if time.monotonic()-began>48*3600:raise TimeoutError('48-hour queue/startup bound; no duplicate submission')
            time.sleep(20)
        save(O/'worker_ready.json',s['worker_ready.json'])
        ready_seen=True
        # First four public images are pre-fetched to avoid wasting agent time on startup network setup.
        for name in P['tasks'][:4]:
            receipt=H.parent/'images'/(name+'.json')
            while not receipt.exists():
                if time.monotonic()-began>48*3600:raise TimeoutError('Image preparation did not close')
                time.sleep(5)
            if json.loads(receipt.read_text())['status']!='PASS':
                receipt=H.parent/'images_recovery'/(name+'.json')
                while not receipt.exists():
                    if time.monotonic()-began>48*3600:raise TimeoutError('Image recovery did not close')
                    time.sleep(5)
            assert json.loads(receipt.read_text())['status']=='PASS',name
        wh='/srv/encbank/legacy_workspace/'+H.relative_to(Path('/srv/encbank/legacy_workspace')).as_posix()
        command=f'cd /srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919 && DOCKER_HOST=unix:///var/run/docker.sock LITELLM_LOCAL_MODEL_COST_MAP=True GIT_CEILING_DIRECTORIES=/srv/encbank/legacy_workspace/.runtime PYTHONPATH={wh} /srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918/harbor_env/bin/harbor run -c {wh}/dense_harbor_rpc.json'
        argv=['wsl','-d','Ubuntu','--','bash','-lc',command]
        with (O/'harbor.stdout.log').open('wb') as out,(O/'harbor.stderr.log').open('wb') as err,ThreadPoolExecutor(max_workers=4) as pool:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            save(O/'status.json',dict(state='harbor_running',pid=child.pid,argv=argv,at=now()))
            seen=set();futures={}
            while child.poll() is None:
                for req in sorted(BOX.glob('*.request.json')):
                    if req.name in seen:continue
                    seen.add(req.name);futures[req.name]=pool.submit(handle,req,child)
                time.sleep(.15)
            code=child.wait()
        errors={name:str(f.exception()) for name,f in futures.items() if f.exception()}
        save(O/'harbor_receipt.json',dict(exit_code=code,actual_parent_wait=True,ended_at=now(),requests_seen=len(seen),transport_errors=errors))
        save(O/'status.json',dict(state='harbor_exited_results_require_validation',exit_code=code,at=now()))
    except BaseException:
        save(O/'controller_failure.json',dict(error=traceback.format_exc(),at=now()));raise
    finally:
        if ready_seen and (child is None or child.poll() is not None):
            try:save(O/'stop_receipt.json',ctl('stop'))
            except Exception:save(O/'stop_error.json',dict(error=traceback.format_exc()))
    save(O/'owner_complete.json',dict(at=now(),actual_child_wait=True))
if __name__=='__main__':main()
