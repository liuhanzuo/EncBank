"""One CoMem batch8 GPU service with independent task worlds and shared host RAM admission."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64,datetime,hashlib,json,os,subprocess,sys,time,traceback
import host_admission
import fair_share
H=Path(__file__).resolve().parent;R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919')
P=json.loads((H/'plan.json').read_text());REMOTE=P['remote_root'];PY='/srv/encbank/qcomem_runtime_20260911/python312/bin/python'
O=H/'execution';BOX=R/'local_rpc_dense_no_task_deadline_r5_20260920/dense';RESULTS=R/'results/dense_no_task_deadline_r5_20260920'
O.mkdir(exist_ok=True);BOX.mkdir(parents=True,exist_ok=True)
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+str(os.getpid())+'-'+str(time.time_ns()))
    t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
    for attempt in range(10):
        try:t.replace(p);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.2)
def run(argv,timeout=60):
    p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if p.returncode:raise RuntimeError(json.dumps(dict(argv=argv,exit=p.returncode,stdout=p.stdout[-2000:],stderr=p.stderr[-3000:])))
    return p.stdout
def ctl(action,rid=None):
    attempts=3 if action=='status' else 1
    for attempt in range(attempts):
        cursor=0
        if action=='status' and (O/'transport_snapshot.json').exists():
            cursor=json.loads((O/'transport_snapshot.json').read_text())['event_cursor']
        arg=str(cursor) if action=='status' else rid
        try:
            raw=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gpu-node1',f'{PY} -u {REMOTE}/file_bridge.py {action} dense'+(' '+arg if arg else '')])
            result=json.loads(raw)
        except (ValueError,RuntimeError,OSError,subprocess.TimeoutExpired) as exc:
            save(O/('status_read_failure_'+str(time.time_ns())+'.json'),dict(action=action,attempt=attempt,error=str(exc),at=now(),no_model_retry=True))
            if attempt+1==attempts:raise
            time.sleep(2);continue
        if action in ['status','stop'] and 'transport' in result:
            save(O/('transport_status_'+str(time.time_ns())+'.json'),result['transport'])
            save(O/'transport_snapshot.json',result['transport'])
        return result

def handle(req,child):
    q=json.loads(req.read_text());rid=q['request_id'];started=time.monotonic();target=BOX/(rid+'.response.json')
    try:
        pub=subprocess.run(['ssh','-o','BatchMode=yes','gpu-node1',f'{PY} {REMOTE}/file_bridge.py publish dense {rid}'],input=req.read_bytes(),capture_output=True,timeout=60)
        assert pub.returncode==0,pub.stderr.decode(errors='replace')
        cmd=f'{PY} {REMOTE}/file_bridge.py wait dense {rid}'
        transfer=target.with_suffix('.transfer');stream=transfer.open('wb')
        wait=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gpu-node1',cmd],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.PIPE,text=True,encoding='utf8',errors='replace')
        cancelled=False
        while wait.poll() is None:
            if not cancelled and ((BOX/(rid+'.cancel.json')).exists() or child.poll() is not None):ctl('cancel',rid);cancelled=True
            if time.monotonic()-started>q['remaining_seconds']+180:wait.terminate();wait.wait();raise TimeoutError('Remote response closure exceeded deadline')
            time.sleep(.2)
        _,err=wait.communicate();stream.close();out=transfer.read_text(encoding='utf8')
        if wait.returncode:raise RuntimeError(f'SSH wait {wait.returncode}: {err[-2000:]}')
        proof=json.loads(out);assert proof['path']==REMOTE+'/run_dense/mailbox/'+rid+'.response.json'
        raw=base64.b64decode(proof.pop('payload_base64'),validate=True);assert len(raw)==proof['bytes'] and hashlib.sha256(raw).hexdigest()==proof['sha256']
        incoming=target.with_suffix('.download')
        with incoming.open('wb') as durable:durable.write(raw);durable.flush();os.fsync(durable.fileno())
        incoming.replace(target)
        save(BOX/(rid+'.broker.json'),dict(status='delivered',seconds=time.monotonic()-started,proof=proof,cancel_sent=cancelled))
    except BaseException:
        try:ctl('cancel',rid)
        except Exception:pass
        save(BOX/(rid+'.error.json'),dict(error=traceback.format_exc(),at=now(),no_automatic_request_retry=True));raise
def host_capacity(active):
    # Reserve the maximum RAM of both unmodified Dense worlds, including future tasks.
    dense_closed=(H.parent/'toolchain_fix/execution/harbor_receipt.json').exists()
    reserve=0 if dense_closed else 16*1024
    available=int(run(['wsl','-d','Ubuntu','--','awk','/MemAvailable:/{print $2}','/proc/meminfo']).strip())/1024
    used=sum(x['memory_mb'] for x in active.values())
    ceiling=16*1024 if dense_closed else 4*1024
    # Conservatively charge full limits even for presently near-empty containers.
    return max(0,min(ceiling-used,available-reserve-used-2*1024)),dict(at=now(),available_mb=available,
        dense_reserved_mb=reserve,active_comem_reserved_mb=used,additional_ceiling_mb=ceiling,safety_mb=2048)
def start_task(row):
    name=row['task'];cfg=json.loads((H/'dense_harbor_template.json').read_text())
    cfg.update(job_name=name,jobs_dir='/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/results/dense_no_task_deadline_r5_20260920')
    cfg['tasks']=[dict(path=P['task_root']+'/'+name)]
    config=O/'configs'/(name+'.json');assert not config.exists() and not (RESULTS/name).exists();save(config,cfg)
    wh='/srv/encbank/legacy_workspace/'+H.relative_to(Path('/srv/encbank/legacy_workspace')).as_posix()
    cmd=f'cd / && DOCKER_HOST=unix:///var/run/docker.sock LITELLM_LOCAL_MODEL_COST_MAP=True GIT_CEILING_DIRECTORIES=/srv/encbank/legacy_workspace/.runtime PYTHONPATH={wh} /srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918/harbor_env/bin/harbor run -c {wh}/execution/configs/{name}.json'
    argv=['wsl','-d','Ubuntu','--','bash','-lc',cmd]
    out=(O/(name+'.stdout.log')).open('wb');err=(O/(name+'.stderr.log')).open('wb')
    child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
    info=dict(**row,child=child,out=out,err=err,argv=argv)
    save(O/'launches'/(name+'.json'),dict(**row,pid=child.pid,argv=argv,at=now()))
    return info
def main():
    resume=(O/'owner_registration.json').exists()
    if not resume:save(O/'owner_registration.json',dict(pid=os.getpid(),at=now(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    active={};futures={};seen=set();completed=[];ready=False;began=time.monotonic();control=None
    try:
        from startup_wait import wait_for_worker
        status=wait_for_worker(ctl,save,O)
        ready=True;save(O/'worker_ready.json',status['worker_ready.json']);pending=list(P['resource_inventory']);last_status=0
        if resume:
            from owner_recovery import recover
            active,completed,pending,seen=recover(H,O,BOX,RESULTS,P,host_admission,save)
        from control_plane_recovery import ControlPlane
        control=ControlPlane(ctl,host_admission,H.name,O,save)
        with ThreadPoolExecutor(max_workers=P['concurrent_tasks']) as pool:
            while pending or active:
                fair_share.update(bool(pending) and len(active)<P['concurrent_tasks'],save)
                for name,info in list(active.items()):
                    child=info['child']
                    if child.poll() is None:continue
                    # Cancel/deliver any unfinished owned call before releasing its bank.
                    task_futures=[f for task,f in futures.values() if task==name]
                    if any(not f.done() for f in task_futures):continue
                    code=child.wait();info['out'].close();info['err'].close()
                    task_requests=[json.loads(p.read_text()) for p in BOX.glob('*.request.json') if json.loads(p.read_text())['task']==name]
                    release=[]
                    save(O/'parent_waits'/(name+'.json'),dict(task=name,pid=child.pid,exit_code=code,actual_parent_wait=getattr(child,'original_parent',True),actual_process_handle_wait=(not isinstance(code,str)),at=now()))
                    if P.get('arm','comem')=='comem':
                        for sid in sorted({q['task_id'] for q in task_requests}):
                            try:release.append(ctl('release',sid))
                            except Exception:release.append(dict(error=traceback.format_exc(),released=False))
                    results=list((RESULTS/name).glob('*/result.json'))
                    save(O/'receipts'/(name+'.json'),dict(task=name,exit_code=code,actual_parent_wait=getattr(child,'original_parent',True),actual_process_handle_wait=(not isinstance(code,str)),at=now(),
                        requests=len(task_requests),release=release,result_paths=[str(p) for p in results],
                        transport_errors=[str(f.exception()) for f in task_futures if f.exception()]))
                    completed.append(name);del active[name]
                    host_admission.release(H.name,name)
                    if code!=0 and not results:save(O/'pre_agent_failures'/(name+'.json'),dict(task=name,exit_code=code,model_requests=len(task_requests),automatic_retry=False,reason='Harbor exited without a trial result; independent tasks continue'))
                for path in sorted(BOX.glob('*.request.json')):
                    if path.name in seen:continue
                    q=json.loads(path.read_text())
                    if q['task'] not in active:
                        save(BOX/(q['request_id']+'.error.json'),dict(error='Request arrived after task/owner closure; no model request replayed',at=now(),no_automatic_request_retry=True))
                        seen.add(path.name);continue
                    seen.add(path.name);futures[path.name]=(q['task'],pool.submit(handle,path,active[q['task']]['child']))
                healthy=control.tick()
                enabled=bool(pending) and len(active)<P['concurrent_tasks'] and not (O/'pause_new_tasks.json').exists()
                row=control.admit(pending[0] if enabled else None,healthy,enabled)
                if row is not None:
                    assert pending[0]==row
                    pending.pop(0)
                    try:active[row['task']]=start_task(row)
                    except BaseException:host_admission.release(H.name,row['task']);raise
                if time.monotonic()-last_status>20:
                    state=('waiting_remote_health' if not healthy else ('paused_storage_new_tasks' if (O/'pause_new_tasks.json').exists() else ('harbor_tasks_running' if active else 'waiting_host_capacity')))
                    save(O/'status.json',dict(state=state,at=now(),remote_health_age_seconds=time.monotonic()-control.last_healthy,
                        active={k:dict(pid=v['child'].pid,memory_mb=v['memory_mb']) for k,v in active.items()},completed=completed,pending=[x['task'] for x in pending]))
                    last_status=time.monotonic()
                time.sleep(.2)
        save(O/'harbor_receipt.json',dict(all_task_parents_waited=all(json.loads(p.read_text()).get('actual_parent_wait',False) for p in (O/'receipts').glob('*.json')),all_task_launches_accounted=True,tasks=len(completed),at=now()))
    except BaseException:
        save(O/'controller_failure.json',dict(error=traceback.format_exc(),at=now(),active={k:v['child'].pid for k,v in active.items()}));raise
    finally:
        if control is not None:control.close()
        fair_share.update(False,save)
        # Never shut down a service while a task can still be generating.
        if ready and all(x['child'].poll() is not None for x in active.values()):
            try:save(O/'stop_receipt.json',ctl('stop','successor' if P.get('adopts_existing_service') else None))
            except Exception:save(O/'stop_error.json',dict(error=traceback.format_exc()))
    save(O/'owner_complete.json',dict(at=now(),all_task_launches_accounted=True,actual_child_wait=all(json.loads(p.read_text()).get('actual_parent_wait',False) for p in (O/'receipts').glob('*.json'))))
if __name__=='__main__':main()
