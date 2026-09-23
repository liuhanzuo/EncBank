"""Native Windows SSH/SCP owner; Linux agents use only local files, never interop."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime,hashlib,json,os,subprocess,time,traceback
H=Path(__file__).resolve().parent;R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918')
O=H/'execution';O.mkdir(exist_ok=True);RPC=R/'local_rpc_dense_sdpa_fix2';RPC.mkdir(exist_ok=True)
PY='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
ROOTS=json.loads((H/'transport_roots.json').read_text())
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');t.replace(p)
def run(argv,timeout=40):
 p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
 if p.returncode:raise RuntimeError(json.dumps(dict(argv=argv,exit=p.returncode,stdout=p.stdout[-2000:],stderr=p.stderr[-4000:])))
 return p.stdout
def ctl(arm,action,rid=None):
 cmd=f'{PY} {ROOTS[arm]}/file_bridge.py {action} {arm}'+(' '+rid if rid else '')
 return json.loads(run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',cmd]))
def transfer_request(argv):
 for attempt,delay in enumerate((0,1,2,4,8)):
  if delay:time.sleep(delay)
  try:return run(argv,timeout=60)
  except RuntimeError as exc:
   detail=json.loads(str(exc))
   if 'Remote I/O error' not in detail.get('stderr','') or attempt==4:raise

def handle(arm,req,child):
 d=json.loads(req.read_text(encoding='utf8'));rid=d['request_id'];assert req.name==rid+'.request.json'
 started=time.monotonic();target=req.with_name(rid+'.response.json');errfile=req.with_name(rid+'.error.json')
 try:
  transfer_request(['scp','-o','BatchMode=yes',str(req),'gpu-node1:'+ROOTS[arm]+'/incoming/'+rid+'.json'])
  ctl(arm,'publish',rid)
  cmd=f'{PY} {ROOTS[arm]}/file_bridge.py wait {arm} {rid}'
  wait=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',cmd],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',errors='replace')
  cancel_sent=False
  while wait.poll() is None:
   if not cancel_sent and (req.with_name(rid+'.cancel.json').exists() or child.poll() is not None):
    ctl(arm,'cancel',rid);cancel_sent=True
   if time.monotonic()-started>1020:wait.terminate();raise TimeoutError('Remote request exceeded its recorded deadline')
   time.sleep(.2)
  out,err=wait.communicate()
  if wait.returncode:raise RuntimeError(f'Native SSH wait exit {wait.returncode}: {err[-4000:]}')
  proof=json.loads(out);incoming=target.with_suffix('.download')
  assert proof['path']==ROOTS[arm]+'/run_'+arm+'/mailbox/'+rid+'.response.json'
  run(['scp','-o','BatchMode=yes','gpu-node1:'+proof['path'],str(incoming)],timeout=60)
  raw=incoming.read_bytes();assert hashlib.sha256(raw).hexdigest()==proof['sha256'] and len(raw)==proof['bytes']
  incoming.replace(target)
  save(req.with_name(rid+'.broker.json'),dict(status='delivered',seconds=time.monotonic()-started,proof=proof,cancel_sent=cancel_sent))
 except BaseException:
  save(errfile,dict(error=traceback.format_exc(),at=now(),no_automatic_request_retry=True));raise
def arm_run(arm):
 status=O/(arm+'_status.json');box=RPC/arm;box.mkdir(exist_ok=True);child=None
 try:
  begin=time.monotonic()
  while True:
   v=ctl(arm,'status')
   if 'worker_failure.json' in v:raise RuntimeError(v['worker_failure.json'])
   if 'worker_ready.json' in v and 'process_receipt.json' not in v:break
   if 'process_receipt.json' in v:raise RuntimeError('GPU worker already exited; reconcile, do not resubmit')
   save(status,dict(state='waiting_gpu_worker',at=now()))
   if time.monotonic()-begin>7200:raise TimeoutError('Two-hour bounded queue wait exceeded')
   time.sleep(20)
  save(O/(arm+'_worker_ready.json'),v['worker_ready.json'])
  wh='/srv/encbank/legacy_workspace/'+H.relative_to(Path('/srv/encbank/legacy_workspace')).as_posix()
  command=f'cd /srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918 && DOCKER_HOST=unix:///var/run/docker.sock LITELLM_LOCAL_MODEL_COST_MAP=True GIT_CEILING_DIRECTORIES=/srv/encbank/legacy_workspace/.runtime PYTHONPATH={wh} /srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918/harbor_env/bin/harbor run -c {wh}/{arm}_harbor_rpc.json'
  argv=['wsl','-d','Ubuntu','--','bash','-lc',command]
  with (O/(arm+'.stdout.log')).open('wb') as out,(O/(arm+'.stderr.log')).open('wb') as err:
   child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
   save(status,dict(state='harbor_running',pid=child.pid,argv=argv,started_at=now()))
   seen=set()
   while child.poll() is None:
    for req in sorted(box.glob('*.request.json')):
     if req.name in seen:continue
     seen.add(req.name)
     try:handle(arm,req,child)
     except Exception:pass  # Error is delivered to the current task; never replay a request.
    time.sleep(.2)
   code=child.wait()
  save(O/(arm+'_harbor_receipt.json'),dict(exit_code=code,actual_parent_wait=True,ended_at=now(),requests_seen=len(seen)))
  save(status,dict(state='harbor_exited_results_require_validation',exit_code=code,ended_at=now(),result_path=str(R/'results'/'dense_sdpa_fix2_rpc'/'result.json')))
 except BaseException:
  save(status,dict(state='controller_failure_no_retry',error=traceback.format_exc(),at=now()));raise
 finally:
  if child is not None and child.poll() is not None:
   try:save(O/(arm+'_stop_receipt.json'),ctl(arm,'stop'))
   except Exception:save(O/(arm+'_stop_error.json'),dict(error=traceback.format_exc()))
if __name__=='__main__':
 assert not (O/'owner_registration.json').exists(),'Already registered; never duplicate'
 record=dict(directory='dense_sdpa_fix2/execution',pid=os.getpid(),host='Windows',started_at=now(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 save(O/'owner_registration.json',record);save(H.parent/'controller_current.json',record)
 outcomes={}
 with ThreadPoolExecutor(max_workers=1) as pool:
  futures={pool.submit(arm_run,a):a for a in ['dense']}
  for f in as_completed(futures):
   a=futures[f]
   try:f.result();outcomes[a]='controller_closed'
   except BaseException:outcomes[a]=traceback.format_exc()
 save(O/'owner_complete.json',dict(ended_at=now(),outcomes=outcomes))
