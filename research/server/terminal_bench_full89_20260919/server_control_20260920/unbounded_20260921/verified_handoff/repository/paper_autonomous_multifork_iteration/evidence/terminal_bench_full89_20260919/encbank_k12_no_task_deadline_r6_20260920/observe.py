"""Read-only capacity continuation and per-task transport/verification status."""
from pathlib import Path
from collections import Counter
import ctypes,datetime,hashlib,json,shlex,subprocess
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
k.CloseHandle.argtypes=[wintypes.HANDLE]
def alive(pid):
    handle=k.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        code=wintypes.DWORD();assert k.GetExitCodeProcess(handle,ctypes.byref(code));return code.value==259
    finally:k.CloseHandle(handle)
H=Path(__file__).resolve().parent;ROOT=Path('/srv/encbank/legacy_workspace');RT=ROOT/'.runtime/terminal_bench_full89_20260919'
PY='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
configs={}
for name,arm,job in [(H.name,'encbank',json.loads((H/'submission.json').read_text())['job_id'])]:
    h=H.parent/name;p=json.loads((h/'plan.json').read_text());box=RT/('local_rpc_'+name)/arm
    reqs={x.stem.removesuffix('.request'):json.loads(x.read_text()) for x in box.glob('*.request.json')}
    configs[name]=dict(remote=p['remote_root'],arm=arm,job=job,request_ids=list(reqs))
code=f'''from pathlib import Path
from collections import deque,Counter
import hashlib,json,subprocess
configs={configs!r};out={{}}
for name,p in configs.items():
 h=Path(p['remote']);r=h/('run_'+p['arm']);d={{}}
 for n in ['process_start.json','worker_ready.json','worker_failure.json','process_receipt.json','memory_cap_failure.json','worker_complete.json','capacity_admission.json','admission.json','engine_args.json']:
  if (r/n).exists():d[n]=json.loads((r/n).read_text())
 for n in ['events.jsonl','owned_nvml.jsonl','worker.stdout.log','worker.stderr.log']:
  if (r/n).exists():
   with (r/n).open() as f:d[n]=list(deque(f,maxlen=6))
 counts=Counter();tokens=0
 for rid in p['request_ids']:
  path=r/'mailbox'/(rid+'.response.json')
  if path.exists():
   q=json.loads(path.read_text());counts[q['status']]+=1;tokens+=q.get('generated_tokens',0)
 d['responses']=dict(counts);d['generated_tokens']=tokens
 if not (r/'process_receipt.json').exists() and (r/'transport_endpoint.json').exists():
  live=subprocess.run(['/srv/encbank/qencbank_runtime_20260911/python312/bin/python',str(h/'file_bridge.py'),'status',p['arm']],capture_output=True,text=True,timeout=40)
  if live.returncode==0:
   d['live_transport']=json.loads(live.stdout).get('transport',{{}});d['service_lifetime_responses']=d['live_transport'].get('response_counts',{{}});d['service_lifetime_generated_tokens']=d['live_transport'].get('generated_tokens',0)
  else:d['live_transport_error']=live.stderr[-2000:]
 d['worker_sha256']=hashlib.sha256((h/'agent_worker.py').read_bytes()).hexdigest()
 d['bridge_sha256']=hashlib.sha256((h/'file_bridge.py').read_bytes()).hexdigest()
 d['accounting']=subprocess.check_output(['sacct','-j',p['job'],'-n','-P','--format=JobID,State,ExitCode,Elapsed,NodeList'],text=True)
 out[name]=d
out['queue']=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%M|%R'],text=True)
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','gpu-node1',shlex.join([PY,'-c',code])],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=90);assert r.returncode==0,r.stderr
out=dict(at=datetime.datetime.now().astimezone().isoformat(),remote=json.loads(r.stdout),local={})
for name,c in configs.items():
    h=H.parent/name;box=RT/('local_rpc_'+name)/c['arm'];d={}
    for n in ['native_launch.json','execution/owner_registration.json','execution/resume_registration.json','execution/status.json','execution/host_admission.json','execution/controller_failure.json','execution/harbor_receipt.json','execution/owner_complete.json']:
        if (h/n).exists():d[n]=json.loads((h/n).read_text())
    registration=d.get('execution/resume_registration.json',d.get('execution/owner_registration.json',{}))
    d['current_controller_pid']=registration.get('pid')
    d['current_controller_pid_running']=alive(registration['pid']) if registration.get('pid') else False
    errors=[]
    for p in box.glob('*.error.json'):
        q=json.loads(p.with_name(p.name.replace('.error.','.request.')).read_text());e=json.loads(p.read_text())
        errors.append(dict(task=q['task'],request_id=q['request_id'],at=e.get('at'),error=e['error']))
    d['transport_errors']=errors;d['delivered_responses']=len(list(box.glob('*.broker.json')))
    cfg=json.loads((h/('dense_harbor_template.json' if c['arm']=='dense' else 'encbank_harbor_template.json')).read_text())
    # Template jobs_dir is overridden by owner per-task to its registered group.
    group=name
    rows=[]
    for p in (RT/'results'/group).glob('*/*/result.json'):
        v=json.loads(p.read_text());e=v.get('exception_info') or {};task=p.parent.name.split('__')[0]
        rows.append(dict(task=task,trial=p.parent.name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            verifier=v.get('verifier_result'),exception_type=e.get('exception_type'),exception_message=e.get('exception_message'),
            agent_execution=v.get('agent_execution'),transport_failure=any(x['task']==task for x in errors)))
    d['closed_trials']=rows;d['pre_agent_no_result_failures']=[json.loads(p.read_text()) for p in (h/'execution/pre_agent_failures').glob('*.json')]
    out['local'][name]=d
out['host_reservations']=json.loads((RT/'host_admission/reservations.json').read_text())
(H/'latest_observation.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
brief={}
for name in configs:
    d=out['local'][name];r=out['remote'][name];st=d.get('execution/status.json',{})
    r['responses']=dict(Counter(json.loads(p.read_text())['status'] for p in (RT/('local_rpc_'+name)/configs[name]['arm']).glob('*.response.json')))
    brief[name]=dict(accounting=r['accounting'].splitlines()[0] if r['accounting'].splitlines() else '',ready='worker_ready.json' in r,
        worker_failure='worker_failure.json' in r,worker_receipt=r.get('process_receipt.json'),controller_pid=d['current_controller_pid'],controller_running=d['current_controller_pid_running'],
        controller_state=st.get('state'),active=list(st.get('active',{})),delivered=d['delivered_responses'],
        responses=r['responses'],transport_errors=len(d['transport_errors']),closed=len(d['closed_trials']))
print(json.dumps(dict(at=out['at'],services=brief),ensure_ascii=False,indent=2))
