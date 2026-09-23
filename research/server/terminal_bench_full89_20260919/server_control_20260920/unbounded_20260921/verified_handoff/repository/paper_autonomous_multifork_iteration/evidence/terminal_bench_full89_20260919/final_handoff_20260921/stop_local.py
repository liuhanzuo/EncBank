"""User-authorized local controller closure only; never stops remote services."""
from pathlib import Path
import ctypes,datetime,hashlib,json,os,subprocess,time,tomllib
from ctypes import wintypes
H=Path(__file__).resolve().parent;B=H.parent;ROOT=Path('/srv/encbank/legacy_workspace')
NAMES=['comem_k12_no_task_deadline_r6_20260920','comem_k48_no_task_deadline_r6_20260920']
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def load(p):return json.loads(p.read_text(encoding='utf8'))
def ps():
 p=subprocess.run(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine,CreationDate | ConvertTo-Json -Depth 3'],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30);assert p.returncode==0;return json.loads(p.stdout)
def norm(s):return str(s or '').replace('\\','/').lower()
def owned(p):
 c=norm(p['CommandLine']);return p['Name'].lower()=='python.exe' and any(norm(B/n/'windows_owner.py') in c for n in NAMES)
def monitor(p):return p['Name'].lower()=='python.exe' and norm(B/'runtime_binding_recovery_20260920/resource_monitor.py') in norm(p['CommandLine'])
cfg=tomllib.loads((Path(os.environ.get('CODEX_HOME','/srv/encbank/client/.codex'))/'automations/q-comem/automation.toml').read_text(encoding='utf8'));assert cfg['status']=='PAUSED'
save(H/'automation_paused.json',dict(at=now(),id=cfg['id'],status=cfg['status'],prompt=cfg['prompt']))
before=ps();targets=[p for p in before if owned(p) or monitor(p)];assert {45952,39996}<={p['ProcessId'] for p in targets}
snap={}
for name in NAMES:
 d=B/name;owner=load(d/'execution/owner_registration.json');assert owner['source_sha256']==hashlib.sha256((d/'windows_owner.py').read_bytes()).hexdigest()
 snap[name]={n:load(d/'execution'/n) for n in ['status.json','owner_registration.json','transport_snapshot.json'] if (d/'execution'/n).exists()}
 old=d/'execution/pause_new_tasks.json';snap[name]['previous_pause']=load(old) if old.exists() else None
 save(old,dict(at=now(),owner=H.name,reason='User-directed final handoff; permanently disable old continuation',resume_authorized=False))
 save(d/'execution/HANDOFF_STOPPED_DO_NOT_RESUME.json',dict(at=now(),reason='User-directed local handoff stop',remote_stop_owner='user/new computer',final_handoff=str(H)))
save(H/'before_stop.json',dict(at=now(),targets=targets,campaigns=snap,processes=[p for p in before if any(n in str(p['CommandLine']) for n in NAMES) or monitor(p)]))
k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.CloseHandle.argtypes=[wintypes.HANDLE]
closed=[]
# The registered Python owners go first, preventing any cancel/stop calls in their finally blocks.
targets.sort(key=lambda p:0 if p['ProcessId'] in [45952,39996] else 1)
for p in targets:
 fresh=next((r for r in ps() if r['ProcessId']==p['ProcessId']),None)
 if fresh is None:closed.append(dict(pid=p['ProcessId'],already_exited=True,actual_parent_wait=False));continue
 assert fresh['CommandLine']==p['CommandLine'] and fresh['CreationDate']==p['CreationDate']
 h=k.OpenProcess(0x100000|0x1000|1,False,p['ProcessId']);assert h
 try:
  assert k.TerminateProcess(h,75);assert k.WaitForSingleObject(h,10000)==0
  code=wintypes.DWORD();assert k.GetExitCodeProcess(h,ctypes.byref(code))
 finally:k.CloseHandle(h)
 closed.append(dict(pid=p['ProcessId'],command=p['CommandLine'],exit_code=code.value,actual_process_handle_wait=True,actual_parent_wait=False,reason='User-directed external termination, not scientific completion'))
 save(H/'windows_stop_receipt.json',dict(at=now(),closed=closed,remote_mutations=0))
after=ps();assert not any(owned(p) or monitor(p) for p in after)
save(H/'windows_after_stop.json',dict(at=now(),old_controllers_absent=True,old_monitor_absent=True,processes=[p for p in after if any(n in str(p['CommandLine']) for n in NAMES)]))
print(json.dumps(dict(status='STOPPED',closed=[x['pid'] for x in closed],remote_mutations=0)))
