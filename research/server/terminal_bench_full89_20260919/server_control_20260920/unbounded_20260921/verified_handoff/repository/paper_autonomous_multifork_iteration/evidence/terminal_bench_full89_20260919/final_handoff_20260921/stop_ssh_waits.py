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

before=ps();targets=[p for p in before if p["Name"].lower()=="ssh.exe" and p["ParentProcessId"] in [45952,39996] and any("/"+n+"/file_bridge.py wait comem " in str(p["CommandLine"]) for n in NAMES)]
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
 save(H/'ssh_wait_stop_receipt.json',dict(at=now(),closed=closed,remote_mutations=0))

after=ps();assert not any(p["Name"].lower()=="ssh.exe" and p["ParentProcessId"] in [45952,39996] for p in after)
print(json.dumps(dict(closed=len(closed),status="PASS",remote_worker_actions=0)))
