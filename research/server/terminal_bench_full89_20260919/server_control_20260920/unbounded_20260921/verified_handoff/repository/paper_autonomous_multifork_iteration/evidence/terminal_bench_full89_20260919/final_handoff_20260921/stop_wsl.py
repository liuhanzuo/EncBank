"""Stop only local Harbor processes for the two frozen old Encbank roots."""
from pathlib import Path
import datetime,json,os,signal,time
H=Path(__file__).resolve().parent
NAMES=['encbank_k12_no_task_deadline_r6_20260920','encbank_k48_no_task_deadline_r6_20260920']
def snapshot():
 out={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   out[int(p.name)]=dict(pid=int(p.name),ppid=int(tail[1]),state=tail[0],starttime=tail[19],command=cmd)
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return out
before=snapshot();roots={pid for pid,p in before.items() if '/harbor run -c ' in p['command'] and any('/'+n+'/execution/configs/' in p['command'] for n in NAMES)}
targets=set(roots)
while True:
 new={pid for pid,p in before.items() if p['ppid'] in targets}-targets
 if not new:break
 targets.update(new)
assert os.getpid() not in targets
record=dict(at=datetime.datetime.now().astimezone().isoformat(),roots=sorted(roots),targets=[before[p] for p in sorted(targets)],signals=[],actual_parent_wait=False,remote_mutations=0)
def live():
 s=snapshot();return {pid:p for pid,p in s.items() if pid in targets and p['starttime']==before[pid]['starttime'] and p['state']!='Z'}
for sig in [signal.SIGTERM,signal.SIGKILL]:
 for pid,p in live().items():
  try:os.kill(pid,sig);record['signals'].append(dict(pid=pid,signal=int(sig)))
  except ProcessLookupError:pass
 for _ in range(20):
  if not live():break
  time.sleep(.25)
record['remaining']=list(live().values());record['status']='PASS' if not record['remaining'] else 'FAILED'
(H/'wsl_stop_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],roots=len(roots),targets=len(targets),remaining=record['remaining'])))
assert not record['remaining']
