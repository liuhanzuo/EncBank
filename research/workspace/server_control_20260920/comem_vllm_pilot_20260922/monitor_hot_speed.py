"""Read-only snapshot for the isolated Hot/PEFT throughput pilot."""
import json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
SCRIPT=r'''
import json,subprocess,time
from pathlib import Path
h=Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/hot_peft_r1')
def read(p):return json.loads(p.read_text()) if p.exists() else None
job=read(h/'submission.json')['job_id']
r=dict(epoch=time.time(),job_id=job,queue=subprocess.check_output(['squeue','-r','-h','-j',job,'-o','%i|%T|%N|%M|%l'],text=True),accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,Timelimit,NodeList'],text=True),active=read(h/'active_case.json'),completed=[],partial=[],failure=read(h/'failure.json'),complete=read(h/'complete.json'))
for case in read(h/'protocol.json')['runs']:
 p=h/'results'/case['id'];result=read(p/'result.json');ready=read(p/'model_ready.json')
 row=dict(case=case,ready=bool(ready),merge_seconds=(ready or {}).get('merge_seconds'),parent=read(h/(case['id']+'.parent.json')),calls=[])
 for n in [1,2]:
  d=read(p/f'call_{n}.json')
  if d:row['calls'].append({k:d[k] for k in ['call_index','batch','prefill_wall_seconds','decode_and_refresh_wall_seconds','row_tokens_per_second','aggregate_tokens_per_second','peak_allocated_gib','peak_reserved_gib','existing_h_unchanged']})
 if result:r['completed'].append(row)
 elif ready:r['partial'].append(row)
if r['failure']:
 rid=r['active']['case']['id']
 r['error_tail']=(h/(rid+'.stderr.log')).read_text()[-8000:]
 r['stdout_tail']=(h/(rid+'.stdout.log')).read_text()[-8000:]
print(json.dumps(r))
'''
r=subprocess.run(['ssh','gpu-node4','/srv/encbank/qcomem_runtime_20260911/python312/bin/python','-'],input=SCRIPT,text=True,encoding='utf-8',capture_output=True,check=True)
d=json.loads(r.stdout)
(P/'attempts/hot_peft_r1/progress_snapshot.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps(d,ensure_ascii=False))
