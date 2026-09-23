"""Read-only live observations. Never launch, restart, cancel or modify runs."""
from pathlib import Path
import datetime,json,shlex,subprocess
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());REMOTE=P['remote_root']
PY='/srv/encbank/qcomem_runtime_20260911/python312/bin/python'
def run(args):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
    assert p.returncode==0,p.stderr;return p.stdout
code=f'''from pathlib import Path
from collections import deque,Counter
import json,subprocess
h=Path({REMOTE!r});r=h/'run_dense';out={{}}
for n in ['submission.json','cpu_preflight_remote.json']:
 if (h/n).exists():out[n]=json.loads((h/n).read_text())
if 'cpu_preflight_remote.json' in out:out['cpu_preflight_remote.json'].pop('files',None)
for n in ['process_start.json','worker_ready.json','worker_failure.json','memory_cap_failure.json','process_receipt.json','worker_complete.json']:
 if (r/n).exists():out[n]=json.loads((r/n).read_text())
for n in ['worker.stdout.log','worker.stderr.log','events.jsonl','owned_nvml.jsonl']:
 if (r/n).exists():
  with (r/n).open(errors='replace') as f:out[n]=list(deque(f,maxlen=15))
counts=Counter();total_tokens=0
for p in (r/'mailbox').glob('*.response.json'):
 d=json.loads(p.read_text());counts[d['status']]+=1;total_tokens+=d.get('generated_tokens',0)
out['responses']=dict(counts);out['generated_tokens']=total_tokens
out['queue']=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%M|%R'],text=True)
if 'submission.json' in out:out['accounting']=subprocess.check_output(['sacct','-j',out['submission.json']['job_id'],'-n','-P','--format=JobID,State,ExitCode,Elapsed,NodeList'],text=True)
print(json.dumps(out))
'''
result=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),remote=json.loads(run(['ssh','-o','BatchMode=yes','gpu-node1',shlex.join([PY,'-c',code])])),local={})
for n in ['native_launch.json','execution/status.json','execution/controller_failure.json','execution/harbor_receipt.json','execution/owner_complete.json']:
    if (H/n).exists():result['local'][n]=json.loads((H/n).read_text())
R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/results/dense_vllm_batch2_headers_fix')
rows=[]
for p in R.glob('*/result.json'):
    d=json.loads(p.read_text());rows.append(dict(trial=p.parent.name,verifier=d.get('verifier_result'),exception=d.get('exception_info'),agent_started=d.get('agent_execution',{})))
result['closed_trials']=rows
result['images']={s:sum(json.loads(p.read_text()).get('status')==s for p in (H.parent/'images').glob('*.json') if p.name not in ['completion.json','owner.json']) for s in ['PASS','FAIL']}
result['images']['recovered']=sum(json.loads(p.read_text()).get('status')=='PASS' for p in (H.parent/'images_recovery').glob('*.json'))
result['images']['effective_pass']=result['images']['PASS']+result['images']['recovered']
(H/'latest_observation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
