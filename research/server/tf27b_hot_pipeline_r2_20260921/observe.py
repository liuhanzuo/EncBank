"""Read-only snapshot of this research experiment; never submits or retries."""
import hashlib,json,subprocess,time
from pathlib import Path
B=Path('/srv/encbank/qencbank_align_codex_20260911')
def read(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return None
out=dict(epoch=time.time(),queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True),qualification={},pipeline={},trials=[])
for root in sorted(B.glob('tf27b_hot_qualification_*_20260921')):
    out['qualification'][root.name]={n:read(root/(n+'.json')) for n in ['submission','qualification_status','qualification_failure','numerical_qualification','batch_qualification','memory_stress','qualification_complete','parent_exit']}
for root in sorted(B.glob('tf27b_hot_pipeline_*_20260921')):
    out['pipeline'][root.name]={n:read(root/(n+'.json')) for n in ['submission','parent_exit']}
live=B/'tf27b_hot_live_20260921'
for row in read(live/'submissions.json') or []:
    r=Path(row['root']);item=dict(row)
    item['worker']={n:read(r/'worker'/(n+'.json')) for n in ['ready','status','failure','complete','model_parent_exit','integrated_failure']}
    item['controller']=read(r/'jobs/run_complete.json');results=[]
    for path in (r/'results').glob('*/execution_receipt.json'):
        x=read(path);results.append(dict(task=path.parent.name,receipt=x))
    item['results']=results;item['requests']=len(list((r/'mailbox').glob('*/*.request.json')))
    item['responses']=len(list((r/'mailbox').glob('*/*.response.json')))
    manifest=read(r/'source_manifest.json') or {}
    item['source_sha_ok']=all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
    out['trials'].append(item)
print(json.dumps(out,indent=2))
