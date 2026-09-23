from pathlib import Path
import json,subprocess
root=Path('/srv/encbank/comem_frozen_j12_20260912')
queue=subprocess.run(['squeue','-j','25633','-h','-o','%i %t %M %N %R'],text=True,capture_output=True)
out={'scheduler':queue.stdout.strip(),'scheduler_note':queue.stderr.strip(),'shards':{}}
if queue.returncode or not queue.stdout.strip():
    account=subprocess.run(['sacct','-X','-j','25633','--format=JobID,State,ExitCode,Elapsed,NodeList','-P'],text=True,capture_output=True)
    out['accounting']=account.stdout.strip()
for i in range(4):
    p=root/'results'/f'shard_{i:02d}'
    item={}
    for name in ('progress','complete','correctness'):
        f=p/(name+'.json')
        if f.exists(): item[name]=json.loads(f.read_text())
    f=p/'metadata.json'
    if f.exists():
        m=json.loads(f.read_text()); item['environment']={k:m[k] for k in ('host','gpu_name','compute_capability','torch','transformers')}
    item['closed_cells']=len(list(p.glob('*.summary.json')))
    out['shards'][str(i)]=item
print(json.dumps(out))
