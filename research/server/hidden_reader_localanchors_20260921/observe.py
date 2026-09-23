import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
jobs=json.loads((root/'submissions.json').read_text())
print(subprocess.check_output(['squeue','-j',','.join(j['job'] for j in jobs),'-h','-o','%i|%j|%T|%N|%R'],text=True).strip())
for job in jobs:
    r=root/job['arm']/'results';view=dict(arm=job['arm'],job=job['job'])
    for name in ['status','parent_exit','failure']:
        p=r/(name+'.json')
        if p.exists():view[name]=json.loads(p.read_text())
    p=r/'validation_selection.json'
    if p.exists():view['validation']=[{k:v for k,v in a.items() if k!='documents'} for a in json.loads(p.read_text())]
    p=r/'test.json'
    if p.exists():
        rows=json.loads(p.read_text());view['test_documents']=len(rows)
        view['mean_kl_so_far']=sum(x['kl'] for x in rows)/len(rows)
    p=r/'runtime.json'
    if p.exists():view['runtime_cells']=len(json.loads(p.read_text()))
    print(json.dumps(view))
