import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;jobs=[]
for p in R.glob('submissions_*.json'):jobs+=json.loads(p.read_text())
for j in jobs:
    out=R/j['run']/'results';row=dict(run=j['run'],job=j['job'])
    for name in ['status','parent_exit','failure']:
        p=out/(name+'.json')
        if p.exists():row[name]=json.loads(p.read_text())
    print(json.dumps(row))
if jobs:print(subprocess.check_output(['squeue','-j',','.join(j['job'] for j in jobs),'-h','-o','%i|%T|%N|%R'],text=True).strip())
