import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
jobs=json.loads((r/'submissions.json').read_text())
print(subprocess.check_output(['squeue','-j',','.join(x['job'] for x in jobs),'-h','-o','%i|%j|%T|%N|%R'],text=True).strip())
for j in jobs:
    out=r/j['cell']/'results';x=dict(cell=j['cell'],job=j['job'])
    for name in ['status','failure','parent_exit','exact_cache_control','summary']:
        p=out/(name+'.json')
        if p.exists():x[name]=json.loads(p.read_text())
    print(json.dumps(x,ensure_ascii=False))
