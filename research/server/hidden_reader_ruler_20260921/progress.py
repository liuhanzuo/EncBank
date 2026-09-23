import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
C=R/'numerical_control'
cfg=json.loads((C/'configs.json').read_text())
def state(root,cell):
    out=root/cell/'results';p=out/'predictions.jsonl'
    n=len(p.read_text().splitlines()) if p.exists() else 0
    st=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    rc=json.loads((out/'parent_exit.json').read_text()).get('returncode') if (out/'parent_exit.json').exists() else None
    return dict(items=n,phase=st.get('phase','QUEUED'),exit=rc)
items=0;controls=0
for cell,c in cfg.items():
    primary=state(R if c['control_only'] else C,cell);control=state(C,cell)
    items+=primary['items'];controls+=control['items']
    print(json.dumps(dict(cell=cell,primary=primary,control=control)))
print(json.dumps(dict(primary_items=items,control_items=controls,target_each=900)))
jobs=json.loads((R/'submissions.json').read_text())+json.loads((C/'submissions.json').read_text())
print(subprocess.check_output(['squeue','-j',','.join(j['job'] for j in jobs),'-h','-o','%i|%j|%T|%N|%R'],text=True).strip())
