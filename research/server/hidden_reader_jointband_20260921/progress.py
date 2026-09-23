import json
from pathlib import Path
R=Path(__file__).resolve().parent
configs=json.loads((R/'configs.json').read_text())
totals={}
for phase in ['screen','confirm']:
    if not (R/('submissions_'+phase+'.json')).exists():continue
    counts=0;complete=0;failed=[];rows=[]
    for j in json.loads((R/('submissions_'+phase+'.json')).read_text()):
        out=R/j['run']/'results';p=out/'predictions.jsonl'
        n=len(p.read_text().splitlines()) if p.exists() else 0;counts+=n
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        done=(out/'parent_exit.json').exists() and json.loads((out/'parent_exit.json').read_text())['returncode']==0
        complete+=int(done)
        if (out/'failure.json').exists():failed.append(j['run'])
        rows.append(dict(cell=j['cell'],items=n,phase=status.get('phase','QUEUED'),done=done))
    totals[phase]=dict(items=counts,target=288 if phase=='screen' else 612,complete_jobs=complete,failures=failed,cells=rows)
print(json.dumps(totals,ensure_ascii=False))
runtime=R/'screen_single8k/results/runtime.json'
if runtime.exists():
    rows=json.loads(runtime.read_text())
    print(json.dumps(dict(first_cell_reconstruction_ms={str(x['n']):round(x['median_wall_ms'],2) for x in rows if x['chunks']==12})))
