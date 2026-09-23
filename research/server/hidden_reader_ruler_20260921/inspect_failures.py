import json
from pathlib import Path
R=Path(__file__).resolve().parent
for cell in ['multikey8k','vt8k']:
    rows=[json.loads(x) for x in (R/cell/'results/predictions.jsonl').read_text().splitlines()]
    labels=json.loads((R/'scoring_only'/(cell+'.json')).read_text());shown=0
    for x in rows:
        refs=labels[x['id']];a=x['arms']['comem']['prediction'];b=x['arms']['kd_selected']['prediction']
        if all(r.lower() in a.lower() for r in refs) and not all(r.lower() in b.lower() for r in refs):
            print(json.dumps(dict(cell=cell,id=x['id'],references=refs,comem=a,kd640=b,kd2048=x['arms']['kd2048']['prediction'],
                stop=x['arms']['kd_selected']['stop_reason']),ensure_ascii=False));shown+=1
            if shown==3:break
