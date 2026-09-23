"""Descriptive interim scores for fully completed 100-item cells only."""
import json
from pathlib import Path
from official_scoring import string_match_all
R=Path(__file__).resolve().parent;C=R/'numerical_control'
cfg=json.loads((C/'configs.json').read_text())
for cell,c in cfg.items():
    out=(R if c['control_only'] else C)/cell/'results'
    if not (out/'parent_exit.json').exists():continue
    if json.loads((out/'parent_exit.json').read_text())['returncode']!=0:continue
    rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()];assert len(rows)==100
    labels=json.loads((R/'scoring_only'/(cell+'.json')).read_text())
    scores={arm:string_match_all([r['arms'][arm]['prediction'] for r in rows],[labels[r['id']] for r in rows]) for arm in rows[0]['arms']}
    co=C/cell/'results'
    if c['control_only'] and (co/'parent_exit.json').exists() and json.loads((co/'parent_exit.json').read_text())['returncode']==0:
        controls=[json.loads(x) for x in (co/'predictions.jsonl').read_text().splitlines()];assert len(controls)==100
        scores['comem_split']=string_match_all([r['arms']['comem_split']['prediction'] for r in controls],[labels[r['id']] for r in controls])
    print(json.dumps(dict(cell=cell,items=100,scores=scores,interim_only=True)))
