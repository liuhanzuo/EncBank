"""Show completed cells only; depth selection remains in choose_depth.py."""
import json,sys
from pathlib import Path
from official_scoring import string_match_all
R=Path(__file__).resolve().parent
phase=sys.argv[1] if len(sys.argv)>1 else 'screen'
assert phase in ['screen','confirm']
for j in json.loads((R/('submissions_'+phase+'.json')).read_text()):
    out=R/j['run']/'results'
    if not (out/'parent_exit.json').exists():continue
    if json.loads((out/'parent_exit.json').read_text())['returncode']!=0:continue
    rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    labels=json.loads((R/'scoring_only'/(j['cell']+'.json')).read_text())
    arms=json.loads((out/'summary.json').read_text())['arms']
    scores={arm:string_match_all([r['arms'][arm]['prediction'] for r in rows],[labels[r['id']] for r in rows]) for arm in arms}
    print(json.dumps(dict(cell=j['cell'],items=len(rows),scores=scores)))
