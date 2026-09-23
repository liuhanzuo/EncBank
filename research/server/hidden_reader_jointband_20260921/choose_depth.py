"""Choose joint depth on the disjoint screening set only."""
import datetime,hashlib,json
from pathlib import Path
from metrics import load_phase,groups
R=Path(__file__).resolve().parent
assert not (R/'selection.json').exists()
data,outputs=load_phase('screen');scores=groups(data)
candidates=[]
for n in [12,14,16,18,20,24,28,32,36]:
    arm='n'+str(n);task_delta={task:x['vs_native'][arm]['delta_pp'] for task,x in scores['tasks'].items()}
    cell_delta={cell:x['vs_native'][arm]['delta_pp'] for cell,x in scores['cells'].items()}
    candidates.append(dict(n=n,task_delta_pp=task_delta,worst_cell_delta_pp=min(cell_delta.values()),
        eligible=all(d>=-2-1e-8 for d in task_delta.values()) and min(cell_delta.values())>=-5-1e-8))
eligible=[x['n'] for x in candidates if x['eligible']];selected=min(eligible) if eligible else 36
arms=list(dict.fromkeys(['native','n12','n'+str(selected),'n36']))
record=dict(at=datetime.datetime.now().astimezone().isoformat(),selected_n=selected,confirm_arms=arms,
    candidates=candidates,fallback_used=not bool(eligible),
    rule='minimum joint depth with all task-mean losses<=2pp and all cell losses<=5pp on first32/cell only',
    confirmation='remaining68/cell; no confirmation outputs read for selection')
for name,x in [('screen_scores.json',scores),('screen_scored_items.json',outputs),('selection.json',record)]:
    (R/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(selection=record,task_scores={t:x['score'] for t,x in scores['tasks'].items()}),indent=2))
