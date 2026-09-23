import json,sys
import controller
from common import ROOT,PLAN as P,save,verify_sources
verify_sources();phase=sys.argv[1];task=P['tasks'][0];rows=[]
controller.QUALIFY=phase=='qualify'
arms=['qualify'] if controller.QUALIFY else (P['arms'] if P['global_task_order'].index(task)%2==0 else list(reversed(P['arms'])))
for arm in arms:
    code=controller.trial(task,arm);rows.append(dict(arm=arm,exit_code=code))
    if code:break
save(ROOT/'jobs'/(phase+'_trials.json'),rows)
assert len(rows)==len(arms) and all(r['exit_code']==0 for r in rows),rows
