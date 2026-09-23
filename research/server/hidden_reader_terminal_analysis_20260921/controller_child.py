"""One task, two fresh trials; runs inside its model's allocated Slurm job."""
import json,sys
import controller
from common import ROOT,PLAN as P,save,verify_sources
task=sys.argv[1];verify_sources()
controller.QUALIFY=True
code=controller.trial(task,'qualify')
assert code==0,'Actual-node environment qualification failed'
receipt=json.loads((ROOT/'qualification'/(task+'--qualify')/'execution_receipt.json').read_text())
assert receipt['closure']['cgroup_empty'] and receipt['model_calls']==0 and not receipt['exception']
controller.QUALIFY=False
rows=[];index=P['global_task_order'].index(task)
for arm in (P['arms'] if index%2==0 else list(reversed(P['arms']))):
    code=controller.trial(task,arm);rows.append(dict(arm=arm,exit_code=code))
    if (ROOT/'pairs'/task/'worker_failure.json').exists():break
save(ROOT/'jobs'/('step-trials-'+task+'.json'),dict(task=task,trials=rows))
assert len(rows)==2 and all(r['exit_code']==0 for r in rows),rows
