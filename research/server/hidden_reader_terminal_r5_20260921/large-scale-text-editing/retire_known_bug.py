"""Close only this revision for the identified inference-tensor audit bug."""
import json,subprocess,time
from common import ROOT,PLAN as P,save
jobs=json.loads((ROOT/'jobs'/'submissions_run.json').read_text());actions=[]
for row in jobs:
    if row['kind']!='worker':continue
    task=row['task'];pair=ROOT/'pairs'/task;box=ROOT/'mailbox'/task
    state=subprocess.check_output(['squeue','-j',row['job'],'-h','-o','%T'],text=True).strip()
    if (pair/'parent_exit.json').exists():continue
    reason='Orchestrator retirement: known InferenceTensor _version audit incompatibility. No quality verdict; actual model parent exit is recorded separately.'
    if not (pair/'worker_failure.json').exists():save(pair/'worker_failure.json',dict(error=reason,origin='research_orchestrator',epoch=time.time()))
    for path in box.glob('*.request.json'):
        if not path.with_name(path.name.replace('.request.','.response.')).exists():save(path.with_name(path.name.replace('.request.','.cancel.')),dict(reason=reason,epoch=time.time()))
    save(box/'stop.json',dict(reason=reason,epoch=time.time()))
    if state=='PENDING':
        subprocess.run(['scancel',row['job']],check=True)
        save(pair/'not_started_cancellation.json',dict(job=row['job'],previous_state=state,reason=reason,actual_model_parent_wait=False))
    actions.append(dict(task=task,job=row['job'],state=state,requests=len(list(box.glob('*.request.json')))))
save(ROOT/'jobs'/'known_bug_retirement.json',dict(actions=actions,epoch=time.time()))
print(json.dumps(actions,indent=2))
