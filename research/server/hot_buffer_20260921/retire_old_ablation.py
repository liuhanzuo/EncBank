"""Retire only obsolete research ablations after the user's scope correction."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'hidden_reader_terminal_r5_20260921'
reason='User clarified the target is hot-buffer experiments. Retire the obsolete no-hot-buffer depth ablation; preserve all evidence, assign no final quality score.'
rows=[]
for task,root in json.loads((OLD/'roots.json').read_text()).items():
    C=Path(root);pair=C/'pairs'/task;box=C/'mailbox'/task
    if (C/'jobs/integrated_complete.json').exists():continue
    submission=next(r for r in json.loads((OLD/'submissions.json').read_text()) if r['task']==task)
    job=submission['job'];state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T|%j'],text=True).strip()
    if not state:continue
    assert 'bandtb5-'+task in state,state
    receipt=dict(origin='research_orchestrator',reason=reason,epoch=time.time(),job=job,state=state,
        prior_status=json.loads((pair/'status.json').read_text()) if (pair/'status.json').exists() else None)
    (pair/'scope_retirement.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if state.startswith('PENDING|'):subprocess.run(['scancel',job],check=True)
    else:
        marker=pair/'worker_failure.json'
        if not marker.exists():marker.write_text(json.dumps(dict(origin='research_orchestrator',error=reason,not_a_model_exception=True,epoch=time.time()),indent=2)+'\n')
        for request in box.glob('*.request.json'):
            if not request.with_name(request.name.replace('.request.','.response.')).exists():
                request.with_name(request.name.replace('.request.','.cancel.')).write_text(json.dumps(receipt)+'\n')
    rows.append(receipt)
(R/'old_ablation_retirement.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
