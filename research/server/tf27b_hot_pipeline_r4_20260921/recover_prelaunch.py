"""Preserve a failed, never-submitted staging directory before a fresh launch."""
import json,subprocess,time
from pathlib import Path
S=Path(__file__).resolve().parent;B=S.parent
T=B/'tf27b_hot_live_20260921';A=B/'tf27b_hot_live_prelaunch_failure_119855_20260922'
Q=B/'tf27b_hot_qualification_r4_20260921';failed=B/'tf27b_hot_pipeline_r3_20260921'
assert json.loads((failed/'submission.json').read_text())['job']=='119855'
assert json.loads((failed/'parent_exit.json').read_text())['exit_code']==1
assert json.loads((S/'task_integrity_preflight.json').read_text())['passed']
assert json.loads((Q/'parent_exit.json').read_text())['exit_code']==0
assert T.resolve().parent==B.resolve() and not A.exists()
assert not (T/'submissions.json').exists()
assert not list(T.rglob('*.request.json')) and not list(T.rglob('*launch*.json'))
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert 'tf27b-' not in queue
T.rename(A)
(A/'prelaunch_retirement.json').write_text(json.dumps(dict(reason='Original manifest predates user-authorized agent timeout removal; no submitted/live trials',queue=queue,epoch=time.time()),indent=2)+'\n')
print(A)
