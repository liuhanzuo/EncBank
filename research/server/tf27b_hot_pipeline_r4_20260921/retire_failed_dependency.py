"""Remove our never-started downstream job after its numerical gate failed."""
import json,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'tf27b_hot_pipeline_r2_20260921';Q=B/'tf27b_hot_qualification_r2_20260921'
assert json.loads((Q/'parent_exit.json').read_text())['exit_code']==1
job=json.loads((R/'submission.json').read_text())['job'];assert job=='116827'
state=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
assert 'JobState=PENDING' in state and 'Reason=DependencyNeverSatisfied' in state and 'WorkDir='+str(R) in state
assert not (R/'parent_exit.json').exists() and not (B/'tf27b_hot_live_20260921').exists()
res=subprocess.run(['scancel',job],capture_output=True,text=True);assert res.returncode==0,res.stderr
(R/'retirement.json').write_text(json.dumps(dict(job=job,verified_never_started=True,
    reason='Numerical batch-equivalence qualification failed; keep all evidence and diagnose before new live jobs',
    pre_state=state,scancel_returncode=res.returncode,epoch=time.time()),indent=2)+'\n')
print('Retired unstartable dependent submitter '+job)
