"""Retire ONLY our never-started submitter after detecting a resource-list bug."""
import json,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'tf27b_hot_pipeline_r1_20260921'
sub=json.loads((R/'submission.json').read_text());job=sub['job'];assert job=='116800'
state=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
assert 'JobState=PENDING' in state and 'JobName=tf27b-stage-live' in state and 'WorkDir='+str(R) in state,state
assert not (R/'parent_exit.json').exists() and not (B/'tf27b_hot_live_20260921').exists()
res=subprocess.run(['scancel',job],capture_output=True,text=True);assert res.returncode==0,res.stderr
(R/'retirement.json').write_text(json.dumps(dict(job=job,reason='Replace incomplete inherited continuation resource inventory with full official task resource selection',
    verified_never_started=True,pre_state=state,scancel_returncode=res.returncode,epoch=time.time()),indent=2)+'\n')
print('Retired only pending experiment submitter '+job)
