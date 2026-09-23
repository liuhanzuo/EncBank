"""Queue the resumed LoCoMo scope once, respecting four GPUs across both arrays."""
import datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
launch=json.loads((root/'launch.json').read_text())
assert launch['evaluation_array']=='45014'
assert not launch.get('locomo_array')
receipt=root/'locomo_submission.json'
assert not receipt.exists(), 'Inspect any existing attempt before retrying'
q=subprocess.run(['squeue','-u','liuhanzuo','-n','midcache-formal-locomo','-h','-o','%i %T'],capture_output=True,text=True,check=True)
assert not q.stdout.strip(),q.stdout
for model in ('Qwen3.5-9B','Qwen3.8-27B'):
    assert json.loads((root/'training'/model/'complete.json').read_text())['steps']==4000
    assert json.loads((root/'samples'/model/'complete.json').read_text())['samples']==7236
assert not list((root/'results_locomo').glob('*/shard*/predictions.jsonl'))
cmd=['sbatch','--parsable','--dependency=afterok:'+launch['evaluation_array'],'locomo.slurm']
record=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,
    total_gpu_limit=4,reason='Four-benchmark array runs first at throttle4; LoCoMo array starts only after it succeeds, also at throttle4.')
receipt.write_text(json.dumps(record,indent=2)+'\n')
r=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
receipt.write_text(json.dumps(record,indent=2)+'\n')
assert r.returncode==0,r.stderr
job=r.stdout.strip().split(';')[0];assert job.isdigit()
launch.update(locomo_array=job,maximum_concurrent_gpus=4,status='EVALUATING_FOUR_BENCHMARKS_WITH_LOCOMO_QUEUED',
    requested_benchmarks=['ruler','longeval','longbench','babilong','locomo'],
    judge_model='gpt-6-astra',locomo_scope_id='locomo-resumed-astra-20260916',
    total_requested_records=101304)
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(record|dict(locomo_array=job)))
