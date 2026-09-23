"""Submit the two fixed-depth training jobs once; keep a durable receipt."""
import datetime, json, subprocess
from pathlib import Path
root = Path(__file__).resolve().parent
receipt = root/'training_submission.json'
assert not receipt.exists(), 'Submission already attempted; inspect receipt and Slurm before retry'
for command in (['squeue','-u','liuhanzuo','-n','midcache-formal-train','-h','-o','%i %T'],
                ['sacct','-u','liuhanzuo','--name=midcache-formal-train','--starttime=2026-09-15','-n','-P','--format=JobID,State']):
    check = subprocess.run(command, capture_output=True, text=True, check=True)
    assert not check.stdout.strip(), check.stdout
record = {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'script':'train.slurm',
          'splits':{'Qwen3.5-9B':6,'Qwen3.8-27B':21}, 'steps':4000, 'maximum_concurrent_gpus':2}
receipt.write_text(json.dumps(record, indent=2)+'\n')
result = subprocess.run(['sbatch','--parsable','train.slurm'], cwd=root, text=True, capture_output=True)
record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
receipt.write_text(json.dumps(record, indent=2)+'\n')
assert result.returncode == 0, result.stderr
job = result.stdout.strip().split(';')[0]
assert job.isdigit(), result.stdout
launch = {'training_array':job, 'evaluation_array':None, 'maximum_concurrent_gpus':2,
          'training_steps_per_model':4000, 'fixed_splits':record['splits'], 'status':'TRAINING_SUBMITTED'}
(root/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
print(json.dumps(launch))
