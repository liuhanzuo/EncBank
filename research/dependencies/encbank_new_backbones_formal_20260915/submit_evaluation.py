"""Data preparation on CPU, followed by evaluation after data AND training succeed."""
import datetime, json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
launch=json.loads((root/'launch.json').read_text())
assert launch['training_array'].isdigit() and not launch.get('evaluation_array')
for kind,script in [('data','prepare.slurm'),('evaluation','evaluate.slurm')]:
    receipt=root/f'{kind}_submission.json'
    assert not receipt.exists(), 'Submission previously attempted; inspect before any retry'
    name='midcache-formal-data' if kind=='data' else 'midcache-formal-eval'
    check=subprocess.run(['squeue','-u','liuhanzuo','-n',name,'-h','-o','%i %T'],text=True,capture_output=True,check=True)
    assert not check.stdout.strip(),check.stdout
    cmd=['sbatch','--parsable']
    if kind=='evaluation': cmd.append('--dependency=afterok:'+launch['training_array']+':'+launch['data_array'])
    cmd.append(script)
    record=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    result=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
    record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    assert result.returncode==0,result.stderr
    job=result.stdout.strip().split(';')[0];assert job.isdigit()
    launch[kind+'_array']=job
    launch['status']='TRAINING_AND_DATA_SUBMITTED' if kind=='data' else 'TRAINING_WITH_DEPENDENT_FULL_EVALUATION'
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps({kind+'_array':job,'dependency':cmd[2:-1]}),flush=True)
