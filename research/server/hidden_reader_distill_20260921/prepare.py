import datetime,fcntl,hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
assert Path('/srv/encbank') in R.resolve().parents
assert not (R/'submissions.json').exists()
prior=R.parent/'hidden_reader_pilot_20260921'
shutil.copytree(prior/'vendor',R/'vendor')
for name in ['dataset.json','dataset_manifest.json']:shutil.copyfile(prior/name,R/name)
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records=[]
with (R.parent/'hidden_reader_distill_submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
    assert 'hreader-kd-' not in queue
    (R/'queue_before.txt').write_text(queue)
    for arm in ['lr1e5','lr5e5']:
        out=R/arm;out.mkdir()
        command=['sbatch','--parsable','--no-requeue','--job-name=hreader-kd-'+arm,
            '--partition=gpu','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1',
            '--cpus-per-task=8','--mem=96G','--time=01:00:00','--chdir='+str(R),
            '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err')]
        script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(R/'launch.py')+' '+arm+'\n'
        (out/'job.sh').write_text(script)
        (out/'submission_intent.json').write_text(json.dumps(dict(command=command,at=datetime.datetime.now().astimezone().isoformat()),indent=2))
        p=subprocess.run(command,input=script,text=True,capture_output=True,timeout=45)
        record=dict(arm=arm,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,command=command)
        if p.returncode==0:record['job']=p.stdout.strip().split(';')[0]
        records.append(record);(R/'submissions.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True);assert p.returncode==0,p.stderr
