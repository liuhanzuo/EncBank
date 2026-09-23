import datetime,fcntl,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ['screen','confirm']
records_path=R/('submissions_'+phase+'.json');assert not records_path.exists()
if phase=='confirm':assert (R/'selection.json').exists()
configs=json.loads((R/'configs.json').read_text());records=[]
with (R/'submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
    assert 'hreader-band-'+phase not in queue
    (R/('queue_before_'+phase+'.txt')).write_text(queue)
    for run,c in configs.items():
        if c['phase']!=phase:continue
        out=R/run;out.mkdir()
        cmd=['sbatch','--parsable','--no-requeue','--job-name=hreader-band-'+run,'--partition=gpu',
            '--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1','--cpus-per-task=8','--mem=96G','--time=02:00:00',
            '--chdir='+str(R),'--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err')]
        script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(R/'launch.py')+' '+run+'\n'
        (out/'job.sh').write_text(script)
        (out/'submission_intent.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),command=cmd),indent=2))
        p=subprocess.run(cmd,input=script,text=True,capture_output=True,timeout=45)
        row=dict(run=run,cell=c['cell'],phase=phase,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        if p.returncode==0:row['job']=p.stdout.strip().split(';')[0]
        records.append(row);records_path.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(row),flush=True)
        assert p.returncode==0,p.stderr
