"""Retain original attempts; retry failed initial gates and add exact-KV control."""
import datetime,fcntl,hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
assert Path('/srv/encbank') in R.parents
assert not (R/'submissions.json').exists()
configs=json.loads((P/'configs.json').read_text());original=json.loads((P/'submissions.json').read_text())
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
(R/'queue_before.txt').write_text(queue)
for j in original:
    out=P/j['cell']/'results';failed=(out/'failure.json').exists()
    if failed:
        rec=json.loads((out/'parent_exit.json').read_text());assert rec['actual_wait'] and rec['returncode']==1
        assert (out/'predictions.jsonl').stat().st_size==0
        assert not any(line.startswith(j['job']+'|') for line in queue.splitlines())
    configs[j['cell']]['control_only']=not failed
    configs[j['cell']]['original_job']=j['job']
assert sum(not c['control_only'] for c in configs.values())==4
(R/'configs.json').write_text(json.dumps(configs,indent=2)+'\n')
shutil.copytree(P/'vendor',R/'vendor');shutil.copyfile(P/'checkpoints.json',R/'checkpoints.json')
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records=[]
with (P/'numerical_control_submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);assert 'hreader-ctl-' not in queue
    for cell in configs:
        out=R/cell;out.mkdir()
        cmd=['sbatch','--parsable','--no-requeue','--job-name=hreader-ctl-'+cell,'--partition=gpu','--nodes=1','--ntasks=1',
            '--gres=gpu:nvidia_l20d:1','--cpus-per-task=8','--mem=96G','--time=02:00:00','--chdir='+str(R),
            '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err')]
        script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(R/'launch.py')+' '+cell+'\n'
        (out/'job.sh').write_text(script)
        p=subprocess.run(cmd,input=script,text=True,capture_output=True,timeout=45)
        rec=dict(cell=cell,control_only=configs[cell]['control_only'],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,command=cmd)
        if p.returncode==0:rec['job']=p.stdout.strip().split(';')[0]
        records.append(rec);(R/'submissions.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(rec),flush=True)
        assert p.returncode==0,p.stderr
