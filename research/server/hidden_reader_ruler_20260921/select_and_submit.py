"""Choose without benchmark scores, then freeze and submit nine paired cells."""
import datetime,fcntl,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
assert not (R/'submissions.json').exists()
long=R.parent/'hidden_reader_longkd_20260921';candidates=[]
for arm in ['cosine3e6','cosine1e5']:
    out=long/arm/'results'
    assert json.loads((out/'parent_exit.json').read_text())['returncode']==0
    x=json.loads((out/'summary.json').read_text());t=x['distillation']
    candidates.append(dict(arm=arm,validation_kl=t['selected_validation_kl'],total_steps=t['selected_total_steps'],
        path=str(long/arm/'heads.pt'),sha256=x['checkpoint_sha256']))
selected=min(candidates,key=lambda x:x['validation_kl'])
p256=R.parent/'hidden_reader_four_distill_20260921/fixed_lr1e5/heads.pt'
pend=long/selected['arm']/'heads_total2048.pt'
checkpoints=dict(at=datetime.datetime.now().astimezone().isoformat(),selection_metric='held-out validation KL, no benchmark access',
    candidates=candidates,arms=dict(comem=None,kd256=dict(path=str(p256),sha256=sha(p256),total_steps=256),
        kd_selected=selected,kd2048=dict(path=str(pend),sha256=sha(pend),total_steps=2048,arm=selected['arm'])))
save(R/'checkpoints.json',checkpoints)
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
manifest={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
save(R/'source_manifest.json',manifest)
configs=json.loads((R/'configs.json').read_text());records=[]
with (R.parent/'hidden_reader_ruler_submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
    assert 'hreader-ruler-' not in queue
    (R/'queue_before.txt').write_text(queue)
    for cell in configs:
        out=R/cell;out.mkdir()
        cmd=['sbatch','--parsable','--no-requeue','--job-name=hreader-ruler-'+cell,
            '--partition=gpu','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1',
            '--cpus-per-task=8','--mem=96G','--time=02:00:00','--chdir='+str(R),
            '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err')]
        script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(R/'launch.py')+' '+cell+'\n'
        (out/'job.sh').write_text(script);save(out/'submission_intent.json',dict(command=cmd,at=datetime.datetime.now().astimezone().isoformat()))
        p=subprocess.run(cmd,input=script,text=True,capture_output=True,timeout=45)
        record=dict(cell=cell,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        if p.returncode==0:record['job']=p.stdout.strip().split(';')[0]
        records.append(record);save(R/'submissions.json',records);print(json.dumps(record),flush=True)
        assert p.returncode==0,p.stderr
