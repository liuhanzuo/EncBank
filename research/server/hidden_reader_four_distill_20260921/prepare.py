"""Expand training only; preserve prior validation/test and add untouched diagnostics."""
import datetime,fcntl,hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
assert Path('/srv/encbank') in R.resolve().parents
assert not (R/'submissions.json').exists()
prior=R.parent/'hidden_reader_localanchors_20260921'
if not (R/'vendor').exists():shutil.copytree(prior/'vendor',R/'vendor')
old=json.loads((prior/'dataset.json').read_text())
source=Path('/srv/encbank/encbank_sparse_slurm_20260912/data/qasper_pilot')
def load(split):
    rows=[json.loads(line) for line in (source/(split+'.jsonl')).read_text().splitlines()]
    eligible={}
    for row in rows:
        chunks=row['document_chunks'];did=row['document_id']
        if len(chunks)>=5 and all(len(c)==512 for c in chunks[:4]) and len(chunks[4])>=129:
            eligible[did]=dict(id=did,memory=chunks[:4],continuation=chunks[4][:129])
    return eligible,{r['document_id'] for r in rows}
train,_=load('train');dev,all_dev=load('dev')
order=lambda did:hashlib.sha256(('hreader-pilot-20260921:'+did).encode()).hexdigest()
oldtrain={r['id'] for r in old['train']};used_dev={r['id'] for s in ['validation','test'] for r in old[s]}
extra=sorted(set(train)-all_dev-oldtrain,key=order)
assert oldtrain.isdisjoint(all_dev)
assert len(extra)>=96,(len(train),len(extra))
fresh=sorted(set(dev)-used_dev,key=order)[:16]
payload=dict(train=old['train']+[train[k] for k in extra[:96]],validation=old['validation'],test=old['test'],fresh_test=[dev[k] for k in fresh])
ids={split:{r['id'] for r in rows} for split,rows in payload.items()}
for a in ids:
    for b in ids:
        if a!=b:assert ids[a].isdisjoint(ids[b]),(a,b)
(R/'dataset.json').write_text(json.dumps(payload)+'\n')
(R/'dataset_manifest.json').write_text(json.dumps(dict(source=str(source),
    source_sha256={s:hashlib.sha256((source/(s+'.jsonl')).read_bytes()).hexdigest() for s in ['train','dev']},
    split_ids={s:[r['id'] for r in rows] for s,rows in payload.items()},
    split_counts={s:len(rows) for s,rows in payload.items()},answers_used=False,document_disjoint=True,
    scope='QASPER body-only internal split; train expanded from32 to128; old4 validation and8 test preserved; remaining eligible dev documents as fresh diagnostic, not official QASPER scores'),indent=2)+'\n')
print(json.dumps(dict(split_counts={s:len(rows) for s,rows in payload.items()})),flush=True)
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
configs=json.loads((R/'configs.json').read_text());records=[]
with (R.parent/'hidden_reader_four_distill_submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
    assert 'hreader-fourkd-' not in queue
    (R/'queue_before.txt').write_text(queue)
    for arm in configs:
        out=R/arm;out.mkdir()
        command=['sbatch','--parsable','--no-requeue','--job-name=hreader-fourkd-'+arm,
            '--partition=gpu','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1',
            '--cpus-per-task=8','--mem=96G','--time=01:00:00','--chdir='+str(R),
            '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err')]
        script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(R/'launch.py')+' '+arm+'\n'
        (out/'job.sh').write_text(script)
        (out/'submission_intent.json').write_text(json.dumps(dict(command=command,at=datetime.datetime.now().astimezone().isoformat()),indent=2))
        p=subprocess.run(command,input=script,text=True,capture_output=True,timeout=45)
        record=dict(arm=arm,config=configs[arm],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,command=command)
        if p.returncode==0:record['job']=p.stdout.strip().split(';')[0]
        records.append(record);(R/'submissions.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True);assert p.returncode==0,p.stderr
