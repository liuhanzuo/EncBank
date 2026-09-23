"""Stage body-only, document-disjoint pilot data and submit independent GPU arms."""
import datetime,fcntl,hashlib,json,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
assert Path('/srv/encbank') in R.resolve().parents
assert not (R/'submissions.json').exists(), 'Inspect existing ownership before continuation'
old=R.parent/'kv_dual_depth_probe_20260921/vendor'
if (R/'vendor').exists():
    for p in old.rglob('*'):
        if p.is_file():assert (R/'vendor'/p.relative_to(old)).read_bytes()==p.read_bytes()
else:shutil.copytree(old,R/'vendor',dirs_exist_ok=False)
data=Path('/srv/encbank/encbank_sparse_slurm_20260912/data/qasper_pilot')
def load(split):
    unique={}
    for line in (data/(split+'.jsonl')).read_text().splitlines():
        row=json.loads(line);did=row['document_id'];chunks=row['document_chunks']
        if did in unique or len(chunks)<5 or any(len(c)!=512 for c in chunks[:4]) or len(chunks[4])<129:continue
        # No answer_ids, references, questions or benchmark verifiers are retained.
        unique[did]=dict(id=did,memory=chunks[:4],continuation=chunks[4][:129])
    return unique
train,dev=load('train'),load('dev')
order=lambda did:hashlib.sha256(('hreader-pilot-20260921:'+did).encode()).hexdigest()
devkeys=sorted(dev,key=order)
trainkeys=sorted(set(train)-set(dev),key=order)
assert len(trainkeys)>=32 and len(devkeys)>=12,(len(trainkeys),len(devkeys))
payload=dict(train=[train[k] for k in trainkeys[:32]],validation=[dev[k] for k in devkeys[:4]],
    test=[dev[k] for k in devkeys[4:12]])
(R/'dataset.json').write_text(json.dumps(payload)+'\n')
(R/'dataset_manifest.json').write_text(json.dumps(dict(source=str(data),
    source_sha256={s:hashlib.sha256((data/(s+'.jsonl')).read_bytes()).hexdigest() for s in ['train','dev']},
    split_ids={k:[r['id'] for r in rows] for k,rows in payload.items()},
    scope='QASPER article-body continuation only; source train/dev are historical internal splits, not official benchmark scores',
    train_fit_tokens=32*512,selected_memory_tokens_per_doc=2048,query_tokens=128,
    answers_used=False,document_disjoint=True),indent=2)+'\n')
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
arms=['exact','h12','dual','h24_late']
records=[]
with (R.parent/'hidden_reader_pilot_submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
    assert 'hreader-p1-' not in queue, 'Existing experiment owner found'
    (R/'queue_before.txt').write_text(queue)
    for arm in arms:
        out=R/arm;out.mkdir()
        command=['sbatch','--parsable','--no-requeue','--job-name=hreader-p1-'+arm,
            '--partition=gpu','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1',
            '--cpus-per-task=8','--mem=96G','--time=01:00:00','--chdir='+str(R),
            '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err')]
        script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(R/'launch.py')+' '+arm+'\n'
        (out/'job.sh').write_text(script)
        (out/'submission_intent.json').write_text(json.dumps(dict(command=command,at=datetime.datetime.now().astimezone().isoformat()),indent=2))
        p=subprocess.run(command,input=script,text=True,capture_output=True,timeout=45)
        record=dict(arm=arm,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,command=command)
        if p.returncode==0:record['job']=p.stdout.strip().split(';')[0]
        records.append(record)
        (R/'submissions.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        assert p.returncode==0,p.stderr
