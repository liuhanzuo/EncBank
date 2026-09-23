"""User-requested Hot24 arm; keep failed Hot32 and running arms immutable."""
import ast,fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path

S=Path(__file__).resolve().parent;T=S.parent/'tf27b_hot_live_20260921'
OLD=T/'hot32';R=T/'hot24'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

lock=(T/'hot24_submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=read(T/'submissions.json')
assert not R.exists() and not any(x['label']=='hot24' for x in rows),'Refuse duplicate Hot24 submission'
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert 'tf27b-tb-hot24' not in queue
assert not any(x.startswith('120000|') for x in queue.splitlines())
parent=read(OLD/'worker/model_parent_exit.json')
assert parent['actual_parent_wait'] and parent['exit_code']==1
receipts=[read(p) for p in (OLD/'results').glob('*/execution_receipt.json')]
assert len(receipts)==32 and all(x['closure']['cgroup_empty'] for x in receipts)
manifest=read(OLD/'source_manifest.json')
for name,h in manifest.items():assert sha(OLD/name)==h,name
P=read(OLD/'plan.json');Q=Path(P['qualification_root'])
proof=read(Q/'qualification_complete.json')
assert proof['passed'] and proof['numerical'] and proof['batch']
assert read(Q/'parent_exit.json')['exit_code']==0
capacity=next(x for x in proof['stress'] if x['arm']=='hot' and x['batch']==24 and x['hot_chunks']==24 and x['passed'])
assert P['task_concurrency']==P['decode_batch_size']==32 and P['hot_chunks']==24 and len(P['tasks'])==32
for path,h in read(OLD/'task_manifest.json').items():assert sha(Path(path))==h,path

R.mkdir()
for name in manifest:
    target=R/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,target)
new=dict(P,task_concurrency=24,decode_batch_size=24)
save(R/'plan.json',new)
changes={key:dict(before=P[key],after=new[key]) for key in P if P[key]!=new[key]}
assert set(changes)=={'task_concurrency','decode_batch_size'}
for d in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs','worker']:(R/d).mkdir(exist_ok=True)
for task in P['tasks']:
    (R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
for p in R.glob('*.py'):
    ast.parse(p.read_text());assert sha(p)==sha(OLD/p.name),p.name
save(R/'hot24_preflight.json',dict(passed=True,epoch=time.time(),changes=changes,
    scientific_code_identical_to=str(OLD),inherited_backend_gpu_validation=True,
    capacity_evidence=capacity,capacity_scope='Short synthetic test; dynamic lifetime stability remains to be measured',
    accounting='Fresh independent Hot24 configuration on all32 tasks; do not merge with failed Hot32',
    authorization='User: 那你先试试hot24吧',queue=queue))
save(R/'source_manifest.json',{str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file()})
cmd=['sbatch','--parsable','--job-name=tf27b-tb-hot24','--partition=gpu','--time=UNLIMITED','--no-requeue',
    '--cpus-per-task=56','--mem=160G','--gres=gpu:nvidia_l20d:1','--exclude=gpu-node1,gpu-node3,gpu-node6',
    '--chdir='+str(R),'--output='+str(R/'logs/slurm-%j.out'),'--error='+str(R/'logs/slurm-%j.err'),
    '--wrap',P['engine_python']+' -B '+str(R/'launch_trial_group.py')]
res=subprocess.run(cmd,capture_output=True,text=True)
save(R/'submission_attempt.json',dict(argv=cmd,returncode=res.returncode,stdout=res.stdout,stderr=res.stderr,epoch=time.time()))
assert res.returncode==0,res.stderr
record=dict(label='hot24',arm='hot',concurrency=24,hot_chunks=24,root=str(R),job=res.stdout.strip().split(';')[0],argv=cmd,epoch=time.time())
save(R/'submission.json',record);save(T/'submissions.json',rows+[record])
print(json.dumps(record,indent=2))
