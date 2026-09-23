"""Recover only Cold8's zero-model-call GPU admission failure, preserving sources."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
S=Path(__file__).resolve().parent;T=S.parent/'tf27b_hot_live_20260921'
rows=json.loads((T/'submissions.json').read_text());old=next(x for x in rows if x['label']=='cold8')
assert old['job']=='119998'
R=Path(old['root']);N=T/'cold8_r2';assert not N.exists()
failure=json.loads((R/'worker/failure.json').read_text())
assert 'GPU not freshly available' in failure['error']
assert json.loads((R/'worker/model_parent_exit.json').read_text())['actual_parent_wait']
assert json.loads((R/'worker/model_parent_exit.json').read_text())['exit_code']==1
assert json.loads((R/'jobs/integrated_complete.json').read_text())['exit_code']==1
assert not (R/'worker/ready.json').exists() and not (R/'worker/trials_launch.json').exists()
assert not list((R/'mailbox').glob('*/*.request.json')) and not list((R/'results').glob('*'))
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert not any(line.startswith(old['job']+'|') for line in queue.splitlines())
N.mkdir()
manifest=json.loads((R/'source_manifest.json').read_text())
for name,expected in manifest.items():
    path=R/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    target=N/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
shutil.copy2(R/'source_manifest.json',N/'source_manifest.json')
plan=json.loads((N/'plan.json').read_text())
for d in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs','worker']:(N/d).mkdir(exist_ok=True)
for task in plan['tasks']:
    (N/'pairs'/task).mkdir();(N/'mailbox'/task).mkdir()
proof=dict(previous=old,zero_model_requests=True,zero_model_forward=True,reason='GPU admission failed before model loading; occupied device on gpu-node1',queue=queue,epoch=time.time())
(N/'environment_recovery.json').write_text(json.dumps(proof,indent=2)+'\n')
cmd=[x.replace(str(R),str(N)) for x in old['argv']]
cmd=[('--exclude=gpu-node1,gpu-node3,gpu-node6' if x.startswith('--exclude=') else x) for x in cmd]
res=subprocess.run(cmd,capture_output=True,text=True);assert res.returncode==0,res.stderr
record=dict(old,root=str(N),job=res.stdout.strip().split(';')[0],argv=cmd,previous_attempt=old['job'])
(T/'cold8_submission_119998.json').write_text(json.dumps(old,indent=2)+'\n')
rows=[record if x['label']=='cold8' else x for x in rows]
tmp=T/'submissions.json.tmp';tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(T/'submissions.json')
print(json.dumps(record,indent=2))
