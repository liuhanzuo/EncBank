import hashlib,json,shutil,subprocess,time,sys
from pathlib import Path
S=Path(__file__).resolve().parent;B=S.parent;Q=B/'tf27b_hot_qualification_r3_20260921'
revision=sys.argv[1] if len(sys.argv)>1 else 'r1';assert revision in ['r1','r2','r3','r4']
R=B/('tf27b_hot_batch_diagnostic_'+revision+'_20260921')
assert not R.exists();assert json.loads((Q/'parent_exit.json').read_text())['exit_code']==1
R.mkdir()
for p in Q.iterdir():
    if p.suffix=='.py' or p.name in ['plan.json','trace_messages.json','trace_source.json']:shutil.copy2(p,R/p.name)
shutil.copy2(S/'diagnose_batch.py',R/'diagnose_batch.py')
launch=(R/'launch_qualification.py').read_text().replace("'qualify_gpu.py'","'diagnose_batch.py'")
(R/'launch_qualification.py').write_text(launch)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
py='/srv/encbank/Paper_Evolve/.venv/bin/python'
cmd=['sbatch','--parsable','--job-name=tf27b-batch-diagnostic','--partition=gpu','--time=UNLIMITED','--no-requeue',
    '--cpus-per-task=8','--mem=80G','--gres=gpu:nvidia_l20d:1','--exclude=gpu-node3,gpu-node6','--chdir='+str(R),
    '--output='+str(R/'slurm-%j.out'),'--error='+str(R/'slurm-%j.err'),
    '--wrap',py+' -B '+str(R/'launch_qualification.py')]
res=subprocess.run(cmd,capture_output=True,text=True);assert res.returncode==0,res.stderr
receipt=dict(job=res.stdout.strip().split(';')[0],root=str(R),argv=cmd,epoch=time.time())
(R/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
