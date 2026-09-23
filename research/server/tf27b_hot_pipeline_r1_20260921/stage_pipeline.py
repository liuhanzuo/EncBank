"""Freeze a one-shot dependent experiment launcher; not a recurring monitor."""
import ast,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;Q=Path(sys.argv[1]);R=S.parent/'tf27b_hot_pipeline_r1_20260921'
assert not R.exists();submission=json.loads((Q/'submission.json').read_text());R.mkdir()
for p in S.iterdir():
    if p.suffix in ['.py','.json','.md']:shutil.copy2(p,R/p.name)
for p in R.glob('*.py'):ast.parse(p.read_text())
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file()}
(R/'pipeline_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
launcher='''import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
for name,h in json.loads((R/'pipeline_manifest.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
start=time.time()
p=subprocess.Popen([sys.executable,'-B',str(R/'stage_trials.py'),sys.argv[1]],cwd=R)
code=p.wait()
(R/'parent_exit.json').write_text(json.dumps(dict(actual_parent_wait=True,pid=p.pid,exit_code=code,start_epoch=start,end_epoch=time.time()),indent=2)+'\\n')
sys.exit(code)
'''
(R/'launch.py').write_text(launcher)
py='/srv/encbank/Paper_Evolve/.venv/bin/python'
cmd=['sbatch','--parsable','--job-name=tf27b-stage-live','--partition=gpu','--time=UNLIMITED','--no-requeue',
    '--cpus-per-task=2','--mem=8G','--dependency=afterok:'+submission['job'],
    '--chdir='+str(R),'--output='+str(R/'slurm-%j.out'),'--error='+str(R/'slurm-%j.err'),
    '--wrap',py+' -B '+str(R/'launch.py')+' '+str(Q)]
res=subprocess.run(cmd,capture_output=True,text=True);assert res.returncode==0,res.stderr
record=dict(job=res.stdout.strip().split(';')[0],depends_on=submission['job'],qualification_root=str(Q),root=str(R),argv=cmd,epoch=time.time())
(R/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
