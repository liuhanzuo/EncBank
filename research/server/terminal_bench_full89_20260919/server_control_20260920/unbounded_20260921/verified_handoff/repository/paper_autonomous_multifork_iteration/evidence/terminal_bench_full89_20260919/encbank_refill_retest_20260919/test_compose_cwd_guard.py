import json,os,shutil,subprocess
from pathlib import Path
h=Path('/srv/encbank/legacy_workspace/paper_autonomous_multifork_iteration/evidence/terminal_bench_full89_20260919/encbank_refill_retest_20260919');assert shutil.which('docker')==str(h/'bin/docker')
rows=[]
for task in ['adaptive-rejection-sampler','build-cython-ext']:
 path='/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/tasks/'+task+'/environment'
 for i in range(3):
  args=['docker','compose','--project-directory',path,'version']
  p=subprocess.run(args,cwd=path,capture_output=True,text=True,timeout=20)
  rows.append(dict(task=task,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
  assert p.returncode==0,p.stderr
print(json.dumps(dict(status='PASS',calls=rows,docker=shutil.which('docker'),model_calls=0)))
