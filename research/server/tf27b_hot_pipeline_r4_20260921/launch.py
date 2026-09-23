import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
for name,h in json.loads((R/'pipeline_manifest.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
start=time.time()
p=subprocess.Popen([sys.executable,'-B',str(R/'stage_trials.py'),sys.argv[1]],cwd=R)
code=p.wait()
(R/'parent_exit.json').write_text(json.dumps(dict(actual_parent_wait=True,pid=p.pid,exit_code=code,start_epoch=start,end_epoch=time.time()),indent=2)+'\n')
sys.exit(code)
