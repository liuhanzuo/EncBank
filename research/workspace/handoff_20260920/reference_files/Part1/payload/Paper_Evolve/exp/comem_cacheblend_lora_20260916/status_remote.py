import datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
launch=read(root/'launch.json') or {}
status=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=launch,
    training=read(root/'training/progress.json'),training_complete=read(root/'training/complete.json'),
    evaluation=read(root/'evaluation/progress.json'),complete=read(root/'COMPLETE.json'),
    failures=[read(p) for p in sorted(root.glob('failure-*.json'))],
    slot_released=read(root/'gpu_slot_released.json'))
jobs=','.join(str(v) for k,v in launch.items() if k in ('pipeline_job','release_job'))
if jobs:
    for key,cmd in [('queue',['squeue','-j',jobs,'-h','-o','%i|%T|%M|%R']),
        ('accounting',['sacct','-j',jobs,'-n','-P','--format=JobID,State,ExitCode,Elapsed'])]:
        r=subprocess.run(cmd,text=True,capture_output=True);status[key]=r.stdout;status[key+'_error']=r.stderr
print(json.dumps(status))
