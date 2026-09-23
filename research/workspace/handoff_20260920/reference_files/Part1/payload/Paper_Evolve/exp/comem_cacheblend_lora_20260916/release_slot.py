"""CPU-only dependent job restores four-way evaluation after our GPU is freed."""
import datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
job=json.loads((root/'launch.json').read_text())['pipeline_job']
q=subprocess.run(['squeue','-j',job,'-h','-o','%T'],text=True,capture_output=True)
assert not any(s in q.stdout.split() for s in ('RUNNING','COMPLETING','PENDING'))
results={}
for array in ('45014','59046'):
    active=subprocess.run(['squeue','-j',array,'-h','-o','%T'],text=True,capture_output=True)
    if active.stdout.strip():
        r=subprocess.run(['scontrol','update','JobId='+array,'ArrayTaskThrottle=4'],capture_output=True,text=True)
        results[array]=dict(returncode=r.returncode,stderr=r.stderr)
        assert r.returncode==0,r.stderr
    else:results[array]=dict(already_finished=True)
(root/'gpu_slot_released.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=results),indent=2)+'\n')
print(json.dumps(results))
