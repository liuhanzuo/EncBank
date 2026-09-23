"""Read this pilot's remote scheduler/progress without touching other jobs."""
import json,subprocess
from pathlib import Path

code=r'''
from pathlib import Path
import datetime,json,subprocess
root=Path('/srv/encbank/encbank_new_backbones_20260915')
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote':str(root),'array_job':'33916','pilot_steps':200,'models':{}}
report['queue']=subprocess.run(['squeue','-j','33916','-h','-o','%i %T %M %R'],text=True,capture_output=True).stdout.strip().splitlines()
for name in ['Qwen3.5-9B','Qwen3.8-27B']:
    runs=sorted((root/'results'/name).glob('job_*'),key=lambda p:p.stat().st_mtime)
    info={'download_complete':(root/'models'/name/'download_complete.json').exists()}
    if runs:
        run=runs[-1];info['run']=str(run);info['files']=[p.name for p in run.iterdir()]
        for fn in ['progress.json','failure.json','correctness.json','correctness_adapted.json']:
            if (run/fn).exists():info[fn]=json.loads((run/fn).read_text())
    report['models'][name]=info
print(json.dumps(report,indent=2))
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
                       '/srv/encbank/Paper_Evolve/.venv/bin/python','-'],
                      input=code,text=True,encoding='utf-8',capture_output=True,timeout=45,check=True)
payload=json.loads(result.stdout)
root=Path(__file__).resolve().parent
(root/'STATUS.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
for model,entry in payload['models'].items():
    print(model,json.dumps(entry.get('progress.json',entry.get('failure.json',{'phase':'initialization'}))))
print('Scheduler:',payload['queue'])
