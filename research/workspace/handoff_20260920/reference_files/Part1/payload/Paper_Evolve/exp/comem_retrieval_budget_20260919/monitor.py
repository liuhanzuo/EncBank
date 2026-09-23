import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import datetime,json,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['stage'])
def read(path):return json.loads(path.read_text()) if path.exists() else None
report=dict(at=datetime.datetime.utcnow().isoformat()+'Z',job=p['job'],stage=str(root),
 sacct=subprocess.check_output(['sacct','-j',p['job'],'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,End'],universal_newlines=True),
 queue=subprocess.check_output(['squeue','-j',p['job'],'-h','-o','%i|%j|%T|%M|%R'],universal_newlines=True),files={},points={})
for name in ['launch.json','child.json','environment.json','access_patterns.json','correctness.json','status.json','failure.json','complete.json','parent_exit.json','order.json']:
 value=read(root/name)
 if value is not None:report['files'][name]=value
for d in (root/'results').glob('*'):
 if d.is_dir():report['points'][d.name]={name:read(d/name) for name in ['complete.json','progress.json','offline.json'] if (d/name).exists()}
for key,path in [('stderr',root/'stderr.log'),('stdout',root/'stdout.log'),('bootstrap_err',Path('/tmp/qcm-budget-'+p['job']+'.err'))]:
 if path.exists():
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-3500));report[key]=f.read().decode('utf-8','replace')
print(json.dumps(report))
'''
def main():
    job=json.loads((ROOT/'submission.json').read_text())
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],input=json.dumps(job),capture_output=True,encoding='utf-8',timeout=40)
    assert r.returncode==0,r.stderr
    state=json.loads(r.stdout)
    (ROOT/'monitor.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    observation=ROOT/'observations'/state['job'];observation.mkdir(parents=True,exist_ok=True)
    for name,value in state['files'].items():
        (observation/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(at=state['at'],job=state['job'],queue=state['queue'],sacct=state['sacct'],status=state['files'].get('status.json'),correctness=state['files'].get('correctness.json'),failure=state['files'].get('failure.json'),points=state['points'],stderr=state.get('stderr'),bootstrap_err=state.get('bootstrap_err'))))
if __name__=='__main__':main()
