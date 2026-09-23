"""Local-only pre-agent CWD mitigation, no GPU restart or scientific retry."""
from pathlib import Path
import datetime,hashlib,json,subprocess
H=Path(__file__).resolve().parent;D=H.parent/'dense_restore37_20260919';ROOT=Path('/srv/encbank/legacy_workspace');RT=ROOT/'.runtime/terminal_bench_full89_20260919'
for h in [H,D]:
    if h==D:
        (h/'bin').mkdir(exist_ok=True)
        for n in ['bin/docker','sitecustomize.py']:(h/n).write_bytes((H/n).read_bytes())
    path='/srv/encbank/legacy_workspace/'+h.relative_to(ROOT).as_posix()+'/bin/docker'
    subprocess.run(['wsl','-d','Ubuntu','--','chmod','+x',path],check=True,capture_output=True)
results=[]
for p in (RT/'results'/H.name).glob('*/*/result.json'):
    q=json.loads(p.read_text());e=q.get('exception_info') or {}
    if q.get('agent_execution') is None and 'getwd: no such file or directory' in e.get('exception_message',''):
        task=p.parent.name.split('__')[0]
        requests=[r for r in (RT/('local_rpc_'+H.name)/'encbank').glob('*.request.json') if json.loads(r.read_text())['task']==task]
        assert not requests
        results.append(dict(task=task,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_requests=0,classification='PRE_AGENT_INFRASTRUCTURE_FAILURE_NOT_SCORE',error=e))
assert results
script='''import json,os,shutil,subprocess
from pathlib import Path
h=Path(HERE);assert shutil.which('docker')==str(h/'bin/docker')
rows=[]
for task in ['adaptive-rejection-sampler','build-cython-ext']:
 path='/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/tasks/'+task+'/environment'
 for i in range(3):
  args=['docker','compose','--project-directory',path,'version']
  p=subprocess.run(args,cwd=path,capture_output=True,text=True,timeout=20)
  rows.append(dict(task=task,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
  assert p.returncode==0,p.stderr
print(json.dumps(dict(status='PASS',calls=rows,docker=shutil.which('docker'),model_calls=0)))
'''.replace('HERE',repr('/srv/encbank/legacy_workspace/'+H.relative_to(ROOT).as_posix()))
(H/'test_compose_cwd_guard.py').write_text(script,encoding='utf8',newline='\n')
wh='/srv/encbank/legacy_workspace/'+H.relative_to(ROOT).as_posix()
argv=['wsl','-d','Ubuntu','--','env','PYTHONPATH='+wh,'/srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918/harbor_env/bin/python',wh+'/test_compose_cwd_guard.py']
p=subprocess.run(argv,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=160)
assert p.returncode==0,p.stderr
record=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_LOCAL_CWD_GUARD_INSTALLED',argv=argv,exit_code=p.returncode,result=json.loads(p.stdout),stderr=p.stderr,
    failures=results,scope='Future local Harbor processes only; no package-global edits, GPU service restart, changed task timeout, sampling or task content. Existing8 task worlds continue. Explicit Compose project path and every argument preserved; only host process cwd changed to slash.',
    limitation='Mitigation for observed intermittent getwd; not proof underlying WSL/DrvFS cause repaired.',
    source_sha256={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['bin/docker','sitecustomize.py']},
    next='After current batch, retry only these no-model pre-agent failures in new per-task output roots on the existing service if available; never record verifier-null as zero.')
(H/'compose_cwd_guard_registration.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
pause=H/'execution/pause_new_tasks.json'
assert pause.resolve().is_relative_to(H.resolve()) and 'Diagnose pre-agent Docker getwd' in pause.read_text()
pause.rename(H/'compose_cwd_pause_closed.json')
print(json.dumps(dict(status=record['status'],pre_agent_failures=[r['task'] for r in results],new_task_admission_resumed=True)))
