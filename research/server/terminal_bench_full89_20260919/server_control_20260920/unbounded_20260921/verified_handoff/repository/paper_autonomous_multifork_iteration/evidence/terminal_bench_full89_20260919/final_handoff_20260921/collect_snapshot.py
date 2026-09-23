from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent;B=H.parent;PY='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
NAMES=dict(dense='dense_no_task_deadline_r6_20260921',top12='encbank_k12_no_task_deadline_r6_20260920',top48='encbank_k48_no_task_deadline_r6_20260920')
def load(p):return json.loads(p.read_text(encoding='utf8'))
cfg={k:dict(root=load(B/n/'plan.json')['remote_root'],arm=load(B/n/'plan.json')['arm'],job=load(B/n/'submission.json')['job_id']) for k,n in NAMES.items()}
code=f'''from pathlib import Path
import hashlib,json,subprocess
cfg={cfg!r};out={{}}
for arm,c in cfg.items():
 h=Path(c['root']);h.resolve().relative_to(Path('/srv/encbank').resolve())
 raw=json.dumps(dict(scheduler_terminal='CANCELLED by 0',worker_parent_wait_available=False)) if arm=='dense' else subprocess.check_output([{PY!r},str(h/'file_bridge.py'),'status',c['arm']],text=True)
 out[arm]=dict(status=json.loads(raw),source_sha256={{n:hashlib.sha256((h/n).read_bytes()).hexdigest() for n in ['agent_worker.py','file_bridge.py','plan.json']}},accounting=subprocess.check_output(['sacct','-j',c['job'],'-n','-P','--format=JobID,State,ExitCode,Elapsed,NodeList'],text=True))
print(json.dumps(dict(services=out,queue=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True))))
'''
argv=['ssh','-o','BatchMode=yes','gpu-node1',shlex.join([PY,'-c',code])]
p=subprocess.run(argv,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=90);assert p.returncode==0,p.stderr
out=dict(at=datetime.datetime.now().astimezone().isoformat(),remote=json.loads(p.stdout),local={},read_command=argv,exit_code=p.returncode)
for arm,n in NAMES.items():
 d=B/n;local={}
 for f in ['owner_registration.json','status.json','host_admission.json','worker_ready.json','owner_complete.json','controller_failure.json']:
  if (d/'execution'/f).exists():local[f]=load(d/'execution'/f)
 out['local'][arm]=local
 for f,sha in out['remote']['services'][arm]['source_sha256'].items():assert hashlib.sha256((d/f).read_bytes()).hexdigest()==sha
(H/'live_snapshot.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps(dict(at=out['at'],queue=out['remote']['queue'],states={k:v['status.json']['state'] for k,v in out['local'].items()})))
