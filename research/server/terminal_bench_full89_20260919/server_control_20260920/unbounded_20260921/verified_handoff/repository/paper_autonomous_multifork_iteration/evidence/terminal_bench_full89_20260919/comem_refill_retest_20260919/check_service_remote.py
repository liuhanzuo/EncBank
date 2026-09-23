"""Bounded CPU-only service test with actual child wait and source hashes."""
from pathlib import Path
import hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());remote=P['remote_root']
names=['test_service_cpu.py'];subprocess.run(['scp',*[str(H/n) for n in names],'gpu-node1:'+remote+'/'],check=True)
code=f'''from pathlib import Path
import os,subprocess,json,time,hashlib
h=Path({remote!r}).resolve();h.relative_to(Path('/srv/encbank').resolve())
assert hashlib.sha256((h/'test_service_cpu.py').read_bytes()).hexdigest()=={hashlib.sha256((H/'test_service_cpu.py').read_bytes()).hexdigest()!r}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/srv/encbank/comem_infra_recheck_20260912/deps',CPATH='/srv/encbank/.cache/python/include/python3.12',TMPDIR=str(h/'tmp'),XDG_CACHE_HOME=str(h/'cache'),HF_HOME=str(h/'hf_cache'))
argv=[{P['engine_python']!r},str(h/'test_service_cpu.py')];start=time.time()
with (h/'service_cpu.stdout.log').open('xb') as out,(h/'service_cpu.stderr.log').open('xb') as err:
 p=subprocess.Popen(argv,env=env,stdout=out,stderr=err)
 (h/'service_cpu_start.json').write_text(json.dumps(dict(pid=p.pid,argv=argv,epoch=start)))
 try:rc=p.wait(timeout=180);timeout=False
 except subprocess.TimeoutExpired:p.kill();rc=p.wait();timeout=True
receipt=dict(exit_code=rc,actual_parent_wait=True,timeout=timeout,seconds=time.time()-start)
(h/'service_cpu_receipt.json').write_text(json.dumps(receipt))
print(json.dumps(dict(receipt=receipt,stdout=(h/'service_cpu.stdout.log').read_text(),stderr=(h/'service_cpu.stderr.log').read_text())))
'''
argv=['ssh','-o','BatchMode=yes','gpu-node1',shlex.join(['/srv/encbank/qcomem_runtime_20260911/python312/bin/python','-c',code])]
r=subprocess.run(argv,capture_output=True,text=True,encoding='utf8',timeout=220)
record=dict(argv=argv,exit_code=r.returncode,stderr=r.stderr,result=json.loads(r.stdout) if r.returncode==0 else r.stdout)
(H/'service_cpu_execution.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps(record['result']))
assert r.returncode==0 and record['result']['receipt']['exit_code']==0
subprocess.run(['scp','gpu-node1:'+remote+'/service_cpu_check.json',str(H/'service_cpu_check.json')],check=True)
