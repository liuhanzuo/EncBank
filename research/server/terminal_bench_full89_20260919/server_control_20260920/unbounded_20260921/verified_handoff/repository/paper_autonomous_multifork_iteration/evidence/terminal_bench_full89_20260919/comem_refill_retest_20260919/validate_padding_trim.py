"""Hold existing pending job while checking the unused-padding reclamation fix."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent;C=H.parent/'comem_batch8_refill_preparation';P=json.loads((H/'plan.json').read_text());remote=P['remote_root']
job=json.loads((H/'submission.json').read_text())['job_id'];assert job=='108870'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(argv,timeout=60):
    r=subprocess.run(argv,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    assert r.returncode==0,r.stderr
    return r.stdout
def ssh(code,timeout=60):return run(['ssh','-o','BatchMode=yes','gpu-node1',shlex.join(['/srv/encbank/qcomem_runtime_20260911/python312/bin/python','-c',code])],timeout)
before=json.loads(ssh(f"from pathlib import Path;import json,subprocess;h=Path({remote!r});s=subprocess.check_output(['squeue','-j',{job!r},'-h','-o','%T|%R'],text=True);assert 'PENDING|(JobHeldUser)' in s and not (h/'run_comem').exists();print(json.dumps(dict(queue=s,old_cache=(h/'batch_cache.py').read_text(),old_manifest=json.loads((h/'remote_source_manifest.json').read_text()))))"))
(H/'padding_trim_before.json').write_text(json.dumps(before,indent=2)+'\n',encoding='utf8',newline='\n')
code=(C/'test_refill_cpu.py').read_text()
code=code.replace("PREV=H.parent/'comem_batch8_service';sys.path.insert(0,str(PREV))","PREV=H;sys.path.insert(0,str(PREV))")
code=code.replace('errors=[];sizes=[]','errors=[];sizes=[];trimmed=[]')
code=code.replace("idx=torch.tensor([7,4,2,0]);qp,lp=compact(low,qp,lp,idx);upos,upad=compact(up,upos,upad,idx);rows=[rows[i] for i in idx.tolist()]", "idx=torch.tensor([2,0]);before_width=up.layers[7].keys.shape[-2];qp,lp=compact(low,qp,lp,idx);upos,upad=compact(up,upos,upad,idx);rows=[rows[i] for i in idx.tolist()]\n            assert int(lp.min())==int(upad.min())==0\n            after_width=up.layers[7].keys.shape[-2];assert after_width<before_width\n            trimmed.append(dict(before=before_width,after=after_width))")
code=code.replace("print(json.dumps(out))","out.update(padding_trim=trimmed,cache_source_sha256=__import__('hashlib').sha256((H/'batch_cache.py').read_bytes()).hexdigest());assert trimmed\n(H/'padding_trim_cpu_check.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))")
(H/'test_padding_trim_cpu.py').write_text(code,encoding='utf8',newline='\n')
run(['scp',str(H/'batch_cache.py'),str(H/'test_padding_trim_cpu.py'),'gpu-node1:'+remote+'/'])
test=f'''from pathlib import Path
import os,json,subprocess,time,hashlib
h=Path({remote!r});h.resolve().relative_to(Path('/srv/encbank').resolve())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/srv/encbank/comem_infra_recheck_20260912/deps',CPATH='/srv/encbank/.cache/python/include/python3.12',TMPDIR=str(h/'tmp'),XDG_CACHE_HOME=str(h/'cache'),HF_HOME=str(h/'hf_cache'))
start=time.time();argv=[{P['engine_python']!r},str(h/'test_padding_trim_cpu.py')]
with (h/'padding_trim_cpu.stdout.log').open('xb') as out,(h/'padding_trim_cpu.stderr.log').open('xb') as err:
 p=subprocess.Popen(argv,env=env,stdout=out,stderr=err)
 (h/'padding_trim_cpu_start.json').write_text(json.dumps(dict(pid=p.pid,argv=argv,epoch=start)))
 try:rc=p.wait(timeout=180);timeout=False
 except subprocess.TimeoutExpired:p.kill();rc=p.wait();timeout=True
receipt=dict(exit_code=rc,actual_parent_wait=True,timeout=timeout,seconds=time.time()-start)
(h/'padding_trim_cpu_receipt.json').write_text(json.dumps(receipt))
print(json.dumps(dict(receipt=receipt,stdout=(h/'padding_trim_cpu.stdout.log').read_text(),stderr=(h/'padding_trim_cpu.stderr.log').read_text())))
'''
result=json.loads(ssh(test,220));(H/'padding_trim_execution.json').write_text(json.dumps(dict(command=test,result=result),indent=2)+'\n',encoding='utf8',newline='\n')
assert result['receipt']['exit_code']==0,result
run(['scp','gpu-node1:'+remote+'/padding_trim_cpu_check.json',str(H/'padding_trim_cpu_check.json')])
check=json.loads((H/'padding_trim_cpu_check.json').read_text());assert check['status']=='PASS' and check['cache_source_sha256']==sha(H/'batch_cache.py')
manifest=before['old_manifest'];assert sha(H/'batch_cache.py')!=manifest['batch_cache.py']
manifest['batch_cache.py']=sha(H/'batch_cache.py')
for n,d in manifest.items():assert sha(H/n)==d,n
(H/'remote_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
record=dict(at=datetime.datetime.now().astimezone().isoformat(),job_id=job,state='PENDING_SOURCE_FIX_VERIFIED_BEFORE_FIRST_MODEL_LOAD',reason='Reclaim common masked KV columns after row removal so continuous cohorts do not accumulate unused padding.',old_cache_sha256=before['old_manifest']['batch_cache.py'],new_cache_sha256=manifest['batch_cache.py'],new_manifest_sha256=sha(H/'remote_source_manifest.json'),test=check,old_outputs_overwritten=False,new_gpu_jobs=0)
(H/'padding_trim_registration.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
run(['scp',str(H/'remote_source_manifest.json'),str(H/'padding_trim_registration.json'),'gpu-node1:'+remote+'/'])
release=ssh(f"from pathlib import Path;import hashlib,json,subprocess;h=Path({remote!r});s=subprocess.check_output(['squeue','-j',{job!r},'-h','-o','%T|%R'],text=True);assert 'PENDING|(JobHeldUser)' in s and not (h/'run_comem').exists();m=json.loads((h/'remote_source_manifest.json').read_text());assert all(hashlib.sha256((h/n).read_bytes()).hexdigest()==d for n,d in m.items());p=subprocess.run(['scontrol','release',{job!r}],capture_output=True,text=True);print(json.dumps(dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)));assert p.returncode==0")
(H/'padding_trim_release.json').write_text(release,encoding='utf8',newline='\n');print(json.dumps(dict(status='PASS_RELEASED_SAME_JOB',job_id=job,check=check)))
