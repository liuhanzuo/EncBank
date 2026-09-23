"""Focused immutable deployment via the existing SSH/Slurm admission lock."""
from pathlib import Path
import ast,datetime,hashlib,json,shlex,subprocess,sys
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());REMOTE=P['remote_root']
PY='/srv/encbank/qcomem_runtime_20260911/python312/bin/python';calls=[]
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def run(argv,timeout=60):
    r=subprocess.run(argv,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    calls.append(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;return r.stdout
def ssh(code,timeout=60):return run(['ssh','-o','BatchMode=yes','gpu-node1',shlex.join([PY,'-c',code])],timeout)
names=['test_live_service_cpu.py','live_mailbox.py','plan.json','common.py','batch_cache.py','batch_cache_refill.py','service_loop.py','hybrid_reader.py','memory_selectors.py','session.py','agent_worker.py','holder.py','file_bridge.py','persistence.py','storage_preflight.py','cpu_preflight_remote.py','comem.slurm']
def prepare():
    for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    assert json.loads((H/'recovery_check.json').read_text())['status']=='PASS'
    ssh(f"from pathlib import Path;h=Path({REMOTE!r});h.parent.resolve().relative_to(Path('/srv/encbank').resolve());h.mkdir(exist_ok=True);h.resolve().relative_to(Path('/srv/encbank').resolve());assert not (h/'submission.json').exists() and not (h/{('run_'+P['arm'])!r}).exists();[(h/n).mkdir(exist_ok=True) for n in ['incoming','tmp','cache','hf_cache','triton','cuda']]")
    run(['scp',*[str(H/n) for n in names],'gpu-node1:'+REMOTE+'/'])
    print('Frozen source prepared; fresh worker preflight remains mandatory.')

def submit():
    assert not (H/'submission.json').exists()
    check=json.loads((H/'recovery_check.json').read_text());assert check['status']=='PASS'
    for n,d in check['source_sha256'].items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==d,n
    assert json.loads((H/'cpu_preflight_local.json').read_text())['status']=='PASS'
    manifest={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in names};save(H/'remote_source_manifest.json',manifest)
    run(['scp',*[str(H/n) for n in names],str(H/'remote_source_manifest.json'),'gpu-node1:'+REMOTE+'/'])
    code=f'''from pathlib import Path
import fcntl,hashlib,json,subprocess,time,os
h=Path({REMOTE!r}).resolve();h.relative_to(Path('/srv/encbank').resolve())
for n,d in json.loads((h/'remote_source_manifest.json').read_text()).items():assert hashlib.sha256((h/n).read_bytes()).hexdigest()==d,n
with Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_20260918/admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not (h/'submission.json').exists() and not (h/'run_comem').exists()
 q=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
 own=[l for l in q.splitlines() if 'qcomem-agentmem-' in l or 'qcomem-tb-' in l]
 assert len(own)<4 and not any('qcomem-tb-k48-recover-codex' in l for l in own),own
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader'],text=True)
 processes=subprocess.check_output(['ps','-u','liuhanzuo','-o','pid,ppid,etime,args'],text=True)
 duplicate=[l for l in processes.splitlines() if str(h) in l and ' -c ' not in l and ('agent_worker.py' in l or 'holder.py' in l)]
 assert not duplicate,duplicate
 r=dict(epoch=time.time(),prior_own_queue=own,login_gpu_snapshot=gpu,duplicate_processes=duplicate,argv=['sbatch','--parsable',str(h/'comem.slurm')],status='intent')
 (h/'submission.json').write_text(json.dumps(r))
 p=subprocess.run(r['argv'],capture_output=True,text=True,timeout=30)
 r.update(exit_code=p.returncode,actual_parent_wait=True,stdout=p.stdout,stderr=p.stderr)
 if p.returncode==0:r.update(job_id=p.stdout.strip().split(';')[0],status='submitted')
 (h/'submission.json').write_text(json.dumps(r,indent=2)+'\\n');assert p.returncode==0,p.stderr;print(json.dumps(r))
'''
    sub=json.loads(ssh(code));save(H/'submission.json',sub)
    with (H/'owner.stdout.log').open('wb') as out,(H/'owner.stderr.log').open('wb') as err:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(H/'windows_owner.py')],cwd=H,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    save(H/'native_launch.json',dict(pid=p.pid,job_id=sub['job_id'],at=datetime.datetime.now().astimezone().isoformat()))
    print(json.dumps(dict(job_id=sub['job_id'],owner_pid=p.pid)))
try:{'prepare':prepare,'submit':submit}[sys.argv[1]]()
finally:save(H/('deploy_'+sys.argv[1]+'_calls.json'),calls)
