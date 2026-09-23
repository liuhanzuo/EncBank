"""One-shot delivery of real completed node-local results to the canonical experiment paths."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import datetime,fcntl,json,os,shutil,subprocess,sys,uuid
from pathlib import Path
payload=json.load(sys.stdin)
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
h=r/'maintenance_history/node-local-final-delivery-20260919-0835'
assert not h.exists(),'Already attempted: inspect history before any repeat'
for pid,needle in [(1599472,b'control/coordinator.py'),(2005194,b'judge_watch.py')]:
 p=Path('/proc')/str(pid)/'cmdline';assert not(p.exists() and needle in p.read_bytes())
locks=[]
for p in [r/'coordinator.lock',r/'judge_gpt6_astra/watch.lock',r.parent/'qcomem_gpu_admission.lock']:
 f=p.open('rb');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=35).stdout
q=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b']);assert 'qcm-q18-' not in q
jobs=[x['task']['job'] for x in payload]
acct=run(['sacct','-j',','.join(jobs+['106640']),'-n','-P','-o','JobIDRaw,State,ExitCode,End'])
for job in jobs+['106640']:assert any(l.startswith(job+'|COMPLETED|0:0|') for l in acct.splitlines())
for item in payload:
 t=item['task'];out=r/'runs'/t['task'];dest=r/'results/large-final/Qwen3.8-27B'/('shard%d'%t['shard'])
 assert json.loads((out/'submission.json').read_text())['job']==t['previous_job']
 assert not (dest/'complete.json').exists()
 assert item['predictions'].encode('utf-8').startswith((dest/'predictions.jsonl').read_bytes())
 parent=item['files']['parent_exit.json']
 assert parent['actual_wait'] and parent['returncode']==0 and parent['job']==t['job'] and parent['completion_exists']
 assert item['files']['evaluation_complete.json']['records']==1809
 for p in [out/'worker.lock',dest/'worker.lock']:
  if p.exists():
   f=p.open('rb');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
def write(p,data):
 temp=p.with_name(p.name+'.delivery-'+uuid.uuid4().hex+'.tmp')
 with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 temp.replace(p);assert p.read_bytes()==data
def dump(p,v):write(p,(json.dumps(v,indent=2)+'\n').encode('utf-8'))
h.mkdir();dump(h/'intent.json',dict(at=datetime.datetime.utcnow().isoformat()+'Z',jobs=jobs,source='real node-local outputs, no generated exit receipts'))
(h/'sacct.txt').write_text(acct)
for item in payload:
 t=item['task'];out=r/'runs'/t['task'];dest=r/'results/large-final/Qwen3.8-27B'/('shard%d'%t['shard'])
 old=out/'attempt_history'/('failed-'+t['previous_job']);old.mkdir(parents=True)
 for p in list(out.iterdir()):
  if p.is_file() and p.name!='worker.lock':shutil.copy2(str(p),str(old/p.name))
 for name in ['predictions.jsonl','protocol.json','progress.json','failure.json']:
  if (dest/name).exists():shutil.copy2(str(dest/name),str(old/('evaluation_'+name)))
 write(dest/'predictions.jsonl',item['predictions'].encode('utf-8'))
 for name in ['protocol.json','complete.json']:dump(dest/name,item['files']['evaluation_'+name])
 for name in ['start.json','parent_exit.json']:dump(out/name,item['files'][name])
 dump(out/'submission.json',t)
 dump(out/'node_local_delivery.json',dict(source=t['stage'],node=t['node'],source_job=t['job'],previous_job=t['previous_job'],copied_actual_parent_receipt=True))
 dump(h/(t['task']+'.json'),dict(delivered=True,job=t['job'],records=1809))
for name in ['coordinator_launch.json','status.json','coordinator_failure.json']:
 p=r/name
 if p.exists():shutil.copy2(str(p),str(h/name))
dump(h/'complete.json',dict(delivered=True,at=datetime.datetime.utcnow().isoformat()+'Z',jobs=jobs,real_receipts_preserved=True))
for f in locks:f.close()
# Refresh final status with the original controller; all existing tasks have real receipts.
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
result=subprocess.run(['/srv/encbank/Paper_Evolve/.venv/bin/python','-B',str(r/'control/coordinator.py')],cwd=str(r),env=env,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
dump(h/'coordinator_refresh_exit.json',dict(actual_wait=True,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
status=json.loads((r/'status.json').read_text());assert status['phase']=='GENERATION_COMPLETE' and status['completed']==28 and status['failed']==0
print(json.dumps(dict(history=str(h),canonical_generation_complete=True,completed=28,failed=0,refresh_actual_wait=result.returncode)))
'''
data=(ROOT/'delivery/qwen38_final_generation/canonical_sync_payload.json').read_text(encoding='utf-8')
out=ROOT/'delivery/qwen38_final_generation'
assert not (out/'canonical_sync_result.json').exists()
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],
                 input=data,text=True,encoding='utf-8',capture_output=True,timeout=58)
(out/'canonical_sync_stdout.txt').write_text(p.stdout,encoding='utf-8')
(out/'canonical_sync_stderr.txt').write_text(p.stderr,encoding='utf-8')
assert p.returncode==0,p.stderr
v=json.loads(p.stdout);(out/'canonical_sync_result.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
