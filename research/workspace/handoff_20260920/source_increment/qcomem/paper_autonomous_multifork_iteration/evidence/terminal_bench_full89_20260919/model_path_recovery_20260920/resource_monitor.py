"""Read-only allocated-GPU, dispatcher, and host-memory sampling; no submissions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,shlex,subprocess,sys,time,traceback
H=Path(__file__).resolve().parent;B=H.parent;RT=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919')
NAMES=['comem_k12_no_task_deadline_r5_20260920','comem_k48_no_task_deadline_r5_20260920','dense_no_task_deadline_r5_20260920']
PY='/srv/encbank/qcomem_runtime_20260911/python312/bin/python'
def now():return datetime.datetime.now().astimezone().isoformat()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
 t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');t.replace(p)
def run(args,timeout=50):
 p=subprocess.run(args,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
 return dict(argv=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
def snapshot():
 cfg={name:dict(job=load(B/name/'submission.json')['job_id'],remote=load(B/name/'plan.json')['remote_root'],arm=load(B/name/'plan.json')['arm']) for name in NAMES}
 code='''from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json,subprocess
cfg=CONFIG
q=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
queue={l.split('|')[0]:l.split('|') for l in q.splitlines()}
def one(item):
 name,c=item;h=Path(c['remote']);r=h/('run_'+c['arm']);d=dict(job=c['job'],queue=queue.get(c['job']))
 for n in ['process_start.json','worker_ready.json','process_receipt.json','worker_failure.json','memory_cap_failure.json']:
  if (r/n).exists():d[n]=json.loads((r/n).read_text())
 for n in ['owned_nvml.jsonl','events.jsonl']:
  if (r/n).exists():
   with (r/n).open() as f:d[n]=list(deque(f,maxlen=4))
 if c['job'] in queue and queue[c['job']][2]=='RUNNING':
  cmd=['srun','--jobid='+c['job'],'--overlap','--ntasks=1','--cpus-per-task=1','--time=00:01:00','nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);d['allocated_node_gpu_snapshot']=dict(argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
  if d.get('owned_nvml.jsonl'):
   nv=json.loads(d['owned_nvml.jsonl'][-1]);uuids={x.split(',')[-1].strip() for x in nv['rows']}
   d['owned_gpu_rows']=[x for x in p.stdout.splitlines() if x.split(',')[0].strip() in uuids]
 return name,d
with ThreadPoolExecutor(max_workers=3) as pool:out=dict(pool.map(one,cfg.items()))
print(json.dumps(dict(queue=q,services=out)))
'''.replace('CONFIG',repr(cfg))
 remote=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',shlex.join([PY,'-c',code])],timeout=55)
 out=dict(at=now(),remote_query=remote,local={},automatic_submissions=0,automatic_cancellations=0)
 if remote['exit_code']==0:out['remote']=json.loads(remote.pop('stdout'))
 for name,c in cfg.items():
  D=B/name;d={}
  for n in ['execution/status.json','execution/owner_registration.json','execution/host_admission.json','execution/transport_snapshot.json','execution/controller_failure.json','execution/owner_complete.json']:
   if (D/n).exists():d[n]=load(D/n)
  box=RT/('local_rpc_'+name)/c['arm'];d['delivered_responses']=len(list(box.glob('*.broker.json')))
  d['outstanding_model_requests']=sum(not (box/(q.name.replace('.request.','.response.'))).exists() and not (box/(q.name.replace('.request.','.error.'))).exists() for q in box.glob('*.request.json'))
  out['local'][name]=d
 if (H/'resource_latest.json').exists():
  prior=load(H/'resource_latest.json');elapsed=(datetime.datetime.fromisoformat(out['at'])-datetime.datetime.fromisoformat(prior['at'])).total_seconds()
  for name,d in out['local'].items():
   current=d.get('execution/transport_snapshot.json',{}).get('generated_tokens');old=prior.get('local',{}).get(name,{}).get('execution/transport_snapshot.json',{}).get('generated_tokens')
   if current is not None and old is not None and elapsed>0:d['completed_response_tokens_per_second_window']=(current-old)/elapsed;d['throughput_scope']='Tokens credited on completed responses, not instantaneous decode; long in-flight responses can give zero.'
 out['host_reservations']=load(RT/'host_admission/reservations.json')
 out['host_memory']=run(['wsl','-d','Ubuntu','--exec','cat','/proc/meminfo'],timeout=30)
 with (H/'resource_samples.jsonl').open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
 save(H/'resource_latest.json',out)
 print(json.dumps(dict(at=out['at'],services={n:dict(queue=x.get('queue'),ready='worker_ready.json' in x,gpu=x.get('owned_gpu_rows'),events=x.get('events.jsonl',[])[-1:]) for n,x in out.get('remote',{}).get('services',{}).items()})),flush=True)
 return out
if __name__=='__main__':
 if '--once' in sys.argv:snapshot()
 else:
  save(H/'resource_monitor_registration.json',dict(pid=os.getpid(),at=now(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interval_seconds=120,read_only=True))
  while True:
   try:
    s=snapshot()
    if s.get('remote') and all('process_receipt.json' in x for x in s['remote']['services'].values()):break
   except Exception:save(H/'resource_monitor_last_error.json',dict(at=now(),error=traceback.format_exc()))
   time.sleep(120)
  save(H/'resource_monitor_complete.json',dict(at=now(),all_registered_workers_closed=True))
