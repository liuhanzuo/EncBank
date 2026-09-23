"""Freeze local raw evidence after user-directed stop. No model or remote actions."""
from pathlib import Path
from collections import Counter
import csv,datetime,hashlib,json,os,tarfile
ROOT=Path('/srv/encbank/legacy_workspace');H=Path(__file__).resolve().parent;B=H.parent;RT=ROOT/'.runtime/terminal_bench_full89_20260919'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def safe_walk(d):
 for here,dirs,files in os.walk(d):
  dirs[:]=[n for n in dirs if n not in {'artifacts','tests','ground_truth','__pycache__','.git'}]
  for n in files:
   p=Path(here)/n
   if not p.is_symlink():yield p
assert load(H/'automation_paused.json')['status']=='PAUSED'
assert load(H/'windows_after_stop.json')['old_controllers_absent'] and load(H/'wsl_stop_receipt.json')['status']=='PASS'
v=load(H/'closed_verification.json');before=load(H/'before_stop.json');rows=[];roots=set();normal_refs=[]
for arm in ['top12','top48']:
 a=v['arms'][arm];name=a['root'];D=B/name;old=load(D/'partition_final.json');st=before['campaigns'][name]['status.json']
 normal={x['task']:x for x in a['retained_normal']};excluded={x['task']:x for x in a['excluded']}
 categories={'VERIFIED_COMPLETE':set(normal),'CLOSED_INCOMPLETE':set(excluded),'INTERRUPTED_BY_USER_HANDOFF':set(st['active']),
  'NEVER_STARTED_THIS_CONTINUATION':set(st['pending']),'OLDER_EXIT_NEEDS_ADJUDICATION':set(old['pending_adjudication']),
  'ENVIRONMENT_BLOCKED_NEW_COMPUTER_OWNS_SUPPLEMENT':set(old['blocked_environment'])}
 seen=set()
 for status,tasks in categories.items():
  assert not seen&tasks,(arm,status,seen&tasks);seen|=tasks
  for task in sorted(tasks):
   r=dict(arm=arm,task=task,status=status,do_not_repeat_verified_result=status=='VERIFIED_COMPLETE',reward=None)
   if task in normal:
    n=normal[task];p=Path(n['path']);assert sha(p)==n['sha256'];r.update(reward=n['verifier']['rewards']['reward'],result=ref(p),evidence=n)
    roots.add(p.relative_to(RT/'results').parts[0]);normal_refs.append(p)
   elif task in excluded:r['evidence']=excluded[task]
   elif status=='INTERRUPTED_BY_USER_HANDOFF':r['pre_stop_pid']=st['active'][task]['pid'];r['scientific_completion']=False
   rows.append(r)
 assert len(seen)==89,(arm,len(seen))
 roots.add(name)
save(H/'TASK_STATUS.json',dict(at=datetime.datetime.now().astimezone().isoformat(),scope='Two old Encbank arms only, 89 tasks each',rows=rows,counts={a:dict(Counter(r['status'] for r in rows if r['arm']==a)) for a in ['top12','top48']},
 constraints=['Incomplete tasks have no assigned zero score.','Do not repeat any VERIFIED_COMPLETE result, including negative outcomes.','New computer two-task supplement and Dense are externally owned; reconcile their outputs separately.','Interrupted requests may still exist server-side; user owns server stop.']))
with (H/'TASK_STATUS.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['arm','task','status','reward','do_not_repeat_verified_result']);w.writeheader();w.writerows({k:r.get(k) for k in w.fieldnames} for r in rows)
save(H/'DO_NOT_RERUN_VERIFIED_TASKS.json',{a:[r['task'] for r in rows if r['arm']==a and r['status']=='VERIFIED_COMPLETE'] for a in ['top12','top48']})
files=set();rpc_index=[]
for name in sorted(roots):
 d=B/name;box=RT/('local_rpc_'+name)/'encbank';results=RT/'results'/name
 for p in d.iterdir():
  if p.is_file() and p.suffix in {'.py','.json','.md','.sh','.log'}:files.add(p)
 for p in safe_walk(d/'execution'):
  if not p.name.startswith('transport_status_'):files.add(p)
 for p in safe_walk(results):
  # Includes agent trajectories, raw terminal panes and verifier outputs; excludes task artifacts/tests.
  files.add(p)
 for p in box.iterdir():
  if p.is_file() and p.suffix in {'.json','.transfer','.download'}:files.add(p)
 for p in sorted(box.glob('*.request.json')):
  q=load(p);rid=q['request_id'];r=dict(root=name,request_id=rid,task=q['task'],task_id=q['task_id'],step=q['step'],request=ref(p))
  for typ in ['response.json','error.json','broker.json','cancel.json']:
   f=box/(rid+'.'+typ)
   if f.exists():r[typ]=ref(f)
  r['local_response_available']='response.json' in r
  if 'response.json' in r:
   ans=load(box/(rid+'.response.json'));r['response_status']=ans.get('status');r['generated_tokens']=ans.get('generated_tokens')
  rpc_index.append(r)
save(H/'REQUEST_RESPONSE_INDEX.json',dict(roots=sorted(roots),requests=rpc_index,requests_count=len(rpc_index),locally_missing_response_count=sum(not r['local_response_available'] for r in rpc_index),note='A missing local response is preserved, not fabricated or counted as completed; inspect server snapshot separately.'))
for p in H.iterdir():
 if p.is_file() and p.suffix in {'.py','.json','.csv','.md'} and p.name not in {'FILE_SHA256.json','package_receipt.json','upload_receipt.json'}:files.add(p)
manifest=[];archive=H/'LOCAL_RAW_EVIDENCE.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=2) as tf:
 for i,p in enumerate(sorted(files)):
  p.resolve().relative_to(ROOT.resolve());rel=p.relative_to(ROOT).as_posix();assert not any(x in p.parts for x in ['ko2','matched_v2','ground_truth'])
  st=p.stat();digest=sha(p);arc='repository/'+rel
  tf.add(p,arcname=arc,recursive=False);after=p.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns),(rel,'changed during snapshot')
  manifest.append(dict(path=rel,archive_path=arc,bytes=st.st_size,sha256=digest))
  if i and i%2000==0:print(json.dumps(dict(packaged=i,total=len(files))),flush=True)
save(H/'FILE_SHA256.json',dict(schema=1,files=manifest,count=len(manifest),total_uncompressed_bytes=sum(x['bytes'] for x in manifest),roots=sorted(roots)))
known={r['archive_path']:r for r in manifest};verified=0
with tarfile.open(archive,'r:gz') as tf:
 for m in tf:
  assert m.isfile() and m.name in known
  h=hashlib.sha256();f=tf.extractfile(m)
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
  assert m.size==known[m.name]['bytes'] and h.hexdigest()==known[m.name]['sha256'];verified+=1
assert verified==len(manifest)
save(H/'package_receipt.json',dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS',archive=ref(archive),file_manifest=ref(H/'FILE_SHA256.json'),verified_files=verified,raw_requests=len(rpc_index),arm_complete_counts={a:v['arms'][a]['normal_count'] for a in ['top12','top48']},excluded='Task artifacts, test sources, hidden ground truth, model tensors, credentials and prohibited directories are not accessed or packaged.'))
print(json.dumps(load(H/'package_receipt.json')),flush=True)
