"""Keep old remote queues intact if shared storage rejects the staged copy."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,uuid
ROOT=Path('/srv/encbank/legacy_workspace');H=Path(__file__).resolve().parent;tag=uuid.uuid4().hex[:12];calls=[];rows=[]
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=40)
 calls.append(dict(argv=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
for host,dest,names in [
 ('longjing-1','/data/liuhanzuo/qcomem_codex_queue_20260908',['EXPERIMENT_QUEUE.md','BENCHMARK_ROADMAP.md']),
 ('gpu-node1','/srv/encbank/qcomem_align_codex_20260911/plan',['EXPERIMENT_QUEUE.md','BENCHMARK_ROADMAP.md','COMEM_BENCHMARK_ALIGNMENT.md'])]:
 row=dict(host=host,directory=dest)
 try:
  assert run(['ssh',host,shlex.join(['realpath',dest])]).strip()==dest
  paths={n:dest+'/'+n+'.pending-'+tag for n in names}
  for n,p in paths.items():run(['scp',str(ROOT/n),host+':'+p])
  hashes=run(['ssh',host,shlex.join(['sha256sum',*paths.values()])])
  actual={line.split()[1]:line.split()[0] for line in hashes.splitlines()}
  expected={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
  for n,p in paths.items():assert actual[p]==expected[n]
  for n,p in paths.items():run(['ssh',host,shlex.join(['mv','--',p,dest+'/'+n])])
  hashes=run(['ssh',host,shlex.join(['sha256sum',*[dest+'/'+n for n in names]])])
  actual={line.split()[1]:line.split()[0] for line in hashes.splitlines()}
  assert all(actual[dest+'/'+n]==expected[n] for n in names)
  row.update(status='SYNCED_SHA_VERIFIED',hashes=actual)
 except Exception as exc:row.update(status='SYNC_FAILED_OLD_QUEUE_NOT_REPLACED',error=str(exc))
 rows.append(row)
record=dict(at=datetime.datetime.now().astimezone().isoformat(),destinations=rows,calls=calls,new_gpu_requests=0)
(H/'queue_sync.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print(json.dumps(rows,ensure_ascii=False))
