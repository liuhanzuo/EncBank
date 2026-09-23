"""Root-owned persistent native holder. Preparation never invokes this script."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, traceback
H=Path(__file__).resolve().parent
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['first','remaining'],required=True);p.add_argument('--expected-plan-sha256',required=True);a=p.parse_args()
 assert sha(H/'plan.json')==a.expected_plan_sha256
 out=H/('held_'+a.phase);out.mkdir(exist_ok=False)
 argv=[sys.executable,'-X','utf8','-B',str(H/'launch.py'),'--expected-plan-sha256',a.expected_plan_sha256,'--max-new-batches','1' if a.phase=='first' else '0']
 r={'status':'STARTING','argv':argv,'started_at':now(),'automatic_retry':False,'holder_pid':__import__('os').getpid(),'holder_source_sha256':sha(Path(__file__)),'launcher_sha256':sha(H/'launch.py'),'plan_sha256':a.expected_plan_sha256};save(out/'receipt.json',r)
 try:
  with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
   child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,cwd=H,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));r.update(pid=child.pid,status='RUNNING');save(out/'receipt.json',r);code=child.wait()
  r.update(actual_exit_code=code,actual_parent_wait=True,process_exit_observed=True,status='COMPLETED_ACTUAL0' if code==0 else 'FAILED_PRESERVED_NO_RETRY')
 except BaseException as error:r.update(status='HOLDER_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc());raise
 finally:r.update(finished_at=now(),stdout_sha256=sha(out/'stdout.log'),stderr_sha256=sha(out/'stderr.log'));save(out/'receipt.json',r)
 raise SystemExit(r['actual_exit_code'])
if __name__=='__main__':main()
