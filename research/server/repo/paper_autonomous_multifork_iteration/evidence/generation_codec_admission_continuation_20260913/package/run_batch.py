"""One allocated GPU; five real children then pinned official-scoring CPU analysis."""
import argparse,datetime,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from protocol import ARMS,HERE,ROOT,REMOTE_ROOT,PYTHON,local,read,save,sha,preflight
from resource_guard import query,require_identity,stable_admission
active_child=None
def now():return datetime.datetime.now().astimezone().isoformat()
def interrupted(signum,frame):
 global active_child
 if active_child is not None and active_child.poll() is None:
  # Only the child process group this supervisor created, never other jobs/services.
  os.killpg(active_child.pid,signal.SIGTERM)
 raise SystemExit(128+signum)
def run_child(argv,out,record):
 global active_child
 with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  active_child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=os.environ.copy(),start_new_session=True)
  record.update(worker_pid=active_child.pid,actual_argv=argv);save(out/'execution.json',record)
  try:code=active_child.wait()
  except BaseException:
   if active_child.poll() is None:
    os.killpg(active_child.pid,signal.SIGTERM)
    try:active_child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(active_child.pid,signal.SIGKILL);active_child.wait()
   record.update(worker_exit_code=active_child.returncode,actual_parent_wait=True,process_exit_observed=True);save(out/'execution.json',record)
   raise
 record.update(worker_exit_code=code,actual_parent_wait=True,process_exit_observed=True);save(out/'execution.json',record)
 active_child=None
 return code
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--expected-plan-sha256',required=True);a=p.parse_args()
 assert sys.executable==PYTHON and ROOT==Path(REMOTE_ROOT) and Path(os.environ['QCOMEM_REPO_ROOT'])==ROOT
 assert sha(a.plan)==a.expected_plan_sha256
 plan=read(a.plan)
 from backend_gate import validate_backend_binding
 validate_backend_binding(plan)  # Fail before any output directory or GPU call.
 root=local(plan['batch_output']);assert not root.exists();root.mkdir(parents=True,exist_ok=False)
 batch={'status':'starting','started_at':now(),'pid':os.getpid(),'plan_sha256':a.expected_plan_sha256,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'node':socket.gethostname(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'arm_order':list(ARMS),'stages':[],'automatic_retry':False,'local_services_modified':False}
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 save(root/'batch_status.json',batch);success=False
 try:
  snapshot=query();batch['initial_admission_snapshot']=snapshot;save(root/'batch_status.json',batch)
  identity=require_identity(snapshot,None);batch['physical_gpu_identity']=identity;save(root/'batch_status.json',batch)
  for arm in ARMS:
   out=local(plan['outputs'][arm]);args=argparse.Namespace(plan=a.plan,expected_plan_sha256=a.expected_plan_sha256,arm=arm,output=str(out))
   preflight(args);out.mkdir(parents=True,exist_ok=False)
   record={'status':'admission_pending','arm':arm,'plan_sha256':a.expected_plan_sha256,'started_at':now(),'gpu_worker_started':False,'physical_gpu_identity':identity,'slurm_job_id':os.environ['SLURM_JOB_ID'],'automatic_retry':False,'quality_evidence':False}
   save(out/'execution.json',record)
   try:
    def progress(samples,elapsed):
     record.update(admissions=samples,stable_interval_seconds=elapsed);save(out/'execution.json',record)
    stable_admission(identity,progress)
    preflight(args,envelope=True)
    record.update(status='worker_running',gpu_worker_started=True);save(out/'execution.json',record)
    argv=[PYTHON,'-u','-B',str(HERE/'resource_worker.py'),'--plan',a.plan,'--expected-plan-sha256',a.expected_plan_sha256,'--arm',arm,'--output',str(out)]
    code=run_child(argv,out,record)
    assert code==0,('Actual native worker exit is nonzero',code)
    worker=read(out/'worker.json');assert worker['status']=='completed' and worker['scientific_result_sha256']==sha(out/'result.json')
    assert all(sha(local(name))==digest for name,digest in plan['source_sha256'].items())
    record.update(status='completed_pending_independent_analysis',source_stable=True,worker_sha256=sha(out/'worker.json'))
   except BaseException as error:
    record.update(status='infrastructure_or_execution_failure_not_complete_evidence',error={'type':type(error).__name__,'message':str(error)});raise
   finally:
    record['finished_at']=now();save(out/'execution.json',record)
    batch['stages'].append({'arm':arm,'execution_sha256':sha(out/'execution.json'),'status':record['status'],'actual_exit_code':record.get('worker_exit_code')});save(root/'batch_status.json',batch)
  batch.update(status='all_five_arms_exited_complete_pending_analysis');save(root/'batch_status.json',batch)
  analysis_out=root/'analysis_cpu';analysis_out.mkdir(exist_ok=False)
  rec={'status':'analysis_running','started_at':now(),'plan_sha256':a.expected_plan_sha256}
  argv=[PYTHON,'-u','-B',str(HERE/'analyze.py'),'--plan',a.plan,'--expected-plan-sha256',a.expected_plan_sha256,'--output',str(root/'completed_analysis.json')]
  code=run_child(argv,analysis_out,rec)
  rec.update(finished_at=now(),status='completed' if code==0 else 'failed');save(analysis_out/'execution.json',rec)
  assert code==0 and (root/'completed_analysis.json').is_file()
  batch.update(analysis_sha256=sha(root/'completed_analysis.json'),analysis_execution_sha256=sha(analysis_out/'execution.json'))
  success=True
 except BaseException as error:
  batch['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
 finally:
  batch.update(status='five_arms_and_analysis_complete_shell_and_slurm_exit_pending' if success else 'stopped_without_retry_not_complete_five_arm_evidence',finished_at=now())
  save(root/'batch_status.json',batch)
 raise SystemExit(0 if success else 1)
if __name__=='__main__':main()
