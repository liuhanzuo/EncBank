"""Single allocated GPU metadata only; CPU parent holds the one real worker."""
import argparse,datetime,os,socket,subprocess,sys,traceback
from pathlib import Path
from protocol import HERE,ROOT,REMOTE_ROOT,PYTHON,RESOURCE,local,read,save,sha,preflight
from resource_guard import query,require_identity,stable_admission
def now():return datetime.datetime.now().astimezone().isoformat()
def worker(plan,expected,out):
 parent=read(out/'execution.json');identity=parent['physical_gpu_identity']
 assert parent['status']=='worker_running' and parent['stable_interval_seconds']>=45
 rec={'status':'preparing','pid':os.getpid(),'started_at':now(),'plan_sha256':expected,'slurm_job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'physical_gpu_identity':identity,'model_loaded':False,'numerical_cases_executed':0,'quality_evidence':False,'profile_evidence':False}
 save(out/'worker.json',rec)
 try:
  import torch,importlib.metadata
  rec['versions']={name:importlib.metadata.version(name) for name in plan['versions']}
  assert rec['versions']==plan['versions'],rec['versions']
  torch.set_num_threads(2);torch.set_grad_enabled(False)
  assert torch.cuda.device_count()==1 and torch.cuda.get_allocator_backend()=='native'
  torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
  # Save first: a mismatch must preserve the actual descriptor, unlike failed attempt24085.
  rec['torch_device_properties']={'major':int(props.major),'minor':int(props.minor),'name':props.name,'uuid':str(props.uuid),'total_memory':int(props.total_memory)}
  save(out/'worker.json',rec)
  assert str(props.uuid).lower().removeprefix('gpu-')==identity['uuid'].lower().removeprefix('gpu-'),'Torch/NVML UUID mismatch'
  total=int(props.total_memory);assert total>=RESOURCE['minimum_free_bytes']
  torch.cuda.set_per_process_memory_fraction(RESOURCE['allocator_cap_bytes']/total,0)
  assert abs(torch.cuda.get_per_process_memory_fraction(0)*total-RESOURCE['allocator_cap_bytes'])<1
  fresh=query();require_identity(fresh,identity);free,actual_total=torch.cuda.mem_get_info(0)
  assert free>=RESOURCE['minimum_free_bytes']
  rec['fresh_post_context_admission']={**fresh,'cuda_free_bytes':int(free),'cuda_total_bytes':int(actual_total)}
  rec['cuda_policy']={'allocator_backend':'native','allocator_cap_bytes':RESOURCE['allocator_cap_bytes'],'allocated_bytes':int(torch.cuda.memory_allocated()),'reserved_bytes':int(torch.cuda.memory_reserved()),'tensor_allocations_requested':False,'model_or_kernel_imports':False}
  assert rec['cuda_policy']['allocated_bytes']==0 and rec['cuda_policy']['reserved_bytes']==0
  preflight(expected)
  rec['status']='metadata_complete_parent_and_Slurm_exit_pending'
 except BaseException as error:
  rec.update(status='failed_metadata_probe_not_quality',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
 finally:
  rec['finished_at']=now();save(out/'worker.json',rec)
def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-plan-sha256',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 plan=preflight(a.expected_plan_sha256)
 if a.check_only:
  assert not local(plan['batch_output']).exists();print('{"status":"PASS_CPU_paths_hashes_no_GPU"}');return
 assert sys.executable==PYTHON and ROOT==Path(REMOTE_ROOT) and Path(os.environ['QCOMEM_REPO_ROOT']).resolve()==ROOT
 assert os.environ['PYTORCH_CUDA_ALLOC_CONF']=='backend:native'
 for key in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCH_HOME','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH'):
  path=Path(os.environ[key]).resolve();path.relative_to(Path('/srv/encbank').resolve());path.mkdir(parents=True,exist_ok=True)
 out=local(plan['batch_output'])
 if a.worker:worker(plan,a.expected_plan_sha256,out);return
 out.mkdir(parents=True,exist_ok=False)
 rec={'status':'admission_pending','pid':os.getpid(),'started_at':now(),'plan_sha256':a.expected_plan_sha256,'slurm_job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),'automatic_retry':False,'quality_evidence':False,'profile_evidence':False}
 save(out/'execution.json',rec)
 try:
  identity=require_identity(query(),None);rec['physical_gpu_identity']=identity
  def progress(samples,elapsed):rec.update(admissions=samples,stable_interval_seconds=elapsed);save(out/'execution.json',rec)
  stable_admission(identity,progress);preflight(a.expected_plan_sha256)
  rec['fresh_before_worker']=query();require_identity(rec['fresh_before_worker'],identity)
  rec['status']='worker_running';save(out/'execution.json',rec)
  argv=[PYTHON,'-u','-B',str(HERE/'run_probe.py'),'--expected-plan-sha256',a.expected_plan_sha256,'--worker']
  with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
   child=subprocess.Popen(argv,stdout=stdout,stderr=stderr)
   rec.update(worker_pid=child.pid,actual_argv=argv);save(out/'execution.json',rec);code=child.wait()
  rec.update(worker_exit_code=code,actual_parent_wait=True,process_exit_observed=True);save(out/'execution.json',rec)
  assert code==0,('real metadata worker nonzero',code)
  result=read(out/'worker.json');assert result['status']=='metadata_complete_parent_and_Slurm_exit_pending'
  preflight(a.expected_plan_sha256);rec.update(status='metadata_complete_shell_and_Slurm_exit_pending',worker_sha256=sha(out/'worker.json'))
 except BaseException as error:
  rec.update(status='failed_metadata_probe_not_quality',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
 finally:
  rec['finished_at']=now();save(out/'execution.json',rec)
if __name__=='__main__':main()
