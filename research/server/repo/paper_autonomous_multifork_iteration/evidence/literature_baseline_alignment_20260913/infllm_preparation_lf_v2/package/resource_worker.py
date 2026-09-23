"""Qualified single-GPU resource envelope; one FP16 single2 methods and natural timing."""
import datetime,gc,os,socket,sys,time,traceback
from pathlib import Path
from protocol import ROOT,HERE,RESOURCE,local,model_path,parser,preflight,read,save,sha,resolve_driver
from resource_guard import query,require_identity
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
 started_clock=time.perf_counter()
 args=parser().parse_args();plan,activation=preflight(args,envelope=not args.check_only)
 assert args.arm=='infllm_resident_qwen3' and not args.arm.startswith('h')
 if args.check_only:print('{"status":"native_file_bindings_checked_no_cuda"}');return
 run,natural_read=resolve_driver(plan)  # Exact origin before sys.path mutations and model imports.
 assert Path(os.environ['QCOMEM_REPO_ROOT']).resolve()==ROOT
 output=Path(args.output);parent=read(output/'execution.json')
 assert parent['status']=='worker_running' and parent['gpu_worker_started'] and parent['stable_interval_seconds']>=45
 identity=parent['physical_gpu_identity'];record={'status':'preparing','arm':args.arm,'pid':os.getpid(),'plan_sha256':args.expected_plan_sha256,'started_at':now(),'model_loaded':False,'training_allowed':False,'runtime_offload_allowed':False,'fallback_allowed':False,'quality_evidence':False,'physical_gpu_identity':identity}
 save(output/'worker.json',record)
 try:
  assert os.environ['PYTORCH_CUDA_ALLOC_CONF']=='backend:native' and os.environ['HF_HUB_OFFLINE']=='1' and os.environ['TRANSFORMERS_OFFLINE']=='1'
  # Bind the staged same local reader; never substitute the other remote Base directory.
  for name,digest in activation['model_file_sha256'].items():assert sha(model_path(plan,'model')/name)==digest,name
  from backend_gate import qualified_node_identity
  record['node_identity']=qualified_node_identity(plan,os.environ,socket.gethostname())
  import torch,transformers
  assert torch.__version__=='2.10.0+cu128' and transformers.__version__=='5.5.4'
  torch.set_num_threads(2);torch.set_grad_enabled(False)
  def forbidden(*a,**kw):raise RuntimeError('Backward/autograd.grad forbidden')
  torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
  assert torch.cuda.device_count()==1 and torch.cuda.get_allocator_backend()=='native'
  torch.cuda.set_device(0);properties=torch.cuda.get_device_properties(0)
  assert torch.cuda.current_stream(0)==torch.cuda.default_stream(0)
  assert str(properties.uuid).lower().removeprefix('gpu-')==identity['uuid'].lower().removeprefix('gpu-')
  expected_cc=plan['backend_qualification']['compute_capability']
  assert [properties.major,properties.minor]==expected_cc,'Allocated compute capability differs from qualified binary'
  record['qualified_compute_capability']=expected_cc
  total=int(properties.total_memory);assert total>=RESOURCE['minimum_free_bytes']
  torch.cuda.set_per_process_memory_fraction(RESOURCE['allocator_cap_bytes']/total,0)
  assert abs(torch.cuda.get_per_process_memory_fraction(0)*total-RESOURCE['allocator_cap_bytes'])<1
  record['cuda_policy']={'allocator_backend':'native','allocator_cap_bytes':RESOURCE['allocator_cap_bytes'],'device_total_bytes':total,'device_name':properties.name,'device_uuid':str(properties.uuid),'fraction':torch.cuda.get_per_process_memory_fraction(0)}
  def gate():
   snapshot=query();require_identity(snapshot,identity);free,actual_total=torch.cuda.mem_get_info(0)
   snapshot.update(cuda_free_bytes=int(free),cuda_total_bytes=int(actual_total))
   record.setdefault('pre_migration_checks',[]).append(snapshot);save(output/'worker.json',record)
   assert free>=RESOURCE['minimum_free_bytes'],'Fresh CUDA free-memory admission denied'
  gate()
  sys.path[:0]=[str(ROOT),str(ROOT/'tmp_external_baselines/comem_official'),str(ROOT/'gpu/kvquant_quality_balanced')]
  from eval._common import load_backbone
  from phase_profile import Profiler
  profiler=Profiler(torch)
  with torch.inference_mode(),profiler.phase('outer_model_load_native_quality_and_release'):
   # Original common-loader math/LoRA setup; CPU construction is not runtime offload.
   model_load_clock=time.perf_counter()
   model,tokenizer=load_backbone(str(model_path(plan,'model')),dtype='float16',attn_impl='sdpa',device='cpu',lora_adapter=None)
   assert all(p.device.type=='cpu' for p in model.parameters())
   gate();model=model.to('cuda:0')
   assert not model.training and all(p.device.type=='cuda' for p in model.parameters())
   assert all(b.device.type=='cuda' for b in model.buffers())
   assert not any(hasattr(m,'_hf_hook') for m in model.modules()) and not getattr(model,'hf_device_map',None)
   model.requires_grad_(False)
   assert model.config.num_hidden_layers==36 and model.config.hidden_size==4096 and model.config.num_key_value_heads==8 and model.config.head_dim==128
   record.update(model_loaded=True,model_parameters_all_cuda=True,model_load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_load_peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_only_allocated_bytes=torch.cuda.memory_allocated(),model_only_reserved_bytes=torch.cuda.memory_reserved(),model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
   record['model_load_and_migration_seconds']=time.perf_counter()-model_load_clock
   record['worker_cold_start_through_model_ready_seconds']=time.perf_counter()-started_clock
   save(output/'worker.json',record)
   fixture=read(local(plan['fixture']['path']));assert model.config.bos_token_id==151643 and tokenizer.eos_token_id==151645
   for rel,digest in plan['backbone_runtime_files'].items():
    assert sha(Path(transformers.__file__).parent/rel)==digest,rel
   assert all(p.dtype==torch.float16 for p in model.parameters())
   assert not any('lora_' in n for n,_ in model.named_parameters()) and not getattr(model,'peft_config',None)
   assert model.config.model_type=='qwen3'
   provenance={'plan_sha256':args.expected_plan_sha256,'fixture_sha256':plan['fixture']['sha256'],
    'fixture':plan['fixture'],'activation_sha256':plan['activation']['sha256'],
    'source_sha256':plan['source_sha256'],'model_file_sha256':activation['model_file_sha256'],
    'adapter_file_sha256':{},'adapter_active':False,'LoRA_parameter_dtypes':[],
    'torch':torch.__version__,'transformers':transformers.__version__,
    'gpu':torch.cuda.get_device_name(0),'physical_gpu_identity':identity,
    'slurm_job_id':os.environ['SLURM_JOB_ID'],
    'attention_backend':'infllm_upstream_torch_joint_softmax','backbone_dtype':'torch.float16',
    'model_load':{k:v for k,v in record.items() if k.startswith('model_') or k.startswith('worker_cold')},
    'resource_policy':RESOURCE,'reader_binding':plan['reader_binding'],
    'method_configuration':plan['method_configuration'],'configuration':plan['configuration'],
    'exact_paper_checkpoint_replication':False,'runtime_offload':False,
    'actual_BOS_token_id':151643,'EOS_token_id':151645,'first_step_EOS_suppressed':True,
    'original_tokenizer_BOS':tokenizer.bos_token_id,
    'original_generation_config_EOS':read(model_path(plan,'model')/'generation_config.json')['eos_token_id'],
    'model_and_tokenizer_configs_mutated':False,'backbone_runtime_files':plan['backbone_runtime_files'],
    'backend_qualification':plan['backend_qualification'],'node_identity':record['node_identity'],
    'backend_qualification_scope':'Prior platform identity only; no prior InfLLM GPU execution or SDPA equivalence claim',
    'executed_attention_policy':'Pinned upstream PyTorch multistage joint softmax; Qwen3 q/k norms; InfLLM RoPE',
    'host_transfers':'Block IDs and scalar selection metadata; entry audit hashing outside natural timing. No KV backing or model runtime offload.',
    'resolved_quality_driver_file':run.__code__.co_filename,
    'resolved_natural_qa_file':natural_read.__code__.co_filename,
    'scope':'One adapted InfLLM Single-needle16K100 cell; no original published checkpoint/protocol replication claim'}
   kernels=None
   result=run(model,tokenizer,fixture,args.arm,plan=plan,output=output/'result.json',provenance=provenance,profiler=profiler,natural_read=natural_read,kernels=kernels)
   assert len(result['rows'])==plan['items_per_arm']==100 and len(result['documents'])==plan['documents_per_arm']==100
   assert len(profiler.records)==plan['expected_complete_phases_per_arm']==601 and all(x['completed'] for x in profiler.records[1:])
   del kernels
   del model,tokenizer;gc.collect();torch.cuda.synchronize()
   record.update(after_model_release_allocated_bytes=torch.cuda.memory_allocated(),after_model_release_reserved_bytes=torch.cuda.memory_reserved())
  assert all(x['completed'] for x in profiler.records) and not profiler.stack
  result.update(status='worker_complete_parent_exit_pending',outer_closed=True);save(output/'result.json',result)
  record.update(scientific_result_sha256=sha(output/'result.json'),status='completed')
 except BaseException as error:
  record.update(status='failed',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
 finally:
  if 'profiler' in locals():record['phases']=profiler.records
  record['finished_at']=now();save(output/'worker.json',record)
if __name__=='__main__':main()
