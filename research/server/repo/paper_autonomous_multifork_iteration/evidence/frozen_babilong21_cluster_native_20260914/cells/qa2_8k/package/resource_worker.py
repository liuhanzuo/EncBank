"""Six-method native resource envelope with frozen FP16 LongEval driver and timing."""
import datetime,gc,os,socket,sys,traceback
from pathlib import Path
from protocol import ROOT,HERE,RESOURCE,local,model_path,parser,preflight,read,save,sha,resolve_driver
from resource_guard import query,require_identity
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
 args=parser().parse_args();plan,activation=preflight(args,envelope=not args.check_only)
 assert args.arm=='encbank_frozen_j12' and not args.arm.startswith('h')
 if args.check_only:print('{"status":"native_file_bindings_checked_no_cuda"}');return
 run,natural_read=resolve_driver(plan)  # Exact origin before sys.path mutations and model imports.
 assert Path(os.environ['QENCBANK_REPO_ROOT']).resolve()==ROOT
 output=Path(args.output);parent=read(output/'execution.json')
 assert parent['status']=='worker_running' and parent['gpu_worker_started'] and parent['stable_interval_seconds']>=45
 identity=parent['physical_gpu_identity'];record={'status':'preparing','arm':args.arm,'pid':os.getpid(),'plan_sha256':args.expected_plan_sha256,'started_at':now(),'model_loaded':False,'training_allowed':False,'runtime_offload_allowed':False,'fallback_allowed':False,'quality_evidence':False,'physical_gpu_identity':identity}
 save(output/'worker.json',record)
 try:
  assert os.environ['PYTORCH_CUDA_ALLOC_CONF']=='backend:native' and os.environ['HF_HUB_OFFLINE']=='1' and os.environ['TRANSFORMERS_OFFLINE']=='1'
  # Bind the staged same local reader; never substitute the other remote Base directory.
  for name,digest in activation['model_file_sha256'].items():assert sha(model_path(plan,'model')/name)==digest,name
  for name,info in activation['files'].items():assert sha(model_path(plan,'adapter')/name)==info['sha256'],name
  from backend_gate import qualified_node_identity
  record['node_identity']=qualified_node_identity(plan,os.environ,socket.gethostname())
  from qencbank_triton_cache import install as install_triton_cache
  record['triton_cache_cleanup_fix']=install_triton_cache(plan['task_cache_root']+'/triton')
  save(output/'worker.json',record)
  import torch,transformers,peft,triton
  assert torch.__version__=='2.10.0+cu128' and transformers.__version__=='5.5.4' and peft.__version__=='0.20.0' and triton.__version__=='3.6.0'
  torch.set_num_threads(2);torch.set_grad_enabled(False)
  def forbidden(*a,**kw):raise RuntimeError('Backward/autograd.grad forbidden')
  torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
  assert torch.cuda.device_count()==1 and torch.cuda.get_allocator_backend()=='native'
  torch.cuda.set_device(0);properties=torch.cuda.get_device_properties(0)
  assert identity['name']==properties.name=='NVIDIA L20D','Allocated GPU differs from cluster-native policy'
  assert torch.cuda.current_stream(0)==torch.cuda.default_stream(0)
  assert str(properties.uuid).lower().removeprefix('gpu-')==identity['uuid'].lower().removeprefix('gpu-')
  expected_cc=plan['backend_qualification']['compute_capability']
  assert [properties.major,properties.minor]==expected_cc,'Allocated compute capability differs from cluster-native inventory policy'
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
  sys.path[:0]=[str(ROOT),str(ROOT/'tmp_external_baselines/encbank_official'),str(ROOT/'gpu/kvquant_quality_balanced')]
  from eval._common import load_backbone
  from phase_profile import Profiler
  profiler=Profiler(torch)
  with torch.inference_mode(),profiler.phase('outer_model_load_native_quality_and_release'):
   # Original common-loader math/LoRA setup; CPU construction is not runtime offload.
   model,tokenizer=load_backbone(str(model_path(plan,'model')),dtype='float16',attn_impl='sdpa',device='cpu',lora_adapter=str(model_path(plan,'adapter')) if args.arm.startswith('h') else None)
   assert all(p.device.type=='cpu' for p in model.parameters())
   gate();model=model.to('cuda:0')
   assert not model.training and all(p.device.type=='cuda' for p in model.parameters())
   assert all(b.device.type=='cuda' for b in model.buffers())
   assert not any(hasattr(m,'_hf_hook') for m in model.modules()) and not getattr(model,'hf_device_map',None)
   model.requires_grad_(False)
   assert model.config.num_hidden_layers==36 and model.config.hidden_size==4096 and model.config.num_key_value_heads==8 and model.config.head_dim==128
   record.update(model_loaded=True,model_parameters_all_cuda=True,model_load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_load_peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_only_allocated_bytes=torch.cuda.memory_allocated(),model_only_reserved_bytes=torch.cuda.memory_reserved(),model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
   save(output/'worker.json',record)
   fixture=read(local(plan['fixture']['path']));assert model.config.bos_token_id==151643 and tokenizer.eos_token_id==151645
   provenance={'plan_sha256':args.expected_plan_sha256,'fixture_sha256':plan['fixture']['sha256'],'activation_sha256':plan['activation']['sha256'],'source_sha256':plan['source_sha256'],'model_file_sha256':activation['model_file_sha256'],'adapter_file_sha256':{},'adapter_files_audited_but_unused':activation['files'],'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':torch.cuda.get_device_name(0),'physical_gpu_identity':identity,'slurm_job_id':os.environ['SLURM_JOB_ID'],'attention_backend':'sdpa','backbone_dtype':'torch.float16','model_load':{k:v for k,v in record.items() if k.startswith('model_')},'resource_policy':RESOURCE,'reader_binding':plan['reader_binding'],'adapter_active':args.arm.startswith('h'),'same_custom_reader_for_H16_H8_H4':False,'method_configuration':plan['method_configuration'],'exact_paper_checkpoint_replication':False,'first_step_EOS_suppressed':True,'single_EOS_override':151645,'explicit_BOS_from_model_config':151643,'original_tokenizer_BOS':tokenizer.bos_token_id,'original_generation_config_EOS':read(model_path(plan,'model')/'generation_config.json')['eos_token_id'],'lexical_metadata_CPU_is_not_H_or_KV_offload':True,'device_block':'cluster-native Slurm single physical GPU; actual node/hostname/UUID retained, never RTX5090 timing evidence'}
   provenance.update(fixture=plan['fixture'],physical_gpu_identity=identity,resource_policy=RESOURCE,scope='New independent frozen Encbank j12 LoRA OFF qa2_8k100; exact original inputs and cap20; native FP16 SDPA.')
   assert model.config._attn_implementation=='sdpa'
   assert model.config.num_hidden_layers==36 and model.config.hidden_size==4096
   assert model.config.bos_token_id==151643 and tokenizer.eos_token_id==151645
   from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
   from transformers.integrations.sdpa_attention import sdpa_attention_forward
   assert ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',None) is sdpa_attention_forward
   assert sum(type(m).__name__=='Qwen3Attention' for m in model.modules())==model.config.num_hidden_layers
   assert all(m.config._attn_implementation=='sdpa' for m in model.modules() if type(m).__name__=='Qwen3Attention')
   for rel,digest in plan['native_attention_runtime_files'].items():assert sha(Path(transformers.__file__).parent/rel)==digest,rel
   provenance.update(native_attention_policy=plan['native_attention_policy'],native_attention_entrypoint='transformers.integrations.sdpa_attention.sdpa_attention_forward',native_attention_runtime_files=plan['native_attention_runtime_files'],CUDA_SDPA_dispatcher_identity='not_observed_or_pinned')
   dtypes={n:str(p.dtype) for n,p in model.named_parameters() if 'lora_' in n}
   assert all(p.dtype==torch.float16 for n,p in model.named_parameters() if 'lora_' not in n)
   if args.arm.startswith('h'):assert dtypes and set(dtypes.values())=={'torch.float32'}
   else:assert not dtypes and not getattr(model,'peft_config',None)
   provenance.update(backbone_dtype='torch.float16',attention_backend='sdpa',LoRA_parameter_dtypes=sorted(set(dtypes.values())),configuration={'resume_j':12,'H_group_size':64,'KIVI_group_size':32,'KIVI_residual_length':128,'query_tokens_per_call':1,'max_new_tokens':20,'BOS_token_id':151643,'EOS_token_id':151645,'first_step_EOS_suppressed':True})
   provenance['triton_cache_cleanup_fix']=record['triton_cache_cleanup_fix']
   provenance['backend_qualification']=plan['backend_qualification']
   provenance['node_identity']=record['node_identity']
   provenance.update(actual_BOS_token_id=151643,EOS_token_id=151645,first_step_EOS_suppressed=True,original_tokenizer_BOS=tokenizer.bos_token_id,original_generation_config_EOS=read(model_path(plan,'model')/'generation_config.json')['eos_token_id'],model_and_tokenizer_configs_mutated=False)
   provenance['executed_attention_policy']=('KIVI SDPA full-document prefill and official Half cached GEMV' if args.arm.startswith('kivi') else 'Native HF SDPA attention, dispatcher not pinned')
   sys.path.insert(0,str(local(plan['tokenwise_runtime_root'])/'runtime'))
   kernels=None
   if args.arm.startswith('kivi'):
       backend=plan['kivi_backend'];binary=local(backend['binary']['path'])
       assert sha(binary)==backend['binary']['sha256']
       sys.path.insert(0,str(binary.parent))
       from bridge import OfficialHalfKernels
       import bridge
       assert Path(bridge.__file__).resolve()==local(plan['tokenwise_runtime_root'])/'runtime/bridge.py'
       kernels=OfficialHalfKernels(local(backend['root']),backend['source_sha256'])
       import kivi_gemv
       assert Path(kivi_gemv.__file__).resolve()==binary and sha(binary)==backend['binary']['sha256']
       provenance['actual_kivi_backend']={'binary':backend['binary'],'module_file':str(Path(kivi_gemv.__file__).resolve()),'source_sha256':backend['source_sha256'],'performance_backend':kernels.performance_backend}
   provenance['resolved_quality_driver_file']=run.__code__.co_filename
   provenance['resolved_natural_qa_file']=natural_read.__code__.co_filename
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
