"""Isolate the first abnormal SDPA operation on one saved Dense trajectory."""
from pathlib import Path
import copy,gc,hashlib,inspect,json,os,platform,subprocess,time,traceback
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from common import MODELS,load_model,dump
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
R=H/'run_dense';plan=json.loads((H/'agent_plan.json').read_text())
raw=(H/'fixed_history.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a55f762a1fa3448170a51443f96b3e2b5e500f4363b96d0393527e508d4fb3b7'
history=json.loads(raw);context={};events=(R/'events.jsonl').open('x');started=time.time()
original_sdpa=F.scaled_dot_product_attention
def event(kind,**data):
 row=dict(event=kind,epoch=time.time(),**data);line=json.dumps(row);events.write(line+'\n');events.flush();print(line,flush=True)
def stats(t):
 return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),finite=bool(torch.isfinite(t).all()),absmax=float(t.abs().max()))
def tensors(v,prefix=''):
 if torch.is_tensor(v):yield prefix,v
 elif isinstance(v,dict):
  for k,x in v.items():yield from tensors(x,prefix+'/'+str(k))
 elif isinstance(v,(list,tuple)):
  for i,x in enumerate(v):yield from tensors(x,prefix+'/'+str(i))
def cache_stats(cache):
 return [dict(layer=i,field=n,path=p,**stats(t)) for i,l in enumerate(cache.layers)
         for n in ('keys','values','conv_states','recurrent_states')
         for p,t in tensors(getattr(l,n,None))]
def compare(a,b):
 x=a.float();y=b.float();d=x-y
 return dict(max_abs_error=float(d.abs().max()),mean_abs_error=float(d.abs().mean()),
             relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y).clamp_min(1e-20)),
             cosine=float(F.cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)).item()))
captured=[]
def watched_sdpa(q,k,v,*args,**kw):
 if context.get('capture') and not captured:
  baseline=original_sdpa(q,k,v,*args,**kw)
  descriptor=dict(context=dict(context),query=stats(q),key=stats(k),value=stats(v),
      args=[stats(x) if torch.is_tensor(x) else x for x in args],
      kwargs={n:stats(x) if torch.is_tensor(x) else x for n,x in kw.items()},default=stats(baseline))
  # CPU copies are diagnostic input exports, never runtime cache/model offload.
  blob=dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),
            args=tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in args),
            kwargs={n:x.detach().cpu() if torch.is_tensor(x) else x for n,x in kw.items()},
            default=baseline.detach().cpu())
  torch.save(blob,R/'first_attention_inputs.pt');del blob
  descriptor['capture_sha256']=hashlib.sha256((R/'first_attention_inputs.pt').read_bytes()).hexdigest()
  controls={}
  for name in ['MATH','EFFICIENT_ATTENTION','FLASH_ATTENTION','CUDNN_ATTENTION']:
   try:
    with sdpa_kernel(getattr(SDPBackend,name)):
     result=original_sdpa(q,k,v,*args,**kw)
    torch.cuda.synchronize();controls[name]=dict(status='computed',stats=stats(result),difference_from_default=compare(result,baseline))
    torch.save(result.detach().cpu(),R/('first_attention_'+name+'.pt'));del result
   except RuntimeError as exc:
    controls[name]=dict(status='unavailable_or_error',error=str(exc))
  descriptor['controls']=controls;captured.append(descriptor);event('first_attention_backend_controls',**descriptor)
  return baseline
 return original_sdpa(q,k,v,*args,**kw)
try:
 assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
 torch.set_num_threads(2);torch.manual_seed(plan['seed']);free,total=torch.cuda.mem_get_info()
 assert free>=70*2**30;cap=min(80*2**30,int(total*.9));torch.cuda.set_per_process_memory_fraction(cap/total)
 event('admission',free=free,total=total,cap=cap,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),processes=subprocess.check_output(['ps','-u','liuhanzuo','-o','pid,ppid,comm'],text=True))
 model=load_model(MODELS[1]);model.requires_grad_(False);assert all(p.device.type=='cuda' for p in model.parameters())
 import transformers,transformers.models.qwen3_5.modeling_qwen3_5 as impl
 sources={}
 for obj in [impl,transformers.cache_utils]:
  p=Path(inspect.getsourcefile(obj));sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 dump(R/'worker_ready.json',dict(pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],host=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,sources=sources,input_sha256=hashlib.sha256(raw).hexdigest(),diagnostic_only=True))
 F.scaled_dot_product_attention=watched_sdpa
 def reset():
  for obj in [model,model.model]:
   if hasattr(obj,'rope_deltas'):obj.rope_deltas=None
 def forward(ids,cache,length=None):
  kw=dict(input_ids=torch.tensor([ids],device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1)
  if length is not None:kw['attention_mask']=torch.ones((1,length),device='cuda',dtype=torch.long)
  out=model(**kw);return out.logits,out.past_key_values
 def sample(logits):
  assert bool(torch.isfinite(logits).all()),'Nonfinite before saved-history sample'
  values,indices=(logits[0,-1].float()/plan['temperature']).topk(plan['top_k'])
  mask=values.softmax(-1).cumsum(-1)>plan['top_p'];mask[1:]=mask[:-1].clone();mask[0]=False
  values[mask]=-float('inf');return int(indices[torch.multinomial(values.softmax(-1),1)].item())
 with torch.inference_mode():
  reset();torch.manual_seed(plan['seed']);cache=None;matches=0;compared=0
  for row in history['steps'][:10]:
   context.update(step=row['step'],phase='saved_replay',capture=False)
   gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
   assert (cache.get_seq_length() if cache else 0)==row['cached_prefix_tokens']
   logits,cache=forward(row['new_prefill_ids'],cache,len(row['ledger_ids']))
   ids=row['saved_generated_ids']
   for j,t in enumerate(ids):
    matches+=int(sample(logits)==t);compared+=1
    if j<len(ids)-1:logits,cache=forward([t],cache)
   event('saved_step_replayed',step=row['step'],matches=matches,compared=compared)
   del logits
  row=history['steps'][10];assert cache.get_seq_length()==row['cached_prefix_tokens']
  event('cache_before_failure_prefill',cache=cache_stats(cache))
  control_cache=copy.deepcopy(cache)
  for a,b in zip(cache.layers,control_cache.layers):
   for n in ('keys','values','conv_states','recurrent_states'):
    aa=list(tensors(getattr(a,n,None)));bb=list(tensors(getattr(b,n,None)));assert len(aa)==len(bb)
    for (pa,ta),(pb,tb) in zip(aa,bb):assert pa==pb and torch.equal(ta,tb) and ta.data_ptr()!=tb.data_ptr()
  context.update(step=10,phase='default_cached_prefill',capture=True)
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  baseline,cache=forward(row['new_prefill_ids'],cache,len(row['ledger_ids']))
  event('default_cached_prefill',logits=stats(baseline),cache=cache_stats(cache))
  baseline_copy=baseline.detach().cpu();torch.save(baseline_copy,R/'default_logits.pt')
  del baseline;cache=None;gc.collect();torch.cuda.empty_cache()
  context.update(phase='math_cached_prefill',capture=False)
  with sdpa_kernel(SDPBackend.MATH):
   repaired,control_cache=forward(row['new_prefill_ids'],control_cache,len(row['ledger_ids']))
  event('math_cached_prefill',logits=stats(repaired),cache=cache_stats(control_cache))
  repaired_copy=repaired.detach().cpu();torch.save(repaired_copy,R/'math_logits.pt')
  del repaired;control_cache=None;gc.collect();torch.cuda.empty_cache();reset()
  context.update(phase='fresh_default_prefill',capture=False)
  fresh,cache=forward(row['ledger_ids'],None,len(row['ledger_ids']))
  fresh_copy=fresh.detach().cpu();torch.save(fresh_copy,R/'fresh_logits.pt')
  result=dict(diagnostic_only=True,saved_sample_matches=matches,saved_sample_compared=compared,
              backend_controls=captured,default_logits=stats(baseline_copy),math_logits=stats(repaired_copy),fresh_logits=stats(fresh_copy),
              default_vs_fresh=compare(baseline_copy,fresh_copy),math_vs_fresh=compare(repaired_copy,fresh_copy),
              elapsed_seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
  dump(R/'diagnostic_result.json',result);event('diagnostic_complete',**result)
except BaseException:
 dump(R/'worker_failure.json',dict(traceback=traceback.format_exc(),context=context));raise
finally:
 F.scaled_dot_product_attention=original_sdpa;events.close()
