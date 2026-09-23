import hashlib, json, os, subprocess, time, types
from pathlib import Path
import torch
from native_common import MODELS, load_model, load_state, tokenizer
from hybrid_reader import HybridReader
from runtime_identity import resolve_configs

def stabilize_decode_matmuls(model,rows=32):
    """Use the same GEMM row shape for 1..32 one-token decode requests.

    Dummy rows enter only row-independent linear projections and are discarded
    immediately. They never enter attention or DeltaNet state. Both dense and
    CoMem use this path, including LoRA and lm_head. No parameter is modified.
    """
    count=0
    def wrap(original):
        def forward(module,x,*args,**kwargs):
            if x.ndim==3 and x.shape[1]==1 and 0<x.shape[0]<rows:
                n=x.shape[0]
                padded=torch.cat([x,x.new_zeros((rows-n,1,x.shape[-1]))],dim=0)
                return original(padded,*args,**kwargs)[:n]
            return original(x,*args,**kwargs)
        return forward
    for module in model.modules():
        if isinstance(module,torch.nn.Linear) or type(module).__name__=='LoRALinear':
            assert not hasattr(module,'_fixed_decode_rows')
            module.forward=types.MethodType(wrap(module.forward),module)
            module._fixed_decode_rows=rows;count+=1
    return count

def stabilize_decode_norms(model,rows=32):
    def wrap(original,target):
        def forward(module,x,*args,**kwargs):
            if (x.ndim>=3 and x.shape[1]==1 and x.shape[0]<rows) or (x.ndim==2 and x.shape[0]<target):
                n=x.shape[0];size=rows if x.ndim>=3 else target
                def pad(t):return torch.cat([t,t.new_zeros((size-n,*t.shape[1:]))],0)
                return original(pad(x),*(pad(a) if torch.is_tensor(a) and a.shape==x.shape else a for a in args),**kwargs)[:n]
            return original(x,*args,**kwargs)
        return forward
    for module in model.modules():
        if type(module).__name__=='Qwen3_5GatedDeltaNet':
            module.norm.forward=types.MethodType(wrap(module.norm.forward,rows*module.num_v_heads),module.norm)
        elif type(module).__name__=='Qwen3_5RMSNorm':
            module.forward=types.MethodType(wrap(module.forward,rows),module)

def install_row_sdpa(reader):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    def forward(module,query,key,value,attention_mask,**kwargs):
        pads=getattr(attention_mask,'_row_left_pads',None)
        if query.shape[2]!=1 or pads is None:
            return sdpa_attention_forward(module,query,key,value,attention_mask,**kwargs)
        # Project Q/K/V and MLP in batches; only the attention reduction is
        # row-local, with native GQA and no padding-dependent kernel shapes.
        outputs=[]
        for b,pad in enumerate(pads):
            out,_=sdpa_attention_forward(module,query[b:b+1].contiguous(),
                key[b:b+1,:,pad:].contiguous(),value[b:b+1,:,pad:].contiguous(),None,**kwargs)
            outputs.append(out)
        return torch.cat(outputs,0),None
    ALL_ATTENTION_FUNCTIONS.register('tf27b_row_sdpa',forward)
    reader.config._attn_implementation='tf27b_row_sdpa'

def load(plan,adapter=True):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.backends.cuda.enable_cudnn_sdp(False)
    free,total=torch.cuda.mem_get_info()
    from common import ROOT,save
    save(ROOT/'gpu_admission.json',dict(epoch=time.time(),free_gib=free/2**30,total_gib=total/2**30,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),
        device=str(torch.cuda.get_device_properties(0)),
        nvml=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv'],text=True)))
    assert free/2**30>=250,('GPU not freshly available',free/2**30)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_gib']*2**30/total)
    identity,cfg=resolve_configs(plan,MODELS[1])
    tok=tokenizer(cfg);model=load_model(cfg);reader=HybridReader(model,21)
    if adapter:
        p=Path(plan['adapter_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==plan['adapter_sha256']
        saved=torch.load(p,map_location='cpu',weights_only=False)
        resolve_configs(plan,identity,saved);reader.attach();load_state(reader,saved,identity);del saved
    model.requires_grad_(False)
    if plan.get('fixed_decode_linear_rows'):
        count=stabilize_decode_matmuls(model,plan['fixed_decode_linear_rows'])
        save(ROOT/'decode_numerics.json',dict(fixed_linear_rows=plan['fixed_decode_linear_rows'],wrapped_modules=count,
            scope='row-independent linear projections only; no attention/state padding; all comparison arms use same setting'))
    if plan.get('stable_decode_norms'):stabilize_decode_norms(model,plan['fixed_decode_linear_rows'])
    if plan.get('row_decode_sdpa'):install_row_sdpa(reader)
    stop=model.generation_config.eos_token_id
    stop=set(stop if isinstance(stop,list) else [stop if stop is not None else tok.eos_token_id])
    return model,reader,tok,stop

def tokens(tok,messages):
    return tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,
        enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=True)
