"""Candidate Qwen3 attention/Cache bridge for original KIVI Half kernels.

No loading, dtype conversion, CUDA build, scheduler or profiling entry point.
The CPU reference kernel is never an admissible performance backend.
"""
from __future__ import annotations
from dataclasses import dataclass
import importlib.metadata
import math
import types
import torch
from torch.nn import functional as F

FIELDS=('key_codes','key_tail','key_scale','key_min','value_codes','value_tail','value_scale','value_min')

def repeat_heads(x,groups):
    if groups==1:return x
    b,h,n,d=x.shape
    return x[:,:,None,:,:].expand(b,h,groups,n,d).reshape(b,h*groups,n,d)

def require_half(*values):
    for value in values:
        if value is not None and value.dtype!=torch.float16:
            raise TypeError('KIVI official Half path requires FP16 values/scales/tails')

def append(old,new,dim):return new if old is None else torch.cat((old,new),dim=dim)

@dataclass
class LayerState:
    bits:int
    group_size:int=32
    residual_length:int=128
    sequence_length:int=0
    key_codes:object=None
    key_tail:object=None
    key_scale:object=None
    key_min:object=None
    value_codes:object=None
    value_tail:object=None
    value_scale:object=None
    value_min:object=None
    frozen:bool=False
    released:bool=False

    def __post_init__(self):
        if self.bits not in (2,4) or self.group_size!=32 or self.residual_length!=128:
            raise ValueError('Frozen candidate: K/V same2or4bit, group32, residual128')

    def live_mutable(self):
        if self.released or self.frozen:raise RuntimeError('Use a fresh request fork; entry is immutable/released')

    def tensor_items(self):return tuple((k,getattr(self,k)) for k in FIELDS if getattr(self,k) is not None)

    def fork(self):
        if self.released or not self.frozen:raise RuntimeError('Only a frozen live entry can fork')
        result=LayerState(self.bits,self.group_size,self.residual_length,self.sequence_length)
        for name,tensor in self.tensor_items():setattr(result,name,tensor.clone())
        return result

    def release(self):
        for name in FIELDS:setattr(self,name,None)
        self.sequence_length=0;self.released=True

    def prefill(self,key,value,kernels):
        self.live_mutable();require_half(key,value)
        if self.sequence_length or key.shape!=value.shape or key.ndim!=4 or key.shape[0]!=1:
            raise ValueError('One initial batch1 full-document Write only')
        length=key.shape[-2];packed_length=(length//128)*128
        if length<1 or key.shape[-1]%32:raise ValueError('Nonempty document and group-divisible head_dim required')
        if packed_length:
            self.key_codes,self.key_scale,self.key_min=kernels.pack(key[:,:,:packed_length,:].transpose(2,3).contiguous(),32,self.bits)
        if packed_length<length:self.key_tail=key[:,:,packed_length:,:].clone()
        if length>128:
            self.value_codes,self.value_scale,self.value_min=kernels.pack(value[:,:,:-128,:].contiguous(),32,self.bits)
        self.value_tail=value[:,:,-min(length,128):,:].clone()
        self.sequence_length=length

    def decode_attention(self,query,key,value,mask,kernels,groups):
        self.live_mutable();require_half(query,key,value)
        if not self.sequence_length or query.shape[-2]!=1 or key.shape[-2]!=1 or value.shape[-2]!=1:
            raise ValueError('Official128-tail update supports one token per cached call')
        past=self.sequence_length
        # K cache is already post-RoPE; only new key receives RoPE upstream.
        qk_packed=None
        if self.key_codes is not None:
            qk_packed=kernels.gemv(32,query,repeat_heads(self.key_codes,groups),repeat_heads(self.key_scale,groups),repeat_heads(self.key_min,groups),self.bits)
        key_full=key if self.key_tail is None else torch.cat((self.key_tail,key),dim=2)
        qk_full=torch.matmul(query,repeat_heads(key_full,groups).transpose(2,3))
        weights=(qk_full if qk_packed is None else torch.cat((qk_packed,qk_full),dim=-1))/math.sqrt(query.shape[-1])
        if weights.shape[-1]!=past+1:raise RuntimeError('K cache length mismatch')
        if mask is not None:
            if mask.shape!=(1,1,1,past+1):raise ValueError('Expected full-position additive cached mask')
            if mask.dtype==torch.bool:
                weights=weights.masked_fill(~mask,torch.finfo(weights.dtype).min)
            else:
                require_half(mask)
                weights=torch.clamp(weights+mask,min=torch.finfo(weights.dtype).min)
        if key_full.shape[-2]==128:
            codes,scale,mn=kernels.pack(key_full.transpose(2,3).contiguous(),32,self.bits)
            self.key_codes=append(self.key_codes,codes,3);self.key_scale=append(self.key_scale,scale,3);self.key_min=append(self.key_min,mn,3);self.key_tail=None
        else:
            if key_full.shape[-2]>128:raise RuntimeError('K residual overflow; query must be tokenwise')
            self.key_tail=key_full
        probabilities=F.softmax(weights,dim=-1,dtype=torch.float32).to(query.dtype)
        value_full=torch.cat((self.value_tail,value),dim=2);native_length=value_full.shape[-2]
        if self.value_codes is None:output=torch.matmul(probabilities,repeat_heads(value_full,groups))
        else:
            output=kernels.gemv(32,probabilities[:,:,:,:-native_length].contiguous(),repeat_heads(self.value_codes,groups),repeat_heads(self.value_scale,groups),repeat_heads(self.value_min,groups),self.bits)
            output+=torch.matmul(probabilities[:,:,:,-native_length:],repeat_heads(value_full,groups))
        if native_length>128:
            if native_length!=129:raise RuntimeError('V residual overflow; query must be tokenwise')
            codes,scale,mn=kernels.pack(value_full[:,:,:1,:].contiguous(),32,self.bits)
            self.value_codes=append(self.value_codes,codes,2);self.value_scale=append(self.value_scale,scale,2);self.value_min=append(self.value_min,mn,2)
            self.value_tail=value_full[:,:,1:,:].clone()
        else:self.value_tail=value_full
        self.sequence_length+=1
        if not torch.isfinite(output).all():raise RuntimeError('Nonfinite KIVI attention output')
        return output,probabilities

def project_qkv(attention,hidden,position_embeddings,apply_rotary):
    if hidden.dtype!=torch.float16:raise TypeError('FP16 backbone hidden required; do not globally cast LoRA')
    shape=(*hidden.shape[:-1],-1,attention.head_dim)
    query=attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1,2)
    key=attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1,2)
    value=attention.v_proj(hidden).view(shape).transpose(1,2)
    require_half(query,key,value)
    query,key=apply_rotary(query,key,*position_embeddings)
    return query,key,value

def make_cache(num_layers,bits):
    """Use HF5.5.4 mask/position queries; no native full-KV update fallback."""
    if importlib.metadata.version('transformers')!='5.5.4':
        raise RuntimeError('Actual HF5.5.4 Cache qualification required; otherversions refused')
    from transformers.cache_utils import Cache,CacheLayerMixin
    class KiviLayer(CacheLayerMixin):
        is_compileable=False
        is_sliding=False
        def __init__(self):
            super().__init__();self.state=LayerState(bits)
        def lazy_initialization(self,key_states,value_states):
            raise RuntimeError('Only explicit KIVI attention can initialize this cache')
        def update(self,*args,**kwargs):
            raise RuntimeError('Native Cache.update would materialize fullKV; use KIVI attention')
        def get_seq_length(self):return self.state.sequence_length
        def get_mask_sizes(self,query_length):return self.get_seq_length()+query_length,0
        def get_max_cache_shape(self):return -1
        def offload(self):raise RuntimeError('No cache offload')
        def prefetch(self):raise RuntimeError('No cache offload/prefetch')
        def reset(self):self.state.release()
        def reorder_cache(self,*args):raise RuntimeError('No beam-search cache mutation')
    class KiviCache(Cache):
        def __init__(self):
            super().__init__(layers=[KiviLayer() for _ in range(num_layers)],offloading=False)
            self._qencbank_kivi_bridge=True;self.frozen=False;self.released=False
        def tensor_items(self):
            return tuple((f'layer.{i}.{n}',t) for i,layer in enumerate(self.layers) for n,t in layer.state.tensor_items())
        def freeze(self):
            lengths={layer.get_seq_length() for layer in self.layers}
            if self.released or len(lengths)!=1 or min(lengths)<=0:raise RuntimeError('Incomplete cache')
            for layer in self.layers:layer.state.frozen=True
            self.frozen=True;return self
        def fork(self):
            if not self.frozen or self.released:raise RuntimeError('Only complete immutable document entry can fork')
            result=KiviCache()
            for left,right in zip(self.layers,result.layers):right.state=left.state.fork()
            return result
        def release(self):
            for layer in self.layers:layer.state.release()
            self.released=True
    return KiviCache()

class OfficialHalfKernels:
    """Lazy import fixed upstream operators; constructing this is NOT CUDA qualification."""
    performance_backend=True
    def __init__(self,kivi_root,expected_hashes):
        import hashlib,sys
        from pathlib import Path
        root=Path(kivi_root).resolve()
        for relative,digest in expected_hashes.items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:raise ValueError('Upstream KIVI source changed: '+relative)
        sys.path.insert(0,str(root))
        from quant.new_pack import triton_quantize_and_pack_along_last_dim
        from quant.matmul import cuda_bmm_fA_qB_outer
        import quant.new_pack,quant.matmul
        for module in (quant.new_pack,quant.matmul):
            Path(module.__file__).resolve().relative_to(root)
        self._pack=triton_quantize_and_pack_along_last_dim;self._gemv=cuda_bmm_fA_qB_outer
    def pack(self,value,group,bits):
        require_half(value)
        if value.device.type!='cuda':raise RuntimeError('Official Triton pack needs separately qualified CUDA')
        return self._pack(value,group,bits)
    def gemv(self,group,query,codes,scale,mn,bits):
        require_half(query,scale,mn)
        if query.device.type!='cuda' or query.shape[-2]!=1 or codes.dtype!=torch.int32:raise RuntimeError('Official Half GEMV only, one cached query token')
        return self._gemv(group,query,codes,scale,mn,bits)

def install_attention_bridge(model,kernels):
    """Temporarily replace forward methods on existing shared-weight attention modules.

    KIVI/Dense have no adapter in the selected FP16 block. This never calls .half()
    or mutates another method's adapter/backbone dtype. Return handle restores it.
    """
    if importlib.metadata.version('transformers')!='5.5.4':raise RuntimeError('HF5.5.4 bridge only; actual API qualification pending')
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    if model.config.model_type!='qwen3' or model.training:raise ValueError('One eval-mode Qwen3 model required')
    if getattr(model,'peft_config',None) or any(hasattr(m,'lora_A') for m in model.modules()):raise ValueError('KIVI arm is LoRAoff; do not cast or silently disable adapters')
    if getattr(model.config,'use_sliding_window',False) or any(x!='full_attention' for x in model.config.layer_types):raise ValueError('Only full-attention checkpoint supported')
    if any(p.dtype!=torch.float16 for p in model.parameters()):raise TypeError('Load backbone as FP16; this bridge never recasts parameters')
    if len({p.device for p in model.parameters()})!=1 or getattr(model,'hf_device_map',None):raise ValueError('One device, no model offload')
    previous=[]
    def forward(attention,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
        if not getattr(past_key_values,'_qencbank_kivi_bridge',False) or past_key_values.released:raise TypeError('Explicit live KIVI HF Cache required')
        state=past_key_values.layers[attention.layer_idx].state
        state.live_mutable()
        query,key,value=project_qkv(attention,hidden_states,position_embeddings,apply_rotary_pos_emb)
        qlen=query.shape[-2]
        positions=kwargs.get('position_ids')
        if positions is not None:
            expected=torch.arange(state.sequence_length,state.sequence_length+qlen,device=positions.device).unsqueeze(0)
            if not torch.equal(positions,expected):raise ValueError('Absolute contiguous position contract changed')
        if state.sequence_length:
            output,_=state.decode_attention(query,key,value,attention_mask,kernels,attention.num_key_value_groups)
        else:
            state.prefill(key,value,kernels)
            output=F.scaled_dot_product_attention(query,repeat_heads(key,attention.num_key_value_groups),repeat_heads(value,attention.num_key_value_groups),attn_mask=attention_mask,dropout_p=0.0,is_causal=attention_mask is None and qlen>1,scale=attention.scaling)
        if not torch.isfinite(output).all():raise RuntimeError('Nonfinite FP16 prefill/Read')
        output=output.transpose(1,2).contiguous().reshape(*hidden_states.shape[:-1],-1)
        return attention.o_proj(output),None
    for layer in model.model.layers:
        attention=layer.self_attn;previous.append((attention,attention.forward))
        attention.forward=types.MethodType(forward,attention)
    class Handle:
        def restore(self):
            for attention,method in previous:attention.forward=method
            previous.clear()
    return Handle()
