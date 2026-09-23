"""New common FP16 query schedule; existing BF16 runtime is never modified.

All methods: request materialization -> complete tokenwise query (head only last)
-> fixed32 sampled tokens /31 tokenwise decode forwards. No loader or launcher.
"""
from __future__ import annotations
import copy,sys,weakref
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers.cache_utils import DynamicCache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
from honly_fp16 import HOnlyMemory

def backbone(model):return model.get_base_model() if callable(getattr(model,'get_base_model',None)) else model
def cache_tensors(cache):
    if cache is None:return ()
    if getattr(cache,'_qencbank_kivi_bridge',False):return cache.tensor_items()
    if not isinstance(cache,DynamicCache):raise TypeError('Only bound native DynamicCache/KIVI Cache')
    return tuple((f'layer.{i}.{name}',t) for i,layer in enumerate(cache.layers) for name in ('keys','values') if isinstance(t:=getattr(layer,name,None),torch.Tensor))
def storage(t):return str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()
def weak_refs(items):return tuple((name,weakref.ref(t)) for name,t in items)
def released(refs):return {'all_tracked_tensor_objects_released':all(r() is None for _,r in refs),'tracked_tensor_objects':len(refs),'alive_names':[n for n,r in refs if r() is not None]}

@dataclass
class NativeEntry:
    raw_ids:torch.Tensor|None
    cache:object
    document_id:str
    representation:str
    def tensor_items(self):return (() if self.raw_ids is None else (('raw_ids',self.raw_ids),))+cache_tensors(self.cache)
    def inventory(self):
        items=self.tensor_items();keys={storage(t) for _,t in items};by_device={}
        for device,_,nbytes in keys:by_device[device]=by_device.get(device,0)+nbytes
        return {'representation':self.representation,'unique_tensor_storage_bytes':sum(x[2] for x in keys),'tensor_storage_bytes_by_device':by_device,'raw_id_cpu_tensor_bytes':self.raw_ids.untyped_storage().nbytes(),'document_tokens':self.raw_ids.numel(),'prefix_tokens':self.cache.get_seq_length(),'model_state_included':False,'query_state_included':False,'native_hidden_or_KV_host_offload':False}
    def release(self):
        if getattr(self.cache,'_qencbank_kivi_bridge',False):self.cache.release()
        self.raw_ids=None;self.cache=None

class NativeRequest:
    def __init__(self,model,entry):
        self.model=model;self.device=next(model.parameters()).device
        self.cache=entry.cache.fork() if getattr(entry.cache,'_qencbank_kivi_bridge',False) else copy.deepcopy(entry.cache)
        sourcekeys={storage(t) for _,t in entry.tensor_items()}
        if any(storage(t) in sourcekeys for _,t in cache_tensors(self.cache)):raise RuntimeError('Request aliases immutable native/packed entry')
        self.position=self.cache.get_seq_length();self.selected_indices=None;self.prefix_tokens=self.position
        self.query_calls=0;self.decode_calls=0;self.head_calls=0
    def tensor_items(self):return cache_tensors(self.cache)
    @torch.inference_mode()
    def consume(self,token,*,emit_head,phase):
        ids=torch.tensor([[int(token)]],dtype=torch.long,device=self.device)
        positions=torch.tensor([[self.position]],device=self.device)
        output=self.model.model(input_ids=ids,position_ids=positions,past_key_values=self.cache,use_cache=True,return_dict=True)
        if output.past_key_values is not self.cache:raise RuntimeError('Cache owner changed')
        self.position+=1
        if self.cache.get_seq_length()!=self.position:raise RuntimeError('Native absolute cache position mismatch')
        self.query_calls+=phase=='query';self.decode_calls+=phase=='decode'
        if emit_head:
            self.head_calls+=1;return self.model.lm_head(output.last_hidden_state[:, -1:])
        return None
    def close(self):
        if getattr(self.cache,'_qencbank_kivi_bridge',False):self.cache.release()
        self.cache=None

class HRequest:
    def __init__(self,memory,entry,bare_question_ids):
        self.memory=memory;self.engine=memory.engine;engine=self.engine
        if engine.top_prepay_b or engine.block_diagonal or not 0<engine.resume_j<engine.num_layers:raise ValueError('Exact interior H split only')
        self.selected_indices=memory.select(entry,bare_question_ids)
        self.selected=memory.materialize_selected(entry,self.selected_indices)
        # Query-independent selected prefix is reconstructed only now, within
        # TTFT. It cannot depend on later query tokens under causal attention.
        packed=torch.cat((self.selected.sink_hidden,*self.selected.hidden),dim=1)
        self.prefix_tokens=packed.shape[1]
        positions=torch.arange(self.prefix_tokens,device=engine.device).unsqueeze(0)
        mask,rope=engine._make_mask_and_rope(packed,positions)
        self.top_cache=DynamicCache(config=engine.config)
        unused=engine._run_layers(packed,slice(engine.resume_j,engine.num_layers),mask,positions,rope,past_key_values=self.top_cache,use_cache=True)
        del unused,packed
        self.bottom_cache=DynamicCache(config=engine.config)
        self.query_position=0;self.pack_position=self.prefix_tokens
        self.query_calls=0;self.decode_calls=0;self.head_calls=0
        self._check_lengths()
    def _check_lengths(self):
        j=self.engine.resume_j
        if any(self.bottom_cache.get_seq_length(i)!=self.query_position for i in range(j)):raise RuntimeError('Lower query-only cache length mismatch')
        if any(self.top_cache.get_seq_length(i)!=self.pack_position for i in range(j,self.engine.num_layers)):raise RuntimeError('Upper contiguous packed position mismatch')
        if any(self.bottom_cache.get_seq_length(i) for i in range(j,self.engine.num_layers)):raise RuntimeError('Lower cache incorrectly stores upper state')
        if any(self.top_cache.get_seq_length(i) for i in range(j)):raise RuntimeError('Upper cache incorrectly stores document lower KV')
    def tensor_items(self):
        return (tuple(('selected.'+n,t) for n,t in self.selected.tensor_items()) if self.selected is not None else ())+tuple(('bottom.'+n,t) for n,t in cache_tensors(self.bottom_cache))+tuple(('top.'+n,t) for n,t in cache_tensors(self.top_cache))
    @torch.inference_mode()
    def consume(self,token,*,emit_head,phase):
        e=self.engine;ids=e._as_ids([int(token)]);hidden=e.embed_tokens(ids)
        bpos=torch.tensor([[self.query_position]],device=e.device)
        brope=e.rotary_emb(hidden,position_ids=bpos)
        hj=e._run_layers(hidden,slice(0,e.resume_j),e._decode_attn_mask(self.query_position+1),bpos,brope,past_key_values=self.bottom_cache,use_cache=True)
        tpos=torch.tensor([[self.pack_position]],device=e.device)
        trope=e.rotary_emb(hj,position_ids=tpos)
        last=e._run_layers(hj,slice(e.resume_j,e.num_layers),e._decode_attn_mask(self.pack_position+1),tpos,trope,past_key_values=self.top_cache,use_cache=True)
        self.query_position+=1;self.pack_position+=1
        self.query_calls+=phase=='query';self.decode_calls+=phase=='decode'
        if emit_head:
            self.head_calls+=1;return e.lm_head(e.norm(last))
        return None
    def close(self):
        if self.selected is not None:self.selected.release()
        self.selected=None;self.bottom_cache=None;self.top_cache=None

class Method:
    def __init__(self,model,tokenizer,arm,*,resume_j=12,reader_binding='unbound',bos_token_id=151643,eos_token_id=151645,kernels=None,group_size=64):
        if arm not in ('h16', 'h8', 'h4', 'h4g32', 'h4g128'):raise ValueError('Six fixed methods only')
        self.model=backbone(model);self.arm=arm;self.bos=bos_token_id;self.handle=None
        if self.model.training or next(self.model.parameters()).dtype!=torch.float16:raise ValueError('Already loaded eval FP16 backbone required')
        if len({p.device for p in self.model.parameters()})!=1:raise ValueError('No mixed-device/offload model')
        self.memory=None
        if arm.startswith('h'):
            self.memory=HOnlyMemory(self.model,tokenizer,resume_j=resume_j,bits=16 if arm=='h16' else 8 if arm=='h8' else 4,group_size=group_size,reader_binding=reader_binding,bos_token_id=bos_token_id,eos_token_id=eos_token_id)
        else:
            if getattr(self.model,'peft_config',None) or any(hasattr(m,'lora_A') for m in self.model.modules()):raise ValueError('Dense/KIVI are LoRA off; never silently cast/disable adapters')
            if arm.startswith('kivi'):
                if kernels is None or not kernels.performance_backend:raise ValueError('Actual qualified Half backend required, no CPU reference fallback')
                from bridge import install_attention_bridge
                self.handle=install_attention_bridge(self.model,kernels)
    @torch.inference_mode()
    def write(self,document_ids,document_id=''):
        if self.memory:return self.memory.encode_ids(document_ids,document_id=document_id)
        if self.arm.startswith('kivi'):
            from bridge import make_cache
            cache=make_cache(self.model.config.num_hidden_layers,int(self.arm[-1]))
        else:cache=DynamicCache(config=self.model.config)
        raw=torch.tensor(document_ids,dtype=torch.long,device='cpu')
        ids=torch.tensor([[self.bos]+raw.tolist()],dtype=torch.long,device=next(self.model.parameters()).device)
        output=self.model.model(input_ids=ids,past_key_values=cache,use_cache=True,return_dict=True)
        if output.past_key_values is not cache:raise RuntimeError('Write cache ownership changed')
        if self.arm.startswith('kivi'):cache.freeze()
        return NativeEntry(raw,cache,str(document_id),'FP16 full-depth native KV' if self.arm=='dense' else 'KIVI packed full-depth KV; Half metadata/tail')
    @torch.inference_mode()
    def open_request(self,entry,bare_question_ids):
        return HRequest(self.memory,entry,bare_question_ids) if self.memory else NativeRequest(self.model,entry)
    def close(self):
        if self.handle:self.handle.restore();self.handle=None

def greedy(logits):
    values=logits[0,-1].float()
    if not bool(torch.isfinite(values).all()):raise RuntimeError('Nonfinite fixed-work logits')
    return int(values.argmax())

@torch.inference_mode()
def read_fixed(method,entry,query_ids,bare_question_ids,tokenizer,profiler,*,capture_logits=False):
    if not query_ids:raise ValueError('Complete query cannot be empty')
    captured=[]
    with profiler.phase('fullRead') as full:
        with profiler.phase('warmEntryReadTTFT') as ttft:
            request=method.open_request(entry,bare_question_ids)
            refs=weak_refs(request.tensor_items())
            for index,token in enumerate(query_ids):logits=request.consume(token,emit_head=index==len(query_ids)-1,phase='query')
            token=greedy(logits)
            if capture_logits:captured.append(logits.detach().clone())
            del logits
        generated=[token]
        with profiler.phase('steadyDecode') as steady:
            for _ in range(31):
                logits=request.consume(token,emit_head=True,phase='decode');token=greedy(logits);generated.append(token)
                if capture_logits:captured.append(logits.detach().clone())
                del logits
        with profiler.phase('textDecodeAndRequestRelease'):
            query_calls,decode_calls,head_calls=request.query_calls,request.decode_calls,request.head_calls
            selected=request.selected_indices;prefix=request.prefix_tokens
            refs+=weak_refs(request.tensor_items());prediction=tokenizer.decode(generated,skip_special_tokens=True)
            request.close();del request
    lifecycle=released(refs)
    if not lifecycle['all_tracked_tensor_objects_released']:raise RuntimeError(lifecycle)
    assert query_calls==len(query_ids) and decode_calls==31 and head_calls==32
    result={'generated_token_ids':generated,'fixed_generated_tokens':32,'decode_forward_count':31,'query_prefill_calls':query_calls,'query_tokens_per_call':1,'head_calls':head_calls,'eos_stopping_enabled':False,'first_step_eos_suppressed':False,'selected_chunk_indices':None if selected is None else list(selected),'packed_read_tokens':prefix+len(query_ids),'warm_entry_read_ttft_seconds':ttft['seconds'],'steady_decode_seconds':steady['seconds'],'steady_decoded_tokens_per_second':31/steady['seconds'],'full_read_seconds':full['seconds'],'prediction_fixed_length_not_quality':prediction,'request_release':lifecycle,'request_no_entry_alias':True}
    if capture_logits:result['diagnostic_logits']=captured
    return result
