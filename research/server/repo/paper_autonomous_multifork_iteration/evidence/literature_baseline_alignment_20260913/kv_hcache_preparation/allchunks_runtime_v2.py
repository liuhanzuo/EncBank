"""Pinned-public KV-Direct / HCache-style with explicit local query boundaries.
Only selector/all-chunks materialization and j0 admission differ from native H16.
No persistent document KV; full causal upper recomputation belongs to Read/TTFT.
"""
import torch
from transformers.cache_utils import DynamicCache
from honly_fp16 import HOnlyMemory, HOnlyEntry, SelectedRead, _storage_key
from typing import Sequence

def cache_tensors(cache):
    if cache is None:return ()
    if not isinstance(cache,DynamicCache):raise TypeError('Native DynamicCache required')
    return tuple((f'layer.{i}.{name}',t) for i,layer in enumerate(cache.layers) for name in ('keys','values') if isinstance(t:=getattr(layer,name,None),torch.Tensor))

class AllChunkMemory(HOnlyMemory):
    def select(self,entry,bare_question_ids):
        self._check_entry(entry)
        return tuple(range(len(entry.chunks)))
    @torch.inference_mode()
    def materialize_selected(self, entry: HOnlyEntry, indices: Sequence[int]) -> SelectedRead:
        self._check_entry(entry)
        indices = tuple(indices)
        if (indices != tuple(range(len(entry.chunks))) or
            any(type(i) is not int or not 0 <= i < len(entry.chunks) for i in indices) or
            list(indices) != sorted(set(indices))):
            raise ValueError('Public all-chunks selection must include every chunk exactly once in source order')
        hidden = tuple(entry.chunks[i].dequantize() for i in indices)
        sink_hidden = entry.sink.dequantize()
        entry_keys = {_storage_key(t) for _, t in entry.tensor_items()}
        for h in (*hidden, sink_hidden):
            if h.device != self.device or h.dtype != entry.dtype:
                raise RuntimeError('Read hidden dtype/device changed')
            if _storage_key(h) in entry_keys:
                raise RuntimeError('Read state aliases immutable entry backing')
        return SelectedRead(indices, hidden, sink_hidden)
class AllChunkRequest:
    def __init__(self,memory,entry,bare_question_ids):
        self.memory=memory;self.engine=memory.engine;engine=self.engine
        if engine.top_prepay_b or engine.block_diagonal or not 0<=engine.resume_j<engine.num_layers:raise ValueError('Public exact j0 or interior split only')
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
    def __init__(self,model,tokenizer,arm,*,resume_j,reader_binding,bos_token_id,eos_token_id,kernels=None):
        expected={'public_kvdirect_j0':0,'public_hcache_j12':12}
        if arm not in expected or resume_j!=expected[arm]:raise ValueError('Frozen public baseline/depth mismatch')
        self.model=model.get_base_model() if callable(getattr(model,'get_base_model',None)) else model
        if self.model.training or next(self.model.parameters()).dtype!=torch.float16:raise ValueError('Eval FP16 required')
        if len({p.device for p in self.model.parameters()})!=1:raise ValueError('No mixed-device/offload model')
        if getattr(self.model,'peft_config',None) or any('lora_' in n for n,_ in self.model.named_parameters()):raise ValueError('Public comparators require adapter OFF')
        self.arm=arm;self.bos=bos_token_id
        self.memory=AllChunkMemory(self.model,tokenizer,resume_j=resume_j,bits=16,group_size=64,reader_binding=reader_binding,bos_token_id=bos_token_id,eos_token_id=eos_token_id)
    @torch.inference_mode()
    def write(self,document_ids,document_id=''):
        return self.memory.encode_ids(document_ids,document_id=document_id)
    @torch.inference_mode()
    def open_request(self,entry,bare_question_ids):
        return AllChunkRequest(self.memory,entry,bare_question_ids)
    def close(self):pass
