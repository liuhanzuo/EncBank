"""Native CoMem whole ordered upper-prefix reuse, with private query-only KV.

In-process prototype for immutable h_j tensor objects. New but equal tensors
conservatively miss. There is no persistence, content-addressing, partial-prefix
reuse, eviction or offload. Normal PyTorch mutations are version-checked; callers
MUST invalidate() after .data/external writes that bypass version counters.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from types import MappingProxyType

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from native_infra_readers import NativeCoMemReader,NativeQueryState


def _tensor_signature(tensor):
    if tensor is None:
        return None
    return (id(tensor),tensor.data_ptr(),tensor._version,tuple(tensor.shape),tuple(tensor.stride()),
            tensor.storage_offset(),str(tensor.dtype),str(tensor.device))


def _bytes(tensor):
    return tensor.numel()*tensor.element_size()


def _unique_storage_bytes(tensors):
    storages={}
    for tensor in tensors:
        storage=tensor.untyped_storage()
        storages[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    return sum(storages.values())


@dataclass(frozen=True)
class PrefixKV:
    k:torch.Tensor
    v:torch.Tensor


@dataclass(frozen=True)
class NativePrefix:
    signature:tuple
    pairs:object
    tensor_signatures:tuple
    token_count:int
    document_lengths:tuple
    sink_tokens:int
    source_refs:tuple
    model_refs:tuple
    prefix_kv_bytes:int
    prefix_storage_bytes:int
    hj_tensor_bytes:int
    hj_storage_bytes:int
    build_layer_calls:int

    def assert_intact(self):
        actual=tuple((layer,_tensor_signature(pair.k),_tensor_signature(pair.v))
                     for layer,pair in self.pairs.items())
        if actual!=self.tensor_signatures:
            raise RuntimeError("Read-only prefix tensors were changed; invalidate and rebuild")


class PrefixQueryCache:
    """Only query KV persists here; full attention keys are temporary cat outputs.

    Qwen3's native attention calls update(). The native mask factory additionally
    consumes get_query_offset/get_mask_sizes. No native attention is overridden.
    """
    is_compileable=False

    def __init__(self,prefix,config,j):
        self.prefix=prefix
        self.j=j
        self.query_cache=DynamicCache(config=config)
        self.is_sliding=[False]*config.num_hidden_layers

    @property
    def layers(self):
        # Deliberately query-only, never combined document+query storage.
        return self.query_cache.layers

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        if layer_idx<self.j:
            raise ValueError("Upper prefix cache cannot be used in query bottom layers")
        own_k,own_v=self.query_cache.update(key_states,value_states,layer_idx,*args,**kwargs)
        pair=self.prefix.pairs.get(layer_idx)
        if pair is None:
            return own_k,own_v
        return torch.cat((pair.k,own_k),dim=-2),torch.cat((pair.v,own_v),dim=-2)

    def get_seq_length(self,layer_idx=None):
        index=self.j if layer_idx is None else layer_idx
        return self.prefix.token_count+int(self.query_cache.get_seq_length(index))

    def get_query_offset(self,layer_idx=None):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self,query_length,layer_idx):
        return self.get_seq_length(layer_idx)+query_length,0

    def query_tensor_bytes(self):
        return sum(_bytes(t) for layer in self.layers[self.j:] if layer.is_initialized
                   for t in (layer.keys,layer.values))


class NativePrefixCoMemReader(NativeCoMemReader):
    """Exact causal graph reuse for one complete ordered sink+document pack."""
    implementation="native-comem-whole-upper-prefix-query-only-cache"

    def __init__(self,comem):
        super().__init__(comem)
        self._prefix=None
        self._epoch=0
        self.build_count=0
        self.hit_count=0
        self.miss_count=0
        self._check_contract()

    @property
    def prefix(self):
        return self._prefix

    def invalidate(self):
        """Required after untracked writes; also invalidates outstanding queries."""
        self._prefix=None
        self._epoch+=1

    def _check_contract(self):
        self._check_eval()
        cm=self.comem
        if self.model is not cm.model or not 0<self.j<self.L:
            raise ValueError("Fixed CoMem/model binding with0<j<L required")
        if cm.config.model_type!="qwen3" or getattr(cm.config,"_attn_implementation",None)!="sdpa":
            raise ValueError("This prototype validates dense Qwen3 native SDPA only")
        if cm.top_prepay_b!=0 or cm.block_diagonal or cm.resume_j!=self.j:
            raise ValueError("Ordinary unchanged native exact-resume graph required")
        if any(layer.training or layer.self_attn.attention_dropout!=0.
               or getattr(layer.self_attn,"sliding_window",None) is not None for layer in cm.layers):
            raise ValueError("Eval,zero dropout and full attention required")
        if getattr(cm.rotary_emb,"rope_type","default") not in ("default","linear","yarn"):
            raise ValueError("Length-dependent dynamic RoPE is not supported")
        first=next(self.model.parameters())
        if first.device!=cm.device or first.dtype!=cm.dtype:
            raise ValueError("CoMem cached device/dtype is stale; construct a new wrapper")

    def _model_signature(self):
        cm=self.comem
        parameters=tuple((name,_tensor_signature(t)) for name,t in self.model.named_parameters())
        buffers=tuple((name,_tensor_signature(t)) for name,t in self.model.named_buffers())
        modules=tuple((name,id(module),type(module).__module__,type(module).__qualname__,
                       tuple(sorted((key,value) for key,value in vars(module).items()
                                    if not key.startswith("_") and type(value) in (str,int,float,bool,type(None)))))
                      for name,module in self.model.named_modules())
        backend=tuple(bool(getattr(torch.backends.cuda,name)()) for name in
                      ("math_sdp_enabled","flash_sdp_enabled","mem_efficient_sdp_enabled","cudnn_sdp_enabled"))
        return (self._epoch,id(self.model),id(cm),self.j,self.L,parameters,buffers,modules,
                json.dumps(cm.config.to_dict(),sort_keys=True,default=str),
                getattr(cm.config,"_attn_implementation",None),str(cm.device),str(cm.dtype),
                torch.is_autocast_enabled(cm.device.type),str(torch.get_autocast_dtype(cm.device.type)),backend)

    def _signature(self,sink,documents,position_offset):
        self._check_contract()
        if position_offset!=0:
            raise ValueError("Native pack positions must start at0; offset reuse is not supported")
        tensors=([sink] if sink is not None else [])+list(documents)
        for tensor in tensors:
            if (not torch.is_tensor(tensor) or tensor.ndim!=3 or tensor.shape[0]!=1
                    or tensor.shape[1]<1 or tensor.shape[2]!=self.comem.hidden_size):
                raise ValueError("Each h_j must be nonempty[1,T,hidden_size]")
        return (self._model_signature(),position_offset,_tensor_signature(sink),
                tuple(_tensor_signature(t) for t in documents))

    @torch.no_grad()
    def build_prefix(self,sink,documents,*,position_offset=0):
        """Document-only native[j,L) pass. This API accepts no query or answer."""
        documents=list(documents)
        signature=self._signature(sink,documents,position_offset)
        source=tuple(([sink] if sink is not None else [])+documents)
        converted_sink,converted=self._prepare_documents(sink,documents)
        pieces=([converted_sink] if converted_sink is not None else [])+converted
        count=sum(int(t.shape[1]) for t in pieces)
        if count>self.comem.config.max_position_embeddings:
            raise ValueError("Prefix exceeds declared model window")
        # Release the reader's stale pack before constructing another one.
        # Active requests retain their own immutable prefix; do not change epoch.
        self._prefix=None
        cache=DynamicCache(config=self.comem.config)
        if count:
            packed=torch.cat(pieces,dim=1)
            positions=torch.arange(count,device=self.comem.device).unsqueeze(0)
            mask,rope=self.comem._make_mask_and_rope(packed,positions)
            self.comem._run_layers(packed,slice(self.j,self.L),mask,positions,rope,
                                   past_key_values=cache,use_cache=True)
        pairs={i:PrefixKV(cache.layers[i].keys,cache.layers[i].values) for i in range(self.j,self.L)} if count else {}
        for pair in pairs.values():
            if pair.k.shape[1]!=self.comem.config.num_key_value_heads or pair.v.shape[1]!=self.comem.config.num_key_value_heads:
                raise RuntimeError("Persistent prefix KV unexpectedly expanded heads")
        tensors=[t for pair in pairs.values() for t in (pair.k,pair.v)]
        model_refs=tuple(self.model.parameters())+tuple(self.model.buffers())+tuple(self.model.modules())
        entry=NativePrefix(signature,MappingProxyType(pairs),
            tuple((i,_tensor_signature(pair.k),_tensor_signature(pair.v)) for i,pair in pairs.items()),
            count,tuple(int(t.shape[1]) for t in documents),0 if sink is None else int(sink.shape[1]),
            source,model_refs,sum(_bytes(t) for t in tensors),_unique_storage_bytes(tensors),
            sum(_bytes(t) for t in source),_unique_storage_bytes(source),self.L-self.j if count else 0)
        # Refuse caching if a writer/model mutation happened during construction.
        if signature!=self._signature(sink,documents,position_offset):
            raise RuntimeError("Prefix inputs/model changed while building")
        self._prefix=entry
        self.build_count+=1
        return entry

    def _assert_request(self,state):
        prefix=state.top_cache.prefix
        prefix.assert_intact()
        sources=prefix.source_refs
        sink=sources[0] if prefix.sink_tokens else None
        docs=sources[1:] if prefix.sink_tokens else sources
        if self._signature(sink,docs,0)!=prefix.signature:
            raise RuntimeError("Document/model/adapter/backend context changed during request")

    @torch.no_grad()
    def prefill(self,sink,documents,prompt,probe_indices=None,*,position_offset=0):
        documents=list(documents)
        signature=self._signature(sink,documents,position_offset)
        prompt_ids=self.comem._as_ids(prompt)
        prompt_length=int(prompt_ids.shape[1])
        prefix_length=sum(int(t.shape[1]) for t in documents)+(0 if sink is None else int(sink.shape[1]))
        if prompt_length<1 or prefix_length+prompt_length>self.comem.config.max_position_embeddings:
            raise ValueError("Nonempty query within declared model window required")
        prefix=self._prefix
        hit=prefix is not None and prefix.signature==signature
        if hit:
            try:
                prefix.assert_intact()
            except RuntimeError:
                hit=False
        if not hit:
            self.miss_count+=1
            self._prefix=None
            prefix=None
            prefix=self.build_prefix(sink,documents,position_offset=position_offset)
        else:
            self.hit_count+=1
        q_h,bottom,q_position=self.comem.write_prefill(prompt_ids)
        if q_position<1 or prefix.token_count+q_position>self.comem.config.max_position_embeddings:
            raise ValueError("Nonempty query within declared model window required")
        top=PrefixQueryCache(prefix,self.comem.config,self.j)
        positions=torch.arange(prefix.token_count,prefix.token_count+q_position,device=self.comem.device).unsqueeze(0)
        mask=create_causal_mask(config=self.comem.config,inputs_embeds=q_h,attention_mask=None,
                                past_key_values=top,position_ids=positions,layer_idx=self.j)
        rope=self.comem.rotary_emb(q_h,position_ids=positions)
        hidden=self.comem._run_layers(q_h,slice(self.j,self.L),mask,positions,rope,past_key_values=top,use_cache=True)
        logits=self.comem.lm_head(self.comem.norm(hidden[:,-1:,:]))
        candidate=sum(prefix.document_lengths)
        stats={"implementation":self.implementation,"selection_source":"all-blocks",
            "selected_indices":list(range(len(documents))),"candidate_blocks":len(documents),
            "candidate_tokens":candidate,"selected_tokens":candidate,"sink_tokens":prefix.sink_tokens,
            "resume_j":self.j,"query_tokens":q_position,"original_query_start":prefix.token_count,
            "probe_indices":list(probe_indices or []),"routing_computed":False,
            "cross_request_kv_reuse":True,"prefix_cache_hit":hit,"prefix_build_count":self.build_count,
            "prefix_build_layer_calls_this_request":0 if hit else prefix.build_layer_calls,
            "query_online_document_layer_calls":0,"prefix_kv_bytes":prefix.prefix_kv_bytes,
            "prefix_storage_bytes":prefix.prefix_storage_bytes,"hj_tensor_bytes":prefix.hj_tensor_bytes,
            "hj_storage_bytes":prefix.hj_storage_bytes,"prefix_plus_hj_tensor_bytes":prefix.prefix_kv_bytes+prefix.hj_tensor_bytes,
            "request_upper_query_kv_bytes":top.query_tensor_bytes(),
            "document_kv_bytes":prefix.prefix_kv_bytes,
            "document_kv_tokens_by_layer":{str(i):prefix.token_count for i in range(self.j,self.L)},
            "storage_scope":"Shared upper document prefix plus original h_j separately; request top state holds query KV only. Attention concatenations are transient.",
            "reuse_scope":"Same-process immutable tensor identities/version counters; entire ordered pack at positions0..D-1 only",
            "formal_inference_timing":False,"formal_inference_memory":False}
        state=NativeQueryState(bottom,top,int(q_position),int(prefix.token_count+q_position),stats)
        self._assert_request(state)
        return logits,state

    @torch.no_grad()
    def decode_step(self,token_id,state):
        self._check_contract()
        self._assert_request(state)
        window=self.comem.config.max_position_embeddings
        if state.query_position>=window or state.pack_position>=window:
            raise ValueError("Next decode token exceeds declared model window")
        logits=super().decode_step(token_id,state)
        state.top_cache.prefix.assert_intact()
        state.route_stats["request_upper_query_kv_bytes"]=state.top_cache.query_tensor_bytes()
        return logits
