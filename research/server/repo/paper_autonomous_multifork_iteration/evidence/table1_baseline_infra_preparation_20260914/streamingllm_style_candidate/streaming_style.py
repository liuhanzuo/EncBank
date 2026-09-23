"""Pinned Encbank StreamingLLM-style prompt policy, explicitly not a rolling cache.

Question-independent Write stores raw document IDs. Query-dependent prefix
selection and all model prefill occur inside Read/TTFT. No prompt token is
silently omitted beyond the original first4/last4096 baseline policy.
"""
def select_prompt_prefix(document_ids,query_ids):
 if not query_ids or len(query_ids)>4096:
  raise ValueError('Complete query must fit the upstream retained tail; no query truncation permitted')
 prompt=[151643]+list(document_ids)+list(query_ids)
 selected=prompt[:4]+prompt[-4096:] if len(prompt)>4100 else prompt
 assert selected[-len(query_ids):]==list(query_ids)
 prefix=selected[:-len(query_ids)]
 assert prefix
 return prefix
class Entry:
 def __init__(self,raw,document_id):self.raw_ids=raw;self.document_id=document_id
 def tensor_items(self):return (('raw_document_ids',self.raw_ids),)
 def inventory(self):
  n=self.raw_ids.untyped_storage().nbytes()
  return {'unique_tensor_storage_bytes':n,'tensor_storage_bytes_by_device':{'cpu':n},'raw_id_cpu_tensor_bytes':n,'persistent_GPU_bytes':0,
   'scope':'Question-independent raw document IDs only; no persistent document model state. Query-dependent truncated-context prefill belongs to Read.'}
 def release(self):self.raw_ids=None
class Method:
 def __init__(self,model,tokenizer,*,plan):
  import torch
  assert model.config.model_type=='qwen3' and not model.training
  assert all(p.device.type=='cuda' and p.dtype==torch.float16 and not p.requires_grad for p in model.parameters())
  assert not getattr(model,'peft_config',None) and not any('lora_' in n for n,_ in model.named_parameters())
  assert model.config._attn_implementation=='sdpa' and not getattr(model,'hf_device_map',None)
  assert not any(hasattr(m,'_hf_hook') for m in model.modules())
  self.model=model
 def write(self,document_ids,document_id=''):
  import torch
  return Entry(torch.tensor(document_ids,dtype=torch.long,device='cpu'),document_id)
 def open_request(self,entry,bare_question_ids,*,query_ids):
  return Request(self.model,select_prompt_prefix(entry.raw_ids.tolist(),query_ids))
 def close(self):self.model=None
class Request:
 def __init__(self,model,prefix_ids):
  import torch
  from transformers.cache_utils import DynamicCache
  self.model=model;self.cache=DynamicCache(config=model.config)
  ids=torch.tensor([prefix_ids],dtype=torch.long,device=model.model.embed_tokens.weight.device)
  output=model.model(input_ids=ids,past_key_values=self.cache,use_cache=True,return_dict=True)
  assert output.past_key_values is self.cache
  self.position=self.prefix_tokens=len(prefix_ids);self.selected_indices=None
  self.query_calls=self.decode_calls=self.head_calls=0
 def tensor_items(self):
  import torch
  return tuple((f'layer.{i}.{name}',t) for i,layer in enumerate(self.cache.layers) for name in ('keys','values') if isinstance(t:=getattr(layer,name,None),torch.Tensor)) if self.cache is not None else ()
 def consume(self,token,*,emit_head,phase):
  import torch
  assert phase in ('query','decode')
  device=self.model.model.embed_tokens.weight.device
  ids=torch.tensor([[int(token)]],dtype=torch.long,device=device)
  positions=torch.tensor([[self.position]],device=device)
  output=self.model.model(input_ids=ids,position_ids=positions,past_key_values=self.cache,use_cache=True,return_dict=True)
  assert output.past_key_values is self.cache
  self.position+=1;assert self.cache.get_seq_length()==self.position
  self.query_calls+=phase=='query';self.decode_calls+=phase=='decode'
  if emit_head:
   self.head_calls+=1;return self.model.lm_head(output.last_hidden_state[:,-1:])
  return None
 def close(self):self.cache=None;self.model=None
