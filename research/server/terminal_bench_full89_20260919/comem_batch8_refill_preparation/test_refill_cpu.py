"""Exact installed runtime, tiny FP32 CPU model: live refill vs independent serial."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import copy,json,sys,time
H=Path(__file__).resolve().parent
PREV=H.parent/'comem_batch8_service';sys.path.insert(0,str(PREV))
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
from hybrid_reader import HybridReader
from batch_cache import merge_caches,decode_layers,compact
from batch_cache_refill import join_cache_groups
P=json.loads((PREV/'plan.json').read_text());torch.set_num_threads(2);torch.manual_seed(1719)
cfg=copy.deepcopy(AutoConfig.from_pretrained(P['model'],local_files_only=True).text_config)
cfg.vocab_size=128;cfg.hidden_size=64;cfg.intermediate_size=128;cfg.num_hidden_layers=8
cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=32
cfg.linear_num_key_heads=2;cfg.linear_num_value_heads=4;cfg.linear_key_head_dim=8;cfg.linear_value_head_dim=8
cfg.layer_types=['linear_attention','linear_attention','linear_attention','full_attention']*2
cfg.rope_parameters['mrope_section']=[1,1,2];cfg.rope_parameters['partial_rotary_factor']=.25
cfg._attn_implementation='sdpa';cfg.tie_word_embeddings=False
model=Qwen3_5ForCausalLM(cfg).float().eval();reader=HybridReader(model,4)
assert all(p.device.type=='cpu' for p in model.parameters())
def prefill(i):
    q=list(range(5,8+i));doc=list(range(20,25+2*i));fixed=reader.write(doc)
    low=DynamicCache(config=cfg);up=DynamicCache(config=cfg)
    qh=reader.layers(reader.core.embed_tokens(reader.tensor(q)),0,4,cache=low)
    reader.layers(torch.cat([fixed,qh],dim=1),4,8,cache=up)
    return dict(i=i,low=low,up=up,qpos=len(q),upos=len(q)+len(doc),fixed=fixed,saved=fixed.clone())
def pack(rows):
    low,lp=merge_caches([r['low'] for r in rows],[r['qpos'] for r in rows],cfg,0,4)
    up,upad=merge_caches([r['up'] for r in rows],[r['upos'] for r in rows],cfg,4,8)
    return low,lp,up,upad,torch.tensor([r['qpos'] for r in rows]),torch.tensor([r['upos'] for r in rows])
errors=[];sizes=[]
with torch.inference_mode():
    rows=[prefill(i) for i in range(4)]
    low,lp,up,upad,qp,upos=pack(rows)
    for step in range(8):
        if step in [2,5]:
            new=[prefill(i) for i in (range(4,8) if step==2 else range(8,12))]
            nl,np,nu,nup,nq,npos=pack(new)
            low,lp=join_cache_groups([low,nl],[lp,np],cfg,0,4,consume=True)
            up,upad=join_cache_groups([up,nu],[upad,nup],cfg,4,8,consume=True)
            qp=torch.cat([qp,nq]);upos=torch.cat([upos,npos]);rows+=new
        tokens=[60+r['i']+step for r in rows];reference=[]
        for r,t in zip(rows,tokens):
            z=reader.core.embed_tokens(reader.tensor([t]));z=reader.layers(z,0,4,cache=r['low'],offset=r['qpos'])
            z=reader.layers(z,4,8,cache=r['up'],offset=r['upos']);reference.append(reader.logits(z));r['qpos']+=1;r['upos']+=1
        z=reader.core.embed_tokens(torch.tensor(tokens)[:,None]);z=decode_layers(reader,z,0,4,low,qp,lp)
        z=decode_layers(reader,z,4,8,up,upos,upad);actual=reader.logits(z);expected=torch.cat(reference)
        error=float((actual-expected).abs().max());errors.append(error);sizes.append(len(rows))
        assert torch.isfinite(actual).all() and torch.allclose(actual,expected,atol=2e-5,rtol=2e-4),(step,error)
        assert all(torch.equal(r['fixed'],r['saved']) for r in rows)
        qp+=1;upos+=1
        if step==3:
            idx=torch.tensor([7,4,2,0]);qp,lp=compact(low,qp,lp,idx);upos,upad=compact(up,upos,upad,idx);rows=[rows[i] for i in idx.tolist()]
out=dict(status='PASS',gpu_calls=0,model='tiny random8-layer FP32 CPU only',refill_after_decode_steps=[2,5],batch_sizes=sizes,max_abs_by_step=errors,row_removal_and_reversal=True,H_immutable=True,formal_service_deployed=False)
print(json.dumps(out))
