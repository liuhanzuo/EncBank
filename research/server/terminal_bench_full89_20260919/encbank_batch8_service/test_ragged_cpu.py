"""Installed-runtime miniature CPU test of real KV/GDN cache merging and row removal."""
from pathlib import Path
import copy,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
from hybrid_reader import HybridReader
from batch_cache import merge_caches,decode_layers,compact
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text())
torch.set_num_threads(2);torch.manual_seed(1709)
cfg=copy.deepcopy(AutoConfig.from_pretrained(P['model'],local_files_only=True).text_config)
cfg.vocab_size=128;cfg.hidden_size=64;cfg.intermediate_size=128;cfg.num_hidden_layers=8
cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=32
cfg.linear_num_key_heads=2;cfg.linear_num_value_heads=4;cfg.linear_key_head_dim=8;cfg.linear_value_head_dim=8
cfg.layer_types=['linear_attention','linear_attention','linear_attention','full_attention']*2
cfg.rope_parameters['mrope_section']=[1,1,2];cfg.rope_parameters['partial_rotary_factor']=.25
cfg._attn_implementation='sdpa';cfg.tie_word_embeddings=False
model=Qwen3_5ForCausalLM(cfg).float().eval();reader=HybridReader(model,4)
assert all(p.device.type=='cpu' for p in model.parameters())
rows=[];max_errors=[]
with torch.inference_mode():
    for i in range(8):
        q=list(range(5,8+i));doc=list(range(20,25+2*i));fixed=reader.write(doc)
        lower=DynamicCache(config=cfg);upper=DynamicCache(config=cfg)
        qh=reader.layers(reader.core.embed_tokens(reader.tensor(q)),0,4,cache=lower)
        reader.layers(torch.cat([fixed,qh],dim=1),4,8,cache=upper)
        rows.append(dict(i=i,lower=lower,upper=upper,qpos=len(q),upos=len(q)+len(doc),fixed=fixed,saved=fixed.clone()))
    low,lp=merge_caches([r['lower'] for r in rows],[r['qpos'] for r in rows],cfg,0,4)
    up,upad=merge_caches([r['upper'] for r in rows],[r['upos'] for r in rows],cfg,4,8)
    qp=torch.tensor([r['qpos'] for r in rows]);upos=torch.tensor([r['upos'] for r in rows])
    initial_Q=qp.tolist();initial_U=upos.tolist()
    for step in range(4):
        tokens=[60+r['i']+step for r in rows];reference=[]
        for r,t in zip(rows,tokens):
            hidden=reader.core.embed_tokens(reader.tensor([t]))
            hidden=reader.layers(hidden,0,4,cache=r['lower'],offset=r['qpos'])
            hidden=reader.layers(hidden,4,8,cache=r['upper'],offset=r['upos'])
            reference.append(reader.logits(hidden));r['qpos']+=1;r['upos']+=1
        hidden=reader.core.embed_tokens(torch.tensor(tokens)[:,None])
        hidden=decode_layers(reader,hidden,0,4,low,qp,lp)
        hidden=decode_layers(reader,hidden,4,8,up,upos,upad)
        actual=reader.logits(hidden);expected=torch.cat(reference,dim=0)
        error=float((actual-expected).abs().max());max_errors.append(error)
        assert torch.isfinite(actual).all() and torch.allclose(actual,expected,atol=2e-5,rtol=2e-4),(step,error)
        qp+=1;upos+=1
        assert all(torch.equal(r['fixed'],r['saved']) for r in rows)
        if step in [0,1]:
            ids=torch.tensor(list(reversed(range(len(rows))))) if step==0 else torch.tensor([0,2,4,6])
            qp,lp=compact(low,qp,lp,ids);upos,upad=compact(up,upos,upad,ids)
            rows=[rows[i] for i in ids.tolist()]
out=dict(status='PASS',epoch=time.time(),gpu_calls=0,model='miniature random8-layer FP32 CPU model, not a benchmark model/result',
    batch_size=8,variable_query_lengths=initial_Q,variable_upper_lengths=initial_U,decode_steps=4,max_abs_by_step=max_errors,
    row_reversal=True,row_removal=True,H_immutable=True,independent_serial_reference=True)
(H/'ragged_cpu_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
