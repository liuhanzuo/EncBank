import torch
from pathlib import Path
from transformers.cache_utils import DynamicCache
from adaptive import boundary_attention_mass
from io_utils import save as dump
ROOT=Path(__file__).resolve().parent
@torch.inference_mode()
def validate_attention(engine):
    """Compare probe to actual HF eager attention with prefix KV and a causal mask."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    reader=engine.reader;layer=reader.layers[12];attn=layer.self_attn
    memory=engine.bank[:2].reshape(1,1024,-1)
    qh=reader.write_chunk(engine.docs[0]['queries'][0]['query'])
    predicted=boundary_attention_mass(reader,memory,qh)
    prefix=torch.cat([engine.sink,memory,qh[:,:-32]],dim=1)
    x=layer.input_layernorm(prefix);length=x.shape[1]
    key=attn.k_norm(attn.k_proj(x).view(1,length,-1,attn.head_dim)).transpose(1,2)
    value=attn.v_proj(x).view(1,length,-1,attn.head_dim).transpose(1,2)
    positions=torch.arange(length,device=x.device)[None,:]
    cos,sin=reader.rotary_emb(x,position_ids=positions)
    key=key*cos.unsqueeze(1)+rotate_half(key)*sin.unsqueeze(1)
    cache=DynamicCache(config=reader.config);cache.update(key,value,12)
    x=layer.input_layernorm(qh[:,-32:]);positions=torch.arange(length,length+32,device=x.device)[None,:]
    pe=reader.rotary_emb(x,position_ids=positions)
    blocked=torch.arange(length+32,device=x.device)[None,:]>positions[0,:,None]
    mask=torch.zeros(1,1,32,length+32,dtype=x.dtype,device=x.device).masked_fill(blocked[None,None],torch.finfo(x.dtype).min)
    old=reader.config._attn_implementation
    try:
        reader.config._attn_implementation='eager'
        _,weights=attn(x,position_embeddings=pe,attention_mask=mask,past_key_values=cache)
    finally:reader.config._attn_implementation=old
    assert weights is not None
    conditional=weights[:,:,:,1:1025].float();conditional/=conditional.sum(-1,keepdim=True).clamp_min(1e-30)
    expected=conditional.reshape(1,conditional.shape[1],32,2,512).sum(-1).mean((0,1,2))
    delta=predicted-expected
    result=dict(predicted=predicted.cpu().tolist(),hf_attention_reference=expected.cpu().tolist(),max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),scope='actual first-suffix attention; memory-conditional chunk mass; two candidate blocks')
    dump(ROOT/'correctness_progress.json',result)
    assert result['max_abs']<.01 and result['rms']<.005,result
    dump(ROOT/'correctness.json',dict(passed=True,check=result))
