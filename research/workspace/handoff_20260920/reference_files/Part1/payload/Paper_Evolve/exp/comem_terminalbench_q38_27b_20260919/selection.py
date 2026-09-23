"""27B hybrid-aware selection. Probe only at an actual full-attention layer."""
import torch
from memory_selectors import bm25_scores,iter_bm25_indices

def nucleus(scores,ids,p=.9,maximum=48,minimum=4):
    import math
    assert len(scores)==len(ids) and all(math.isfinite(float(s)) and s>=0 for s in scores)
    total=math.fsum(scores);cap=min(maximum,len(ids));floor=min(minimum,cap)
    if not total:
        return dict(selected=sorted(ids)[-floor:] if floor else [],zero_mass=True,target_met=False,achieved_mass=0.,cap_limited=False)
    selected=[];mass=0.
    for i,s in sorted(zip(ids,scores),key=lambda t:(-t[1],t[0]))[:cap]:
        if len(selected)>=floor and mass>=p-1e-12:break
        selected.append(i);mass+=float(s)/total
    return dict(selected=sorted(selected),zero_mass=False,target_met=mass>=p-1e-12,achieved_mass=mass,cap_limited=len(selected)==cap and mass<p-1e-12)

def lexical(chunks,search,arm,p=.9):
    docs=[torch.tensor(c) for c in chunks]
    if arm in ['iter_k12','iter_k48','iter48_qk_p90']:
        k=12 if arm=='iter_k12' else 48
        ids=iter_bm25_indices(docs,search,k,iter_hop_topk=4,iter_rounds=0) if chunks else []
        return ids,dict(kind='original iterBM25',requested_k=k,actual_candidates=len(ids),recency_fill=False)
    assert arm=='bm25_p90'
    if not chunks:
        result=nucleus([],[],p=p)
        result['empty_history']=True
        return [],result
    result=nucleus(bm25_scores(chunks,search),list(range(len(chunks))),p=p)
    return result['selected'],result

@torch.inference_mode()
def attention_scores(reader,static_states,candidate_states,query_h,probe_tokens=32,verify=False):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    assert candidate_states
    layer_index=next(i for i in range(reader.j,reader.L) if reader.config.layer_types[i]=='full_attention')
    fixed=torch.cat(static_states+candidate_states,dim=1)
    combined=torch.cat([fixed,query_h],dim=1)
    # j=21 is not an attention boundary: execute intervening DeltaNet layers.
    hidden=reader.layers(combined,reader.j,layer_index)
    block=reader.core.layers[layer_index];attn=block.self_attn
    x=block.input_layernorm(hidden);take=min(probe_tokens,query_h.shape[1]);length=x.shape[1]
    begin=sum(t.shape[1] for t in static_states);end=fixed.shape[1]
    query=x[:,-take:];memory=x[:,begin:end]
    q,_=torch.chunk(attn.q_proj(query).view(1,take,-1,attn.head_dim*2),2,dim=-1)
    q=attn.q_norm(q).transpose(1,2)
    k=attn.k_norm(attn.k_proj(memory).view(1,end-begin,-1,attn.head_dim)).transpose(1,2)
    positions=torch.arange(length,device=x.device)[None,:]
    cos,sin=reader.core.rotary_emb(hidden,positions[None].expand(3,-1,-1))
    q,_=apply_rotary_pos_emb(q,q,cos[:,-take:],sin[:,-take:])
    k,_=apply_rotary_pos_emb(k,k,cos[:,begin:end],sin[:,begin:end])
    k=k.repeat_interleave(attn.num_key_value_groups,dim=1)
    weights=(q@k.transpose(-1,-2)*attn.scaling).softmax(-1,dtype=torch.float32)
    lengths=[t.shape[1] for t in candidate_states]
    mass=torch.stack([v.sum(-1).mean() for v in weights.split(lengths,dim=-1)])
    assert bool(torch.isfinite(mass).all()) and abs(float(mass.sum())-1)<1e-5
    detail=dict(attention_layer=layer_index,intervening_layers=list(range(reader.j,layer_index)),memory_conditional=True,
        probe_query_positions=take,candidate_lengths=lengths,scores=mass.cpu().tolist())
    if verify:
        assert length<=2048,'Small probe reference only'
        old=reader.config._attn_implementation
        blocked=torch.arange(length,device=x.device)[None,:]>torch.arange(length,device=x.device)[:,None]
        mask=torch.zeros(1,1,length,length,dtype=x.dtype,device=x.device).masked_fill(blocked[None,None],torch.finfo(x.dtype).min)
        try:
            reader.config._attn_implementation='eager'
            _,actual=attn(x,position_embeddings=(cos,sin),attention_mask=mask,past_key_values=None)
        finally:reader.config._attn_implementation=old
        conditional=actual[:,:,-take:,begin:end].float()
        conditional/=conditional.sum(-1,keepdim=True).clamp_min(1e-30)
        reference=torch.stack([v.sum(-1).mean() for v in conditional.split(lengths,dim=-1)])
        delta=mass-reference
        detail['reference_check']=dict(max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()))
        assert detail['reference_check']['max_abs']<.01 and detail['reference_check']['rms']<.005,detail
    return mass.cpu().tolist(),detail
