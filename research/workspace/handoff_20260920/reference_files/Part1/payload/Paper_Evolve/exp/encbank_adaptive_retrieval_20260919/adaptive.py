"""Deterministic chunk nucleus selection; not output-token sampling."""
import math

def nucleus(scores, p=.9, max_chunks=48, min_chunks=4, ids=None):
    scores=[float(x) for x in scores]
    ids=list(range(len(scores))) if ids is None else list(ids)
    assert len(ids)==len(scores) and len(set(ids))==len(ids)
    assert 0 < p <= 1 and 0 <= min_chunks <= max_chunks
    assert all(math.isfinite(s) and s>=0 for s in scores)
    if not scores:return dict(selected=[],achieved_mass=0.,target_met=False,cap_limited=False,zero_mass=True)
    total=math.fsum(scores);cap=min(max_chunks,len(scores));floor=min(min_chunks,cap)
    if total==0:
        # Explicit recency fallback, never pretend zero BM25 scores are probabilities.
        return dict(selected=sorted(ids)[-floor:] if floor else [],achieved_mass=0.,target_met=False,cap_limited=False,zero_mass=True)
    ranked=sorted(zip(ids,scores),key=lambda pair:(-pair[1],pair[0]))
    picked=[];mass=0.
    for idx,score in ranked[:cap]:
        if len(picked)>=floor and mass>=p-1e-12:break
        picked.append(idx);mass+=score/total
    return dict(selected=sorted(picked),achieved_mass=mass,target_met=mass>=p-1e-12,
                cap_limited=mass<p-1e-12 and len(picked)==cap,zero_mass=False)

def fixed_topk(scores,k=48,ids=None):
    ids=list(range(len(scores))) if ids is None else list(ids)
    return sorted(i for i,s in sorted(zip(ids,scores),key=lambda item:(-float(item[1]),item[0]))[:k])

def boundary_attention_mass(reader, memory_h, query_h, chunk_size=512, probe_tokens=32):
    """Qwen3 first-suffix Q/K, full candidate positions; memory-conditional mass.

    No V/MLP/remaining-layer execution and no persistent projected-key index.
    Both projections include the current LoRA. Mean over heads and the final
    probe_tokens of the query; this is a chunk-scoring heuristic, not Twilight's
    per-head KV pruning or its approximation guarantee.
    """
    import torch
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    assert reader.config.model_type=='qwen3'
    assert memory_h.ndim==query_h.ndim==3 and memory_h.shape[0]==query_h.shape[0]==1
    assert memory_h.shape[1]%chunk_size==0 and memory_h.shape[1]>0
    layer=reader.layers[reader.resume_j];attn=layer.self_attn
    mem_tokens=memory_h.shape[1];q_tokens=query_h.shape[1];take=min(probe_tokens,q_tokens)
    assert take>0
    mem=layer.input_layernorm(memory_h);query=layer.input_layernorm(query_h[:,-take:])
    q=attn.q_norm(attn.q_proj(query).view(1,take,-1,attn.head_dim)).transpose(1,2)
    key=attn.k_norm(attn.k_proj(mem).view(1,mem_tokens,-1,attn.head_dim)).transpose(1,2)
    mp=torch.arange(1,1+mem_tokens,device=mem.device)[None,:]
    qp=torch.arange(1+mem_tokens+q_tokens-take,1+mem_tokens+q_tokens,device=mem.device)[None,:]
    mc,ms=reader.rotary_emb(mem,position_ids=mp);qc,qs=reader.rotary_emb(query,position_ids=qp)
    q=q*qc.unsqueeze(1)+rotate_half(q)*qs.unsqueeze(1)
    key=key*mc.unsqueeze(1)+rotate_half(key)*ms.unsqueeze(1)
    key=key.repeat_interleave(attn.num_key_value_groups,dim=1)
    weights=(torch.matmul(q,key.transpose(-1,-2))*attn.scaling).softmax(-1,dtype=torch.float32)
    mass=weights.reshape(1,q.shape[1],take,mem_tokens//chunk_size,chunk_size).sum(-1).mean((0,1,2))
    assert torch.isfinite(mass).all() and abs(float(mass.sum())-1)<1e-5
    return mass
