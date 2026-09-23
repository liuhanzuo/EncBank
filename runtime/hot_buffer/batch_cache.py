"""Ragged one-token decode: independent prefill, left-masked KV, row-local GDN.

No padding tokens are passed through the recurrent model. Only completed
attention K/V tensors are padded; original per-row RoPE positions are retained.
"""
import copy
import torch
from transformers.cache_utils import DynamicCache


def merge_caches(caches, lengths, config, start, end, consume=False):
    assert len(caches)==len(lengths)>0 and min(lengths)>0
    merged=DynamicCache(config=config);width=max(lengths)
    for i in range(start,end):
        layers=[c.layers[i] for c in caches];dst=merged.layers[i]
        assert all(type(x) is type(dst) for x in layers)
        if config.layer_types[i]=='full_attention':
            assert type(dst).__name__=='DynamicLayer'
            assert all(x.is_initialized and x.keys.shape[0]==1 and x.keys.shape[-2]==n for x,n in zip(layers,lengths))
            dst.keys=torch.cat([torch.nn.functional.pad(x.keys,(0,0,width-n,0)) for x,n in zip(layers,lengths)],dim=0)
            dst.values=torch.cat([torch.nn.functional.pad(x.values,(0,0,width-n,0)) for x,n in zip(layers,lengths)],dim=0)
            dst.dtype=dst.keys.dtype;dst.device=dst.keys.device;dst.is_initialized=True
            if consume:
                for x in layers:x.keys=None;x.values=None;x.is_initialized=False
        else:
            assert config.layer_types[i]=='linear_attention' and type(dst).__name__=='LinearAttentionLayer'
            first=layers[0];assert not first.record_past
            for field in ['number_of_states','is_conv_states_initialized','is_recurrent_states_initialized','has_previous_state','conv_kernel_size','device','dtype','record_past']:
                value=getattr(first,field)
                assert all(getattr(x,field)==value for x in layers),field
                setattr(dst,field,copy.copy(value))
            for field in ['conv_states','recurrent_states']:
                states=getattr(first,field)
                assert all(t is not None and t.shape[0]==1 for t in states.values())
                setattr(dst,field,{key:torch.cat([getattr(x,field)[key] for x in layers],dim=0) for key in states})
                if consume:
                    for x in layers:setattr(x,field,dict.fromkeys(states))
    pads=torch.tensor([width-n for n in lengths],device=next(x.device for x in merged.layers[start:end] if x.device is not None),dtype=torch.long)
    return merged,pads


def decode_layers(reader, hidden, start, end, cache, positions, left_pads):
    assert hidden.shape[1]==1 and positions.shape==(hidden.shape[0],)
    position_ids=positions[:,None]
    rotary=reader.core.rotary_emb(hidden,position_ids[None].expand(3,-1,-1))
    full_layer=next(i for i in range(start,end) if reader.config.layer_types[i]=='full_attention')
    size=cache.layers[full_layer].keys.shape[-2]+1
    allowed=torch.arange(size,device=hidden.device)[None,:]>=left_pads[:,None]
    mask=allowed[:,None,None,:]
    mask._row_left_pads=left_pads.tolist()
    for block in reader.core.layers[start:end]:
        hidden=block(hidden,position_embeddings=rotary,
            attention_mask=mask if block.block_type=='full_attention' else None,
            position_ids=position_ids,past_key_values=cache,use_cache=True)
    return hidden


def compact(cache, positions, pads, indices):
    cache.reorder_cache(indices)
    positions=positions.index_select(0,indices);pads=pads.index_select(0,indices)
    # Continuous refill can keep a cohort alive indefinitely. Remove columns
    # masked for every surviving row when the longest old request exits.
    # Clone releases their storage; a sliced view would retain the allocation.
    common=int(pads.min().item())
    if common:
        for layer in cache.layers:
            if type(layer).__name__=='DynamicLayer' and layer.is_initialized:
                assert layer.keys.shape[-2]>common
                layer.keys=layer.keys[...,common:,:].clone()
                layer.values=layer.values[...,common:,:].clone()
        pads=pads-common
    return positions,pads
