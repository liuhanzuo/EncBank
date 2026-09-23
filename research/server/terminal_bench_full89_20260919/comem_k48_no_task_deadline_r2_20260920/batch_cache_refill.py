"""Join active decode rows and independently prefilled newcomers, without pad tokens."""
import copy
import torch
from transformers.cache_utils import DynamicCache


def join_cache_groups(caches, pads, config, start, end, consume=False):
    """Preserve per-row RoPE positions externally; add only masked KV columns."""
    assert len(caches)==len(pads)>0
    full=next(i for i in range(start,end) if config.layer_types[i]=='full_attention')
    widths=[c.layers[full].keys.shape[-2] for c in caches]
    width=max(widths);merged=DynamicCache(config=config)
    for c,p in zip(caches,pads):
        assert p.ndim==1 and p.shape[0]==c.layers[full].keys.shape[0]
    for i in range(start,end):
        layers=[c.layers[i] for c in caches];dst=merged.layers[i]
        assert all(type(x) is type(dst) for x in layers)
        if config.layer_types[i]=='full_attention':
            assert type(dst).__name__=='DynamicLayer'
            for field in ['keys','values']:
                values=[getattr(x,field) for x in layers]
                assert all(x.shape[-2]==n for x,n in zip(values,widths))
                setattr(dst,field,torch.cat([torch.nn.functional.pad(x,(0,0,width-n,0)) for x,n in zip(values,widths)],dim=0))
                if consume:
                    for x in layers:setattr(x,field,None)
            dst.dtype=dst.keys.dtype;dst.device=dst.keys.device;dst.is_initialized=True
            if consume:
                for x in layers:x.is_initialized=False
        else:
            assert config.layer_types[i]=='linear_attention' and type(dst).__name__=='LinearAttentionLayer'
            first=layers[0];assert not first.record_past
            for field in ['number_of_states','is_conv_states_initialized','is_recurrent_states_initialized','has_previous_state','conv_kernel_size','device','dtype','record_past']:
                value=getattr(first,field);assert all(getattr(x,field)==value for x in layers)
                setattr(dst,field,copy.copy(value))
            for field in ['conv_states','recurrent_states']:
                states=getattr(first,field)
                setattr(dst,field,{key:torch.cat([getattr(x,field)[key] for x in layers],dim=0) for key in states})
                if consume:
                    for x in layers:setattr(x,field,dict.fromkeys(states))
    return merged,torch.cat([p+(width-n) for p,n in zip(pads,widths)])
