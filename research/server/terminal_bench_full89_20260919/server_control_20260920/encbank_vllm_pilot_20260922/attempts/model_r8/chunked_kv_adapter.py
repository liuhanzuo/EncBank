"""Isolated append-capacity experiment; not a vLLM paged-attention backend.

Keep identical K/V values and sequence lengths while avoiding a full history
concatenation on every decode step. Capacity grows by 512 tokens, with no fixed
sequence ceiling. Row reorder/refill/crop invalidates or safely reuses the pool.
"""
import os
import torch
from transformers.cache_utils import DynamicLayer

ORIGINAL_UPDATE = DynamicLayer.update
CHUNK = 512

def reserve_layer(layer, additional=1):
    if type(layer) is not DynamicLayer or not layer.is_initialized or layer.keys.numel() == 0:
        return
    n = layer.keys.shape[-2]
    needed = n + additional
    pool = getattr(layer, '_encbank_append_pool', None)
    valid = pool is not None and (
        layer.keys.data_ptr() == pool[0].data_ptr() and layer.values.data_ptr() == pool[1].data_ptr()
        and layer.keys.shape[:2] == pool[0].shape[:2]
        and layer.keys.stride() == pool[0][..., :n, :].stride()
        and layer.values.stride() == pool[1][..., :n, :].stride())
    if valid and pool[0].shape[-2] >= needed:
        return
    capacity = ((needed+CHUNK-1)//CHUNK)*CHUNK
    keys = layer.keys.new_empty((*layer.keys.shape[:-2], capacity, layer.keys.shape[-1]))
    values = layer.values.new_empty((*layer.values.shape[:-2], capacity, layer.values.shape[-1]))
    keys[..., :n, :].copy_(layer.keys)
    values[..., :n, :].copy_(layer.values)
    layer._encbank_append_pool = (keys, values)
    layer.keys, layer.values = keys[..., :n, :], values[..., :n, :]

def chunked_update(layer, key_states, value_states, *args, **kwargs):
    if os.environ.get('ENCBANK_ISOLATED_KERNEL_PILOT') != '1' or torch.is_grad_enabled():
        raise RuntimeError('Isolated inference only')
    # Multi-token prefill retains the exact original path.
    if (type(layer) is not DynamicLayer or not layer.is_initialized
            or layer.keys.numel() == 0 or key_states.shape[-2] != 1):
        if hasattr(layer, '_encbank_append_pool'):
            del layer._encbank_append_pool
        return ORIGINAL_UPDATE(layer, key_states, value_states, *args, **kwargs)
    n = layer.keys.shape[-2]
    reserve_layer(layer, additional=1)
    keys, values = layer._encbank_append_pool
    keys[..., n:n+1, :].copy_(key_states)
    values[..., n:n+1, :].copy_(value_states)
    layer.keys, layer.values = keys[..., :n+1, :], values[..., :n+1, :]
    return layer.keys, layer.values

def reserve_cache(cache):
    for part in cache[:2]:
        for layer in part.layers:
            reserve_layer(layer)

def check_contract():
    """Append across allocation boundaries and a row selection; exact values."""
    reference, candidate = DynamicLayer(), DynamicLayer()
    original = DynamicLayer.update
    checks = []
    try:
        for step in range(1035):
            length = 7 if step == 0 else 1
            batch = 3 if step < 520 else 2
            if step == 520:
                ids = torch.tensor([2, 0], device='cuda')
                reference.batch_select_indices(ids)
                candidate.batch_select_indices(ids)
            k = torch.randn(batch, 4, length, 8, device='cuda', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            expected = original(reference, k, v)
            actual = chunked_update(candidate, k, v)
            if step in [0, 1, 504, 505, 506, 519, 520, 521, 1017, 1018, 1034]:
                for a,e in zip(actual,expected):
                    torch.testing.assert_close(a,e,rtol=0,atol=0)
                assert candidate.get_seq_length() == reference.get_seq_length()
                checks.append(dict(step=step, length=candidate.get_seq_length(), batch=batch, exact=True))
        # A compacted/copied view must trigger fresh ownership of the buffers.
        reference.keys=reference.keys[..., 5:, :].clone();reference.values=reference.values[..., 5:, :].clone()
        candidate.keys=candidate.keys[..., 5:, :].clone();candidate.values=candidate.values[..., 5:, :].clone()
        expected=original(reference,k,v);actual=chunked_update(candidate,k,v)
        for a,e in zip(actual,expected):torch.testing.assert_close(a,e,rtol=0,atol=0)
        checks.append(dict(operation='compact-and-append',exact=True))
    finally:
        DynamicLayer.update=original
    return dict(status='PASS',checks=checks,capacity_chunk=CHUNK,sequence_limit=None)
