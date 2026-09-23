"""No policy token/time ceiling; native position and memory bounds stay explicit."""
def output_allowance(context_tokens, actual_input_tokens, max_new_tokens=None):
    assert max_new_tokens is None, 'This protocol forbids an arbitrary output token cap'
    return max(0, context_tokens-actual_input_tokens)

def deadline_expired(request):
    assert request.get('deadline_epoch') is None, 'No elapsed-time cancellation in this protocol'
    return False

def native_memory_admissible(allocated, current_cache, projected_h_growth, rows, plan):
    worst_cache=rows*plan['context_tokens']*plan['kv_bytes_per_token']
    projected=allocated-current_cache+projected_h_growth+worst_cache+plan['decode_workspace_gib']*2**30
    return projected <= plan['native_allocator_cap_gib']*2**30, projected
