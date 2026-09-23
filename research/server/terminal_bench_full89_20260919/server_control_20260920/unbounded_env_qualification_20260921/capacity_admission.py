"""Reserve complete prompt + generation budget, never crop the task context."""
def reservation(prompt_tokens, context_tokens, max_new_tokens):
    if prompt_tokens >= context_tokens:
        return 1  # The worker returns context_limit without a model forward.
    assert max_new_tokens is None
    return context_tokens

def admissible(active_count, reserved_tokens, requested_tokens, slots, token_budget):
    return active_count < slots and reserved_tokens + requested_tokens <= token_budget
