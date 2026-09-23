"""Unregistered natural-QA control flow over the frozen FP16 Method API.

No model/tokenizer loading or scoring. Timings are variable-length per-request diagnostics.
Caller owns the immutable entry and Method; this function owns one returned request.
"""
from __future__ import annotations

import weakref
from time import perf_counter


def _ids(values, name, *, nonempty=False):
    ids = list(values)
    if nonempty and not ids:
        raise ValueError(f'{name} must contain the complete nonempty query')
    if any(type(t) is not int or t < 0 for t in ids):
        raise ValueError(f'{name} requires nonnegative integer token IDs')
    return ids


def _argmax(logits, eos_token_id, suppress_eos):
    # Tensor-shaped API; no torch import is needed to exercise control-flow mocks.
    values = logits[0, -1].float()
    if not bool(values.isfinite().all()):
        raise RuntimeError('Nonfinite original QA logits before EOS masking')
    if eos_token_id >= values.numel():
        raise ValueError('Explicit EOS is outside the vocabulary')
    if suppress_eos:
        values = values.clone()  # Do not mutate original logits or a captured reference.
        values[eos_token_id] = float('-inf')
    return int(values.argmax().item())


def _refs(request):
    return tuple((name, weakref.ref(tensor)) for name, tensor in request.tensor_items())


def read_natural(method, entry, query_ids, bare_question_ids, *,
                 max_new_tokens, bos_token_id, eos_token_id,
                 suppress_first_eos, decode_ids, synchronize, clock=perf_counter):
    """Read a complete query and return uncropped IDs plus untrimmed decoded strings.

    ``decode_ids`` is a bound tokenizer.decode-compatible callable supplied by the
    formal caller. This module does not construct/import a tokenizer. No defaults
    choose a BOS or EOS protocol. Current prepared protocol requires scalar151645
    with first-step suppression; another policy needs a separate declaration.
    """
    synchronize()
    read_started = clock()
    first_token_time = last_token_time = final_decision_time = None
    query = _ids(query_ids, 'query_ids', nonempty=True)
    bare = _ids(bare_question_ids, 'bare_question_ids')
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be the positive frozen per-item cap')
    if type(bos_token_id) is not int or bos_token_id not in (151643, 151645):
        raise ValueError('An explicit declared BOS151643 or BOS151645 is required')
    if type(eos_token_id) is not int or eos_token_id != 151645 or suppress_first_eos is not True:
        raise ValueError('This preparation preserves scalarEOS151645 and first-step suppression')
    if method.bos != bos_token_id:
        raise ValueError('Declared BOS does not match the existing Method')
    if getattr(method, 'memory', None) is not None:
        if method.memory.bos_token_id != bos_token_id or method.memory.eos_token_id != eos_token_id:
            raise ValueError('Existing H memory uses a different declared BOS/EOS')
    request = None
    references = ()
    active_error = None
    generated = []
    stopped = False
    try:
        request = method.open_request(entry, bare)
        references = _refs(request)
        for index, token in enumerate(query):
            logits = request.consume(token, emit_head=index == len(query) - 1, phase='query')
        for decision in range(max_new_tokens):
            token = _argmax(logits, eos_token_id, decision == 0)
            final_decision_time = clock()  # _argmax.item() has synchronized this decision.
            logits = None
            if token == eos_token_id:
                stopped = True
                break  # Stop token is neither stored nor forwarded.
            if first_token_time is None:
                first_token_time = final_decision_time
            last_token_time = final_decision_time
            generated.append(token)
            if len(generated) == max_new_tokens:
                break  # Last sampled non-EOS token is pending, not cached again.
            logits = request.consume(token, emit_head=True, phase='decode')
        selected = request.selected_indices
        query_calls, decode_calls, head_calls = request.query_calls, request.decode_calls, request.head_calls
        if query_calls != len(query) or head_calls != decode_calls + 1:
            raise RuntimeError('Complete-query/head execution counts differ from the request contract')
        expected_decode = len(generated) if stopped else len(generated) - 1
        if decode_calls != expected_decode:
            raise RuntimeError('Sampled/pending/EOS cache-consumption counts differ')
        prefix_tokens = request.prefix_tokens
        if hasattr(request, 'position'):
            next_position = int(request.position)
            assert next_position == prefix_tokens + query_calls + decode_calls
            position_usage = {'mode':'full_native_cache', 'next_position':next_position,
                              'maximum_consumed_position':next_position-1}
        else:
            lower_next, upper_next = int(request.query_position), int(request.pack_position)
            assert lower_next == query_calls + decode_calls and upper_next == prefix_tokens + lower_next
            position_usage = {'mode':'chunk_local_Write_selected_contiguous_Read',
                              'lower_query_next_position':lower_next, 'upper_next_position':upper_next,
                              'maximum_lower_query_position':lower_next-1,
                              'maximum_upper_read_position':upper_next-1}
    except BaseException as error:
        active_error = error
        raise
    finally:
        if request is not None:
            # Close even when a consume, finite check or count assertion raises.
            cleanup_errors = []
            try:
                references += _refs(request)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                request.close()
            except BaseException as error:
                cleanup_errors.append(error)
            request = None
            if cleanup_errors:
                if active_error is None:
                    raise cleanup_errors[0]
                for error in cleanup_errors:
                    active_error.add_note(f'Request cleanup also failed: {type(error).__name__}: {error}')
    alive = [name for name, ref in references if ref() is not None]
    if alive:
        raise RuntimeError(f'Returned request retained tracked tensor objects after close: {alive}')
    synchronize()
    cleanup_finished = clock()
    # Decode only after cache release. Neither spelling is stripped/first-lined.
    raw_text = decode_ids(list(generated), skip_special_tokens=False)
    prediction = decode_ids(list(generated), skip_special_tokens=True)
    if not isinstance(raw_text, str) or not isinstance(prediction, str):
        raise TypeError('Decoder must return strings')
    full_read_finished = clock()
    decode_seconds = last_token_time - first_token_time if len(generated) >= 2 else None
    timing = {
        'clock': 'time.perf_counter (caller-injectable for CPU mocks)',
        'pre_read_cuda_synchronized': True,
        'ttft_seconds': first_token_time - read_started if first_token_time is not None else None,
        'last_non_EOS_token_seconds': last_token_time - read_started if last_token_time is not None else None,
        'decode_seconds_first_to_last_non_EOS': decode_seconds,
        'decode_tokens_first_to_last': max(0, len(generated)-1),
        'decode_tokens_per_second': (len(generated)-1)/decode_seconds if decode_seconds is not None and decode_seconds > 0 else None,
        'final_decision_seconds': final_decision_time-read_started if final_decision_time is not None else None,
        'EOS_decision_end_seconds': final_decision_time-read_started if stopped else None,
        'read_through_request_cleanup_seconds': cleanup_finished-read_started,
        'full_Read_including_text_decode_seconds': full_read_finished-read_started,
        'includes': ['argument checks','retrieval','open_request/materialization/fork','full query prefill','natural decode','request cleanup','text decode for full_Read'],
        'excludes': ['Write','entry fingerprints','before-call CUDA synchronization'],
        'scope': 'variable-length quality request diagnostics, not fixed-work infra or cross-device timing',
    }
    return {
        'position_usage': position_usage,
        'quality_timing': {
            'generated_tokens': len(generated),
            'ttft_seconds': timing['ttft_seconds'],
            'decode_seconds': timing['decode_seconds_first_to_last_non_EOS'],
            'decode_tokens': timing['decode_tokens_first_to_last'],
            'decode_tokens_per_second': timing['decode_tokens_per_second'],
            'read_total_seconds': timing['full_Read_including_text_decode_seconds'],
            'EOS_decision_end_seconds': timing['EOS_decision_end_seconds'],
            'scope': timing['scope'],
        },
        'timing': timing,
        'generated_token_ids': generated, 'raw_generated_text': raw_text, 'prediction': prediction,
        'generation_length': len(generated), 'max_new_tokens': max_new_tokens,
        'stopped_on_eos': stopped, 'hit_generation_cap': not stopped,
        'actual_BOS_token_id': bos_token_id, 'EOS_token_id': eos_token_id,
        'first_step_EOS_suppressed': True,
        'query_tokens': len(query), 'query_prefill_calls': query_calls, 'query_tokens_per_call': 1,
        'decode_forward_count': decode_calls, 'head_calls': head_calls,
        'consumed_generated_tokens': decode_calls,
        'pending_token_id_before_request_release': None if stopped else generated[-1],
        'stop_token_not_stored_or_forwarded': True,
        'selected_chunk_indices': None if selected is None else list(selected),
        'packed_read_tokens': prefix_tokens + len(query),
        'request_release': {'all_tracked_tensor_objects_released': True,
                            'tracked_tensor_objects': len(references), 'alive_names': []},
        'selected_read_released': True,
        'scope': 'Natural QA adapter; does not certify model quality, entry provenance, or a formal experiment',
    }
