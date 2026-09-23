"""Bounded model-level backend parity diagnosis, separate from formal timing.

The caller owns GPU admission and the autocast policy. This module neither loads
a model nor acquires/bypasses a GPU gate. Reference and candidate run sequentially
so their large persistent document states are never alive at the same time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

FORMAT = 'sparse-backend-parity-v1'
MAX_ABS = 0.15
MAX_RMS = 0.02


def _now():
    return datetime.now(timezone.utc).isoformat()


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector_metrics(reference, candidate):
    """Pure-stdlib numerical comparison; values are small CPU logit/score arrays."""
    shape_equal = len(reference) == len(candidate)
    finite = all(_finite_number(v) for v in reference) and all(_finite_number(v) for v in candidate)
    result = dict(elements=len(reference), length_equal=shape_equal, finite=finite,
                  max_abs=None, rms=None, mean_abs=None)
    if shape_equal and finite:
        delta = [float(a)-float(b) for a, b in zip(reference, candidate)]
        result.update(max_abs=max(map(abs, delta), default=0.),
                      rms=math.sqrt(math.fsum(d*d for d in delta)/len(delta)) if delta else 0.,
                      mean_abs=math.fsum(map(abs, delta))/len(delta) if delta else 0.)
    return result


def _top_two(values):
    if not values or not all(_finite_number(v) for v in values):
        return dict(greedy_id=None, top1=None, top2=None, top2_margin=None)
    # Python max chooses the first equal maximum, matching torch.argmax.
    first = max(range(len(values)), key=values.__getitem__)
    second = max((i for i in range(len(values)) if i != first), key=values.__getitem__, default=None)
    return dict(greedy_id=first, top1=values[first],
                top2=None if second is None else values[second],
                top2_margin=None if second is None else values[first]-values[second])


def logits_metrics(reference, candidate):
    result = vector_metrics(reference, candidate)
    result['reference'] = _top_two(reference)
    result['candidate'] = _top_two(candidate)
    result['greedy_equal'] = (result['finite'] and result['length_equal']
                             and result['reference']['greedy_id'] is not None
                             and result['reference']['greedy_id'] == result['candidate']['greedy_id'])
    result['within_error_thresholds'] = (result['finite'] and result['length_equal']
        and result['max_abs'] <= MAX_ABS and result['rms'] <= MAX_RMS)
    result['passed'] = bool(result['greedy_equal'] and result['within_error_thresholds'])
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if value is None or type(value) in (str, bool, int, float):
        return value
    return repr(value)


def _all_finite(value):
    if isinstance(value, dict):
        if 'nonfinite' in value:
            return False
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def route_metrics(reference, candidate):
    """Keep every route field, with only block score values permitted to differ."""
    differing = [key for key in sorted(set(reference) | set(candidate))
                 if key not in reference or key not in candidate
                 or (key != 'block_scores' and reference[key] != candidate[key])]
    ids_equal = ('selected_indices' in reference and 'selected_indices' in candidate
                 and reference['selected_indices'] == candidate['selected_indices'])
    scores = vector_metrics(reference.get('block_scores', []), candidate.get('block_scores', []))
    finite = _all_finite(reference) and _all_finite(candidate) and scores['finite']
    return dict(reference=_json_safe(reference), candidate=_json_safe(candidate),
                strict_selected_indices_equal=ids_equal, differing_non_score_fields=differing,
                block_score_error=scores, finite=finite,
                score_policy='Block scores may differ; report the error and require identical route decisions and other fields.',
                passed=bool(ids_equal and not differing and finite and scores['length_equal']))


def _tensor_metadata(tensor):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    storage = tensor.untyped_storage()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                stride=list(tensor.stride()), storage_offset=int(tensor.storage_offset()),
                data_ptr=int(tensor.data_ptr()), storage_ptr=int(storage.data_ptr()),
                storage_bytes=int(storage.nbytes()), tensor_bytes=int(tensor.numel()*tensor.element_size()),
                version=version, version_available=version is not None, tensor_identity=id(tensor))


def _pair_metadata(pair):
    return dict(k=_tensor_metadata(pair.k), v=_tensor_metadata(pair.v))


def _hot_metadata(entries):
    if entries is None:
        return None
    return [None if entry is None else dict(source_key=entry.source_key, token_count=entry.token_count,
        entry_identity=id(entry), signature_metadata_sha256=hashlib.sha256(repr(entry.signature).encode()).hexdigest(),
        h_m=_tensor_metadata(entry.h_m), raw_kv={str(layer): _pair_metadata(pair)
                                                for layer, pair in entry.raw_kv.items()}) for entry in entries]


def _input_metadata(sink, documents, entries):
    return dict(sink=_tensor_metadata(sink), documents=[_tensor_metadata(t) for t in documents],
                hot_cache=_hot_metadata(entries))


def _cache_comparison(before, after):
    def missing_versions(value):
        if isinstance(value, dict):
            return (1 if value.get('version_available') is False else 0) + sum(missing_versions(v) for v in value.values())
        if isinstance(value, list):
            return sum(missing_versions(v) for v in value)
        return 0
    missing = missing_versions(before) + missing_versions(after)
    return dict(metadata_unchanged=before == after, tensors_without_version_evidence=missing,
                passed=bool(before == after and missing == 0),
                evidence_scope='Shape/dtype/stride/storage-address/tensor identity and PyTorch version counters only; no tensor-content hash or full copy. Untracked writes can evade this check; data identity is not proven.')


def _state_metadata(state):
    return dict(query_position=int(state.query_position), pack_position=int(state.pack_position),
                selected_indices=list(state.selected_indices),
                doc_kv={str(layer): _pair_metadata(pair) for layer, pair in state.doc_kv.items()},
                query_kv={str(layer): _pair_metadata(pair) for layer, pair in state.query_kv.items()})


def validate_state(metadata, route, *, layers, resume_j, expected_heads, head_dim, prompt_length, decode_step):
    """Verify persistent 8-head KV and query-only length/storage, never 32-head expansion."""
    errors = []
    query_length = prompt_length + decode_step
    if metadata['query_position'] != query_length:
        errors.append('query_position does not equal prompt length + decode step')
    if metadata['pack_position'] != route['original_query_start'] + query_length:
        errors.append('pack_position does not preserve original document offsets')
    if metadata['selected_indices'] != route['selected_indices']:
        errors.append('state and route selected_indices differ')
    if set(metadata['query_kv']) != {str(i) for i in range(layers)}:
        errors.append('query KV layer set differs')
    if set(metadata['doc_kv']) != {str(i) for i in range(resume_j, layers)}:
        errors.append('document KV layer set differs')
    if expected_heads != 8:
        errors.append('This 8B diagnostic requires 8 persistent KV heads')
    for kind in ('query_kv', 'doc_kv'):
        for layer, pair in metadata[kind].items():
            expected_tokens = query_length if kind == 'query_kv' else route['document_kv_tokens_by_layer'].get(layer)
            for kv in ('k', 'v'):
                t = pair[kv]
                if t['shape'] != [1, expected_heads, expected_tokens, head_dim]:
                    errors.append(f'{kind}.{layer}.{kv}: unexpected persistent shape {t["shape"]}')
                if kind == 'query_kv' and t['storage_bytes'] > t['tensor_bytes']:
                    errors.append(f'{kind}.{layer}.{kv}: query view retains a larger backing storage')
    return dict(passed=not errors, errors=errors, expected_persistent_kv_heads=expected_heads,
                expected_query_tokens=query_length)


def _state_structure(metadata):
    # Different reader runs necessarily own different allocations/identities.
    return {kind: {layer: {kv: {field: tensor[field] for field in ('shape', 'dtype', 'device')}
                          for kv, tensor in pair.items()} for layer, pair in metadata[kind].items()}
            for kind in ('query_kv', 'doc_kv')}


def _hot_arguments(hot_cache, document_keys, document_count):
    if hot_cache is None:
        return None, {}
    if isinstance(hot_cache, dict):
        if set(hot_cache) - {'hot_entries', 'document_keys'}:
            raise ValueError('hot_cache mapping accepts hot_entries and document_keys only')
        entries = hot_cache['hot_entries']
        keys = hot_cache.get('document_keys', document_keys)
        if document_keys is not None and keys != document_keys:
            raise ValueError('Conflicting document_keys')
    else:
        entries, keys = hot_cache, document_keys
    entries = list(entries)
    if keys is None:
        if any(entry is None for entry in entries):
            raise ValueError('Partial hot entries require explicit document_keys')
        keys = [entry.source_key for entry in entries]
    keys = list(keys)
    if len(entries) != document_count or len(keys) != document_count:
        raise ValueError('Hot entries/keys must match candidate document count')
    return entries, dict(hot_entries=entries, document_keys=keys)


def _write_receipt(path, receipt):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(_json_safe(receipt), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def run_backend_parity(reference_reader, candidate_reader, sink, documents, prompt, probe_indices, *,
                       hot_cache=None, document_keys=None, decode_steps=3, out_dir):
    """Return and save a diagnostic receipt with a strict boolean ``passed``.

    Preserve the caller's autocast context. Both readers must already be eval and
    share the very same backbone object. Run reference first; retain only small
    CPU vocabulary logits and metadata before freeing its entire request state.
    Errors are recorded as passed=False; the caller must decline advancement.
    """
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path = folder/f'backend_parity_{stamp}.json'
    receipt = dict(format=FORMAT, started_utc=_now(), receipt_path=str(path.resolve()), passed=False,
        formal_timing_eligible=False, measurement_scope='Diagnostic only; CPU logit transfers and comparisons are excluded from formal latency/memory measurements.',
        thresholds=dict(logits_max_abs=MAX_ABS, logits_rms=MAX_RMS,
                        policy='Provisional BF16 model-scale thresholds; passing also requires finite values, identical greedy IDs/routes, and persistent-cache checks. Changes require an explicit reviewed protocol revision.'),
        decode_steps=decode_steps, stages=[], errors=[], cache_evidence={},
        reference_class=type(reference_reader).__name__, candidate_class=type(candidate_reader).__name__,
        execution='Sequential reference then candidate; candidate consumes the exact reference greedy token at each step, even after a reported flip.')
    try:
        if type(decode_steps) is not int or not 0 <= decode_steps <= 32:
            raise ValueError('decode_steps must be an integer in [0,32] for this bounded diagnostic')
        if reference_reader.training or candidate_reader.training:
            raise ValueError('Both readers must be in eval mode before this diagnostic')
        if reference_reader.model is not candidate_reader.model:
            raise ValueError('Reference and candidate must share the identical backbone object')
        fields = ('j', 'm', 'L', 'rho', 'probe_mode', 'num_heads', 'num_kv_heads', 'head_dim',
                  'attention_chunk_size', 'score_chunk_size')
        config = {key: getattr(reference_reader, key) for key in fields}
        if config != {key: getattr(candidate_reader, key) for key in fields}:
            raise ValueError('The two readers have different attention graphs/model dimensions')
        receipt['reader_graph'] = config
        documents = list(documents)
        entries, kwargs = _hot_arguments(hot_cache, document_keys, len(documents))
        if hasattr(prompt, 'shape'):
            shape = tuple(prompt.shape)
            if len(shape) not in (1, 2) or (len(shape) == 2 and shape[0] != 1):
                raise ValueError('Prompt must describe exactly one request')
            prompt_length = int(shape[-1])
        else:
            prompt_length = len(prompt)
        if prompt_length <= 0:
            raise ValueError('Prompt must be nonempty')
        receipt['inputs'] = dict(prompt_tokens=prompt_length, document_blocks=len(documents),
            probe_indices=None if probe_indices is None else list(probe_indices), hot_cache_provided=entries is not None,
            document_keys=kwargs.get('document_keys'))
        before = _input_metadata(sink, documents, entries)
        receipt['cache_evidence']['before'] = before
        # Torch is imported only by the active diagnosis, not by stdlib tests.
        import torch

        def run(reader, run_name, forced_tokens=None):
            values, records, tokens = [], [], []
            receipt.setdefault('runs', {})[run_name] = dict(stages=records, greedy_ids=tokens)
            state = logits = None
            try:
                with torch.no_grad():
                    for step in range(decode_steps+1):
                        if step == 0:
                            logits, state = reader.prefill(sink, documents, prompt,
                                probe_indices=probe_indices, **kwargs)
                            doc_before = {str(layer): _pair_metadata(pair) for layer, pair in state.doc_kv.items()}
                            route_before = _json_safe(state.route_stats)
                        else:
                            token = tokens[-1] if forced_tokens is None else forced_tokens[step-1]
                            if token is None:
                                raise ValueError('Reference logits are nonfinite; cannot choose a replay token')
                            logits = reader.decode_step(token, state)
                        shape = list(logits.shape)
                        if len(shape) != 3 or shape[:2] != [1, 1] or shape[-1] < 1:
                            raise ValueError(f'Expected one-token vocabulary logits, got {shape}')
                        row = logits.detach().float().cpu().reshape(-1).tolist()
                        values.append(row)
                        tokens.append(_top_two(row)['greedy_id'])
                        meta = _state_metadata(state)
                        check = validate_state(meta, state.route_stats, layers=reader.L, resume_j=reader.j,
                            expected_heads=reader.num_kv_heads, head_dim=reader.head_dim,
                            prompt_length=prompt_length, decode_step=step)
                        records.append(dict(step=step, logits_shape=shape, logits_dtype=str(logits.dtype),
                            consumed_reference_token=None if step == 0 else (tokens[step-1] if forced_tokens is None else forced_tokens[step-1]),
                            state=meta, state_check=check, route_stats=_json_safe(state.route_stats),
                            route_fixed=route_before == _json_safe(state.route_stats)))
                    doc_after = {str(layer): _pair_metadata(pair) for layer, pair in state.doc_kv.items()}
                    unchanged = _cache_comparison(doc_before, doc_after)
                    return values, records, tokens, unchanged
            finally:
                # No GC/empty_cache/synchronization changes to allocator policy;
                # ordinary references to GPU outputs/state end before next run.
                del state, logits

        ref_values, ref_records, ref_tokens, ref_doc_check = run(reference_reader, 'reference')
        after_reference = _input_metadata(sink, documents, entries)
        receipt['cache_evidence'].update(after_reference=after_reference,
            reference_input_cache=_cache_comparison(before, after_reference), reference_doc_cache=ref_doc_check)
        can_values, can_records, can_tokens, can_doc_check = run(candidate_reader, 'candidate', ref_tokens)
        after_candidate = _input_metadata(sink, documents, entries)
        receipt['cache_evidence'].update(after_candidate=after_candidate,
            candidate_input_cache=_cache_comparison(before, after_candidate), candidate_doc_cache=can_doc_check)
        receipt.update(reference_greedy_ids=ref_tokens, candidate_greedy_ids=can_tokens,
                       candidate_input_tokens=ref_tokens[:decode_steps])
        for step, (ref, can) in enumerate(zip(ref_values, can_values)):
            r, c = ref_records[step], can_records[step]
            numeric = logits_metrics(ref, can)
            route = route_metrics(r['route_stats'], c['route_stats'])
            structure = _state_structure(r['state']) == _state_structure(c['state'])
            stage = dict(phase='prefill' if step == 0 else f'decode_{step}', numerical=numeric,
                route=route, persistent_shape_dtype_device_equal=structure,
                reference_state=r, candidate_state=c)
            stage['passed'] = bool(numeric['passed'] and route['passed'] and structure
                and r['state_check']['passed'] and c['state_check']['passed']
                and r['route_fixed'] and c['route_fixed'] and r['logits_dtype'] == c['logits_dtype'])
            receipt['stages'].append(stage)
        cache_checks = [receipt['cache_evidence'][name] for name in
                        ('reference_input_cache', 'candidate_input_cache', 'reference_doc_cache', 'candidate_doc_cache')]
        receipt['passed'] = bool(len(receipt['stages']) == decode_steps+1
                                 and all(stage['passed'] for stage in receipt['stages'])
                                 and all(check['passed'] for check in cache_checks))
        receipt['observed_max_abs'] = max((s['numerical']['max_abs'] for s in receipt['stages']
                                         if s['numerical']['max_abs'] is not None), default=None)
        receipt['observed_max_rms'] = max((s['numerical']['rms'] for s in receipt['stages']
                                         if s['numerical']['rms'] is not None), default=None)
    except Exception as error:
        receipt['errors'].append(dict(type=type(error).__name__, message=str(error)))
        receipt['passed'] = False
    receipt['finished_utc'] = _now()
    _write_receipt(path, receipt)
    return receipt
