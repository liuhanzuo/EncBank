"""Stdlib grouping and paired QA receipts for native whole-prefix reuse.

Grouping uses existing exact ordered document packs. It never retrieves, changes
prompts, selects by answers, or manufactures cache hits. This is quality evidence,
not a timing, memory or numerical-equivalence gate.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict

from backend_quality_results import canonical_hash, score_prediction

BRANCHES = ('native_cold', 'native_prefix')
PLAN_SCHEMA = 'native-prefix-quality-plan-v1'
SUMMARY_SCHEMA = 'native-prefix-quality-summary-v1'
LOCATION_FIELDS = ('ordinal', 'group_index', 'query_index_in_group',
                   'pack_fingerprint', 'expected_prefix_hit')
INPUT_FIELDS = ('document_id', 'source', 'references', 'selected_chunk_indices',
                'document_chunks', 'prompt_ids', 'probe_indices')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(value, *, nonempty=True):
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(type(item) is int and item >= 0 for item in value))


def pack_fingerprint(row):
    """Bind document identity, chunk order/index and the exact existing tokens."""
    document = row.get('document_id')
    indices, chunks = row.get('selected_chunk_indices'), row.get('document_chunks')
    _require(isinstance(document, str) and bool(document), 'Missing document_id')
    _require(_ids(indices) and len(set(indices)) == len(indices), 'Invalid ordered chunk indices')
    _require(isinstance(chunks, list) and len(chunks) == len(indices)
             and all(_ids(chunk) for chunk in chunks), 'Invalid or misaligned document chunks')
    return canonical_hash(dict(document_id=document,
                               selected_chunk_indices=indices, document_chunks=chunks))


def _validate_row(row):
    _require(isinstance(row, dict), 'Prepared row must be an object')
    _require(isinstance(row.get('id'), str) and bool(row['id']), 'Missing row ID')
    _require(row.get('source') == 'allenai/qasper:train:v0.3', 'Use the existing Qasper-train dev pilot')
    if 'split' in row:
        _require(row['split'] == 'dev', 'Quality plan must use development rows')
    _require(not row.get('answer_truncated', False), 'Do not use truncated targets')
    _require(_ids(row.get('prompt_ids')), 'Invalid prompt IDs')
    probes = row.get('probe_indices')
    _require(_ids(probes) and probes == sorted(set(probes))
             and max(probes) < len(row['prompt_ids']), 'Invalid prompt-only probe positions')
    refs = row.get('references')
    _require(isinstance(refs, list) and refs and all(isinstance(x, str) for x in refs),
             'References must be nonempty text list')
    canonical_hash(row)  # Reject nonfinite/non-JSON data without importing Torch.
    return pack_fingerprint(row)


def build_prefix_quality_plan(rows, *, mode='smoke', seed=42):
    """Group exactly 99 existing rows; smoke uses two real two-prompt groups.

    ``ordered_rows`` preserves all prepared fields and adds LOCATION_FIELDS.
    ``groups`` points to those rows using ordered_ids and row_ordinals. Counters
    describe the online prefix reader: each group starts with explicit invalidation.
    """
    rows = list(rows)
    _require(mode in ('smoke', 'full') and type(seed) is int and seed == 42,
             'This fixed pilot supports smoke/full and seed 42 only')
    _require(len(rows) == 99, 'Require the existing complete 99-row development set')
    grouped, identities = defaultdict(list), set()
    for row in rows:
        fingerprint = _validate_row(row)
        _require(row['id'] not in identities, 'Duplicate development ID')
        identities.add(row['id'])
        grouped[fingerprint].append(row)
    ordered_groups = []
    for fingerprint, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: hashlib.sha256(
            f"{seed}:{row['id']}".encode('utf-8')).hexdigest())
        different, seen_prompts = [], set()
        for row in ordered:
            prompt = tuple(row['prompt_ids'])
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                different.append(row)
        ordered_groups.append((fingerprint, ordered, different))
    eligible = [group for group in ordered_groups if len(group[2]) >= 2]
    _require(len(eligible) >= 2,
             'Need two real exact-pack groups with different prompts; do not manufacture hits')
    source_pack_manifest = [dict(pack_fingerprint=fp,
        rows=[dict(id=row['id'], prompt_sha256=canonical_hash(row['prompt_ids'])) for row in ordered])
        for fp, ordered, _different in ordered_groups]
    chosen = [(fp, different[:2], len(ordered), len(different))
              for fp, ordered, different in eligible[:2]] if mode == 'smoke' else [
              (fp, ordered, len(ordered), len(different)) for fp, ordered, different in ordered_groups]
    selected, groups = [], []
    for group_index, (fingerprint, group, source_count, distinct_count) in enumerate(chosen):
        ordinals = []
        for index, row in enumerate(group):
            value = copy.deepcopy(row)
            ordinal = len(selected)
            value.update(ordinal=ordinal, group_index=group_index, query_index_in_group=index,
                         pack_fingerprint=fingerprint, expected_prefix_hit=index > 0)
            selected.append(value)
            ordinals.append(ordinal)
        groups.append(dict(group_index=group_index, pack_fingerprint=fingerprint,
                           document_id=group[0]['document_id'], ordered_ids=[row['id'] for row in group],
                           row_ordinals=ordinals, source_group_size=source_count,
                           source_distinct_prompt_count=distinct_count))
    plan = dict(schema=PLAN_SCHEMA, mode=mode, seed=seed, source_examples=99,
                source_pack_manifest=source_pack_manifest,
                source_pack_sha256=canonical_hash(source_pack_manifest),
                source_rows_sha256=canonical_hash(sorted(rows, key=lambda row: row['id'])),
                groups=groups, ordered_rows=selected, ordered_ids=[row['id'] for row in selected],
                branches=list(BRANCHES), expected_records=2*len(selected),
                expected_prefix_builds=len(groups), expected_prefix_hits=len(selected)-len(groups),
                cold_baseline_independent=True, group_reset='invalidate-prefix-before-each-group',
                selection='pack-fingerprint order; sha256(42:id) within groups; no answer-based selection',
                formal_inference_timing=False, formal_inference_memory=False)
    plan['plan_sha256'] = canonical_hash(plan)
    return plan


def _validate_plan(plan):
    _require(isinstance(plan, dict) and plan.get('schema') == PLAN_SCHEMA, 'Invalid prefix plan')
    _require(plan.get('plan_sha256') == canonical_hash({k:v for k,v in plan.items() if k != 'plan_sha256'}),
             'Plan was changed after construction')
    _require(plan.get('seed') == 42 and plan.get('source_examples') == 99
             and plan.get('branches') == list(BRANCHES) and plan.get('cold_baseline_independent') is True,
             'Unexpected plan protocol')
    rows = plan.get('ordered_rows')
    count = 4 if plan.get('mode') == 'smoke' else 99 if plan.get('mode') == 'full' else -1
    _require(isinstance(rows, list) and len(rows) == count, 'Invalid planned row count')
    _require(plan.get('ordered_ids') == [row['id'] for row in rows]
             and len(set(plan['ordered_ids'])) == count, 'Invalid planned ID order')
    _require(plan.get('source_pack_sha256') == canonical_hash(plan.get('source_pack_manifest')),
             'Source pack binding changed')
    manifest = plan['source_pack_manifest']
    _require(isinstance(manifest, list) and manifest
             and [g['pack_fingerprint'] for g in manifest] == sorted(set(g['pack_fingerprint'] for g in manifest)),
             'Invalid source pack order')
    source_ids, source_lookup, eligible = [], {}, []
    for group in manifest:
        entries = group['rows']
        _require(entries and [r['id'] for r in entries] == sorted(
            [r['id'] for r in entries], key=lambda ident: hashlib.sha256(f'42:{ident}'.encode()).hexdigest()),
            'Invalid source question order')
        different, seen = [], set()
        for entry in entries:
            source_ids.append(entry['id'])
            source_lookup[entry['id']] = (group['pack_fingerprint'], entry['prompt_sha256'])
            if entry['prompt_sha256'] not in seen:
                seen.add(entry['prompt_sha256'])
                different.append(entry['id'])
        if len(different) >= 2:
            eligible.append((group['pack_fingerprint'], different[:2]))
    _require(len(source_ids) == len(set(source_ids)) == 99 and len(eligible) >= 2,
             'Source manifest lacks the complete pilot or genuine smoke hits')
    expected_ids = source_ids if plan['mode'] == 'full' else [ident for _, ids in eligible[:2] for ident in ids]
    _require(plan['ordered_ids'] == expected_ids, 'Selected questions differ from fixed smoke/full selection')
    flattened = []
    for index, group in enumerate(plan.get('groups', [])):
        _require(group['group_index'] == index and bool(group['row_ordinals']), 'Invalid group order')
        for within, ordinal in enumerate(group['row_ordinals']):
            _require(type(ordinal) is int and ordinal == len(flattened), 'Noncontiguous group rows')
            row = rows[ordinal]
            _require(row['ordinal'] == ordinal and row['group_index'] == index
                     and row['query_index_in_group'] == within
                     and row['expected_prefix_hit'] is (within > 0), 'Planned row location mismatch')
            _require(_validate_row(row) == row['pack_fingerprint'] == group['pack_fingerprint'],
                     'Planned pack differs from its exact tokens')
            _require(source_lookup[row['id']] == (row['pack_fingerprint'], canonical_hash(row['prompt_ids'])),
                     'Planned question/pack differs from source manifest')
            flattened.append(row['id'])
        _require(group['ordered_ids'] == [rows[i]['id'] for i in group['row_ordinals']], 'Group IDs mismatch')
    _require(flattened == plan['ordered_ids'], 'Groups do not cover every planned row')
    _require(plan['expected_records'] == 2*count
             and plan['expected_prefix_builds'] == len(plan['groups'])
             and plan['expected_prefix_hits'] == count-len(plan['groups']), 'Expected cache counts mismatch')


def _validate_record(record, planned, stops, budget):
    _require(isinstance(record, dict), 'Record must be an object')
    canonical_hash(record)
    _require(record.get('id') == planned['id'], 'Record ID mismatch')
    _require(record.get('branch') in BRANCHES, 'Unexpected branch')
    for key in INPUT_FIELDS:
        _require(record.get(key) == planned[key], f'Record {key} differs from prepared input')
    for key in LOCATION_FIELDS:
        if key in record:
            _require(record[key] == planned[key], f'Record {key} differs from plan')
    _require(pack_fingerprint(record) == planned['pack_fingerprint'], 'Record pack mismatch')
    for key in ('formal_inference_timing', 'formal_inference_memory'):
        _require(record.get(key, False) is False, 'Quality records cannot claim formal infrastructure measurements')
    _require(record.get('status') in ('complete', 'failed'), 'Invalid record status')
    if record['status'] == 'failed':
        _require(bool(record.get('error')), 'Failed record lacks error')
        return
    _require(record.get('max_new_tokens') == budget, 'Output budget changed')
    _require(_ids(record.get('stop_token_ids'))
             and len(set(record['stop_token_ids'])) == len(record['stop_token_ids'])
             and set(record['stop_token_ids']) == set(stops), 'EOS policy changed')
    generated = record.get('generated_ids')
    _require(_ids(generated) and len(generated) <= budget, 'Invalid generated tokens')
    if record.get('finish_reason') == 'eos':
        _require(generated[-1] in stops and record.get('eos_token_id') == generated[-1]
                 and not any(token in stops for token in generated[:-1]), 'Generation continued after EOS')
    elif record.get('finish_reason') == 'max_new_tokens':
        _require(len(generated) == budget and record.get('eos_token_id') is None
                 and not any(token in stops for token in generated), 'Incomplete or EOS-containing limit output')
    else:
        raise ValueError('Invalid finish reason')
    _require(isinstance(record.get('prediction'), str), 'Prediction must be text')
    for key, expected in score_prediction(record['prediction'], planned['references']).items():
        value = record.get(key)
        _require(type(value) in (int, float) and math.isfinite(value)
                 and math.isclose(value, expected, rel_tol=0., abs_tol=1e-12), f'Incorrect recomputed {key}')
    cache = record.get('cache_observation')
    _require(isinstance(cache, dict), 'Missing cache observation')
    _require(record.get('shared_writer_hj_unchanged') is True, 'Shared writer inputs changed or lack validation')
    for key in ('prefix_unchanged', 'query_private', 'request_state_released'):
        _require(cache.get(key) is True, f'Request invariant failed: {key}')
    if record['branch'] == 'native_cold':
        _require(cache.get('cross_request_kv_reuse') is False, 'Cold baseline reused cross-request KV')
        for stem in ('build', 'hit', 'miss'):
            before, after = cache.get(stem+'_count_before'), cache.get(stem+'_count_after')
            _require(type(before) is int and type(after) is int and before >= 0 and before == after,
                     'Cold request changed prefix counters')
        return
    _require(cache.get('cross_request_kv_reuse') is True, 'Missing actual prefix reuse policy')
    comparison = record.get('first_logit_comparison')
    _require(isinstance(comparison, dict) and comparison.get('numeric_gate_applied') is False,
             'Missing first-logit observation or unexpected numerical gate')
    for key in ('max_abs', 'rms', 'signed_mean_error', 'centered_rms'):
        value = comparison.get(key)
        _require(type(value) in (int, float) and math.isfinite(value), f'Invalid first-logit {key}')
        if key != 'signed_mean_error':
            _require(value >= 0, f'Negative first-logit {key}')
    hit = planned['expected_prefix_hit']
    _require(cache.get('prefix_cache_hit') is hit, 'Expected real pack miss/hit differs')
    for stem, delta in (('build', int(not hit)), ('hit', int(hit)), ('miss', int(not hit))):
        before, after = cache.get(stem+'_count_before'), cache.get(stem+'_count_after')
        _require(type(before) is int and type(after) is int and before >= 0
                 and after-before == delta, f'Incorrect prefix {stem} counter change')


def summarize_prefix_quality(plan, records, *, stop_token_ids, max_new_tokens=128):
    """Return complete/pending/partial/failed/invalid; incomplete metrics are None.

    Each branch must appear in planned question order. Cold and prefix records
    may interleave, but cannot skip/reorder a question within either branch.
    """
    result = dict(schema=SUMMARY_SCHEMA, status='invalid', complete=False, metrics=None,
                  plan_sha256=plan.get('plan_sha256') if isinstance(plan, dict) else None,
                  formal_inference_timing=False, formal_inference_memory=False)
    try:
        _validate_plan(plan)
        stops = list(stop_token_ids)
        _require(_ids(stops) and len(set(stops)) == len(stops), 'Invalid expected EOS tokens')
        _require(type(max_new_tokens) is int and 0 < max_new_tokens <= 128, 'Invalid output budget')
        rows, records = plan['ordered_rows'], list(records)
        positions = {row['id']: row for row in rows}
        seen, next_ordinal = {}, dict.fromkeys(BRANCHES, 0)
        previous_prefix = None
        for record in records:
            _require(isinstance(record, dict) and record.get('id') in positions, 'Unexpected record ID')
            planned = positions[record['id']]
            _validate_record(record, planned, stops, max_new_tokens)
            branch = record['branch']
            key = (record['id'], branch)
            _require(key not in seen, 'Duplicate branch/ID record')
            _require(planned['ordinal'] == next_ordinal[branch], 'Record order differs from fixed plan')
            next_ordinal[branch] += 1
            seen[key] = record
            if branch == 'native_prefix' and record['status'] == 'complete':
                cache = record['cache_observation']
                if previous_prefix is not None:
                    for stem in ('build', 'hit', 'miss'):
                        _require(cache[stem+'_count_before'] == previous_prefix[stem+'_count_after'],
                                 'Prefix counters were reset or changed outside planned requests')
                previous_prefix = cache
        missing = [dict(id=row['id'], branch=branch) for row in rows for branch in BRANCHES
                   if (row['id'], branch) not in seen]
        failed = [dict(id=row['id'], branch=row['branch'], error=row['error'])
                  for row in records if row['status'] == 'failed']
        result.update(ordered_ids=plan['ordered_ids'], expected_records=plan['expected_records'],
                      records_received=len(records), missing=missing, failures=failed,
                      source_pack_sha256=plan['source_pack_sha256'])
        if failed or missing:
            result['status'] = 'failed' if failed else 'partial' if records else 'pending'
            return result
        branches = {}
        for branch in BRANCHES:
            branch_rows = [seen[(row['id'], branch)] for row in rows]
            branches[branch] = dict(examples=len(rows), **{
                key+'_percent': 100*math.fsum(record[key] for record in branch_rows)/len(rows)
                for key in ('token_f1', 'exact_match')}, finish_reasons={
                key:sum(record['finish_reason'] == key for record in branch_rows)
                for key in ('eos', 'max_new_tokens')})
        per_question = []
        for row in rows:
            left, right = (seen[(row['id'], branch)] for branch in BRANCHES)
            per_question.append(dict(id=row['id'], ordinal=row['ordinal'],
                pack_fingerprint=row['pack_fingerprint'], prefix_cache_hit=row['expected_prefix_hit'],
                token_f1_delta=right['token_f1']-left['token_f1'],
                exact_match_delta=right['exact_match']-left['exact_match'],
                generated_ids_equal=right['generated_ids'] == left['generated_ids'],
                prediction_equal=right['prediction'] == left['prediction']))
        paired = dict(examples=len(rows), delta_direction='native_prefix minus native_cold')
        for key in ('token_f1', 'exact_match'):
            values = [row[key+'_delta'] for row in per_question]
            paired[key] = dict(mean_delta_percentage_points=100*math.fsum(values)/len(values),
                               wins=sum(value > 0 for value in values),
                               ties=sum(value == 0 for value in values), losses=sum(value < 0 for value in values))
        for key in ('generated_ids_equal', 'prediction_equal'):
            equal = sum(row[key] for row in per_question)
            paired[key] = dict(equal=equal, different=len(rows)-equal, agreement_percent=100*equal/len(rows))
        result.update(status='complete', complete=True, metrics=dict(branches=branches, paired=paired,
            per_question=per_question, prefix_builds=plan['expected_prefix_builds'],
            prefix_hits=plan['expected_prefix_hits'], cold_requests=len(rows)))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
        result.update(status='invalid', complete=False, metrics=None,
                      error=f'{type(exc).__name__}: {exc}')
    return result
