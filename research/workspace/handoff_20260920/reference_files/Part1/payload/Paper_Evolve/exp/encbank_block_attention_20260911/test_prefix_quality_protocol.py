"""Stdlib fixtures for exact-pack selection and paired quality accounting."""
from __future__ import annotations
import copy
import hashlib
import random
import unittest

from backend_quality_results import canonical_hash, score_prediction
from prefix_quality_protocol import (BRANCHES, LOCATION_FIELDS, build_prefix_quality_plan,
                                     pack_fingerprint, summarize_prefix_quality)


def prepared():
    rows = []
    for index in range(99):
        group = 'a' if index < 3 else 'b' if index < 6 else f'single-{index}'
        prompt = [11, 0 if index == 1 else index, 12]
        rows.append(dict(id=f'q{index}', document_id=group, split='dev',
                         source='allenai/qasper:train:v0.3', references=['red'],
                         selected_chunk_indices=[0, 2], document_chunks=[[3, 4], [7, 8]],
                         prompt_ids=prompt, probe_indices=[1], answer_ids=[21, 99],
                         question=f'Question {index}', answer_truncated=False))
    return rows


def records_for(plan):
    counters = dict(build=0, hit=0, miss=0)
    records = []
    for row in plan['ordered_rows']:
        for branch in BRANCHES:
            cache = {stem+'_count_before':value for stem,value in counters.items()}
            if branch == 'native_prefix':
                hit = row['expected_prefix_hit']
                counters['build'] += int(not hit)
                counters['miss'] += int(not hit)
                counters['hit'] += int(hit)
                cache.update(prefix_cache_hit=hit, prefix_unchanged=True,
                             query_private=True, request_state_released=True)
            cache.update({stem+'_count_after':value for stem,value in counters.items()})
            cache['cross_request_kv_reuse'] = branch == 'native_prefix'
            cache.update(prefix_unchanged=True, query_private=True, request_state_released=True)
            prediction = 'wrong' if row['ordinal'] == 0 and branch == 'native_cold' else 'red'
            record = {**copy.deepcopy(row), 'branch':branch, 'status':'complete',
                      'prediction':prediction, 'generated_ids':[8 if prediction == 'wrong' else 7, 99],
                      'finish_reason':'eos', 'eos_token_id':99, 'stop_token_ids':[99, 100],
                      'max_new_tokens':2, 'cache_observation':cache, 'shared_writer_hj_unchanged':True,
                      'first_logit_comparison':dict(max_abs=0., rms=0., signed_mean_error=0.,
                          centered_rms=0., numeric_gate_applied=False) if branch == 'native_prefix' else None,
                      'formal_inference_timing':False, 'formal_inference_memory':False}
            record.update(score_prediction(prediction, row['references']))
            records.append(record)
    return records


def aggregate(plan, records):
    return summarize_prefix_quality(plan, records, stop_token_ids=[99, 100], max_new_tokens=2)


class SelectionTests(unittest.TestCase):
    def test_smoke_uses_first_two_real_groups_and_different_prompts(self):
        rows = prepared()
        original = copy.deepcopy(rows)
        plan = build_prefix_quality_plan(rows)
        self.assertEqual(rows, original)
        self.assertEqual(len(plan['ordered_rows']), 4)
        self.assertEqual(len(plan['groups']), 2)
        self.assertEqual([r['expected_prefix_hit'] for r in plan['ordered_rows']], [False, True]*2)
        for group in plan['groups']:
            candidates = sorted([r for r in rows if pack_fingerprint(r) == group['pack_fingerprint']],
                                key=lambda r: hashlib.sha256(('42:'+r['id']).encode()).hexdigest())
            expected, prompts = [], set()
            for row in candidates:
                if tuple(row['prompt_ids']) not in prompts:
                    prompts.add(tuple(row['prompt_ids']))
                    expected.append(row['id'])
            self.assertEqual(group['ordered_ids'], expected[:2])
        shuffled = copy.deepcopy(rows)
        random.Random(193).shuffle(shuffled)
        self.assertEqual(plan, build_prefix_quality_plan(shuffled))

    def test_full_has_all_99_in_exact_groups(self):
        plan = build_prefix_quality_plan(prepared(), mode='full')
        self.assertEqual(len(plan['ordered_rows']), 99)
        self.assertEqual(set(plan['ordered_ids']), {f'q{i}' for i in range(99)})
        self.assertEqual(plan['expected_records'], 198)
        self.assertEqual(plan['expected_prefix_builds'], 95)
        self.assertEqual(plan['expected_prefix_hits'], 4)
        self.assertEqual(aggregate(plan, records_for(plan))['status'], 'complete')

    def test_pack_binds_document_identity_order_indices_and_exact_tokens(self):
        row = prepared()[0]
        same = copy.deepcopy(row)
        same.update(prompt_ids=[44, 45], references=['another answer'], id='another-question')
        self.assertEqual(pack_fingerprint(row), pack_fingerprint(same))
        for key, value in (('document_id', 'different'), ('selected_chunk_indices', [2, 0]),
                           ('document_chunks', [[7, 8], [3, 4]]), ('document_chunks', [[3, 9], [7, 8]])):
            changed = copy.deepcopy(row)
            changed[key] = value
            with self.subTest(key=key, value=value):
                self.assertNotEqual(pack_fingerprint(row), pack_fingerprint(changed))

    def test_no_real_distinct_prompt_groups_or_wrong_count_rejected(self):
        rows = prepared()
        for row in rows[:6]:
            row['prompt_ids'] = [11, 11, 12]
        with self.assertRaisesRegex(ValueError, 'two real exact-pack groups'):
            build_prefix_quality_plan(rows)
        for wrong in (prepared()[:98], prepared()+[prepared()[0]]):
            with self.assertRaisesRegex(ValueError, '99-row'):
                build_prefix_quality_plan(wrong)
        rows = prepared()
        rows[-1]['id'] = rows[0]['id']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_prefix_quality_plan(rows)

    def test_references_do_not_select_smoke_or_pack(self):
        rows = prepared()
        expected = build_prefix_quality_plan(rows)
        for row in rows:
            row['references'] = ['candidate-favorable artificial reference']
        actual = build_prefix_quality_plan(rows)
        self.assertEqual(actual['ordered_ids'], expected['ordered_ids'])
        self.assertEqual(actual['source_pack_sha256'], expected['source_pack_sha256'])
        self.assertNotEqual(actual['source_rows_sha256'], expected['source_rows_sha256'])


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_prefix_quality_plan(prepared())
        self.records = records_for(self.plan)

    def test_complete_scores_percent_points_counters_and_token_agreement(self):
        result = aggregate(self.plan, self.records)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['complete'])
        metrics = result['metrics']
        self.assertEqual(metrics['branches']['native_cold']['token_f1_percent'], 75.)
        self.assertEqual(metrics['branches']['native_prefix']['token_f1_percent'], 100.)
        for key in ('token_f1', 'exact_match'):
            self.assertEqual(metrics['paired'][key], dict(mean_delta_percentage_points=25., wins=1, ties=3, losses=0))
        self.assertEqual(metrics['paired']['generated_ids_equal']['agreement_percent'], 75.)
        self.assertEqual((metrics['prefix_builds'], metrics['prefix_hits'], metrics['cold_requests']), (2, 2, 4))

    def test_partial_failed_pending_never_have_complete_metrics(self):
        self.assertEqual(aggregate(self.plan, [])['status'], 'pending')
        partial = aggregate(self.plan, self.records[:-1])
        self.assertEqual(partial['status'], 'partial')
        self.assertIsNone(partial['metrics'])
        failed = copy.deepcopy(self.records)
        failed[-1].update(status='failed', error='OOM')
        for key in ('prediction', 'generated_ids', 'cache_observation', 'token_f1', 'exact_match'):
            failed[-1].pop(key)
        result = aggregate(self.plan, failed)
        self.assertEqual(result['status'], 'failed', result)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['metrics'])

    def test_generation_receipts_must_honor_eos_and_full_budget(self):
        changes = [dict(generated_ids=[99, 7]), dict(generated_ids=[7], finish_reason='max_new_tokens', eos_token_id=None),
                   dict(generated_ids=[7, 99], finish_reason='max_new_tokens', eos_token_id=None),
                   dict(eos_token_id=100), dict(stop_token_ids=[7]), dict(max_new_tokens=1)]
        for change in changes:
            records = copy.deepcopy(self.records)
            records[0].update(change)
            with self.subTest(change=change):
                self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        for record in self.records:
            record.update(generated_ids=[7, 8], finish_reason='max_new_tokens', eos_token_id=None)
        self.assertEqual(aggregate(self.plan, self.records)['status'], 'complete')

    def test_records_cannot_change_inputs_scores_order_or_plan_location(self):
        changes = [dict(references=['wrong']), dict(document_chunks=[[1, 2], [7, 8]]),
                   dict(prompt_ids=[1, 2, 3]), dict(ordinal=99), dict(pack_fingerprint='a'*64),
                   dict(token_f1=100.), dict(exact_match=float('nan'))]
        for change in changes:
            records = copy.deepcopy(self.records)
            records[0].update(change)
            with self.subTest(change=change):
                self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        self.assertEqual(aggregate(self.plan, self.records+[self.records[0]])['status'], 'invalid')
        records = copy.deepcopy(self.records)
        records[0], records[2] = records[2], records[0]
        self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')

    def test_cache_hits_private_state_and_counter_continuity_checked(self):
        changes = [dict(prefix_cache_hit=True), dict(build_count_after=3),
                   dict(query_private=False), dict(prefix_unchanged=False),
                   dict(request_state_released=False), dict(cross_request_kv_reuse=False)]
        for change in changes:
            records = copy.deepcopy(self.records)
            records[1]['cache_observation'].update(change)
            with self.subTest(change=change):
                self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        records = copy.deepcopy(self.records)
        # Rebase all counters in group two; local deltas remain valid but the run is discontinuous.
        for record in records[4:]:
            for key in record['cache_observation']:
                if key.endswith(('_before', '_after')):
                    record['cache_observation'][key] += 10
        self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        records = copy.deepcopy(self.records)
        records[0]['cache_observation']['build_count_after'] += 1
        self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')

    def test_invalid_plan_or_candidate_selected_reordering_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed['ordered_rows'][0]['references'] = ['different']
        self.assertEqual(aggregate(changed, self.records)['status'], 'invalid')

    def test_cold_invariants_and_first_logit_observation_are_required(self):
        for key in ('query_private', 'request_state_released', 'prefix_unchanged'):
            records = copy.deepcopy(self.records)
            records[0]['cache_observation'][key] = False
            with self.subTest(key=key):
                self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        for branch_index in (0, 1):
            records = copy.deepcopy(self.records)
            records[branch_index]['shared_writer_hj_unchanged'] = False
            self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        for comparison in (None, dict(max_abs=0., rms=float('inf'), signed_mean_error=0.,
                                     centered_rms=0., numeric_gate_applied=False),
                           dict(max_abs=0., rms=0., signed_mean_error=0., centered_rms=0., numeric_gate_applied=True)):
            records = copy.deepcopy(self.records)
            records[1]['first_logit_comparison'] = comparison
            self.assertEqual(aggregate(self.plan, records)['status'], 'invalid')
        changed = copy.deepcopy(self.plan)
        changed['ordered_ids'][0], changed['ordered_ids'][1] = changed['ordered_ids'][1], changed['ordered_ids'][0]
        changed['plan_sha256'] = canonical_hash({k:v for k,v in changed.items() if k != 'plan_sha256'})
        self.assertEqual(aggregate(changed, self.records)['status'], 'invalid')

    def test_location_fields_may_be_inferred_and_branch_interleaving_is_permitted(self):
        records = copy.deepcopy(self.records)
        for row in records:
            for key in LOCATION_FIELDS:
                row.pop(key)
        self.assertEqual(aggregate(self.plan, records)['status'], 'complete')
        by_branch = [r for branch in BRANCHES for r in records if r['branch'] == branch]
        self.assertEqual(aggregate(self.plan, by_branch)['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
