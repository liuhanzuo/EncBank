"""Only complete exited new arms enter the frozen official RULER scorer."""
from pathlib import Path
import argparse, ast, datetime, random, re, statistics
from protocol import ROOT, ARMS, local, read, save, sha, module

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--arm', default='dense')  # Uniform wrapper binding, not a scoring filter.
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert sha(a.plan) == a.expected_plan_sha256 and not Path(a.output).exists()
    plan = read(a.plan)
    for name, digest in plan['source_sha256'].items(): assert sha(local(name)) == digest, name
    fixture = read(local(plan['fixture']['path']))
    assert sha(local(plan['fixture']['path'])) == plan['fixture']['sha256']
    assert sha(local(plan['labels']['path'])) == plan['labels']['sha256']
    labels = read(local(plan['labels']['path']))
    assert labels['fixture_sha256'] == plan['original_fixture']['sha256']
    label_map = {x['item_id']: x['references'] for x in labels['items']}
    scorer = module(local(plan['scorer']['path']), 'official_ruler_scorer').string_match_all
    source = local(plan['postprocess']['path'])
    assert sha(source) == plan['postprocess']['sha256']
    node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.FunctionDef) and x.name == 'postprocess_pred')
    scope = {'re': re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    postprocess = scope['postprocess_pred']
    ids = [x['id'] for x in fixture['items']]
    assert len(ids) == len(set(ids)) == len(label_map) == 100 and set(ids) == set(label_map)
    reports, scores = {}, {}
    for arm in ARMS:
        out = local(plan['outputs'][arm])
        execution, worker, result = (read(out / name) for name in ('execution.json', 'worker.json', 'result.json'))
        assert execution['status'] == 'completed_pending_independent_analysis'
        assert execution['worker_exit_code'] == 0 and execution['actual_parent_wait'] and execution['process_exit_observed'] and execution['source_stable']
        assert worker['status'] == 'completed' and worker['scientific_result_sha256'] == sha(out / 'result.json')
        assert result['outer_closed'] and all(x['completed'] for x in result['phases'])
        assert result['phases'][0]['name'] == 'outer_model_load_quality_and_release'
        rows = result['rows']
        assert [x['id'] for x in rows] == ids
        assert all(1 <= len(x['generated_token_ids']) <= 48 for x in rows)
        if arm != 'dense':
            assert all(x['entry_unchanged'] and x['selected_read_released'] and x['entry_release']['all_tracked_tensor_objects_released'] for x in rows)
            assert all(x['dequantized_chunk_indices'] == x['selected_chunk_indices'] and len(x['selected_chunk_indices']) <= 12 for x in rows)
            assert all(x['store']['document_lower_KV_bytes'] == 0 for x in rows)
        refs = [label_map[i] for i in ids]
        predictions = [postprocess(x['prediction'], {}) for x in rows]
        scores[arm] = [scorer([pred], [ref]) for pred, ref in zip(predictions, refs)]
        reports[arm] = {'official_string_match_all_percent': scorer(predictions, refs),
            'n': 100, 'capped': sum(x['hit_generation_cap'] for x in rows),
            'empty_predictions': sum(not x for x in predictions), 'zero_scores': sum(s == 0 for s in scores[arm]),
            'per_item_scores': dict(zip(ids, scores[arm])), 'result_sha256': sha(out / 'result.json'),
            'official_postprocessed_predictions': dict(zip(ids, predictions)),
            'worker_sha256': sha(out / 'worker.json'), 'execution_sha256': sha(out / 'execution.json')}
    rng = random.Random(20260911)
    draws = [[rng.randrange(100) for _ in range(100)] for _ in range(10000)]
    contrasts = {}
    for arm, reference in [('h8','h16'), ('h4','h16'), ('h16','dense'), ('h8','dense'), ('h4','dense')]:
        delta = [x-y for x,y in zip(scores[arm],scores[reference])]
        values = sorted(statistics.fmean(delta[i] for i in draw) for draw in draws)
        contrasts[f'{arm}_minus_{reference}'] = {'mean_points': statistics.fmean(delta),
            'percentile_interval_95': [values[249], values[9749]],
            'interpretation': 'Descriptive paired unique-document bootstrap; includes-zero is unresolved, not equivalence or noninferiority.',
            'causal_quantization_comparison': reference == 'h16'}
    record = {'status': 'complete_400_answers_actual_arm_exits_and_original_scorer_verified_pending_independent_review',
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'plan_sha256': a.expected_plan_sha256, 'arms': reports, 'paired_contrasts': contrasts,
        'resampling': {'rng': 'Python Random MT19937', 'seed': 20260911, 'draws':10000,
                       'unit': '100 unique exact-document clusters, one question each', 'interval_order_indices_zero_based':[249,9749]},
        'scope': 'New official-generator niah_multikey_1 8K/100 cell only; no full RULER macro; no literature checkpoint equivalence; not main infra measurement.',
        'claims_not_established': ['same-checkpoint literature ranking', 'full five-benchmark result', 'quality noninferiority', 'speedup', 'maximum capacity', 'multi-query GPU reuse']}
    save(a.output, record)
    print({k:v['official_string_match_all_percent'] for k,v in reports.items()})

if __name__ == '__main__': main()
