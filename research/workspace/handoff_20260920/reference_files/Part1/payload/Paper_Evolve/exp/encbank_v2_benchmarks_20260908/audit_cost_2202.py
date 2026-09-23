"""CPU/stdlib-only audit and reporting of completed local serving attempts."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'heartbeat_cost_20260908_2202.json'
STATE = ROOT / 'results/local/bootstrap_serving/status.json'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)

def finite(obj):
    if isinstance(obj, float):
        assert math.isfinite(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            finite(value)
    elif isinstance(obj, list):
        for value in obj:
            finite(value)

def token_identity(path):
    # Read only the small CPU token artifact; never load model/cache tensors.
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if '/data/' in name or name.endswith('/data.pkl'):
                h.update(name.encode())
                h.update(z.read(name))
    return h.hexdigest()

state = read(STATE)
jobs = []
rows = []
raw = {}
workloads = {}
tokens = {}
packs = {}
new_count = 0
new_raw_count = 0
new_names = {'j0_131072', 'pub_lora_32768', 'pub_lora_131072'}
query_checks = 0
for key, info in state['jobs'].items():
    if '/full/' not in key or info['status'] != 'complete':
        continue
    folder = Path(info['result'])
    assert not (folder / 'INVALIDATED.json').exists()
    completion = read(folder / 'COMPLETED.json')
    config = read(folder / 'config.json')
    summary = read(folder / 'summary.json')
    assert len(summary) == completion['cells'] == info['cells'] == 12
    name = key.rsplit('/', 1)[-1]
    is_new = name in new_names
    arm, length = summary[0]['arm'], summary[0]['context_tokens']
    hardware = config['hardware']
    assert hardware == completion['hardware']
    assert hardware['device_name'] == 'NVIDIA GeForce RTX 5090'
    assert hardware['timing_eligible'] and hardware['platform'] == 'Windows'
    assert hardware['torch_cpu_threads'] == 2 and hardware['torch_interop_threads'] == 16
    assert hardware['omp_num_threads'] == hardware['mkl_num_threads'] == '2'
    assert hardware['tokenizers_parallelism'] == 'false'
    admission = hardware.get('gpu_admission')
    if admission is not None:
        assert admission['effective_idle_slack_gib'] == 5
        assert admission['comparison'] == 'strictly_less_than'
        assert admission['source'] == 'nvidia-smi MiB / 1024'
        assert admission['initial_used_gib'] < 5 and admission['recheck_used_gib'] < 5
        assert not admission['other_python_compute_processes']
    else:
        assert name in {'fix_all_32768', 'pub_32768'}
        # Early accepted runs predate structured dual-check provenance.
        # Keep this evidence limitation instead of inventing exact values.
        log = ROOT / 'results/local/bootstrap_serving/logs' / f'base_full_{name}_0001.log'
        admission = {'structured_dual_checks': 'not recorded',
                     'legacy_gate_log': next(line for line in log.read_text(encoding='utf-8', errors='replace').splitlines() if 'lock taken' in line)}
    if arm == 'pub_lora':
        assert config['adapter']['step'] == 4000
        assert config['adapter']['load_mode'] == 'peft_unmerged'
    else:
        assert config['adapter'] is None
    assert config['model'] == 'F:\\qencbank\\models\\Qwen3-8B'
    assert config['j'] == 12 and config['chunk_size'] == 512 and config['topk'] == 12
    assert config['generation_lengths'] == [16, 128] and config['query_counts'] == [1, 10, 100]
    assert config['fixed_generation_length'] and config['source_tokens'] == 2202561
    work = read(folder / f'workload_{length}.json')
    assert len(work['questions']) == 100 and not work['source_repeated']
    assert work == workloads.setdefault(length, work)
    store = folder / f'store_{length}_{arm}'
    ident = token_identity(store / 'tokens.pt')
    assert ident == tokens.setdefault(length, ident)
    actual_store_bytes = sum(p.stat().st_size for p in store.iterdir() if p.is_file())
    assert all(s['write']['serialized_bytes'] == actual_store_bytes for s in summary)
    files = list(folder.glob('queries_*.jsonl'))
    assert len(files) == 4
    file_records = {}
    for path in files:
        qs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert [q['id'] for q in qs] == list(range(100))
        tier = path.stem.split('_')[-2]
        g = int(path.stem.split('_')[-1][1:])
        raw[(arm, length, tier, g)] = qs
        file_records[tier, g] = qs
        for q in qs:
            assert q['hardware'] == hardware
            assert q['capture_calls'] == 0
            assert q['generated_tokens'] == len(q['generated_ids']) == g
            assert q['decode_steps'] == g - 1 and q['fixed_generation_length']
            assert len(set(q['selected_indices'])) == 12
            pack_key = length, q['id']
            pack = (q['selected_indices'], q['query_tokens'], q['read_tokens'])
            assert pack == packs.setdefault(pack_key, pack)
            if is_new:
                finite(q)
                assert all(value >= 0 for field, value in q.items() if field.endswith('_s') and isinstance(value, (int, float)))
                near(q['total_s'], q['ttft_s'] + q['decode_s'])
                new_raw_count += 1
            query_checks += 1
    for g in (16, 128):
        assert [q['generated_ids'] for q in file_records['cpu', g]] == [q['generated_ids'] for q in file_records['disk', g]]
    seen = set()
    for s in summary:
        assert s['hardware'] == hardware and not s['source_repeated']
        assert s['source_preprocessing_charged'] is False
        cell = s['tier'], s['G'], s['Q']
        assert cell not in seen
        seen.add(cell)
        qs = file_records[s['tier'], s['G']][:s['Q']]
        # Newly completed rows receive the full arithmetic audit; old rows
        # retain their prior passed audit and are read for matrix comparison.
        if is_new:
            finite(s)
            for metric, total in s['query_totals'].items():
                near(total, sum(q[metric] for q in qs))
            near(s['end_to_end_total_s'], s['write']['write_total_s'] + s['startup']['startup_load_s'] + s['query_totals']['total_s'])
            near(s['mean_ttft_s'], s['query_totals']['ttft_s'] / s['Q'])
            near(s['decode_tokens_per_s'], s['query_totals']['decode_steps'] / s['query_totals']['decode_s'])
            assert s['incremental_peak_bytes'] == max(q['incremental_peak_bytes'] for q in qs)
            assert s['peak_allocated_bytes'] == max(s['write']['write_peak_allocated_bytes'], max(q['peak_allocated_bytes'] for q in qs))
            new_count += 1
        rows.append(s)
    assert seen == {(t, g, q) for t in ('cpu', 'disk') for g in (16, 128) for q in (1, 10, 100)}
    jobs.append({'job': key, 'path': str(folder), 'cells': 12, 'new_full_arithmetic_audit': is_new,
                 'admission': admission, 'serialized_bytes': actual_store_bytes,
                 'token_tensor_identity': ident, 'adapter_mode': (config['adapter'] or {}).get('load_mode')})

index = {(s['arm'], s['context_tokens'], s['tier'], s['G'], s['Q']): s for s in rows}
comparisons = []
for length in (32768, 131072):
    for tier in ('cpu', 'disk'):
        for g in (16, 128):
            for q in (1, 10, 100):
                v, j = (index[a, length, tier, g, q] for a in ('fix_all', 'j0'))
                comparisons.append({'context_tokens': length, 'tier': tier, 'G': g, 'Q': q,
                    'v2_cumulative_s': v['end_to_end_total_s'], 'j0_cumulative_s': j['end_to_end_total_s'],
                    'v2_over_j0_cumulative': v['end_to_end_total_s']/j['end_to_end_total_s'],
                    'j0_over_v2_cumulative_speedup': j['end_to_end_total_s']/v['end_to_end_total_s'],
                    'v2_mean_ttft_s': v['mean_ttft_s'], 'j0_mean_ttft_s': j['mean_ttft_s'],
                    'j0_over_v2_ttft_speedup': j['mean_ttft_s']/v['mean_ttft_s'],
                    'v2_peak_GiB': v['peak_allocated_bytes']/2**30, 'j0_peak_GiB': j['peak_allocated_bytes']/2**30})

crossings = []
for length in (32768, 131072):
    for tier in ('cpu', 'disk'):
        for g in (16, 128):
            cumulative = {}
            for arm in ('fix_all', 'j0'):
                s = index[arm, length, tier, g, 100]
                acc = s['write']['write_total_s'] + s['startup']['startup_load_s']
                vals = []
                for row in raw[arm, length, tier, g]:
                    acc += row['total_s']
                    vals.append(acc)
                cumulative[arm] = vals
            wins = [i + 1 for i, (v, j) in enumerate(zip(cumulative['fix_all'], cumulative['j0'])) if v < j]
            first_sustained = next((q for q in wins if wins[-1] == 100 and wins[wins.index(q):] == list(range(q, 101))), None)
            crossings.append({'context_tokens': length, 'tier': tier, 'G': g,
                'measured_prefix_v2_wins': wins, 'first_observed_faster_Q': wins[0] if wins else None,
                'first_Q_faster_through_100': first_sustained,
                'interpretation': 'Only this measured query ordering and Q<=100; no extrapolation or universal break-even.'})

distributions = []
for (arm, length, tier, g), qs in raw.items():
    if arm != 'fix_all':
        continue
    distribution = {'arm': arm, 'context_tokens': length, 'tier': tier, 'G': g}
    for field in ('total_s', 'ttft_s', 'decode_s', 'load_s', 'transfer_s', 'rotate_prepare_s', 'read_prefill_s'):
        values = sorted(q[field] for q in qs)
        distribution[field] = {'mean': statistics.mean(values), 'median': statistics.median(values), 'p95_nearest_rank': values[94], 'maximum': values[-1]}
    distributions.append(distribution)

compact = []
for s in rows:
    compact.append({k: s[k] for k in ('arm', 'context_tokens', 'tier', 'G', 'Q', 'end_to_end_total_s', 'mean_ttft_s', 'decode_tokens_per_s', 'peak_allocated_bytes', 'incremental_peak_bytes')} |
                   {'write': s['write'], 'startup': s['startup'], 'query_totals': s['query_totals']})

for campaign, agg in [('base', ROOT / 'results/local/serving_reuse/full/summary.json'),
                      ('pub_lora', ROOT / 'results/local/serving_reuse/pub_lora/full/summary.json')]:
    expected = [s for s in rows if (s['arm'] == 'pub_lora') == (campaign == 'pub_lora')]
    actual = read(agg)
    def key(s):
        return s['arm'], s['context_tokens'], s['tier'], s['G'], s['Q']
    assert sorted(expected, key=key) == sorted(actual, key=key)

result = {'checked_at': dt.datetime.now().astimezone().isoformat(), 'success': True,
          'source_status_updated_at': state['updated_at'], 'active_job_at_snapshot': state.get('active_job'),
          'full_cells_complete': len(rows), 'new_cells_full_arithmetic_audited': new_count,
          'new_query_records_full_arithmetic_audited': new_raw_count,
          'matrix_query_identity_metadata_records_checked': query_checks,
          'jobs': jobs, 'rows': compact, 'v2_vs_j0': comparisons, 'measured_prefix_crossings': crossings,
          'v2_untrimmed_distributions': distributions,
          'boundary': 'Pretokenized fixed-length document + raw queries to generated token IDs; one full document store write + each tier/G startup + Q prefix query total. Full-source read/tokenization, prefix preparation, model load/warmup, client question construction and output string decoding excluded.',
          'limitations': ['Unscored controlled excerpt workload, not QA quality or matched QA Pareto point.', 'Ordinary OS page cache, no eviction; tier timing is not a cold-disk claim.', 'Single process run per method/length; untrimmed query order includes long tails.', 'No generalized or extrapolated break-even claim.', 'Peak bytes are PyTorch allocated memory, not nvidia-smi total resident memory.', 'Persistent store bytes are distinct from peak GPU memory.', 'Actual online KV tensor bytes were not directly measured; allocated-memory peaks, incremental peaks and transfer bytes cannot substitute for that quantity.']}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('checked_at', 'success', 'full_cells_complete', 'new_cells_full_arithmetic_audited', 'new_query_records_full_arithmetic_audited', 'active_job_at_snapshot')}, indent=2))
