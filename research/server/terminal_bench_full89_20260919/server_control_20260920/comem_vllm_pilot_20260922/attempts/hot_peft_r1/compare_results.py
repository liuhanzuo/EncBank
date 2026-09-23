"""CPU-only numerical comparisons and matched-pair timing ratios."""
import hashlib
import json
import statistics
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
protocol = json.loads((H/'protocol.json').read_text())
results = {c['id']: json.loads((H/'results'/c['id']/'result.json').read_text()) for c in protocol['runs']}
groups = {}
for result in results.values():
    c = result['case']
    key = f"B{c['batch']}_H{c['history_tokens']}"
    groups.setdefault(key, {}).setdefault(c['variant'], []).append(result)
timings = {}
for key, variants in groups.items():
    row = {}
    for variant, items in variants.items():
        batch = items[0]['case']['batch']
        row[variant] = dict(repetitions=len(items),
            first_prefill_seconds=statistics.median(x['calls'][0]['prefill_wall_seconds'] for x in items),
            first_decode_refresh_seconds=statistics.median(x['calls'][0]['decode_and_refresh_wall_seconds'] for x in items),
            continuation_prefill_seconds=statistics.median(x['calls'][1]['prefill_wall_seconds'] for x in items),
            continuation_decode_seconds=statistics.median(x['calls'][1]['decode_and_refresh_wall_seconds'] for x in items),
            two_call_total_seconds=statistics.median(x['total_measured_seconds'] for x in items),
            peak_allocated_gib=max(c['peak_allocated_gib'] for x in items for c in x['calls']),
            peak_reserved_gib=max(c['peak_reserved_gib'] for x in items for c in x['calls']))
        row[variant]['aggregate_tokens_per_second'] = batch*544/row[variant]['first_decode_refresh_seconds']
        row[variant]['per_row_tokens_per_second'] = 544/row[variant]['first_decode_refresh_seconds']
    row['hot_vs_dense_decode_speedup'] = row['dense']['first_decode_refresh_seconds']/row['hot_merged']['first_decode_refresh_seconds']
    row['hot_vs_dense_two_call_speedup'] = row['dense']['two_call_total_seconds']/row['hot_merged']['two_call_total_seconds']
    if 'cold_merged' in row:
        row['hot_vs_cold_two_call_speedup'] = row['cold_merged']['two_call_total_seconds']/row['hot_merged']['two_call_total_seconds']
        row['merge_hot_decode_speedup'] = row['hot_legacy']['first_decode_refresh_seconds']/row['hot_merged']['first_decode_refresh_seconds']
    timings[key] = row

numeric = []
for left, right in [('main_hot_legacy_r0','main_hot_merged_r0'), ('main_cold_merged_r0','main_hot_merged_r0')]:
    for call in [1,2]:
        a = torch.load(H/'results'/left/f'call_{call}_logits.pt', map_location='cpu', weights_only=True)
        b = torch.load(H/'results'/right/f'call_{call}_logits.pt', map_location='cpu', weights_only=True)
        for point in ['prefill','end']:
            x,y = a[point].float(),b[point].float()
            delta = x-y
            lp,lq = x.log_softmax(-1),y.log_softmax(-1)
            kl = (lp.exp()*(lp-lq)).sum(-1)
            numeric.append(dict(left=left,right=right,call=call,point=point,positions=x.numel()//x.shape[-1],
                bitwise_equal=torch.equal(a[point],b[point]),max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
                kl_mean=float(kl.mean()),kl_max=float(kl.max()),argmax_agree=int((x.argmax(-1)==y.argmax(-1)).sum()),
                strict_prior_max_abs_0125=bool(delta.abs().max()<=.125), prior_hot_cold_kl_002=bool(kl.max()<=.02)))
result=dict(timings=timings,numerical_comparisons=numeric,
    scope='Fixed-trace speed experiment, sampling work included. No Terminal-Bench scores. BF16 merge changes numerical execution.',
    result_sources={str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (H/'results').glob('*/*.json')})
(H/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'COMPLETE','case_groups':len(timings),'numerical_comparisons':len(numeric)}))
