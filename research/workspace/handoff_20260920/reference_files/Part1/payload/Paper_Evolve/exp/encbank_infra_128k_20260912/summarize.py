"""Check completed records and derive the local infrastructure table."""
from pathlib import Path
import collections
import json
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
paper = ROOT / 'Encbank/paper_iclr2027_rewrite_20260912'
records = []
metadata = []
for process in (1, 2, 3):
    d = HERE / f'results/process_{process:02d}'
    complete = json.loads((d/'complete.json').read_text())
    rows = [json.loads(s) for s in (d/'records.jsonl').read_text().splitlines()]
    assert complete['complete'] and complete['records'] == len(rows) == 360
    keys = {(r['workload'], r['phase'], r['arm'], r['rep']) for r in rows}
    expected = {(w, phase, arm, r) for w in range(3) for phase in ('read','ttft')
                for arm in ('encbank_k12','replay_k12','replay_k10') for r in range(20)}
    assert keys == expected and all(r['process'] == process for r in rows)
    assert all(r['latency_ms'] > 0 and r['peak_allocated_bytes'] >= r['baseline_allocated_bytes'] for r in rows)
    assert all(r['pack_tokens'] == (5633 if r['arm']=='replay_k10' else 6657) for r in rows)
    correctness = json.loads((d/'correctness.json').read_text())
    assert correctness['same_top1'] and correctness['max_abs_logit_difference'] <= .125
    metadata.append(json.loads((d/'metadata.json').read_text()))
    records.extend(rows)
for m in metadata[1:]:
    for k in ('gpu','torch','transformers','cuda','adapter','adapter_config','attention'):
        assert m[k] == metadata[0][k], k
selection = collections.defaultdict(set)
for r in records:
    selection[(r['workload'], r['arm'])].add(tuple(r['selected_chunks']))
assert all(len(v)==1 for v in selection.values())
assert all(selection[(w,'encbank_k12')] == selection[(w,'replay_k12')] for w in range(3))
summary = {}
details = []
for arm in ('encbank_k12', 'replay_k12', 'replay_k10'):
    summary[arm] = {}
    for phase in ('read','ttft'):
        rows = [r for r in records if r['arm']==arm and r['phase']==phase]
        procmedians = [statistics.median(r['latency_ms'] for r in rows if r['process']==p) for p in (1,2,3)]
        value = {'latency_ms': statistics.median(procmedians), 'process_medians_ms': procmedians,
                 'peak_GB': max(r['peak_allocated_bytes'] for r in rows)/1e9,
                 'min_peak_GB': min(r['peak_allocated_bytes'] for r in rows)/1e9,
                 'n': len(rows)}
        summary[arm][phase] = value
        for p in (1,2,3):
            for w in range(3):
                group = [r for r in rows if r['process']==p and r['workload']==w]
                details.append({'arm':arm,'phase':phase,'process':p,'workload':w,
                                'median_ms':statistics.median(r['latency_ms'] for r in group),
                                'min_ms':min(r['latency_ms'] for r in group),
                                'max_ms':max(r['latency_ms'] for r in group),
                                'peak_GB':max(r['peak_allocated_bytes'] for r in group)/1e9})
out = {'record_count':len(records), 'summary':summary,'details':details,'metadata':metadata}
(HERE/'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')

assert all(int(m['args']['source_length'])==131072 for m in metadata)
print(json.dumps(summary, indent=2))
