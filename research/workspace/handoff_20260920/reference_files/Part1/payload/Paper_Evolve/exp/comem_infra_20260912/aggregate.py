"""Check completed records and derive the local infrastructure table."""
from pathlib import Path
import collections
import json
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
paper = ROOT / 'COMem/paper_iclr2027_rewrite_20260912'
records = []
metadata = []
for process in (1, 2, 3):
    d = HERE / f'results/process_{process:02d}'
    complete = json.loads((d/'complete.json').read_text())
    rows = [json.loads(s) for s in (d/'records.jsonl').read_text().splitlines()]
    assert complete['complete'] and complete['records'] == len(rows) == 360
    keys = {(r['workload'], r['phase'], r['arm'], r['rep']) for r in rows}
    expected = {(w, phase, arm, r) for w in range(3) for phase in ('read','ttft')
                for arm in ('comem_k12','replay_k12','replay_k10') for r in range(20)}
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
assert all(selection[(w,'comem_k12')] == selection[(w,'replay_k12')] for w in range(3))
summary = {}
details = []
for arm in ('comem_k12', 'replay_k12', 'replay_k10'):
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
lines = ['# 本机补测结果', '', 'RTX 5090；相同 Qwen3-8B + 新训练 final4000 adapter；3 独立进程 × 3 文档 × 20 正式重复。',
         '延迟为三个进程中位数的中位数；显存为全部重复中的最大 allocated peak，含权重，十进制 GB。', '',
         '| 算法 | chunks | Read ms | TTFT ms | Read 峰值 GB | TTFT 峰值 GB |',
         '|---|---:|---:|---:|---:|---:|']
labels = {'comem_k12':'CoMem','replay_k12':'Raw replay','replay_k10':'Raw replay'}
tex_rows = []
for arm, v in summary.items():
    k = 10 if arm=='replay_k10' else 12
    values = [f"{v['read']['latency_ms']:.1f}", f"{v['ttft']['latency_ms']:.1f}",
              f"{v['read']['peak_GB']:.2f}",f"{v['ttft']['peak_GB']:.2f}"]
    lines.append('| '+' | '.join([labels[arm],str(k),*values])+' |')
    tex_rows.append(' & '.join([labels[arm],str(k),*values])+r'\\')
lines.extend(['', '该新训练 checkpoint 未确认等于 ARR 原权重。结果是新本机成本实验，不补写为 H20 历史显存。',
              'Read 与 TTFT 不含 document Write；TTFT 从已 tokenized query 开始，含 CPU iterative BM25、fetch、sink/query Write 和 prefill。',
              'top10 是固定预算，未称为本机等延迟校准。三个进程及文档分组统计见 summary.json。'])
(HERE/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
# Keep the paper's reviewed layout when refreshing numerical aggregates.
display_rows = []
best = [min(summary[arm][phase][key] for arm in summary)
        for phase, key in [('read','latency_ms'),('ttft','latency_ms'),('read','peak_GB'),('ttft','peak_GB')]]
for arm in ('replay_k12', 'replay_k10', 'comem_k12'):
    v = summary[arm]
    numbers = [v['read']['latency_ms'],v['ttft']['latency_ms'],v['read']['peak_GB'],v['ttft']['peak_GB']]
    cells = [f'{x:.1f}' if i < 2 else f'{x:.2f}' for i,x in enumerate(numbers)]
    cells = [r'\textbf{'+s+'}' if x==best[i] else s for i,(s,x) in enumerate(zip(cells,numbers))]
    name = r'\textbf{CoMem}' if arm=='comem_k12' else 'Raw replay'
    if arm=='comem_k12': display_rows.append(r'\rowcolor{resultshade}')
    display_rows.append(' & '.join([name, '10' if arm=='replay_k10' else '12', *cells])+r'\\')
template = (HERE/'tab_infra_template.tex').read_text(encoding='utf-8')
assert template.count('@@LOCAL_ROWS@@') == 1
table = template.replace('@@LOCAL_ROWS@@', '\n'.join(display_rows))
(paper/'sections/tab_infra.tex').write_text(table, encoding='utf-8')
process_rows = []
for phase in ('read','ttft'):
    for arm in ('comem_k12','replay_k12','replay_k10'):
        v = summary[arm][phase]
        k = 10 if arm=='replay_k10' else 12
        process_rows.append(' & '.join([labels[arm]+f' ({k})', 'Read' if phase=='read' else 'TTFT',
                           *[f'{x:.1f}' for x in v['process_medians_ms']], f"{v['peak_GB']:.2f}"])+r'\\')
process_table = r'''\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Method (chunks) & Phase & Process 1 & Process 2 & Process 3 & Peak GB\\
\midrule
''' + '\n'.join(process_rows) + r'''
\bottomrule
\end{tabular}
\caption{Local RTX 5090 supplement: within-process median latency in milliseconds, each over three fixed documents and twenty timed repetitions per document. The main-table latency is the median across these three process medians. Peak GB is the maximum allocated memory across all 180 formal observations for the row. All arms share the same principal unmerged adapter; no accuracy is inferred from this cost workload.}
\label{tab:local-infra-processes}
\end{table}
'''
(paper/'sections/tab_local_infra_processes.tex').write_text(process_table, encoding='utf-8')
(paper/'LOCAL_INFRA_RESULTS_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(summary, indent=2))

# Keep completed source-length and B300 supplements in the final manuscript.
import runpy
extended=ROOT/'exp/comem_infra_128k_20260912'
if (extended/'summary.json').exists():
    runpy.run_path(str(extended/'update_paper.py'),run_name='__main__')
