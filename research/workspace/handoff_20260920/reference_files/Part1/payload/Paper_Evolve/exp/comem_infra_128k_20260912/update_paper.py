"""Integrate the completed 32k and 128k supplements without changing accuracy."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PAPER=ROOT/'COMem/paper_iclr2027_rewrite_20260912'
base=ROOT/'exp/comem_infra_20260912'
summaries={length:json.loads((folder/'summary.json').read_text())['summary'] for length,folder in [('32k',base),('128k',HERE)]}
fields=[('read','latency_ms'),('ttft','latency_ms'),('read','peak_GB'),('ttft','peak_GB')]
display_arms=('replay_k12','comem_k12')
rows=[]
for length,summary in summaries.items():
    best=[min(summary[arm][phase][key] for arm in display_arms) for phase,key in fields]
    if rows: rows.append(r'\midrule')
    for idx,arm in enumerate(display_arms):
        v=summary[arm]
        values=[v[phase][key] for phase,key in fields]
        cells=[f'{x:.1f}' if i<2 else f'{x:.2f}' for i,x in enumerate(values)]
        cells=[r'\textbf{'+s+'}' if values[i]==best[i] else s for i,s in enumerate(cells)]
        if arm=='comem_k12': rows.append(r'\rowcolor{resultshade}')
        name=r'\textbf{CoMem}' if arm=='comem_k12' else 'Raw replay'
        rows.append(' & '.join([length if idx==0 else '',name,'10' if arm=='replay_k10' else '12',*cells])+r'\\')
local=r'''\begin{tabularx}{\linewidth}{llr*{4}{>{\raggedleft\arraybackslash}X}}
\toprule
\multicolumn{7}{l}{\textbf{(a) RTX 5090}\quad Same adapter}\\[2pt]
\multirow{2}{*}{\textbf{Source}} & \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Chunks}} & \multicolumn{2}{c}{\textbf{Latency (ms)} $\downarrow$} & \multicolumn{2}{c}{\textbf{Peak GPU (GB)} $\downarrow$}\\
\cmidrule(lr){4-5}\cmidrule(l){6-7}
 & & & Prefill & TTFT & Prefill & TTFT\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabularx}'''
s=(base/'tab_infra_template.tex').read_text()
start=s.index(r'\begin{tabularx}')
end=s.index(r'\end{tabularx}',start)+len(r'\end{tabularx}')
s=s[:start]+local+s[end:]
s=s.replace('Bold marks the best displayed value within a panel.','Bold marks the best displayed value within each source length and panel.')
s=s.replace('three documents, 20 repeats each','three documents per source length, 20 repeats each')
s=s.replace('; top ten is a fixed budget, not a local equal-TTFT calibration.', '.')
s=s.replace('Read uses GPU-ready inputs;', 'Prefill uses GPU-ready inputs;')
(PAPER/'sections/tab_infra.tex').write_text(s,encoding='utf-8')
procrows=[]
report=['# RTX 5090：32k 与 128k source 补测','','各方法共用 BF16 Qwen3-8B 与未合并的 FP32 final4000 adapter。每个 source 长度、方法及阶段均为 3 个独立进程 × 3 文档 × 20 次正式重复。延迟取进程中位数的中位数；显存取所有正式记录中的最大 allocated peak，含权重，单位为十进制 GB。','','| Source | Method | Chunks | Read ms | TTFT ms | Read GB | TTFT GB |','|---|---|---:|---:|---:|---:|---:|']
for length,summary in summaries.items():
    for arm in display_arms:
        name='CoMem' if arm=='comem_k12' else 'Raw replay'
        k=10 if arm=='replay_k10' else 12
        v=summary[arm]
        values=[v[p][key] for p,key in fields]
        formatted=[f'{x:.1f}' if i<2 else f'{x:.2f}' for i,x in enumerate(values)]
        report.append('| '+' | '.join([length,name,str(k),*formatted])+' |')
    if procrows: procrows.append(r'\midrule')
    for phase in ('read','ttft'):
        for arm in display_arms:
            name='CoMem' if arm=='comem_k12' else 'Replay'
            k=10 if arm=='replay_k10' else 12
            v=summary[arm][phase]
            procrows.append(' & '.join([length,name+f' ({k})','Prefill' if phase=='read' else 'TTFT',*[f'{x:.1f}' for x in v['process_medians_ms']],f"{v['peak_GB']:.2f}"])+r'\\')
appendix=r'''\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lllrrrr@{}}
\toprule
Source & Method ($k$) & Phase & Process 1 & Process 2 & Process 3 & Peak GB\\
\midrule
'''+ '\n'.join(procrows)+r'''
\bottomrule
\end{tabular}
\caption{RTX 5090 source-length supplement. Each process latency is the median over three documents and twenty formal repetitions per document, in milliseconds. Main-table latency is the median of these three process medians; memory is the maximum allocated peak over all 180 formal observations per source length, method, and phase. Both methods use the same top-12 selected pack of 6,657 tokens and the same newly trained adapter; this cost workload supplies no accuracy evidence.}
\label{tab:local-infra-processes}
\end{table}
'''
(PAPER/'sections/tab_local_infra_processes.tex').write_text(appendix,encoding='utf-8')
report += ['', 'Source 分别为 32,768 和 131,072 tokens；top12/top10 的实际 pack 仍为 6,657/5,633 tokens。32k 使用 PG19 文档索引 0、1、2，128k 使用 0、1、32，即各长度下前三个足够长的文档。Query 是 source 后续的 512 tokens，其中前 32 tokens 供检索使用。这不是相同 query 下只改变 source 长度的配对实验。', '', '两个计时都不含 document Write 与 decode；TTFT 包含 CPU selector 构建、检索、fetch、sink/query Write 与 cached prefill。Top10 是固定预算，未做等 TTFT 校准。本地 adapter 尚未确认等于 ARR 权重；这些显存值不能填为 H20 历史结果。', '', 'The original L20A display name is corrected to B200 on the author\'s confirmation; historical numerical observations are unchanged.']
report='\n'.join(report)+'\n'
report=report.replace('top12/top10 的实际 pack 仍为 6,657/5,633 tokens。','两种方法均使用 top12，实际 pack 为 6,657 tokens。').replace('Top10 是固定预算，未做等 TTFT 校准。','')
report=report.replace('| Read ms |','| Prefill ms |').replace('| Read GB |','| Prefill GB |')
(HERE/'RESULTS.md').write_text(report,encoding='utf-8')
(PAPER/'LOCAL_INFRA_RESULTS_zh.md').write_text(report,encoding='utf-8')
f=PAPER/'sections/11_systems.tex'
s=f.read_text()
old='The workload comprises the first three PG19 books in the local 64-book subset with at least 33,280 tokens. Each contributes a 32,768-token source and the following 512-token continuation query.'
new='At each source length, the workload takes the first three eligible PG19 books from the local 64-book subset: book indices 0, 1, and 2 at 32k, and 0, 1, and 32 at 128k. Each contributes a 32,768- or 131,072-token source and the following 512-token continuation query; queries therefore differ across source lengths.'
s=s.replace(old,new).replace('Three independent processes each measure three documents,','At each source length, three independent processes each measure three documents,').replace('180 observations per reported latency.','180 observations per reported latency, for 2,160 measurements across both lengths.').replace('The reported device has 183\\,GB HBM; software is','Software is')
s=s.replace(' Replay at top ten uses 5,633 tokens and is not locally calibrated to equal TTFT.','').replace('for 2,160 measurements across both lengths.','for 1,440 observations in the two displayed methods across both lengths.')
if 'At 128k, the residual tensor occupies 1 GiB' not in s:
    s=s.replace('CoMem writes the complete source before timing and retains its residuals in CPU-pinned memory; replay stores token IDs.', 'CoMem writes the complete source before timing and retains its residuals in CPU-pinned memory; replay stores token IDs. At 128k, the residual tensor occupies 1 GiB per document, computed as $131{,}072\\times4{,}096\\times2$ bytes, separate from GPU peak allocation.')
f.write_text(s,encoding='utf-8')
f=PAPER/'sections/05_experiments.tex'; s=f.read_text(); s=s.replace('Its local RTX 5090 supplement measures Read, TTFT, and both GPU peaks with a shared, newly trained rank-32 adapter.','Its local RTX 5090 supplement measures Read, TTFT, and both GPU peaks at 32k and 128k source lengths with a shared, newly trained rank-32 adapter.')
s=s.replace('The top-12 arms use identical selected evidence; top-10 replay is a fixed-budget reference.','Both methods use the same top-12 selected evidence.')
f.write_text(s,encoding='utf-8')
print(report)

# Preserve a completed B300 recheck when refreshing the local rows.
import runpy
b300=ROOT/'exp/comem_b300_recheck_20260912'
if (b300/'summary.json').exists():
    runpy.run_path(str(b300/'update_paper.py'),run_name='__main__')

# Retain the completed first-request E2E column when regenerating older phases.
e2e=ROOT/'exp/comem_e2e_20260913'
if (e2e/'summary.json').exists():
    runpy.run_path(str(e2e/'update_paper.py'),run_name='__main__')
