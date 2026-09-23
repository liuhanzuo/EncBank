"""Add only complete, measured E2E results to panel (a)."""
from pathlib import Path
import json, re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAPER=ROOT/'COMem/paper_iclr2027_rewrite_20260912'
e2e=json.loads((HERE/'summary.json').read_text())
assert e2e['complete'] and e2e['formal_requests']==108 and e2e['output_tokens']==128
assert e2e['document_write_included']
arms=('replay_k12','comem_k12')
rows=[]; detail=[]; report=[]
for length,old_name,label in [(32768,'comem_infra_20260912','32k'),(131072,'comem_infra_128k_20260912','128k')]:
    old=json.loads((ROOT/'exp'/old_name/'summary.json').read_text())['summary']
    new=e2e['summary'][str(length)]
    values={arm:[old[arm]['read']['latency_ms'],old[arm]['ttft']['latency_ms'],new[arm]['latency_ms'],
                 old[arm]['read']['peak_GB'],old[arm]['ttft']['peak_GB']] for arm in arms}
    best=[min(values[arm][i] for arm in arms) for i in range(5)]
    if rows: rows.append(r'\midrule')
    for idx,arm in enumerate(arms):
        name=r'\textbf{CoMem}' if arm=='comem_k12' else 'Raw replay'
        cells=[f'{v:.1f}' if i<3 else f'{v:.2f}' for i,v in enumerate(values[arm])]
        cells=[r'\textbf{'+s+'}' if values[arm][i]==best[i] else s for i,s in enumerate(cells)]
        if arm=='comem_k12': rows.append(r'\rowcolor{resultshade}')
        rows.append(' & '.join([label if idx==0 else '',name,'12',*cells])+r'\\')
        v=new[arm]
        detail.append(' & '.join([label,'CoMem' if arm=='comem_k12' else 'Raw replay',
          *[f'{m:.1f}' for m in v['process_medians_ms']],f"{v['latency_ms']:.1f}",f"{v['peak_GB']:.2f}"])+r'\\')
        report.append(f"| {label} | {'CoMem' if arm=='comem_k12' else 'Raw replay'} | {v['latency_ms']:.1f} | {v['peak_GB']:.2f} |")
panel=r'''\begin{tabularx}{\linewidth}{llr*{5}{>{\raggedleft\arraybackslash}X}}
\toprule
\multicolumn{8}{l}{\textbf{(a) RTX 5090}\quad Same adapter; E2E includes document Write}\\[2pt]
\multirow{2}{*}{\textbf{Source}} & \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Chunks}} & \multicolumn{3}{c}{\textbf{Latency (ms)} $\downarrow$} & \multicolumn{2}{c}{\textbf{Peak GPU (GB)} $\downarrow$}\\
\cmidrule(lr){4-6}\cmidrule(l){7-8}
 & & & Prefill & TTFT & E2E$_{128}$ & Prefill & TTFT\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabularx}'''
f=PAPER/'sections/tab_infra.tex'; text=f.read_text(encoding='utf-8')
a=text.index(r'\begin{tabularx}'); b=text.index(r'\end{tabularx}',a)+len(r'\end{tabularx}')
text=text[:a]+panel+text[b:]
a=text.index(r'\textbf{(a)} Prefill'); b=text.index(r'\textbf{(b)} Same adapter',a)
note=r'''\textbf{(a)} Prefill uses GPU-ready inputs; TTFT adds BM25, pinned fetch, and sink/query Write. E2E$_{128}$ also includes full document preparation and generation of 128 tokens; model loading, tokenization, and external I/O are excluded. Medians over three process medians: three documents, 20 repeats each for Prefill/TTFT and three for E2E. GPU columns are phase-specific allocated peaks including weights (decimal GB). The shared adapter is newly trained, distinct from Table~\ref{tab:accuracy}.\\[1pt]
'''
text=text[:a]+note+text[b:]; f.write_text(text,encoding='utf-8')
appendix=r'''\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Source & Method & Process 1 & Process 2 & Process 3 & E2E$_{128}$ & Peak GB\\
\midrule
'''+ '\n'.join(detail)+r'''
\bottomrule
\end{tabular}
\caption{RTX 5090 end-to-end supplement, including document preparation and a fixed 128-token output. Times are milliseconds. Each process median covers three documents and three formal requests per document after one warmup; E2E is the median across three independent processes (27 requests per row). Full-request GPU peaks are separate from the Prefill/TTFT peaks in Table~\ref{tab:infra}.}
\label{tab:local-e2e}
\end{table}
'''
(PAPER/'sections/tab_local_e2e.tex').write_text(appendix,encoding='utf-8')
section=r'''\paragraph{End-to-end requests at 32k and 128k.}
\label{app:local-e2e}
\input{sections/tab_local_e2e}
The E2E$_{128}$ column directly times the complete model pipeline from the pretokenized source and query to 128 generated token IDs. Each request prepares a fresh pinned CPU token store; CoMem also independently writes every source chunk into a fresh pinned residual store before retrieval. The timer includes BM25 construction and selection, fetch, sink/query Write, cached prefill, greedy token selection, and 127 subsequent cached decode steps. EOS does not shorten the fixed output budget. It ends after the final token, before store/cache teardown; model loading, tokenization, detokenization, external I/O, service queues, and network transport are outside this model-pipeline boundary. Document Write is therefore included without amortization, unlike store-ready TTFT.

The inputs, selected top-12 chunks, backbone, adapter, and thread/kernel settings are identical to the earlier local measurements. Three fresh processes use one warmup and three formal requests for each document, method, and source length, totaling 108 timed requests. The reported latency is measured directly rather than added from separately summarized stages. The new processes check both cached decoders against full recomputation, as well as the stock-versus-$j=0$ logits. These fixed-length outputs are timing workloads and do not establish answer quality.

'''
f=PAPER/'sections/11_systems.tex'; text=f.read_text(encoding='utf-8')
b=text.index(r'\subsection{B300 full-context recheck}')
if r'\paragraph{End-to-end requests' in text:
    a=text.index(r'\paragraph{End-to-end requests'); text=text[:a]+section+text[b:]
else: text=text[:b]+section+text[b:]
text=text.replace(r'\subsection{Local Read, TTFT, and peak-memory supplement}',r'\subsection{Local prefill, TTFT, E2E, and peak-memory supplement}')
f.write_text(text,encoding='utf-8')
f=PAPER/'sections/05_experiments.tex'; text=f.read_text(encoding='utf-8')
text=text.replace('Its local RTX 5090 supplement measures selected-pack prefill, TTFT, and both GPU peaks at 32k and 128k source lengths',
    'Its local RTX 5090 supplement measures prefill, TTFT, E2E including document Write and 128 output tokens, and phase-specific GPU peaks at 32k and 128k source lengths')
if all(e2e['summary'][str(n)]['comem_k12']['latency_ms'] > e2e['summary'][str(n)]['replay_k12']['latency_ms'] for n in (32768,131072)):
    sentence="With document Write charged to each request, CoMem's measured E2E is higher than replay at both lengths."
    anchor='Appendix~\\ref{app:local-infra} gives the protocol and checkpoint scope.'
    if sentence not in text: text=text.replace(anchor,anchor+' '+sentence)
f.write_text(text,encoding='utf-8')
report='\n'.join(['# RTX 5090：完整流程 E2E（含文档 Write，输出 128 tokens）','',
    '32k、128k 两种 source 均复用原表同一批 token 输入、top12 evidence 与同一 adapter。每格为三进程 × 三文档 × 三次正式请求，合计 108 条；数值为进程中位数的中位数。', '',
    '| Source | Method | E2E ms | E2E peak GB |','|---|---|---:|---:|',*report,'',
    'E2E 直接测量从预分词 source/query 到生成完 128 个 token：含新建 CPU pinned store、完整 CoMem document Write、BM25 构建/检索、fetch、sink/query Write、prefill 与 127 次 decode。每次请求重新准备文档，不摊销 Write；两方法强制相同输出长度，EOS 不提前结束。', '',
    '不包含模型加载、输入分词、输出反分词、网络、队列与外部 I/O。主表原 GPU 列仍对应 Prefill/TTFT 的 peak，附录另报完整流程 peak。准确率、原有 Prefill/TTFT 和 B300/B200 数值不变。'])+'\n'
(HERE/'RESULTS_zh.md').write_text(report,encoding='utf-8')
(PAPER/'E2E_RESULTS_zh.md').write_text(report,encoding='utf-8')
f=PAPER/'README.md'; text=f.read_text(encoding='utf-8')
if '`E2E_RESULTS_zh.md`' not in text:
    text+='\nThe RTX 5090 panel also reports directly measured E2E, including document Write and a fixed 128-token output, at both source lengths. See `E2E_RESULTS_zh.md` and Appendix D.1.\n'
f.write_text(text,encoding='utf-8')
f=PAPER/'CLAIM_EVIDENCE.md'; text=f.read_text(encoding='utf-8')
if 'The RTX 5090 E2E supplement' not in text:
    text+='\nThe RTX 5090 E2E supplement directly times source preparation, including full document Write for CoMem, through 128 generated tokens on the same local workloads and adapter. It covers 108 requests in three independent processes. This is first-use model-pipeline cost with pretokenized inputs; the separately reported store-ready TTFT and historical amortization results have different boundaries. See `E2E_RESULTS_zh.md`.\n'
f.write_text(text,encoding='utf-8')
f=PAPER/'SUPPLEMENTARY_EXPERIMENTS_zh.md'; text=f.read_text(encoding='utf-8')
if '新增 E2E 补测已完成' not in text:
    text='> 2026-09-13：新增 E2E 补测已完成，覆盖 5090 的 32k/128k source、完整文档 Write 和固定 128-token 输出，共 108 次请求。见 E2E_RESULTS_zh.md。下文 repair 后的质量与完整服务曲线仍是后续实验建议。\n\n'+text
f.write_text(text,encoding='utf-8')
for name in ('TABLE_STYLE_REFERENCES_zh.md','REVISION_NOTES_zh.md'):
    f=PAPER/name; text=f.read_text(encoding='utf-8')
    text=text.replace('上方完整宽度显示 RTX 5090 的 Read/TTFT 与两个阶段的峰值显存，下方两个并排面板保留 H20 store-ready 和 L20A pipeline。',
        '上方完整宽度显示 RTX 5090 的 Prefill/TTFT、含文档 Write 和 128-token 输出的 E2E，以及两个阶段的峰值显存；下方并排显示 B300 store-ready 与 B200 pipeline。')
    text=text.replace('汇总本机显存、TTFT、Read 延迟，以及历史 H20/L20A 对照。',
        '汇总本机 Prefill/TTFT、含文档 Write 的 E2E 与阶段显存，以及 B300/B200 对照；历史 H20 结果保留在附录。')
    f.write_text(text,encoding='utf-8')
print(report)

# The author replaced same-pack replay in panel (a) with full-source Dense.
# Keep that newer comparison when earlier E2E data are regenerated.
import runpy
dense=ROOT/'exp/comem_dense5090_20260913'
if (dense/'summary.json').exists() and (dense/'short_summary.json').exists():
    runpy.run_path(str(dense/'update_paper.py'),run_name='__main__')
