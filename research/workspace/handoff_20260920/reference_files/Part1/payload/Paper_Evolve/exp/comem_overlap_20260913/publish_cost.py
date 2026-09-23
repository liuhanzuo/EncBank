"""Insert complete measured overlap costs into the current manuscript."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PAPER=ROOT/'COMem/paper_iclr2027_rewrite_20260912'
s=json.loads((HERE/'cost_summary.json').read_text(encoding='utf-8'))
assert s['complete'] and s['formal_observations']==540 and s['unique_samples']==20
assert all(s['cells'][c][a]['status']=='ok' for c in s['cells'] for a in ('replay','w0','w32'))
lines=[r'\begin{table}[htbp]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
       r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
       r'Method & Quality & Write (ms) & TTFT (ms) & Online (s) & E2E (s) & GPU (GB) \\',r'\midrule']
for cell,label in (('longeval_8k','LongEval 8k'),('longeval_16k','LongEval 16k'),('longeval_32k','LongEval 32k'),('qasper','Qasper')):
    lines.append(r'\multicolumn{7}{@{}l}{\textit{'+label+r' ($n=5$)}} \\')
    for a,name in (('replay','Replay'),('w0','CoMem'),('w32',r'CoMem ($w=32$)')):
        r=s['cells'][cell][a]
        values=[f"{r['local_quality_percent']:.2f}",f"{r['document_write_ms']['median']:.1f}",f"{r['ttft_ms']['median']:.1f}",
                f"{r['online_ms']['median']/1000:.3f}",f"{r['e2e_ms']['median']/1000:.3f}",f"{r['peak_allocated_GB']:.2f}"]
        lines.append(name+' & '+' & '.join(values)+r' \\')
    if cell!='qasper':lines.append(r'\addlinespace[3pt]')
lines +=[r'\bottomrule',r'\end{tabular}',
r'\caption{Paired quality and cost on RTX 5090 for the first five prespecified examples per task/length. Both CoMem variants use $j=12$, with $w=0$ unless indicated; replay uses the same selected tokens at $j=0$. Quality is local accuracy or Qasper F1 (percent), using each task\textquotesingle s generation limit. Online and E2E force 128 output tokens; E2E additionally includes full-document preparation and Write. Latencies are medians of three process medians, each over five examples and three formal repetitions. GPU is maximum allocated memory including weights. These 20 examples are a timing subset, not the full quality evaluation in Table~\ref{tab:overlap-confirmation}.}',
r'\label{tab:overlap-cost}',r'\end{table}']
(PAPER/'sections/tab_overlap_cost.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
para=r'''\paragraph{Quality and Write-inclusive cost on the same examples.}
\label{app:overlap-cost}
\input{sections/tab_overlap_cost}
Table~\ref{tab:overlap-cost} uses the first five examples from each LongEval length and from Qasper, fixed before observing quality, on RTX 5090. All arms share the principal adapter and selected evidence. Each of three independent processes runs one warmup and three formal repetitions per example and arm: 540 formal requests and 180 warmups, with 20 unique quality examples. Software and model placement follow Appendix~\ref{app:local-infra}; the allocator cap for this experiment is 26.08 GB, below the 28 GB budget. No OOM occurs. Every successful request is checked for 128 generated IDs and consistent timing components.

Each request starts from pretokenized CPU inputs and prepares a new CPU-pinned store. CoMem writes every source chunk, then selects, fetches, writes the sink/query, prefills the suffix, and generates 128 tokens without stopping at EOS. Replay prepares token IDs and processes the same selected pack through all layers. TTFT includes selection, fetch, sink/query Write, and prefill; online time adds generation, and E2E additionally includes document preparation. Replay token preparation is charged to E2E; it has no transformer document Write. Loading, tokenization, external I/O, and teardown are excluded. Local quality uses the first 16 generated tokens for LongEval and 128 for Qasper, truncated at the first EOS. Predictions and scores agree across local repetitions; device-dependent predictions cause small Qasper score differences from the B300 subset, so quality here is scored locally.

Relative to $w=0$, overlap raises document Write by 17.9--19.4\% and E2E by 1.1--4.2\% across the four cells. TTFT changes by less than 0.4\% in either direction. Stored residual bytes are identical: 64/128/256 MiB at the three LongEval lengths and 20--32 MiB for these Qasper sources, excluding the retrieval index. GPU peaks are at most 18.92 GB across all arms. CoMem reduces TTFT relative to replay, but charging full Write makes its 32k E2E longer (7.475 versus 6.900 seconds). These five-example cells expose request costs and local quality together; their quality rankings do not replace the complete 500-example evaluation or establish a general serving frontier.

'''
p=PAPER/'sections/09_diagnostics.tex';t=p.read_text(encoding='utf-8')
assert r'\label{app:overlap-cost}' not in t
t=t.replace(r'\subsection{Split depth and suffix adaptation}',para+r'\subsection{Split depth and suffix adaptation}')
p.write_text(t,encoding='utf-8')
p=PAPER/'sections/05_experiments.tex';t=p.read_text(encoding='utf-8')
old=r'This repair is task-dependent (Appendix~\ref{app:overlap-confirmation}).'
new=r'On a prespecified 20-example RTX 5090 subset, overlap adds 17.9--19.4\% to document Write and 1.1--4.2\% to E2E, with unchanged stored bytes (Appendix~\ref{app:overlap-cost}). The repair is task-dependent.'
assert old in t;t=t.replace(old,new);p.write_text(t,encoding='utf-8')
p=PAPER/'README.md';t=p.read_text(encoding='utf-8').replace('and the fixed-w32 LongEval/Qasper evaluation.','and the fixed-w32 LongEval/Qasper evaluation, including paired RTX5090 quality and Write-inclusive costs on a prespecified 20-example subset.')
p.write_text(t,encoding='utf-8')
shutil.copy2(HERE/'cost_summary.json',PAPER/'overlap_cost_summary.json')
p=HERE/'README.md';t=p.read_text(encoding='utf-8').replace('Local cost data are being collected serially','Local cost data are complete and were collected serially')
t+='\nThe complete cost results (540 formal requests plus 180 warmups) have been inserted into the manuscript as `tab:overlap-cost`, with the 20-example quality subset and all measurement boundaries stated.\n'
p.write_text(t,encoding='utf-8')
print('Inserted all 12 measured cost rows and matched local quality.')
