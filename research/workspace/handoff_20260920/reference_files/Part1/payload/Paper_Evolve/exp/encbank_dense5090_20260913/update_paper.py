"""Replace the misplaced same-pack replay baseline with measured full-source Dense."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
PAPER=ROOT/'Encbank/paper_iclr2027_rewrite_20260912'
data=json.loads((HERE/'summary.json').read_text())
assert data['complete'] and data['expected_cells']==54 and data['budget_GB']==28
short=json.loads((HERE/'short_summary.json').read_text())
assert short['complete'] and short['expected_cells']==108 and short['budget_GB']==28
e2e=json.loads((ROOT/'exp/encbank_e2e_20260913/summary.json').read_text())
rows=[]; details=[]; report=[]
fields=[('prefill','latency_ms'),('ttft','latency_ms'),('e2e','latency_ms'),('prefill','peak_GB'),('ttft','peak_GB')]
for length,folder,label in [(8192,None,'8k'),(16384,None,'16k'),(32768,'encbank_infra_20260912','32k'),(131072,'encbank_infra_128k_20260912','128k')]:
    if folder:
        old=json.loads((ROOT/'exp'/folder/'summary.json').read_text())['summary']['encbank_k12']
        end=e2e['summary'][str(length)]['encbank_k12']; dense=data['summary'][str(length)]
        cm={'prefill':{'status':'ok',**old['read']},'ttft':{'status':'ok',**old['ttft']},'e2e':{'status':'ok',**end}}
    else:
        dense=short['summary'][str(length)]['dense']; cm=short['summary'][str(length)]['encbank']
    d=[dense[p][k] if dense[p]['status']=='ok' else None for p,k in fields]
    c=[cm[p][k] if cm[p]['status']=='ok' else None for p,k in fields]
    assert all(v is None or v<28 for v in c[3:])
    def cells(values,other):
        answer=[]
        for i,value in enumerate(values):
            if value is None: answer.append('OOM'); continue
            s=f'{value:.1f}' if i<3 else f'{value:.2f}'
            if other[i] is not None and value<other[i]: s=r'\textbf{'+s+'}'
            answer.append(s)
        return answer
    if rows: rows.append(r'\midrule')
    rows.append(' & '.join([label,'Dense',*cells(d,c)])+r'\\')
    rows.append(r'\rowcolor{resultshade}')
    rows.append(' & '.join(['',r'\textbf{Encbank}',*cells(c,d)])+r'\\')
    report.append('| '+' | '.join([label,'Dense',*[('OOM' if v is None else f'{v:.1f}') for v in d[:3]],*[('OOM' if v is None else f'{v:.2f}') for v in d[3:]]])+' |')
    report.append('| '+' | '.join([label,'Encbank',*[('OOM' if v is None else f'{v:.1f}') for v in c[:3]],*[('OOM' if v is None else f'{v:.2f}') for v in c[3:]]])+' |')
    for method,measurements in ([('Dense',dense)] if folder else [('Dense',dense),('Encbank',cm)]):
        if details: details.append(r'\midrule')
        for phase in ('prefill','ttft','e2e'):
            v=measurements[phase]
            if v['status']=='ok':
                vals=[*[f'{m:.1f}' for m in v['process_medians_ms']],f"{v['peak_GB']:.2f}"]
            else: vals=['OOM']*4
            details.append(' & '.join([label if phase=='prefill' else '',method if phase=='prefill' else '',{'prefill':'Prefill','ttft':'TTFT','e2e':r'E2E$_{128}$'}[phase],*vals,f"{v['oom_cells']}/9"])+r'\\')
panel=r'''\begin{tabularx}{\linewidth}{ll*{5}{>{\raggedleft\arraybackslash}X}}
\toprule
\multicolumn{7}{l}{\textbf{(a) RTX 5090}\quad Full-source Dense; 28 GB memory cap}\\[2pt]
\multirow{2}{*}{\textbf{Source}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c}{\textbf{Latency (ms)} $\downarrow$} & \multicolumn{2}{c}{\textbf{Peak GPU (GB)} $\downarrow$}\\
\cmidrule(lr){3-5}\cmidrule(l){6-7}
 & & Prefill & TTFT & E2E$_{128}$ & Prefill & TTFT\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabularx}'''
f=PAPER/'sections/tab_infra.tex'; text=f.read_text(encoding='utf-8')
a=text.index(r'\begin{tabularx}'); b=text.index(r'\end{tabularx}',a)+len(r'\end{tabularx}')
text=text[:a]+panel+text[b:]
a=text.index(r'\textbf{(a)}',text.index(r'\footnotesize')); b=text.index(r'\textbf{(b)} Same adapter',a)
note=r'''\textbf{(a)} Dense processes the full source; Encbank resumes from cached $j=12$ residual states of 12 selected chunks. Both share the adapter. The 28 GB CUDA allocator cap includes weights, KV, and temporaries; OOM denotes an actual failure. Prefill starts GPU-ready; TTFT adds transfer and Encbank retrieval/query Write. E2E$_{128}$ includes document preparation and 128 output tokens. Model load, tokenization, and external I/O are excluded. Three processes and three documents; protocol and OOM attempts are in Appendix~\ref{app:local-infra}. GPU peaks are allocated bytes including weights (decimal GB); the adapter is shared with Table~\ref{tab:accuracy}; these cost inputs are unscored.\\[1pt]
'''
text=text[:a]+note+text[b:]
text=text.replace('Same adapter; full prompt versus selected pack.',
                  r'Same adapter; full prompt versus cached residual states at $j=12$.')
text=text.replace('Bold marks the best displayed value within each source length and panel.',
                  'Bold marks better results where both methods complete within each source length and panel.')
f.write_text(text,encoding='utf-8')
table=r'''\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{lll*{5}{>{\raggedleft\arraybackslash}X}}
\toprule
Source & Method & Phase & P1 & P2 & P3 & Peak GB & OOM\\
\midrule
'''+ '\n'.join(details)+r'''
\bottomrule
\end{tabularx}
\caption{Fresh RTX 5090 measurements with the shared adapter and a 28 GB total CUDA allocator cap. Dense receives the full source plus 513 query/BOS tokens; Encbank resumes from cached $j=12$ residual states of 12 selected chunks. P1--P3 are independent process medians in milliseconds, each from three documents with one warmup and three formal repetitions per document. OOM counts failed document/process cells out of nine; repetitions stop after an OOM. Failed entries have no completed latency or peak-memory value. Peak GB is the maximum allocated memory including weights. Retained 32k/128k Encbank process results appear in the subsequent diagnostic tables.}
\label{tab:dense5090}
\end{table}
'''
(PAPER/'sections/tab_dense5090.tex').write_text(table,encoding='utf-8')
section=r'''\subsection{Local full-source comparison and depth diagnostics}
\label{app:local-infra}
\input{figures/source_scaling}
The RTX 5090 measurements use Windows, PyTorch 2.7.1+cu128, Transformers 5.16.1, two intra-op and sixteen inter-op CPU threads, and SDPA with explicit KV-head repetition. Qwen3-8B weights remain bf16 and fully GPU resident. All local methods share the unmerged fp32 rank-32 suffix adapter spanning layers 12--35. This separate 4,000-step run uses a 64-book PG19 subset, $\alpha=32$, 4,096-token windows, and batch size one; its costs are not paired with the historical accuracy rows.

At each length, three saved PG19 documents supply the complete source and its next 512 tokens as the query: book indices 0/1/2 at 8k, 16k, and 32k, and 0/1/32 at 128k. Queries therefore differ across source lengths. These are fixed, unscored systems workloads. Encbank's iterative token-ID BM25 uses the first 32 query tokens, hop width two, and top-12 source-order chunks of 512 tokens. Its suffix receives 6,657 residual-state positions at $j=12$: 6,144 cached source positions, 512 query positions, and one BOS sink. Dense instead receives all 8,705, 16,897, 33,281, or 131,585 token IDs without retrieval.

\paragraph{Full-source Dense under the memory limit.}
\label{app:local-dense28}
\input{sections/tab_dense5090}
Dense uses the standard model forward with a complete KV cache and only the final vocabulary logits. There is no truncation, quantization, CPU weight/KV offload, or RoPE extension. Fused SDPA is required and the quadratic-memory math backend is disabled. A 28,000,000,000-byte process CUDA allocator cap includes weights, inputs, KV, and temporary tensors; both allocated and reserved memory remain within it. Desktop and driver-context overhead are separate from this allocator budget. Each of three fresh processes attempts every source length, document, and phase. A CUDA OOM is recorded with its error and full input shape; remaining repetitions of that failed cell are skipped. Other errors fail the worker rather than becoming OOM entries. Any failed prescribed document/process cell makes the corresponding overview entry OOM, without a latency or a selectively reported successful-subset mean.

Dense Prefill starts with the entire ID sequence on GPU and ends at first logits. TTFT starts with pretokenized CPU source/query, assembles and pins the full IDs, transfers them, and runs the same cached forward. E2E continues to 128 generated token IDs, including the first token and 127 cached decode steps; EOS does not shorten this fixed budget. Model loading, tokenization/detokenization, queues, network transport, and external I/O are excluded. Each worker checks cached greedy decoding against full recomputation before formal measurement; the short-source workers also check Encbank's cached decode against pack recomputation. Beyond-native full-source timings are not an accuracy claim.

\paragraph{Encbank preparation and request boundaries.}
At 8k and 16k, both methods are freshly measured in the same three processes under the 28 GB cap, with shuffled method and phase order. At 32k and 128k, Encbank's earlier observations are retained; their original 26 GB allocator cap is below the new limit. The same model, adapter, and saved inputs are used within each source length. Encbank Prefill starts with GPU-ready residuals, while TTFT additionally includes BM25 construction/selection, pinned fetch, and sink/query Write. These two phases exclude document Write and decode. E2E begins with the pretokenized source/query and includes preparing a new pinned store, independently writing every source chunk, retrieval, fetch, sink/query Write, prefill, and a fixed 128-token output. It stops before store/cache teardown. Document Write is charged in full, without amortization. At 128k the residual store is 1 GiB of CPU memory, separate from GPU allocation.

Every successful fresh cell uses one warmup and three formal repetitions: 27 formal observations per source/method/phase across three processes and three documents. Retained 32k/128k Encbank Prefill and TTFT instead use five warmups and twenty formal repetitions per document (180 observations per cell); retained E2E uses one warmup and three repetitions (27 observations). Reported latencies are medians of process medians; GPU peaks are maximum allocated bytes including weights. Phases are timed independently, so small Prefill/TTFT median reversals can occur with run-to-run variation; their difference is not an isolated transfer-cost estimate. E2E is timed directly, not added from separate medians. Earlier observations are retained without increasing their repeat counts. All local inputs and generated IDs, phase boundaries, checks and raw attempts are saved.

\paragraph{Same-evidence depth diagnostics.}
\input{sections/tab_local_infra_processes}
\input{sections/tab_local_e2e}
The earlier raw-replay measurements instead send the same top-12 selected tokens through all layers ($j=0$), using Encbank's adapter, evidence, order, and sink. They are depth diagnostics, separate from the full-source Dense baseline in Table~\ref{tab:infra}. Their GPU inputs remain 6,657 tokens at both source lengths. Three-process Prefill/TTFT and E2E observations use the respective repetition counts above. The E2E diagnostic includes source preparation and fixed-length generation, with a fresh full-document residual store for Encbank; it does not describe Dense on the entire source.

The historical H20 matched-depth and TTFT-calibration sources do not report corresponding GPU peaks. These local results supply independent device-specific observations. Persistent-store capacity, online request peak allocation, and document-Write peak allocation remain separate quantities.

'''
f=PAPER/'sections/11_systems.tex'; text=f.read_text(encoding='utf-8')
a=text.index(r'\subsection{Local '); b=text.index(r'\subsection{B300 full-context recheck}',a)
text=text[:a]+section+text[b:]; f.write_text(text,encoding='utf-8')
f=PAPER/'sections/tab_local_infra_processes.tex'; text=f.read_text(encoding='utf-8')
text=text.replace('RTX 5090 source-length supplement.','RTX 5090 same-evidence depth diagnostic, separate from full-source Dense.')
text=text.replace('Main-table latency is the median of these three process medians;','The retained Encbank main-table latency is the median of these process medians;')
f.write_text(text,encoding='utf-8')
f=PAPER/'sections/tab_local_e2e.tex'; text=f.read_text(encoding='utf-8')
text=text.replace('RTX 5090 end-to-end supplement,','RTX 5090 same-evidence E2E diagnostic, distinct from full-source Dense,')
f.write_text(text,encoding='utf-8')
f=PAPER/'sections/05_experiments.tex'; text=f.read_text(encoding='utf-8')
a=text.index('Table~\\ref{tab:infra} reports device-specific systems measurements.')
b=text.index('\n\nIn the historical H20',a)
paragraph=r'''Table~\ref{tab:infra} reports device-specific systems measurements. On RTX 5090, full-source Dense is compared with Encbank from 8k to 128k under a 28 GB memory limit and a shared adapter. Dense receives the entire source and query; Encbank resumes at $j=12$ using cached residual states of 12 selected chunks and the query states. E2E includes full document preparation and 128 output tokens. This measures their complete configurations, while same-evidence replay isolates depth reuse in Appendix~\ref{app:local-infra}. The adapter is shared with Table~\ref{tab:accuracy}, but these cost inputs do not supply accuracy evidence.'''
if all(short['summary'][str(n)][m]['e2e']['status']=='ok' for n in (8192,16384) for m in ('dense','encbank')):
    ratios=[short['summary'][str(n)]['dense']['e2e']['latency_ms']/short['summary'][str(n)]['encbank']['e2e']['latency_ms'] for n in (8192,16384)]
    paragraph+=rf" The Dense-to-Encbank E2E ratios are ${ratios[0]:.2f}\times$ at 8k and ${ratios[1]:.2f}\times$ at 16k, with document Write charged to each Encbank request."
if all(data['summary'][str(n)][p]['status']=='OOM' for n in (32768,131072) for p in ('prefill','ttft','e2e')):
    paragraph+=' Dense exceeds the memory cap at 32k and 128k; no latency or speedup is inferred from these OOMs.'
text=text[:a]+paragraph+text[b:]
text=text.replace(r'Its speedup reaches $2.74\times$ at 128k, but Encbank is slower at 8k and 16k;',
                  r'On B200, the pipeline speedup reaches $2.74\times$ at 128k, but Encbank is slower at 8k and 16k;')
f.write_text(text,encoding='utf-8')
report='\n'.join(['# 5090 主表基线更正：完整 source Dense 对比 Encbank','',
  '按作者要求，Table 2(a) 的原 top12 raw replay 改为完整 source Dense。旧 replay 仅保留为同证据深度诊断，不改名冒充 Dense。', '',
  '| Source | Method | Prefill ms | TTFT ms | E2E128 ms | Prefill GB | TTFT GB |',
  '|---|---|---:|---:|---:|---:|---:|',*report,'',
  '按作者要求删除 Read tokens 列。Encbank从选中12个片段的j=12中间层残差缓存继续计算；source长度、原始token输入与缓存状态的序列位置分别说明。所有实测延迟、显存和OOM结果保持原值。', '',
  'Dense 通过标准模型 forward 一次处理 BOS + 全部 source + 512-token query。32k/128k的Dense包含54个文档/进程/阶段组合；新增8k/16k同时测Dense与Encbank，包含108个组合。每组使用三个独立进程和每进程三文档；成功组各1次预热、3次正式重复。28GB为整个CUDA allocator预算，包含权重、KV和临时张量；不是额外28GB激活额度。桌面/驱动context开销另计。Dense没有截断、量化、offload或换用检索包。', '',
  'OOM条目来自实际CUDA内存分配失败，错误和完整输入shape均保存；失败没有完成延迟或完成峰值，不能填0或据此计算speedup。32k/128k Encbank保留原26GB上限下已经完成的观测（低于19GB allocated）；没有增加其重复次数。8k/16k两种方法均在新的28GB上限下测量。', '',
  f"完整 source Dense长序列记录 {data['recorded_attempts']} 次实际尝试，其中 {data['oom_attempts']} 次OOM，成功正式重复 {data['formal_successes']} 次。8k/16k新增记录 {short['recorded_attempts']} 次尝试，其中 {short['oom_attempts']} 次OOM，成功正式重复 {short['formal_successes']} 次。预热不计入正式重复。", '',
  'E2E包含文档准备和固定128-token输出，不含模型加载、分词、反分词、网络队列与外部I/O。输入和adapter沿用先前本机实验。B300/B200与准确率表数值不变。'])+'\n'
(HERE/'RESULTS_zh.md').write_text(report,encoding='utf-8')
(PAPER/'DENSE5090_RESULTS_zh.md').write_text(report,encoding='utf-8')
f=PAPER/'README.md'; text=f.read_text(encoding='utf-8')
text=text.replace('A subsequent user-authorized local supplement now measures selected-pack prefill, TTFT, and peak GPU allocation on RTX 5090, using the same newly trained suffix adapter in the Encbank and replay arms. These costs do not substitute for historical H20 measurements or add accuracy evidence. See `LOCAL_INFRA_RESULTS_zh.md` and Appendix D.1.',
    'Subsequent user-authorized local measurements compare full-source Dense with Encbank from 8k to 128k on RTX 5090, using a shared principal suffix adapter and a 28 GB total CUDA allocator budget including weights. The main panel reports Prefill, TTFT, direct E2E through 128 output tokens, and phase-specific GPU peaks; actual failures are OOM. Earlier selected-pack replay remains a separate depth diagnostic. These costs do not substitute for historical H20 measurements or add accuracy evidence. See `DENSE5090_RESULTS_zh.md` and Appendix D.1.')
f.write_text(text,encoding='utf-8')
for name in ('README.md','LOCAL_INFRA_RESULTS_zh.md','E2E_RESULTS_zh.md','POSITIONING_zh.md','CLAIM_EVIDENCE.md',
             'TABLE_STYLE_REFERENCES_zh.md','TABLE_RESTORATION_zh.md','REVISION_NOTES_zh.md','SUPPLEMENTARY_EXPERIMENTS_zh.md'):
    f=PAPER/name; text=f.read_text(encoding='utf-8')
    note=('> Current main-panel correction: Table 2(a) compares full-source Dense with Encbank under a 28 GB total CUDA allocator cap. Earlier top-12 replay values below are retained depth diagnostics, not the Dense baseline. See `DENSE5090_RESULTS_zh.md`.\n\n'
          if name in ('README.md','CLAIM_EVIDENCE.md') else
          '> 当前主表更正：Table 2(a) 改为完整 source Dense 对比 Encbank，使用28GB总CUDA allocator预算。下文早先的top12 replay数值仅为深度诊断，不是Dense基线。见 DENSE5090_RESULTS_zh.md。\n\n')
    if 'Current main-panel correction:' not in text and '当前主表更正：' not in text: text=note+text
    f.write_text(text,encoding='utf-8')
print(report)
