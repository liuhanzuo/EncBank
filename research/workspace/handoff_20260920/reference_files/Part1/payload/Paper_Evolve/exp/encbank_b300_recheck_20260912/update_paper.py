"""Place verified B300 results into the systems table and appendix."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAPER=ROOT/'Encbank/paper_iclr2027_rewrite_20260912'
result=json.loads((HERE/'summary.json').read_text())
s=result['summary']; ratios=result['speedups']
full,cm=s['dense_shared'],s['encbank_read']
f=PAPER/'sections/tab_infra.tex'; table=f.read_text(encoding='utf-8')
start=table.index(r'\multicolumn{4}{l}{\textbf{(b)')
end=table.index(r'\bottomrule',start)
panel=r'''\multicolumn{4}{l}{\textbf{(b) B300}\quad 128k source}\\
\multicolumn{4}{l}{\textit{Store-ready prefill}}\\[2pt]
\textbf{Method} & \textbf{Latency} & \textbf{Peak GPU} & \textbf{Speedup}\\
 & (ms) $\downarrow$ & (GB) $\downarrow$ & $\uparrow$\\
\midrule
'''
panel+=f"Dense & {full['latency_ms']:,.0f} & {full['peak_GB']:.2f} & $1.00\\times$"+r'\\'+'\n'
panel+=r'\rowcolor{resultshade}'+'\n'
panel+=r'\textbf{Encbank} & \textbf{'+f"{cm['latency_ms']:,.1f}"+r'} & \textbf{'+f"{cm['peak_GB']:.2f}"+r'} & $\mathbf{'+f"{ratios['same_adapter_full_vs_encbank_read']:.1f}"+r'}\times$\\'+'\n'
table=table[:start]+panel+table[end:]
table=table.replace(r'\textbf{(b)} Stock dense versus adapted Encbank; prewritten store, fetch excluded.',r'\textbf{(b)} Same adapter; full prompt versus selected pack. Document/query Write and fetch excluded; three processes, three documents, three repeats each.')
f.write_text(table,encoding='utf-8')
readrows=[]
for arm,name,adapter,pack in [('dense_stock','Dense','Off','131,585'),('dense_shared','Dense','Shared','131,585'),('replay_read','Raw replay','Shared','6,657'),('encbank_read','Encbank','Shared','6,657')]:
    v=s[arm]; readrows.append(' & '.join([name,adapter,pack,f"{v['latency_ms']:,.1f}",f"{v['peak_GB']:.2f}"])+r'\\')
ttftrows=[]
for arm,name in [('replay_ttft','Raw replay'),('encbank_ttft','Encbank')]:
    v=s[arm]; ttftrows.append(' & '.join([name,'Shared','6,657',f"{v['latency_ms']:,.1f}",f"{v['peak_GB']:.2f}"])+r'\\')
appendix=r'''\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}lllrr@{}}
\toprule
Method & Adapter & Model tokens & Latency (ms) & Peak GPU (GB)\\
\midrule
\multicolumn{5}{l}{\textit{GPU-ready prefill to first logits}}\\
'''+ '\n'.join(readrows)+r'''
\midrule
\multicolumn{5}{l}{\textit{Selected-pack TTFT with online selection and fetch}}\\
'''+ '\n'.join(ttftrows)+r'''
\bottomrule
\end{tabular}
\caption{B300 recheck on three fixed 128k-source PG19 workloads. Each latency is the median of three independent process medians, each over three documents and three formal repeats after one warmup. Peak GPU allocation is the maximum over all 27 formal repetitions per row, including weights. Only the full-context diagnostic disables LoRA; all principal comparisons share the same adapter. Source length, selected-pack length, and timer boundary are distinct.}
\label{tab:b300-recheck}
\end{table}
'''
(PAPER/'sections/tab_b300_recheck.tex').write_text(appendix,encoding='utf-8')
section=r'''\subsection{B300 full-context recheck}
\label{app:b300-recheck}
\input{sections/tab_b300_recheck}
This new recheck uses the identical source/query tokens, selected chunk IDs, backbone, and unmerged fp32 adapter as the local 128k supplement. One Slurm-allocated B300 reports CUDA compute capability 10.3 and 287.4 GB total device memory; its driver display name is L20D. Software is PyTorch 2.14.0+cu130, Transformers 5.16.1, and PEFT 0.20.0, with two intra-op and four inter-op CPU threads. Native GQA uses fused SDPA; the math attention backend is disabled. Model weights remain bf16 and fully GPU resident.

Dense processes BOS, the complete 131,072-token source, and the 512-token query: 131,585 input tokens. It constructs a full KV cache and projects only the final vocabulary logits. Encbank and raw replay use the same 6,657-token selected pack. All document chunks are written into CPU-pinned memory before the timer; Read begins with GPU-ready tensors. TTFT additionally includes CPU iterative BM25 (including its term-frequency construction), fetch, and sink/query Write. Neither timer includes document Write, tokenization, decode, or external I/O. Source/query inputs are unscored and unextended; these costs do not establish accuracy beyond the native position window.

'''
section+=f"The same-adapter full-context/Encbank Read ratio is ${ratios['same_adapter_full_vs_encbank_read']:.1f}\\times$, but the same-pack depth-only ratio is ${ratios['same_pack_depth_read']:.3f}\\times$ and the selected-pack TTFT ratio is ${ratios['same_pack_ttft']:.3f}\\times$. The large full-context ratio therefore combines selection with depth reuse. The stock, adapter-disabled dense diagnostic gives ${ratios['stock_full_vs_encbank_read']:.1f}\\times$ against Encbank. These new workloads and software do not reproduce the historical H20 conditions exactly; the original 38.3$\\times$ reference remains in Table~\\ref{{tab:online-lengths}}. Each process checks the loaded $j=0$ reader against stock last-position logits on a 256-token input before timing.\n\n"
f=PAPER/'sections/11_systems.tex'; text=f.read_text(encoding='utf-8')
if r'\subsection{B300 full-context recheck}' in text:
    begin=text.index(r'\subsection{B300 full-context recheck}'); finish=text.index(r'\subsection{Write-once serving',begin)
    text=text[:begin]+section+text[finish:]
else: text=text.replace(r'\subsection{Write-once serving',section+r'\subsection{Write-once serving')
f.write_text(text,encoding='utf-8')
f=PAPER/'sections/05_experiments.tex'; text=f.read_text(encoding='utf-8')
old='The two dense-context comparisons have still different meanings. The H20 store-ready group combines selection and depth reuse, with different adapter settings and preprocessing excluded.'
new=f"The B300 recheck compares same-adapter full prefill with store-ready Encbank ({full['latency_ms']:,.0f} versus {cm['latency_ms']:.1f}\\,ms; ${ratios['same_adapter_full_vs_encbank_read']:.1f}\\times$). This combines selection and depth reuse, with preparation and fetch excluded. The historical H20 38.3$\\times$ reference additionally differs in adapter settings (Appendix~\\ref{{app:b300-recheck}})."
if old in text: text=text.replace(old,new)
else:
    begin=text.index('The B300 recheck compares same-adapter full prefill')
    finish=text.index('The B200 pipeline group',begin)
    text=text[:begin]+new+' '+text[finish:]
f.write_text(text,encoding='utf-8')
report=['# B300 128k 复测','','实机 CUDA compute capability 为 10.3，总显存 287.4 GB；驱动别名 L20D。单个 Slurm GPU，3 独立进程 × 3 文档 × 3 次正式重复，每组 27 次。','', '| Operation | Method | Adapter | Latency ms | Peak GB |','|---|---|---|---:|---:|']
for arm in ('dense_stock','dense_shared','replay_read','encbank_read','replay_ttft','encbank_ttft'):
    v=s[arm]; report.append(f"| {arm} | {'Dense' if arm.startswith('dense') else ('Encbank' if arm.startswith('encbank') else 'Raw replay')} | {'Off' if arm=='dense_stock' else 'Shared'} | {v['latency_ms']:.1f} | {v['peak_GB']:.2f} |")
report+=['',f"同 adapter 的完整 dense / Encbank Read：{ratios['same_adapter_full_vs_encbank_read']:.2f}×。同 pack 的 raw replay / Encbank Read：{ratios['same_pack_depth_read']:.3f}×；对应 TTFT：{ratios['same_pack_ttft']:.3f}×。",'',f"关闭 adapter 的 dense / Encbank Read：{ratios['stock_full_vs_encbank_read']:.2f}×，仅作历史口径诊断。H20 的 50.59 / 1.32 = 38.3258× 算术成立；本轮更换了硬件、工作负载、checkpoint 和软件，不能据此证实或否定历史 H20 的精确值。",'','完整输入为 131,585 tokens，选中 pack 为 6,657 tokens；大倍数包含检索压缩上下文和跳过下层两种效应。Read 为 GPU-ready prefill，TTFT 另含 CPU 检索、fetch、sink/query Write；均排除 document Write、tokenization 与 decode。本轮不新增准确率结论。','', '主表中以 B300 同 adapter 复测替换旧 H20 运行点；H20 数值保留在附录原表。原 B200 pipeline 只更正硬件名称，数值不变。']
report='\n'.join(report)+'\n'
(HERE/'RESULTS_zh.md').write_text(report,encoding='utf-8')
(PAPER/'B300_RECHECK_zh.md').write_text(report,encoding='utf-8')
print('Integrated B300 recheck; historical H20 values retained in appendix')
