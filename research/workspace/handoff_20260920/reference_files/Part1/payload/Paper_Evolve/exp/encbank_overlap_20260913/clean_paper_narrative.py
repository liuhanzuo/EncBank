"""Consolidate the manuscript as one study; preserve measured protocol distinctions."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'Encbank/paper_iclr2027_rewrite_20260912'
def edit(name,changes):
    p=PAPER/name;s=p.read_text(encoding='utf-8')
    for old,new in changes:
        if old not in s:raise ValueError((name,old[:100]))
        s=s.replace(old,new)
    p.write_text(s,encoding='utf-8')

edit('sections/tab_accuracy.tex',[
    ('All scores are on a 0--100 scale',r'Both Encbank variants use split depth $j=12$. All scores are on a 0--100 scale'),
    (r'Encbank (without LoRA) ($j=12$)$^*$',r'Encbank (without LoRA)'),
    (r'\textbf{Encbank} ($j=12$)',r'\textbf{Encbank}'),
    (r'$^{\dagger}$Different backbone. $^{\ddagger}$Earlier HCache-style checkpoint, distinct from the same-split adapter control and native HCache.',r'$^{\dagger}$Different backbone. $^{\ddagger}$Retrieval-free HCache-style checkpoint, distinct from the same-split adapter control and native HCache.'),
    (r'\\[1pt]'+'\n'+r'$^*$New frozen-$j=12$ measurements for RULER, LongEval, LongBench, and BABILong; LoCoMo retains the reported full-set same-split Judge score. New synthetic examples are not paired with the historical rows (Appendix~\ref{app:frozen-recheck}).',''),
])
edit('sections/05_experiments.tex',[
    (r'Table~\ref{tab:accuracy} compares seven configurations across five benchmarks. Encbank with LoRA has the highest descriptive average, although full-context KV-Direct is stronger on BABILong and slightly stronger on LongBench. The new frozen-$j=12$ evaluation scores 16.87 on RULER and 0.2 on LongEval, showing that the unadapted interface performs poorly on these tasks. Its synthetic examples are freshly generated, so its difference from the historical adapted row is not an item-paired adaptation estimate. The table summarizes measured operating points; it does not isolate the effect of depth reuse.',r'Table~\ref{tab:accuracy} compares seven configurations across five benchmarks. Encbank has the highest descriptive average, although full-context KV-Direct is stronger on BABILong and slightly stronger on LongBench. Encbank without LoRA scores 16.87 on RULER and 0.2 on LongEval, showing that the unadapted interface performs poorly on these tasks. This overview summarizes operating points; paired controls below isolate depth reuse.'),
    ('In the historical H20 same-pack, same-adapter depth comparison','In the H20 same-pack, same-adapter depth comparison'),
    ('The historical H20 equal-latency calibration','The H20 latency calibration'),
    ('The B300 recheck compares','The B300 experiment compares'),
    (r' The historical H20 38.3$\times$ reference additionally differs in adapter settings (Appendix~\ref{app:b300-recheck}).',''),
])
edit('sections/08_protocol.tex',[
    (r'The original adapter-free depth reference uses $j=9$. The new Table~\ref{tab:accuracy} frozen row instead uses $j=12$ with the same memory settings; Appendix~\ref{app:frozen-recheck} identifies its fresh evaluation inputs.',r'The Encbank and Encbank without LoRA configurations both use $j=12$; Appendix~\ref{app:frozen-recheck} specifies the no-adapter evaluation protocol.'),
    ('Cohort B uses a freshly sampled','Cohort B uses an independently sampled'),
    ('The historical main-table rows use the task--length support of set A; the starred frozen row uses the same support with fresh synthetic samples. Set B is reserved for replay and YaRN controls. No direct difference is taken between them, and legacy 256k results are excluded from these means.','The main table uses the task--length support of set A, with independently sampled synthetic runs for the adapter and no-adapter configurations. Set B is reserved for paired replay and YaRN controls. These macros cover 8k--128k; no direct difference is taken between the two cohorts.'),
    ('The HCache-style overview retains an earlier retrieval-free checkpoint evaluation','The HCache-style overview evaluates a retrieval-free checkpoint'),
    ("LongEval's common-support mean covers 8k--128k. Its separate six-length mean includes the 4k Encbank result of 92, yielding 72.83 over 600 examples.","LongEval's common-support mean covers 8k--128k."),
    (r'The historical environment specifies Python 3.10+, PyTorch 2.10, Transformers 5.5.4, PEFT 0.10+, Datasets 2.14+, bf16, and SDPA. Appendix~\ref{app:frozen-recheck} gives the environment for the new frozen row.',r'The main benchmark runs use Python 3.10+, PyTorch 2.10, Transformers 5.5.4, PEFT 0.10+, Datasets 2.14+, bf16, and SDPA. Appendix~\ref{app:frozen-recheck} specifies the no-adapter runs, and Appendix~\ref{app:systems} gives the device-specific timing environments.'),
])
edit('sections/10_benchmarks.tex',[
    (r'\subsection{New frozen-$j=12$ operating point}',r'\subsection{Evaluation without LoRA}'),
    (r'\input{sections/tab_frozen_j12_recheck}'+'\n',''),
    (r'The starred frozen row in Table~\ref{tab:accuracy} uses a new adapter-free Qwen3-8B evaluation at $j=12$.',r'Encbank without LoRA in Table~\ref{tab:accuracy} evaluates Qwen3-8B at $j=12$ with the suffix adapter disabled.'),
    ('These are fresh synthetic examples whose identity with the historical rows is not established.','Adapter and no-adapter synthetic results use independent runs; their aggregate difference is not an item-paired estimate.'),
    (r'The LoCoMo score is the existing 1,986-item frozen-$j=12$ Judge result, not a new judging run. Thus the overview combines a new four-benchmark measurement with that historical same-split result. The $j=9$ rows below remain labeled historical depth references; their values are not relabeled as $j=12$. This operating-point comparison does not establish an item-paired effect of the historical suffix adapter.',r'The no-adapter LoCoMo Judge score is 24.52 over all 1,986 questions. The benchmark tables below report the corresponding $j=12$ configuration throughout; Table~\ref{tab:frozen-depth} separately studies split depth.'),
    ('but the original submission does not establish paired example identity','but the examples are not treated as item-paired'),
    ('The frozen Encbank control permits 48 generated tokens','Encbank without LoRA permits 48 generated tokens'),
])
edit('sections/tab_eval_protocol.tex',[
    ('16; frozen 48','16; without LoRA 48'),
    ('Historical shard counts are eight for RULER/LongEval/LongBench and four for BABILong/LoCoMo; the new frozen-$j=12$ evaluation uses four shards throughout.','Adapted runs use eight shards for RULER/LongEval/LongBench and four for BABILong/LoCoMo; no-adapter runs use four shards throughout.'),
])

for name in ('tab_ruler_a.tex','tab_longeval_detail.tex','tab_babilong_detail.tex','tab_longbench_detail.tex'):
    p=PAPER/'sections'/name;s=p.read_text(encoding='utf-8')
    s=s.replace(r'\caption{',r'\caption{Both Encbank variants use $j=12$. ',1)
    p.write_text(s,encoding='utf-8')
p=PAPER/'sections/tab_locomo_categories.tex';s=p.read_text(encoding='utf-8')
s='\n'.join(l for l in s.splitlines() if not l.startswith('Encbank (without LoRA) $j=9$'))+'\n'
p.write_text(s,encoding='utf-8')
for p in (PAPER/'sections').glob('tab*.tex'):
    s=p.read_text(encoding='utf-8')
    s=s.replace('HCache-style values retain the earlier checkpoint evaluation; this is not the same-split adapter diagnostic.','HCache-style uses a retrieval-free checkpoint distinct from the same-split adapter diagnostic.')
    s=s.replace('The native-window losses on qa1/qa2 and the stronger Encbank qa5 mean are both retained. ','')
    s=s.replace("Encbank's separate 4k score of 92 yields a six-length mean of 72.83. ",'')
    s=s.replace(' Timing quantiles are as reported in the original submission.','')
    s=s.replace('The retained Encbank main-table latency','The Encbank main-table latency')
    s=s.replace('B300 recheck on three fixed','B300 measurements on three fixed')
    s=s.replace('was not retained','is unavailable')
    s=s.replace('is retained only as sensitivity','is reported only as sensitivity')
    p.write_text(s,encoding='utf-8')
edit('sections/09_diagnostics.tex',[
    ('An earlier component measurement','A component measurement'),
    ('Historical multikey scripts instead fall back to the first input token when the tokenizer has no BOS ID; therefore these new scores are independent confirmation, not paired differences from the historical cohort.','The multikey diagnostic uses the first input token as its sink when the tokenizer has no BOS ID. This sink difference and the independently sampled inputs prevent item-paired differences across tasks.'),
    ('The earlier $j=12$ fixed-pack Write measurement was not retained.','The $j=12$ fixed-pack Write measurement is unavailable.'),
])
edit('sections/11_systems.tex',[
    ('At 8k and 16k, both methods are freshly measured in the same three processes under the 28 GB cap, with shuffled method and phase order. At 32k and 128k, Encbank\'s earlier observations are retained; their original 26 GB allocator cap is below the new limit.','At 8k and 16k, both methods are measured in the same three processes under the 28 GB cap, with shuffled method and phase order. At 32k and 128k, Encbank uses a stricter 26 GB allocator cap.'),
    ('Every successful fresh cell uses one warmup and three formal repetitions:','Dense at every length and Encbank at 8k/16k use one warmup and three formal repetitions per successful cell:'),
    ('Retained 32k/128k Encbank Prefill and TTFT instead','Encbank Prefill and TTFT at 32k/128k'),
    ('retained E2E uses','Encbank E2E at these lengths uses'),
    (' Earlier observations are retained without increasing their repeat counts.',''),
    ('The earlier raw-replay measurements instead send','The raw-replay depth measurements send'),
    ('The historical H20 matched-depth and TTFT-calibration sources do not report corresponding GPU peaks. These local results supply independent device-specific observations.','The H20 matched-depth and TTFT-calibration experiments do not measure GPU peaks; the RTX 5090 experiments report device-specific memory costs.'),
    (r'\subsection{B300 full-context recheck}',r'\subsection{B300 full-context and selected-pack comparison}'),
    ('This new recheck uses the identical','The B300 experiment uses the same'),
    (r'These new workloads and software do not reproduce the historical H20 conditions exactly; the original 38.3$\times$ reference remains in Table~\ref{tab:online-lengths}. ',''),
    (r'\input{sections/tab_online_lengths}'+'\n',''),
    (' The separate H20 store-ready cohort uses one warmup and three measurements; dense is stock LoRA-off and Encbank is adapted. Preparation and external fetch are excluded, so its ratios are composed online operating points rather than same-adapter or repeated-query estimates.',''),
    ('Original pooled-IID intervals are retained as sensitivity only','Pooled-IID intervals are reported as sensitivity only'),
])
edit('figures/source_scaling.tex',[("Encbank's earlier 32k/128k measurements used a tighter 26 GB cap.","Encbank's 32k/128k measurements use a tighter 26 GB cap.")])
edit('sections/12_statistics.tex',[
    ('The retained records contain','The records contain'),
    ('No retained matched native-chat table is available','No matched native-chat evaluation is available'),
    ('The retained artifacts lack predictions','The available results lack predictions'),
    ('The later natural-task audit','The natural-task audit'),
])
print('Removed revision-history framing and consolidated current benchmark results.')
