"""Format the verified remote CPU audit; no local model/timing work."""
from pathlib import Path
import json

root=Path.cwd()
r=json.loads((root/'heartbeat_cacheblend_cost_20260909_0005.json').read_text())
names={'pub':'Encbank','pub_sink':'Encbank + sink','pub_lora':'Encbank + LoRA','j0':'Full recompute (j0)','fix_all':'Encbank V2','cacheblend16':'CacheBlend-style'}
idx={(s['arm'],s['context_tokens'],s['tier'],s['G'],s['Q']):s for s in r['rows']}
lines=['# CacheBlend serving cost audit — 2026-09-09 00:05 heartbeat','',
f"Audited at {r['audited_at_utc']} on the remote CPU using only Python standard-library JSON/ZIP metadata processing. No local model computation, GPU scheduling or GPU process changes were performed by this subtask.",'',
f"**CacheBlend-style 32k is complete and verified: {r['checked_cacheblend_cells']} cells / {r['checked_cacheblend_queries']} query records.** The 128k attempt is active and incomplete at the copied snapshot; no 128k CacheBlend value is filled here. This audit does not wait for it. Functional GPU smoke is separate: eight reused 8B BF16 generations plus eight extra fresh references passed exact token-ID and recomputation-position checks.",'',
'## Verification and comparability','',
'Only this task\'s small JSON/JSONL and raw token artifacts were copied to a dedicated remote CPU audit directory (79 files, 10,219,520-byte TAR). Full KV files and model weights were not copied or loaded. Actual local serialized store bytes came from filesystem stat records; arithmetic, comparisons, token-artifact identities and reporting were computed remotely. CUDA was disabled and no Torch/CUDA module was imported by the audit.','',
'All 400 new raw query records have IDs0–99 per tier/G, exact fixed generation lengths16/128, G−1 decode steps, finite/nonnegative times, no query-time document/query capture, and genuine layer-1 V selection with floor(0.16×6144)=983 context tokens. Sink and every query token are selected. All raw rows have timing_eligible=true, instrumentation_diagnostic=false, no fresh reference and no online-KV inventory probe inside the formal run. CPU and disk produce identical generated IDs.','',
'The new 12-cell summary passes every Q-prefix component sum, cumulative formula, TTFT mean, decode-rate and peak calculation. Whole-document capture count is65 (64chunks + one standalone BOS); measured tensor payload is exactly32769×147456bytes. Store-file stat totals equal every recorded serialized-byte total. No raw record or long tail was dropped.','',
'Across the new method and the five already audited methods, all **2,400 query records** have the same per-query selected indices and query/read token lengths. The token artifacts and all100 query texts match exactly at32k. Model path, source, chunk512, topk12, j12, seed42, BF16, generation settings and CPU/disk ordering match; the final LoRA arm intentionally retains its completed4000-step unmerged adapter. The five older methods\' rows are identical to the previously audited120-cell matrix and were not replaced or rerun.','',
'All six methods record the same local RTX5090 UUID, Windows/Python3.13.5, Torch2.7.1+cu128, Transformers5.16.1, Torch CPU/interop2/16, OMP/MKL2 and tokenizer parallelism false. CB32k actual admission was **3.58203125GiB initially and3.5830078125GiB after the lock**, both strictly below5GiB, with no other Python compute process. The effective threshold and nvidia-smi MiB/1024 units match the existing gate. Earlier V2/Encbank32k have only legacy gate logs; do not invent their precise dual-check values.','',
'## Same accounting boundary','',
'The common cumulative total is **one whole-document write + that tier/G startup + Q-prefix query totals**, from an already-tokenized document and raw query texts to generated token IDs. It includes CPU tensor/chunk construction, full-document precompute/serialization, query tokenization/retrieval, CPU/disk load, H2D, position preparation, read prefill and decode. Full-source read/tokenization, prefix preparation, model loading/warmup, client question construction and output text decoding are excluded. This is not raw-text-document end-to-end latency.','',
'Q1/10/100 are nested prefixes of one100-query trace, not repeated independent runs. CPU startup loads the whole document cache into RAM; disk uses ordinary OS page cache without eviction. FixedG suppresses EOS and is an unscored excerpt workload. Neither its numbers nor its output lengths can be paired with natural-EOS QA F1 as one matched quality/cost point.','',
'## Complete32k comparison','',
'Each cumulative triple is Q1 / Q10 / Q100 seconds. TTFT values are per-query means atQ100.','',
'| Tier | Method | G16 cumulative s | G128 cumulative s | TTFT G16 / G128 s |',
'|---|---|---|---|---|']
for tier in ('cpu','disk'):
 for arm,label in names.items():
  vals=[' / '.join(f"{idx[arm,32768,tier,g,q]['end_to_end_total_s']:.3f}" for q in (1,10,100)) for g in (16,128)]
  ttft=' / '.join(f"{idx[arm,32768,tier,g,100]['mean_ttft_s']:.3f}" for g in (16,128))
  lines.append(f'| {tier} | {label} | {vals[0]} | {vals[1]} | {ttft} |')
lines+=['','## Actual storage and allocated memory at32k','',
'Peak is the maximum PyTorch allocated memory over write and all four query traces, including weights. It is not total nvidia-smi resident memory or directly measured online KV.','',
'| Method | Serialized store bytes | Store GiB | Write s | Peak GiB |',
'|---|---:|---:|---:|---:|']
for arm,label in names.items():
 s=idx[arm,32768,'cpu',16,100];w=s['write']
 peak=max(idx[arm,32768,tier,g,100]['peak_allocated_bytes'] for tier in ('cpu','disk') for g in (16,128))/2**30
 lines.append(f"| {label} | {w['serialized_bytes']} | {w['serialized_bytes']/2**30:.6f} | {w['write_total_s']:.3f} | {peak:.3f} |")
cb=idx['cacheblend16',32768,'cpu',16,100];v=idx['fix_all',32768,'cpu',16,100]
store_reduction=1-v['write']['serialized_bytes']/cb['write']['serialized_bytes']
peak_reduction=1-v['peak_allocated_bytes']/cb['peak_allocated_bytes']
lines+=['',f"At this measured32k workload, V2 uses **{100*store_reduction:.2f}% less serialized document storage** and **{100*peak_reduction:.2f}% less allocated peak memory** than this CacheBlend-style port. This is a qualified comparison to this full-depth KV implementation; it does not create a memory advantage over same-pack full recompute, whose measured peak remains lower than V2. Direct online-KV inventories are null in these timed records and cannot be reconstructed from allocation peaks.",'',
'## What the completed values support','',
'- CPU, Q100, G16: CacheBlend-style is115.595s versus j0123.967s, a6.75% cumulative reduction. It remains slower atQ1/Q10 because of write/startup cost. This is one observed condition, not a universal acceleration claim.',
'- CPU, Q100, G128: CacheBlend-style580.278s is slower than j0563.346s despite its lower TTFT. A short-answer benefit must not be transferred to long answers.',
'- Disk, Q100: CacheBlend-style133.688/594.715s exceeds j0124.491/559.402s for G16/G128. Disk residency does not automatically make the method cheaper.',
'- Against V2 on CPU, CacheBlend-style has lowerQ100 totals; those V2 traces contain the already disclosed substantial32k long tails. On disk, V2 wins G16 (128.156<133.688s), while CacheBlend-style wins G128 (594.715<620.640s). Keep these mixed outcomes and avoid an overall method ranking from one trace.',
'- CacheBlend-style CPU Q100 meanTTFT is0.417/0.409s, compared with j00.655/0.657s. Its disk TTFT is0.610/0.590s. The difference in phase costs and startup must remain visible alongside total latency.','',
'## Implementation and stability limitations','',
'This is the Qwen3 CacheBlend-style adapter already described in the protocol: full36-layer isolated post-RoPE KV, standalone BOS, no per-chunk BOS, contextual layer-1 V deviation, full two-layer bootstrap then16% selective context recomputation. It is not the no-recompute cbos control, and it is not the released CacheBlend serving system. Query suffix KV slots begin at zero and are overwritten at every layer; local8B BF16 smoke verified exact IDs and selected positions against fresh query-prefill references. Formal records include no extra reference inference. Unneeded GPU staging/merged tensors are released after the selective read; this lifetime was not applied retroactively to old methods.','',
'There is one sequential trace per method/length, no repeated-run confidence interval and no cold-disk experiment. Startup varies across tier/G, and unexplained32k V2 tails remain untrimmed. Peak allocation comparisons and storage savings do not establish accuracy or whole-system superiority. No128k or real-QA CacheBlend cost is inferred from this32k completion.','',
'| Arm / tier / G | Median query s | p95 s | Max s |',
'|---|---:|---:|---:|']
for d in r['untrimmed_distributions']:
 if d['arm'] in ('cacheblend16','fix_all'):
  lines.append(f"| {d['arm']} / {d['tier']} / {d['G']} | {d['median']:.3f} | {d['p95_nearest_rank']:.3f} | {d['maximum']:.3f} |")
lines+=['','Machine-readable verified rows, per-condition ratios, hardware admission, store signatures and distributions are in `heartbeat_cacheblend_cost_20260909_0005.json`. Reproduction uses `audit_cacheblend_cost_0005.py` against the copied metadata bundle. Only the32k result is eligible for writeback now; keep128k blank until its full completion and audit.']
(root/'heartbeat_cacheblend_cost_20260909_0005.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('report ready; store reduction=',store_reduction,'peak reduction=',peak_reduction)
