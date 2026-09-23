"""Format verified two-length results on the remote CPU, without model imports."""
from pathlib import Path
import json

root = Path.cwd()
r = json.loads((root/'heartbeat_cacheblend_cost_20260909_0027.json').read_text())
names = {'pub':'Encbank', 'pub_sink':'Encbank + sink', 'pub_lora':'Encbank + LoRA',
         'j0':'Full recompute (j0)', 'fix_all':'Encbank V2', 'cacheblend16':'CacheBlend-style port'}
idx = {(s['arm'],s['context_tokens'],s['tier'],s['G'],s['Q']):s for s in r['rows']}
lines = ['# Complete CacheBlend serving cost audit — 2026-09-09 00:27 heartbeat', '',
f"Audit completed at {r['audited_at_utc']} on the remote CPU. **Both document lengths are now complete: 24 CacheBlend cells and 800 query records.** The new 128k contribution is 12 cells / 400 queries. Previously verified 32k rows are unchanged. No model was rerun, no GPU job was launched, and no paper, sync process or prior result was modified.", '',
'## Verification', '',
'A separate metadata bundle contained 146 small files (25,630,720 bytes): JSON/JSONL and original token artifacts only. Model weights and full KV stores were not copied or loaded. Local filesystem stat records supplied actual serialized store sizes; all reaggregation, identity checks and comparisons ran remotely using the Python standard library with CUDA disabled and no Torch import.', '',
'The new 128k job has an actual complete marker and normal completion log. Its admission was **4.125 GiB initially and 4.1240234375 GiB after locking**, both strictly below 5 GiB at 00:04:36, with no other Python compute process. It completed naturally; the bootstrap reports completed with no active job/child/error at 00:26:42.', '',
'The actual GPU is the same local RTX 5090 UUID as the five existing methods, with stock BF16 Qwen3-8B, Windows/Python 3.13.5, Torch 2.7.1+cu128, Transformers 5.16.1, Torch CPU/interop threads 2/16, OMP/MKL 2 and tokenizer parallelism false. The trained-original comparison retains its intended unmerged final 4000-step adapter. Chunk size 512, topk 12, seed 42, source, document prefix, G and Q settings match.', '',
'Across all six methods and both lengths, **4,800 existing query records** have matching query text, token artifacts, selected chunk indices and query/read token lengths for each document/query. Each method produces identical generated IDs between CPU and disk for the same G. The five earlier methods match their prior audited 120-cell matrix exactly; the earlier CacheBlend 32k rows also match their saved report exactly.', '',
'All 800 CacheBlend formal rows have finite/nonnegative times, IDs 0–99 per tier/G, G generated tokens, G−1 decode calls, zero document/query capture at query time, timing_eligible=true and instrumentation_diagnostic=false. They contain neither fresh-reference generation nor online-inventory probes. Each read records two full bootstrap layers and floor(0.16×6144)=983 selectively recomputed context tokens, plus all sink/query positions. All Q-prefix component sums, cumulative totals, TTFT, decode rates and allocated-peak formulas pass. Whole-document capture counts are 65/257 for 32k/128k, including one standalone BOS; payload and actual serialized file-byte counts agree.', '',
'The separate 8B BF16 GPU smoke remains eight reuse generations plus eight extra fresh references, with exact generated IDs and selected recomputation positions. It is correctness evidence and contributes no formal cost cell.', '',
'## 128k CPU main-table extract', '',
'TTFT is the mean at Q=100, G=16. Totals include one write, the corresponding tier/G startup and the query prefix. Peak is maximum PyTorch allocated GPU memory across write/query phases, including weights; it is not nvidia-smi resident memory.', '',
'| Method | Store GiB | Write s | TTFT s | G16 Q1 / Q100 s | G128 Q1 / Q100 s | Peak GiB |',
'|---|---:|---:|---:|---|---|---:|']
for arm,name in names.items():
    s = idx[arm,131072,'cpu',16,100]
    totals = [' / '.join(f"{idx[arm,131072,'cpu',g,q]['end_to_end_total_s']:.3f}" for q in (1,100)) for g in (16,128)]
    peak = max(idx[arm,131072,'cpu',g,100]['peak_allocated_bytes'] for g in (16,128))/2**30
    lines.append(f"| {name} | {s['write']['serialized_bytes']/2**30:.6f} | {s['write']['write_total_s']:.3f} | {s['mean_ttft_s']:.3f} | {totals[0]} | {totals[1]} | {peak:.6f} |")
lines += ['', '## All cumulative conditions', '',
'Each triple is Q=1 / 10 / 100 in seconds. TTFT means use Q=100; unrounded per-Q TTFT and all component totals are in the JSON.', '',
'| Length | Tier | Method | G16 cumulative s | G128 cumulative s | TTFT G16 / G128 s |',
'|---|---|---|---|---|---|']
for n in (32768,131072):
    for tier in ('cpu','disk'):
        for arm,name in names.items():
            totals = [' / '.join(f"{idx[arm,n,tier,g,q]['end_to_end_total_s']:.3f}" for q in (1,10,100)) for g in (16,128)]
            ttft = ' / '.join(f"{idx[arm,n,tier,g,100]['mean_ttft_s']:.3f}" for g in (16,128))
            lines.append(f'| {n//1024}k | {tier} | {name} | {totals[0]} | {totals[1]} | {ttft} |')
lines += ['', '## Exact CacheBlend storage, peak and startup', '',
'Incremental query peak is relative to PyTorch allocation at query entry. It excludes the separately recorded write increment. Startup is charged separately for each tier/G; it is not assumed identical between answer lengths.', '',
'| Length | Payload bytes / GiB | Serialized bytes / GiB | Write s | Query increment GiB | Peak GiB | CPU startup G16 / G128 s |',
'|---|---|---|---:|---:|---:|---|']
for w in r['writeback_summary']:
    lines.append(f"| {w['context_tokens']//1024}k | {w['payload_bytes']} / {w['payload_GiB']:.9f} | {w['serialized_bytes']} / {w['serialized_GiB']:.9f} | {w['write_total_s']:.9f} | {w['max_query_incremental_peak_GiB']:.9f} | {w['max_allocated_peak_GiB']:.9f} | {w['startup_cpu_G16_s']:.9f} / {w['startup_cpu_G128_s']:.9f} |")
lines += ['', '## V2, CacheBlend and full recompute', '',
'For 128k and Q=100, the CacheBlend cumulative difference relative to same-pack j0 is:', '',
'| Tier | G | CacheBlend total s | j0 total s | CacheBlend / j0 − 1 | V2 total s |',
'|---|---:|---:|---:|---:|---:|']
for tier in ('cpu','disk'):
    for g in (16,128):
        cb,j,v = [idx[a,131072,tier,g,100]['end_to_end_total_s'] for a in ('cacheblend16','j0','fix_all')]
        lines.append(f'| {tier} | {g} | {cb:.3f} | {j:.3f} | {100*(cb/j-1):+.2f}% | {v:.3f} |')
lines += ['', '**The full results contain real conditional CacheBlend gains:** at 128k, Q=100, G128, its cumulative total is lower than j0 on CPU (577.664 versus 587.217 s) and disk (544.300 versus 572.384 s). At the same length/Q with G16 it is slower than j0. At 32k it wins CPU/G16/Q100 but loses CPU/G128 and both disk/G settings at Q100. Do not describe CacheBlend as always slower, or transfer one winning answer length to the others.', '',
'V2 has higher Q100 cumulative cost than CacheBlend in all four 128k tier/G settings, although it starts with a smaller document store and lower write cost. Its early-query comparisons are mixed because write/startup and query costs differ. Preserve those unfavorable V2 observations. At 32k, CPU V2 has the already disclosed untrimmed long tails; disk G16 favors V2, while disk G128 favors CacheBlend. No single overall latency ranking or statistically significant benefit follows from these traces.', '',
'V2 uses less serialized storage and reaches a smaller measured allocated peak than this CacheBlend-style implementation:', '',
'| Length | V2 store GiB | CB store GiB | Storage reduction | V2 peak GiB | CB peak GiB | Peak reduction |',
'|---|---:|---:|---:|---:|---:|---:|']
for n in (32768,131072):
    v,cb = [idx[a,n,'cpu',16,100] for a in ('fix_all','cacheblend16')]
    vb,cb_b=v['write']['serialized_bytes'],cb['write']['serialized_bytes']
    vp,cp=[max(idx[a,n,t,g,100]['peak_allocated_bytes'] for t in ('cpu','disk') for g in (16,128)) for a in ('fix_all','cacheblend16')]
    lines.append(f'| {n//1024}k | {vb/2**30:.6f} | {cb_b/2**30:.6f} | {100*(1-vb/cb_b):.2f}% | {vp/2**30:.6f} | {cp/2**30:.6f} | {100*(1-vp/cp):.2f}% |')
lines += ['', '## Required interpretation limits', '',
'1. **Prototype allocation, not an architectural minimum.** The CacheBlend port holds selected GPU payloads plus merged full-depth KV while cloning the upper 34 layers and retaining fresh two-layer bootstrap KV. It also computes logits for every selected position; V2 read_prefill computes the final-position logits only. Staging and old merged KV are released after the selective read. No cross-query leak or fresh-reference work was found in formal runs, but these within-query copies/logits are optimizable prototype overhead. Attribute the peak difference to these measured implementations, not solely to necessary online KV.',
'2. **Same allocator and phase boundary.** Both methods reset and read torch.cuda.max_memory_allocated during document write and each query, then take the same aggregate maximum. This includes actual temporary tensors, not merely KV. Timed rows contain online_kv_inventory=null. Separate phase inventories can expose active cache storage but cannot fully decompose the transient peak or be inferred from it.',
'3. **Known CacheBlend-style deviation.** The method is full-depth isolated post-RoPE KV with standalone BOS, no chunk-write BOS, layer-1 V-error selection and two full bootstrap layers before 16% selective context recomputation. It is not cbos and not the released CacheBlend serving implementation. Zero query-KV slots avoid redundant isolated query prefill; actual BF16 smoke verified the output equivalence.',
'4. **Identical stated input/cost boundary.** One whole-document write + tier/G startup + query totals starts from pretokenized document and raw queries and ends at token IDs. Full-source read/tokenization, prefix setup, model load/warmup, client question construction and text decoding are excluded. It is not raw-document end-to-end latency.',
'5. **Single traces and fixed output lengths.** CPU/disk runs use ordinary OS page cache without eviction; Q=1/10/100 are nested prefixes, not independent repeats. Startup differs across G. All original V2 32k long tails remain. Do not invent confidence intervals, cold-disk performance, extrapolated break-even, or matched QA quality/cost points from this unscored excerpt workload.', '',
'The complete unrounded machine artifact is `heartbeat_cacheblend_cost_20260909_0027.json`: 144 combined rows (120 original + 24 CacheBlend), 120 CacheBlend-versus-reference ratios, both exact writeback summaries, actual hardware receipts and untrimmed distributions. The prior 0005 audit remains preserved. All CacheBlend cost cells are now eligible for manuscript writeback under these qualifications; no other queue was awaited.']
(root/'heartbeat_cacheblend_cost_20260909_0027.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Complete two-length report written')
