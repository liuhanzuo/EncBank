from pathlib import Path
import json

root = Path(__file__).resolve().parent
r = json.loads((root / 'heartbeat_cost_20260908_2202.json').read_text(encoding='utf-8'))
idx = {(s['arm'], s['context_tokens'], s['tier'], s['G'], s['Q']): s for s in r['rows']}
names = {'pub': 'Encbank', 'pub_sink': 'Encbank + sink', 'pub_lora': 'Encbank + LoRA', 'j0': 'Full recompute (j0)', 'fix_all': 'Encbank V2'}
arms = tuple(names)
lines = [
'# Local serving cost audit — 2026-09-08 22:02 heartbeat', '',
f"Completed CPU/stdlib-only audit at {r['checked_at']}. No GPU process was started, stopped, repeated or reconfigured. No model or cache tensors were loaded into Torch; only JSON/JSONL, small token ZIP members and file-size metadata were read.", '',
'## Completion and integrity', '',
'**All 120/120 formal cells are complete**: stock four methods × two lengths × two tiers × two generation lengths × three query counts = 96, plus 24 final-LoRA cells. Functional smoke results remain separate. The LoRA 128k attempt completed naturally during this audit; the bootstrap status has no active job.', '',
'This patrol fully checked the newly completed **36 cells / 1,200 query records** from j0 128k and LoRA 32k/128k: finite/nonnegative timing values, exact IDs 0–99, fixed generated counts 16/128, decode steps G−1, zero query-time capture, all Q-prefix component sums, cumulative totals, TTFT, decode rates and peak maxima. Every check passed. The older 84 cells retain their prior full arithmetic audits; they were not redundantly rescored here.', '',
'Across the completed matrix, **4,000 query records** passed hardware/thread and selected-pack identity checks. CPU token artifacts and query texts match across methods for each document length; all 100 selected indices, query/read token lengths match across methods, G and tiers. Each arm/G produces identical generated IDs on CPU and disk. Accepted-attempt and campaign aggregate summaries agree exactly. Actual serialized store file sizes match the reported byte totals. Invalid pub_sink 32k attempt 0001 remains excluded and preserved; accepted attempt 0002 is used.', '',
'All actual hardware metadata identify the local RTX 5090, Torch CPU/interop 2/16, OMP/MKL 2/2 and tokenizer parallelism false. New LoRA rows record step 4000 and unmerged PEFT; adapter weights are not included in the serialized document-store column.', '',
'| New job | Initial used GiB | Locked recheck GiB | Admitted at |',
'|---|---:|---:|---|',
]
for job in r['jobs']:
    if job['new_full_arithmetic_audit']:
        a = job['admission']
        lines.append(f"| {job['job']} | {a['initial_used_gib']:.9f} | {a['recheck_used_gib']:.9f} | {a['admitted_at']} |")
lines += ['', 'Each newly completed job has an effective 5.0 GiB ceiling, strict comparison, nvidia-smi MiB/1024 units, and no other Python compute process at admission. The earliest accepted V2 32k and Encbank 32k runs predate structured dual-check recording: their legacy gate logs report lock acquisition with approximately 3.9 and 2.8 GiB used, respectively. Do not claim that their exact initial and locked-recheck values were recorded. All later accepted runs have structured admission evidence.', '',
'## Reporting boundary', '',
'The cumulative total starts at an **already-tokenized, fixed-length document plus raw query text**, and ends at generated token IDs. It contains one whole-document store write (CPU tensor/chunk construction, capture, device-to-CPU copy, serialization), the measured startup load for that tier/G, and Q query totals including query tokenization, retrieval, load, host-to-device transfer, preparation, prefill and decode. It excludes full-source reading/tokenization, source-prefix preparation, model loading/warmup, client question construction and output-token-to-text decoding. Those exclusions must not be relabeled raw-document end-to-end serving.', '',
'Each tier/G has its own measured startup load. CPU tier loads the document cache into CPU RAM; disk tier reads selected payloads on demand, under ordinary OS page cache without eviction. This is a single sequential trace per method/length, with no repeated-run uncertainty estimate. Q=1/10/100 are nested prefixes of the same 100 queries, not three independent trials. No timing outlier was trimmed. Fixed G=16/128 suppresses natural EOS. This is an **unscored controlled excerpt workload**, not a real QA quality experiment and not a matched Pareto point with the QA F1 table.', '',
'## Compact 128k CPU table for main text', '',
'Cumulative columns include write + startup + queries; peak is the maximum PyTorch allocated memory across write and the shown CPU workloads, including model weights. It is not total nvidia-smi resident memory.', '',
'| Method | Store GiB | Write s | G16 Q1 s | G16 Q100 s | G128 Q1 s | G128 Q100 s | Peak GiB |',
'|---|---:|---:|---:|---:|---:|---:|---:|',
]
for arm in arms:
    s = idx[arm, 131072, 'cpu', 16, 100]
    vals = [idx[arm, 131072, 'cpu', g, q]['end_to_end_total_s'] for g in (16, 128) for q in (1, 100)]
    peak = max(idx[arm, 131072, 'cpu', g, 100]['peak_allocated_bytes'] for g in (16, 128)) / 2**30
    lines.append(f"| {names[arm]} | {s['write']['serialized_bytes']/2**30:.6f} | {s['write']['write_total_s']:.3f} | " + ' | '.join(f'{v:.3f}' for v in vals) + f' | {peak:.3f} |')
lines += ['', '**Main conclusion:** V2 has lower mean TTFT than j0 at 128k CPU Q=100 (0.550/0.559 s versus 0.710/0.694 s for G16/G128: 1.291×/1.242× faster). Its measured cumulative totals remain **18.49% / 6.76% higher**, respectively. Writing and loading a 7.003 GiB cache plus generation costs do not yield overall savings at these settings. The full recompute arm stores raw token IDs rather than document KV; V2 uses more storage than that arm and has a slightly higher allocated peak (17.069 versus 16.788 GiB). Do not claim an overall latency or peak-memory advantage over j0.', '',
'The memory comparison with final unmerged LoRA is checkpoint/mode dependent: LoRA peak is 17.273 GiB, slightly above V2, whereas original Encbank and +sink peak near 16.506 GiB. This does not establish a general architectural memory advantage. The separate analytic 61.1% persistent-cache reduction is relative to an isolated **full-depth KV cache**, which is not j0 and is not measured in this five-arm cost matrix. **Actual online KV tensor bytes were not directly measured**: neither allocated-memory peak/increment nor host-to-device transfer bytes is a substitute. Query incremental peak is relative to allocation at query entry and excludes the separately recorded write increment; use the existing fields with those definitions.', '',
'## Complete cumulative matrix', '',
'Each triple is Q=1 / 10 / 100 in seconds. TTFT columns are means at Q=100. Full unrounded per-Q component totals, TTFT, startup, peak, incremental query peak, decode rates and actual bytes are in the paired JSON.', '',
'| Length | Tier | Method | G16 cumulative s (1/10/100) | G128 cumulative s (1/10/100) | TTFT G16/G128 s |',
'|---|---|---|---|---|---|',
]
for n in (32768, 131072):
    for tier in ('cpu', 'disk'):
        for arm in arms:
            triples = [' / '.join(f"{idx[arm,n,tier,g,q]['end_to_end_total_s']:.3f}" for q in (1,10,100)) for g in (16,128)]
            ttft = ' / '.join(f"{idx[arm,n,tier,g,100]['mean_ttft_s']:.3f}" for g in (16,128))
            lines.append(f'| {n//1024}k | {tier} | {names[arm]} | {triples[0]} | {triples[1]} | {ttft} |')
lines += ['', '## Store and write details', '',
'Write is charged once per document in each cumulative scenario. Payload bytes exclude raw IDs, headers and serialization metadata; serialized bytes include all store files and were verified on disk.', '',
'| Length | Method | Serialized bytes | Payload bytes | Write compute s | D2CPU s | Serialize s | Write total s | Peak GiB |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for n in (32768, 131072):
    for arm in arms:
        s = idx[arm, n, 'cpu', 16, 100]
        w = s['write']
        peak = max(idx[arm,n,tier,g,100]['peak_allocated_bytes'] for tier in ('cpu','disk') for g in (16,128))/2**30
        lines.append(f"| {n//1024}k | {names[arm]} | {w['serialized_bytes']} | {w['payload_tensor_bytes']} | {w['write_compute_s']:.3f} | {w['write_device_to_cpu_s']:.3f} | {w['write_serialize_s']:.3f} | {w['write_total_s']:.3f} | {peak:.3f} |")
lines += ['', '## V2 versus same-pack full recompute and observed crossings', '',
'V2 is slower in cumulative total at **all 24 prespecified settings** (two lengths × two tiers × two G × Q=1/10/100). Disk residency does not create a winning prespecified condition. At Q=100, disk V2/j0 cumulative ratios are 1.029/1.109 at 32k and 1.202/1.118 at 128k for G16/G128.', '',
'Reconstructing all intermediate prefixes exposes one qualification: 32k CPU G16 is temporarily faster for **Q=32–70**, then subsequent recorded long tails reverse the comparison. All other seven length/tier/G traces are slower at every measured Q=1–100. Thus there is **no sustained break-even through Q=100** in these recorded traces; do not describe the temporary prefix crossing as a durable benefit or extrapolate beyond 100 queries.', '',
'## Preserved 32k V2 variation', '',
'| Tier | G | Median query total s | p95 s | Maximum s |',
'|---|---:|---:|---:|---:|',
]
for d in sorted(r['v2_untrimmed_distributions'], key=lambda d: (d['tier'], d['G'])):
    if d['context_tokens'] == 32768:
        v = d['total_s']
        lines.append(f"| {d['tier']} | {d['G']} | {v['median']:.3f} | {v['p95_nearest_rank']:.3f} | {v['maximum']:.3f} |")
lines += ['', 'The CPU G16 maximum is 12.609 s and CPU G128 maximum is 28.914 s; their origins are not resolved. Start-time admission does not establish freedom from interference throughout a long run. Neither silently trim these traces nor reinterpret their disk/CPU ranking as intrinsic storage performance. V2 128k CPU startup also differs by G (12.065 s for G16 versus 1.563 s for G128), consistent with the explicitly uncontrolled OS cache state; exact startup values are retained for every cell.', '',
'## Evidence and next step', '',
'- Machine-readable complete matrix and checks: `heartbeat_cost_20260908_2202.json`.',
'- Reproducible stdlib-only audit: `audit_cost_2202.py`.',
'- Accepted attempts: `results/local/serving_reuse/full/*/attempts/` and `results/local/serving_reuse/pub_lora/full/*/attempts/`; exact paths are listed in the JSON.',
'- Bootstrap `status.json`, each accepted `COMPLETED.json`, workload/token store, raw query files and aggregate summaries agree.',
'',
'No continuation of this 120-cell workload is needed. Keep every accepted result and invalid attempt. Remaining QA-matched costs or new cache-system baselines require their own protocol and fresh local 5090 admission; completion here does not mean those distinct experiments are complete.',
]
(root / 'heartbeat_cost_20260908_2202.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(root / 'heartbeat_cost_20260908_2202.md')
