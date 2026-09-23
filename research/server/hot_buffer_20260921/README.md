# Hot KV buffer: real agent trace replay and streaming qualification

This experiment implements the cache proposal rather than only changing reconstruction depth. It uses the same server-resident Qwen3-8B and original COMem LoRA. No weights or adapters are exported.

Cold bank: independent H12 for 512-token chunks. Hot pool: actual upper-layer KV tensors, byte-budgeted LRU. Capacity 12/24/48 chunks corresponds to approximately 576/1152/2304 MiB of BF16 upper-24-layer KV, plus the sink. Active decode KV, H12, model and temporary copies are accounted for separately through measured peak allocation.

Exact mode validates identical ordered memory prefixes and rebuilds only an unmatched suffix; this preserves mathematical attention semantics, subject to BF16/kernel-shape differences. Chunk mode reuses a chunk's stored KV across changed selected prefixes, rebasing RoPE keys from their stored position. It deliberately approximates contextual KV and must be evaluated for quality. Online-generated/prefilled KV enters only chunk mode, never masquerades as an exact independent-H12 reconstruction.

Every 512 generated tokens, the active query is repartitioned, retrieval is rerun, and only selected history participates in attention. Full chunks already computed in the live query are proactively stored before retrieval and at call completion. The currently incomplete chunk stays in active decode KV. Serialized history can change its chunk boundaries between calls; content and chunk-position keys prevent false identity matches, and H12 bank writes are deduplicated by content.

Initial experiment: fixed first-six-request traces from real Terminal-Bench attempts, one forced token per request to measure prefill and next-token distribution differences; two repetitions with reversed variant order. A separate 1056-token forced streaming test exercises both boundaries, current-chunk promotion, and no-promotion ablation. These are performance/implementation tests, not completed Terminal-Bench scores. Existing cold experiments stay intact. Full fresh task evaluations follow qualification.
