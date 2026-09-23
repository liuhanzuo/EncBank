# Joint causal depth ablation

Server-only Qwen3-8B / original Encbank LoRA experiment. Starting from independent chunk H12, run original blocks13 through n jointly and causally over selected memory; run remaining blocks inside each chunk. Preserve packed RoPE positions. Query and decode still access all memory at every upper layer. No learned hidden-to-KV heads or new training in this ablation.

Sweep n=12,14,16,18,20,24,28,32,36 plus native Encbank on first32 of each fixed100-item task/length cell. Select the shallowest joint depth meeting the prespecified2pp task-mean /5pp per-cell loss screen. Test that depth and both endpoints on the remaining68 items. Screen and confirmation are reported separately; all nine cells use exactly the previously frozen retrieval fixtures.

Measure actual history KV reconstruction latency, including local batching and cache packing. Every layer's MLP/projections still run. This does not implement reduced Transformer depth, learned-head distillation, or a cross-call hot buffer.
