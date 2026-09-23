# Experimental CoMem decode kernels

This isolated prototype is derived from the installed vLLM 0.22.1 recurrent kernel. It accepts the existing HF [B,H,K,V] state directly, preserves the input state, and can fuse HF-style intermediate dtype rounding in Q/K normalization. Only single-token decode is supported. It is not a full vLLM CoMem runner or paged-attention backend.

GPU job119955 passed all12 synthetic dtype/shape/normalization cases, including unequal K/V dimensions,16 recurrent steps,zero initial state, and unchanged input state. Actual model BF16 dimensions had about4.5x single-layer speedup with normalization fusion.

Full27B model tests with the original step4000 adapter and1/8 ragged rows kept all544 tested greedy decisions equal and H unchanged. Strict full-logit equivalence did NOT pass. The fused variant measured1.033x (1row/k12),1.007x (8rows/k12),1.075x (8rows/k48), with onlytwo order-balanced timing repeats. These are fixed-token decode tests, not task accuracy or free-generation quality.

Do not install into running benchmark processes. Full integration still requires long-generation quality validation, cache lifecycle tests in the complete runner, scheduling, and efficient attention/cache execution. All weights and adapters remain on the server.

The Python source retains its upstream copyright notices. KERNEL_PROVENANCE data records source hashes and modifications. The opt-in environment marker is COMEM_ISOLATED_KERNEL_PILOT=1.

The separately tested chunked_kv_adapter.py uses512-token capacity increments with no fixed sequence ceiling. Job120013 passed append/reorder/compact contracts and had bitwise-identical complete model logits at all544 tested positions. KV-only8-row speedups were1.024x/1.032x at k12/k48 history sizes; combining it with GDN measured1.082x/1.110x but retained the GDN full-logit gate failure. See ../VALIDATION_RESULTS.md for final evidence and limitations.
