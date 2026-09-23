# Hidden-keyed hot KV buffer functional pilot

One GPU, original H12 ridge heads. LRU capacity eight chunk slots, including a pinned current chunk. Historical lookup keys are frozen hidden SHA plus head version. Cache projected K after K norm but before RoPE, and V; apply RoPE for the current packing positions. Current chunk accumulates its native KV and actual online H12 during decode. On reaching 512 tokens, project its H12 once, store the completed chunk, and permit immediate hidden-space retrieval.

Execute 64 query tokens plus 1024 fixed decode inputs, yielding full chunk boundaries after 448 and 960 generated inputs, with 64 tokens in the final active chunk. Retrieve top4 using cosine of mean H12 among 16 initial chunks plus completed current chunks. Compare cached retrieval with cold projection for exact tensor equality, including repacked RoPE; test LRU and active-slot eviction rules.

This is a functional trajectory, not an agent benchmark or natural cache-hit estimate. Eight slots constrain the buffer only: the attention pack, hidden bank, lower-layer query cache, weights and transition temporaries have additional memory costs. Current native KV is converted to the frozen hidden-projection policy on completion, rather than silently treated as equivalent across changed retrieval contexts.
