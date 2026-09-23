# Chunk-local anchor qualification

Three configurations, caching no depth above H24: dual, H12/H16/H20/H24 and six shallow anchors. Write each historical 512-token chunk independently through layer24 with local positions. Write the sink independently. These cached source states can be packed into new retrieval combinations without having been computed from that selected joint context.

Fit heads anew with the same 32 training documents, per-chunk-local source H and original joint-context teacher KV targets. Validate four documents and evaluate eight reused test documents. Exact anchor projections now inherit the approximation of chunk-local deeper hidden. The exact control in each job concerns true per-layer teacher H, not an assertion that chunk-local deeper anchors reproduce joint-context KV.

This checks source-context mismatch beyond the fixed-context shallow sweep. It still is a small static continuation study, not a long-running agent benchmark, and does not establish quality for arbitrary natural retrieval distributions.
