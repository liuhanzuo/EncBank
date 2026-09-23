# Shallow-only hidden cache sweep

User correction: cache additional depths only from H12 through H24. Never cache H28/H32 in this sweep. H24 supplies every later target layer.

Compare seven schedules in configs.json, including H12/H16/H20/H24. Reuse exactly the prior 32/4/8 document split, ridge fitting and output metrics. Select schedules by validation KL, report all test results; the test set is reused from the earlier pilot and is not a fresh final benchmark.

Each arm measures two separate mechanisms: approximate per-layer affine KV heads, and exact original-transformer segment replay from true cached anchors. Exact replay includes serial and single-GPU threaded streams, with KV equality checks and separately reported extra-anchor preparation cost. All anchors come from the same fixed teacher context; changed retrieval reuse remains unvalidated.

Files used by submitted jobs are immutable. All model computation and fitted weights remain on the server. No user GPU-count cap applies to this research; unrelated benchmark jobs retain their own constraints and owners.
