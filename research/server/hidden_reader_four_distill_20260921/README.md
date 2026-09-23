# Four chunk-local hidden depths: logits distillation

H12/H16/H20/H24 only, no cache point above24. Expand training from32 to128 document-disjoint body-only examples. Keep the four validation and eight diagnostic test documents, and add any remaining eligible dev documents as a fresh diagnostic set. Do not download model data or heads.

Initialize from selected four_local ridge heads. Four runs: learning rates3e-6/1e-5/3e-5 with20 heads and fixed anchor projections; plus1e-5 with three additional zero-initialized residual heads at KV17/21/25. Freeze backbone, writer and original LoRA. Train on full logits KL at128 continuation positions for256 updates, validating every32 steps with step0 eligible. Select checkpoints and the preferred run by validation KL only, then report old and fresh test separately. Additional residual heads cost48MiB BF16 but do not change16MiB/chunk history H storage.

All runs must reproduce the prior untrained validation KL, verify gradients reach every trainable head, preserve zero trainable backbone parameters, and compare resident/on-demand outputs after training. This is a small continuation pilot, not an agent benchmark.
