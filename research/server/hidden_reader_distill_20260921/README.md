# H12 logits distillation continuation

Two learning rates (1e-5, 5e-5), 96 updates / three passes over the same 32 training documents. Initialize from the previous selected H12 ridge heads. Freeze backbone, original LoRA and the exact H12→KV13 projection; train the 23 other affine heads on teacher logits KL at 128 continuation positions.

Validate on the same four validation documents every 16 updates, with the untrained step-zero heads eligible. Evaluate eight test documents only after selecting the checkpoint. This reused held-out set is not a new final benchmark. Keep all trained weights on the server and preserve original pilot artifacts.
