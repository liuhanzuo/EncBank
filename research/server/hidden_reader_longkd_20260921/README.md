# Longer four-depth distillation

Warm-start the previously selected256-step H12/H16/H20/H24 independent-chunk model. Train1792 more updates, for2048 total model updates. Keep128 training documents and4 validation documents; exclude all RULER data. Two peak learning rates3e-6/1e-5, cosine decay to10%. Optimizer state is reinitialized, not resumed, because it was not previously saved.

Save total-step512/1024/2048 snapshots plus validation-selected heads.pt. Choose the preferred long run and checkpoint using validation KL only, before loading RULER scores. Same frozen backbone/LoRA/writer and20 learned heads as before. All model artifacts remain on the server; unrelated jobs are not modified.
