# 27B Hot24 with PEFT-merged LoRA

This directory is a convenient entry point copied from the independent Hot24
merged experiment. The full historical tree is preserved at
[`research/server/tf27b_hot_live_20260921/hot24_peft_merged_r3`](../../research/server/tf27b_hot_live_20260921/hot24_peft_merged_r3/).

| Component | Source |
| --- | --- |
| H21 writing, exact-prefix pool, online refresh | `hybrid_hot.py` |
| Hybrid model reader and legacy adapter | `hybrid_reader.py` |
| Variable-length cache batching | `batch_cache.py` |
| Multi-hop BM25 retrieval | `memory_selectors.py` |
| PEFT conversion and execution modes | `comem_peft.py` |
| Model identity, fixed-row projections/norms, row-local SDPA | `model_setup.py`, `runtime_identity.py` |
| Single shared model and independent sessions | `worker.py` |
| Harbor/Terminus trials and supervision | `tb_agent.py`, `run_trial.py`, `parallel_trials.py`, `launch_trial_group.py` |
| Merged Hot/Cold and batch qualification | `merged_qualification.py` |

The example plan selects 32 tasks with task/decode ceilings of 24, a 24-entry
per-session Hot Buffer, split depth 21, top-12 retrieval, 512-token chunks and
refreshes, and a 32-step scheduling quantum. The merged weights are built in
memory; no merged checkpoint is distributed.

These sibling modules use direct imports. For library inspection or integration,
add this directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/runtime/hot_buffer:$PYTHONPATH"
```

In a fresh server-side process with an already loaded compatible model:

```python
from comem_peft import load_comem_peft
from hybrid_reader import HybridReader

model, _, targets = load_comem_peft(
    model, "/srv/encbank/adapters/my_peft_adapter", mode="merged_bf16"
)
reader = HybridReader(model, 21)
```

Do not attach the original LoRA branch again after merging. `safe_merge=True`
checks weight finiteness; it does not promise equality to unmerged FP32
arithmetic. Compatibility and merged modes are distinct experimental settings.

Before running `worker.py` or submitting jobs, follow the
[reproduction procedure](../../docs/REPRODUCIBILITY.md). The shipped plan is an
example, and genuine environment/numerical qualification receipts, source
manifests, external model artifacts, and task environments are required. The
source release does not bypass those checks.
