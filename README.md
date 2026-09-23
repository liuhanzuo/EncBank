# EncBank

**Under review at ICLR 2027.**

Research code for comprehension memory (CoMem), hidden-state readers, exact-prefix
Hot Buffers, LoRA distillation, and long-context / Terminal-Bench evaluation.
The Python package retains the name `comem` to preserve existing imports.

This release collects the implementation and experiment sources used through
September 23, 2026. It includes the 27B Transformers Hot24 experiment with
PEFT-merged LoRA, along with Dense baselines, reader studies, kernel pilots,
training code, qualification checks, and experiment controllers.

## Code map

| Location | Contents |
| --- | --- |
| [`comem/`](comem/) | Chunk writing, retrieval, resumed decoding, model registry, and MoE support |
| [`train/`](train/) | LoRA self-distillation for the base CoMem implementation |
| [`eval/`](eval/) | RULER, BABILong, LongBench, LongEval, and LoCoMo drivers |
| [`bench/`](bench/) | Dense / CoMem comparisons and correctness checks |
| [`runtime/hot_buffer/`](runtime/hot_buffer/) | 27B Hot24, batched hybrid attention, PEFT loading/merging, and Terminal-Bench controller sources |
| [`research/`](research/) | Experiment source snapshots, including server training, reader studies, Dense/k12/k48 evaluation, kernel pilots, and local orchestration |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Entry points for the experiment families |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Dependencies, configuration, and reproduction boundaries |
| [`docs/source_inventory.json`](docs/source_inventory.json) | Original and published source hashes, plus mappings for duplicate copies |

## Base package

Run training and model inference on a Linux GPU server with your own licensed
model checkpoint and datasets. Neither weights nor adapters are distributed here.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m eval.run --benchmark ruler --model /path/to/model --j auto \
    --lengths 8k,16k,32k --n 100 --selector bm25 --out /path/to/results

python -m train.distill --model /path/to/model --j 12 \
    --data /path/to/pg19_train.jsonl --out /path/to/adapter
```

See the [base implementation guide](docs/legacy_core.md) for its complete API
and benchmark flags. Its historical result statements describe that earlier
implementation, not the new Hot24 merged experiment.

## Hot Buffer and PEFT

The 27B runtime uses a depth-21 split, 512-token chunks, top-12 retrieval,
512-generated-token refreshes, and up to 24 exact-prefix Hot Buffer entries per
session. Its task admission and decode-batch ceilings are both 24; the independent
Terminal-Bench experiment selects 32 tasks. A concurrency ceiling does not imply
that every decode step has 24 active requests.

PEFT loading supports three explicit execution modes:

- `compat`: preserve the original FP32 LoRA arithmetic and rounding order;
- `native`: use PEFT's forward implementation;
- `merged_bf16`: merge once with `merge_and_unload(safe_merge=True)`.

The merged Hot24 arm uses `merged_bf16`. Merging changes numerical behavior and
is not claimed to be bitwise equivalent to the original adapter branch. See
[PEFT usage](docs/peft_zh.md), the [Hot Buffer implementation](docs/hot_buffer_zh.md),
and the [runtime setup notes](runtime/hot_buffer/README.md).

The new experiment was verified to have started real model requests and replies;
this source release does **not** claim a completed accuracy or speed result for
that experiment. Kernel pilots are separate experiments and are not a complete
vLLM serving implementation.

## Source checks

```bash
python tools/validate_release.py
```

This checks source hashes, the primary Python entry points, example configuration,
and forbidden artifact types. It does not run inference, submit jobs, or establish
GPU numerical equivalence. Historical snapshots include superseded experiments;
read their configuration and qualification scripts before adapting them.

Private deployment paths and addresses were replaced with example values such
as `/srv/encbank` and `gpu-node1`. Credentials, checkpoints, benchmark responses,
generated solutions, and compiler caches are excluded. Historical source
manifests refer to the original bytes; the release inventory separately records
the published bytes. See [third-party notices](THIRD_PARTY_NOTICES.md).
