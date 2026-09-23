# Reproduction notes

## Two implementation layers

The root `encbank`, `train`, `eval`, and `bench` packages contain the base decoder
implementation. The 27B hybrid-attention runtime is a separate implementation
under `runtime/hot_buffer`; it depends on Qwen3.5-family full-attention and
DeltaNet cache internals. A local experiment label containing `Qwen3.8-27B`
is retained in source metadata; consult the checkpoint configuration and pinned
revision rather than treating that label as a new model architecture.

The observed hybrid-runtime versions were Python 3.12, PyTorch 2.14.0+cu130,
Transformers 5.16.1, and PEFT 0.20.0. These describe the research environment;
CUDA wheel availability and driver compatibility depend on the deployment.
The general requirements file is not a complete lockfile for every archived
experiment. Terminal-Bench also requires Harbor/Terminus, the benchmark tasks,
and the configured Docker or Apptainer environment backend. Slurm controllers
require Linux, Slurm, and the relevant user-service/container configuration.

## Configuring snapshots

1. Install the model, original adapter, task definitions, and datasets on the
   server. The source inventory contains no model or adapter tensors.
2. Copy the chosen experiment into a **new** run directory. Treat archived
   attempts as references. Replace `/srv/encbank`, example GPU node aliases,
   Python interpreter paths, output paths, and optional proxy settings for your
   deployment. Preserve numerical, sampling, task-resource, and scoring settings
   when making a controlled comparison.
3. For the public Hot24 entry point, start from `plan.example.json`. Its paths
   are examples and some inherited metadata describes older experiments; the
   effective worker uses `task_concurrency=24` and `decode_batch_size=24`.
   Resolve the original checkpoint identity and PEFT conversion receipt against
   your actual artifacts. `encbank_peft.export_adapter` writes standard PEFT files
   and records exact tensor-roundtrip hashes.
4. Generate fresh source/task manifests for the configured copy. Original
   snapshot manifests must not be used to claim that normalized publication
   files are byte-identical to production. The release inventory records both
   hash domains explicitly.
5. Run the real environment, identity, and numerical qualification programs on
   the allocated host. Produce genuine qualification/parent-exit receipts.
   `worker.py` deliberately requires these records before accepting requests;
   the repository does not ship fabricated passing receipts or disable gates.
6. Submit one new controller for each intended task group. Confirm actual task
   launches, request/reply progress, and the model process on its allocated GPU.

The archived staging and qualification implementations are linked from
`EXPERIMENTS.md`; the code is a research source release, not a cluster-independent
one-command provisioning system.

## Comparing runs

Keep model revision, adapter identity, tokenizer, prompt, sampling, task set,
per-task resources, and hardware fixed. Report prefill, decode, refresh/rebuild,
environment, verification, and total wall-clock time separately. Record actual
batch sizes, generated tokens, peak allocated/reserved memory, and GPU identity.
Fixed-length microbenchmarks are not Terminal-Bench scores or production limits.

The unlimited-time protocol removes configured generation, task, request, and
scoring deadlines. Native context and physical memory limits remain. The final
full89 scoring policy counts native-context exhaustion as a distinct failure/0
while preserving the raw exception and evidence. Infrastructure failures and
incomplete evidence are not silently converted to normal quality failures.
Some earlier archived plan comments predate this policy.

Validate response identity and SHA, unmodified H, session release, verifier
output, container shutdown, and actual parent waits before closing an attempt.
Do not pool historical Docker results, Apptainer attempts, or new Hot24 merged
results without explicitly defining the comparison protocol.

## Publication transformations

Only publication copies were changed. Live source and jobs were left untouched.
Machine-specific paths, node aliases, and private-network addresses are replaced
with examples. Project names, Python identifiers, filenames, and origin path
labels use Encbank throughout this publication checkout. The original source
hashes still refer to the unchanged research artifacts; the published hashes
cover the renamed files. Source hashes are retained for traceability; no passing scientific
result is inferred from a static source check. The inventory maps local duplicate
copies to their published equivalents and records which dependency/cache
categories were omitted.
