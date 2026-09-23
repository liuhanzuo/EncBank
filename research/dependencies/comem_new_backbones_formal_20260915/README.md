# MidCache full newer-backbone evaluation

**Current instruction (September16):** use at most four GPUs and resume LoCoMo,
with **gpt-6-astra as Judge**, explicitly confirmed by the user. The existing
four-benchmark array45014 now has throttle4; LoCoMo array59046 is queued after
its successful completion and also has throttle4, so combined GPU use never
exceeds four. Both final4000-step adapters are already complete.

The original four-benchmark scope/results remain unchanged. Its historical
scope ID contains "locomo-deferred", but that describes those saved outputs,
not the current overall task. LoCoMo uses locomo_scope.json and results_locomo:
1,986 inputs/model, seven arms,27,804 additional records (101,304 total).
The model answers come from the seven evaluated arms; Astra only judges them.

The official CLI returned a valid Astra label and passed all8 fixed semantic/
format calibration cases. This sanity check is not human-agreement validation.
locomo_judge_worker.py runs locally, collecting newly verified LoCoMo shards
every60 seconds and scoring independent blinded prompts with up to8 concurrent
CPU client processes. Exact identical question/gold/candidate/category/protocol
inputs share a cached judgment. Category5 retains the original abstention rule;
OOM and failed/malformed Judge calls never become incorrect-answer scores.
See judge_gpt6_astra/formal_new_models/worker_status.json, progress.json and
status.json. Model alias gpt-6-astra is fixed; underlying dated snapshot unverified.

Authorized September 15, 2026 after the completed frozen depth screen. This is
the full quality extension requested by the user, not another depth-training sweep.

- Qwen3.5-9B: 32 text blocks, fixed j=6 (transition candidate with some frozen loss).
- Qwen3.8-27B: 64 text blocks, fixed j=21 (observed LongEval plateau edge).
- Text only; official pinned checkpoints already downloaded. Hybrid text blocks
  preserve native Gated DeltaNet and attention state. No backbone finetuning.
- Fresh suffix LoRA per model: 4,000 PG-19 steps, 4,096 tokens/step, rank/alpha32,
  top64 bidirectional KL(.6), AdamW(.9,.95), lr1e-4, 50-step warmup, cosine decay,
  clipping1, seed42. Same 64-book source and sequential cyclic recipe as the paper;
  tokenizer-specific corpus lengths differ. All suffix linear projections are
  adapted, including the hybrid model's DeltaNet projections. Fixed final checkpoint.
  Do not resume the cancelled 200-step multi-depth training array 34420.

## Full benchmark support

| Benchmark | Task / length support | Samples per model | Scoring |
|---|---|---:|---|
| RULER | single-2, multikey-1, variable-tracking; 8/16/32/64/128k | 1,500 | 15-cell macro |
| LongEval | 8/16/32/64/128k | 500 | Five-length macro |
| LongBench | NarrativeQA, Qasper, HotpotQA, 2WikiMQA, MultiFieldQA-en, Musique | 1,150 | Six-dataset F1 macro |
| BABILong | qa1,qa2,qa5; 0/1/2/4/8/16/32k | 2,100 | 21-cell macro |
| LoCoMo | All ten conversations, five categories | 1,986 | Fixed semantic Judge plus local abstention |

The already running input preparation keeps all7,236 inputs/model. The active
GPU evaluation filters those saved inputs to the first four benchmarks:5,250
inputs/model,7 compatible arms,36,750 output records/model,73,500 across both
models. LoCoMo uses the same1,986 prepared inputs in a separate subsequent
array. Four shards/model in each scope; at most four GPUs concurrently overall.
All arms within a model use identical saved examples; paired cache/replay arms
also share exact selected chunks, ordering, sink and query boundary.

## Baseline implementation and availability

The executable arms are MidCache, MidCache without LoRA, base selected-text replay,
shared-adapter selected-text replay, KV-Direct, StreamingLLM and HCache-style.
KV-Direct runs the native full source, without truncation, adaptation, or RoPE
extension. StreamingLLM is the paper's sink/recent-token recomputation reference
(4+6,653 positions), not a claim of native StreamingLLM serving performance.
HCache-style independently writes all source chunks at the fixed model split and
reads them without retrieval or adaptation; this is a mechanism reference, not a
ported native HCache system or the old Qwen3 checkpoint.

InfLLM is explicitly unsupported by the locally available upstream patcher for
these hybrid blocks (its patcher only supports Llama/Mistral/Qwen2). MemoryLLM has
its own learned-memory backbone and no interchangeable Qwen3.5/3.8 checkpoint.
Neither is silently replaced or assigned old scores as fresh new-model results.
Their old-backbone rows can remain references in the existing paper; the new
same-backbone table must state N/A for these two implementations.

## Protocol details and limits

Native chat with thinking disabled follows the newer-model depth screen. Thus
these are a separate cohort from the original Qwen3-8B plain-text table. All
benchmark tasks and full sample counts match Table1. LongEval uses 48 generated
tokens for every arm here: the earlier no-LoRA limit, applied equally to remove
the original 16-versus-48 budget asymmetry. Other limits match the paper:
RULER48/VT60, LongBench32/64/128, BABILong20, LoCoMo48. Chunks512, iterative BM25
top12, hop4, auto rounds, one BOS-or-EOS sink, trailing prompt chunk as query.

The depth screen was exploratory and included some of these natural QA IDs.
Do not claim an independent held-out depth optimum. Never choose depths or
checkpoints from these full evaluation scores. OOM records retain null scores;
never truncate inputs, silently reduce output budgets, or score OOM as an answer.
Scoring is only aggregated on complete support. LoCoMo lexical metrics do not
replace its semantic Judge. Judge availability is tracked separately.

The user now selected exact model `gpt-6-astra`; see ASTRA_JUDGE.md and
judge_protocol.json. Its actual official CLI connection and eight-case
calibration succeeded. Original Qwen3-8B prediction bundles still need exact
source recovery before paper-wide rejudging. Neither old GPT-4o scores nor
unrelated experiments may be relabeled as Astra results. Credentials remain
outside experiment artifacts. Earlier failed gpt-6 attempts are preserved.

## Execution

Remote: `/srv/encbank/comem_new_backbones_formal_20260915` on gpu-node1.
Training array 43946, CPU data array 45013 and dependent evaluation array 45014
were submitted successfully. `launch.json` contains their job receipts. Data preparation uses CPU-only Slurm jobs. Evaluation
depends on successful training and sample preparation, and runs 8 one-GPU shards
with concurrency4. LoCoMo59046 follows45014 at concurrency4. `verify.py`
independently decodes/rescores every completed shard.
`collect.py` refreshes local STATUS.json. Training checkpoints preserve optimizer
and RNG for explicit recovery. Never duplicate a completed or active shard.

Quality only: Slurm wall times and generation timers are not infrastructure
measurements. Shared-cluster runs do not update paper speedups. New results are
not inserted into the paper until complete and verified.
