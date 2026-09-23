# MidCache: newer-backbone pilot (September 15, 2026)

User requested newer models including Qwen3.8. This is a separate exploratory
extension. Existing manuscript numbers, adapters, and jobs are untouched.

## Models and design

- Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a: 32 blocks, j=11.
- Qwen/Qwen3.8-27B, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0: 64 blocks, j=21.
- Both are official post-trained models; native chat template, thinking disabled.
- Text inputs only. Both use Gated DeltaNet / attention hybrid blocks.
- Fixed split round(0.33 L), chosen before seeing scores; no benchmark tuning.
- Residual-only independent chunk Write, c=512, iterative BM25 top-12, causal
  upper reader. The full rendered prompt's last chunk is the query, matching
  the current paper's boundary convention. Native BOS or EOS is the sink.
- Separate transient caches retain query-side lower state and assembled-pack
  upper state, including each DeltaNet layer's recurrent/convolution state.
- Native stock forward, continuous split, and cached/recomputed decode are
  checked before evaluating the interface. Zero-adapter identity, suffix
  gradients and frozen backbone were checked on the available Qwen3.5-9B-Base
  text tower; this infrastructure smoke is not a benchmark result.

## Bounded first experiment

Each model receives its own suffix LoRA (all suffix linear projections, rank=32,
alpha=32), trained on PG-19 train text only for 200 steps, 2,048-token windows,
512-token query segments, teacher top-64 bidirectional KL (lambda=0.6), one
sample per step. This is a short pilot, not the principal paper's full recipe
or a claim that the Qwen3-8B adapter transfers to another model.

After training, evaluate four paired arms on the exact same selected chunks:
base replay, cache without LoRA, replay with the shared new LoRA, and cache with
the new LoRA. RULER single-2 and multikey-1 plus LongEval at 8k/16k/32k each have
10 prespecified examples per cell; Qasper uses 30 evenly spaced official IDs.
Each model produces 480 predictions over 120 inputs. Numeric scores are only
aggregated after complete cell coverage; metrics retain their task definitions.
No infrastructure latency/speedup is inferred from shared Slurm execution.

## Execution and outputs

Remote directory: /srv/encbank/comem_new_backbones_20260915
Use the gpu partition, one scheduler-allocated GPU per model. The scheduler's
name is nvidia_l20d; the earlier smoke reported 267.69 GiB on this user-designated
B300 pool. GPU identity and host are saved in every protocol record.

- download.slurm / download_models.py fetch official weights, pinned revisions.
- smoke.slurm / smoke.py check the execution interface and training gradients.
- run.slurm / run_pilot.py train and evaluate both models independently.
- results/<model>/job_<SLURM_JOB_ID>/protocol.json, correctness*.json, training.jsonl,
  adapter-*.pt, samples.jsonl.gz, predictions.jsonl, progress.json,
  summary.json or failure.json retain the exact run and its status.

`verify_results.py <run directory>` independently decodes and rescores every
completed prediction on CPU, checks complete support, and computes paired
bootstrap intervals per task/length. No mixed-metric overall score is used.
Run local `collect_status.py` to refresh STATUS.json from SSH and Slurm.

Downloads 33877_0/1 and interface smoke 33885 completed successfully. Initial
pilot 33906_0/1 stopped in sample serialization before loading a model: the
new tokenizer returned BatchEncoding by default. Explicit list extraction and
the selector's tensor input were corrected; check_samples.py then passed for
12 samples of each model. Retry array 33916 preserves these failed attempts
and writes separate per-job output directories. See STATUS.json for the latest
verified scheduler/progress state, not the historical job list here.

## Earlier experiments found

- COMem/paper/sections/tab_scale.tex includes exploratory Qwen3 0.6B, 1.7B,
  4B, 8B, 14B and 32B rows, including partial adapted rows at 1.7B/4B/14B;
  splits and support vary. Qwen3-30B-A3B has separate MoE experiments.
- The current paper reports Qwen/OLMo layer probes and Llama SST2 probes.
- exp/comem_v2_benchmarks_20260908/non_qwen_formal_20260909_0631.md contains
  completed SmolLM2-1.7B-Instruct results (1,750 predictions, five arms), testing
  a different revised reader; do not merge those arms into this paper's primary
  method without explicitly explaining the interface change.
- exp/results/s27_deployable_qwen3-1.7b_pg19.json and its WikiText companion
  contain additional Qwen3-1.7B mechanism diagnostics.

Official model cards:
https://huggingface.co/Qwen/Qwen3.5-9B
https://huggingface.co/Qwen/Qwen3.8-27B
