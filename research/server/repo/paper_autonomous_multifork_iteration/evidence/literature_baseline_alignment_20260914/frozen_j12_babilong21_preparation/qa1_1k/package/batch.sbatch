#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qencbank-lit-frozen-j12-qa1_1k100-v1-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --nodelist=gpu-node1
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_babilong21_preparation/qa1_1k/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_babilong21_preparation/qa1_1k/package/slurm-%j.err
set -euo pipefail
export QENCBANK_REPO_ROOT=/srv/encbank/qencbank_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/task_cache/literature_frozen_j12_qa1_1k_v1_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/literature_frozen_j12_qa1_1k_v1_attempt1/xdg
export HF_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/literature_frozen_j12_qa1_1k_v1_attempt1/hf
export TORCH_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/literature_frozen_j12_qa1_1k_v1_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/task_cache/literature_frozen_j12_qa1_1k_v1_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QENCBANK_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qencbank_runtime_20260911/python312/bin/python -B /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_babilong21_preparation/qa1_1k/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u -B /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_babilong21_preparation/qa1_1k/package/run_batch.py --plan /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_babilong21_preparation/qa1_1k/package/plan.json --expected-plan-sha256 ecd4a534a0d3f47df6fedd146d17b4b94a75f4608aec0aaab752b86e11cfe834
