#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qcomem-lit-frozen-j12-multifieldqa_en-v1-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --nodelist=gpu-node1
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260913/frozen_j12_qa6_preparation_20260913/multifieldqa_en_v2/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260913/frozen_j12_qa6_preparation_20260913/multifieldqa_en_v2/package/slurm-%j.err
set -euo pipefail
export QCOMEM_REPO_ROOT=/srv/encbank/qcomem_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_qa6_multifieldqa_en_v1_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_qa6_multifieldqa_en_v1_attempt1/xdg
export HF_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_qa6_multifieldqa_en_v1_attempt1/hf
export TORCH_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_qa6_multifieldqa_en_v1_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_qa6_multifieldqa_en_v1_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QCOMEM_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260913/frozen_j12_qa6_preparation_20260913/multifieldqa_en_v2/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260913/frozen_j12_qa6_preparation_20260913/multifieldqa_en_v2/package/run_batch.py --plan /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260913/frozen_j12_qa6_preparation_20260913/multifieldqa_en_v2/package/plan.json --expected-plan-sha256 2e9f81e5f49c854615b33607f87a05b72d2244046190915ad4d76470aa63fa1f
