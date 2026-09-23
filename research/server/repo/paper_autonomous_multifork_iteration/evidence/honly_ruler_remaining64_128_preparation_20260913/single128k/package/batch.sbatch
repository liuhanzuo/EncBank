#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qcomem-ruler-single128k100-fp16-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --nodelist=gpu-node1
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_ruler_remaining64_128_preparation_20260913/single128k/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_ruler_remaining64_128_preparation_20260913/single128k/package/slurm-%j.err
set -euo pipefail
export QCOMEM_REPO_ROOT=/srv/encbank/qcomem_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_single128k100_fp16_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_single128k100_fp16_attempt1/xdg
export HF_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_single128k100_fp16_attempt1/hf
export TORCH_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_single128k100_fp16_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_single128k100_fp16_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QCOMEM_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_ruler_remaining64_128_preparation_20260913/single128k/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_ruler_remaining64_128_preparation_20260913/single128k/package/run_batch.py --plan /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_ruler_remaining64_128_preparation_20260913/single128k/package/plan.json --expected-plan-sha256 dae7e91c383ec4871b80903d41158eced5fdb6145a1f78360fe1d44f580ff436
