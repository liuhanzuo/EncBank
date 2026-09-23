#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=qcomem-align-codex
#SBATCH --dependency=singleton
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_kivi_remote_preparation_20260912/gpu_qualification_preparation/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_kivi_remote_preparation_20260912/gpu_qualification_preparation/slurm-%j.err
set -euo pipefail
export QCOMEM_REPO_ROOT=/srv/encbank/qcomem_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/xdg
export HF_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/hf
export TORCH_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export TORCH_EXTENSIONS_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/torch_extensions
export CUDA_CACHE_PATH=/srv/encbank/qcomem_align_codex_20260911/task_cache/kivi_sm89_qualification_attempt1/cuda
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QCOMEM_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_kivi_remote_preparation_20260912/gpu_qualification_preparation/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_kivi_remote_preparation_20260912/gpu_qualification_preparation/run_batch.py --plan /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_kivi_remote_preparation_20260912/gpu_qualification_preparation/plan.json --expected-plan-sha256 35ee63a69cdb3cd1da43cd1352c36fd92efa314231d2bc14e0c804b4c83ad373
