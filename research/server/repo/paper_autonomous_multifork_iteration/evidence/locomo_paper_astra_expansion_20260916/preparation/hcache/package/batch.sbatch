#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --job-name=qcomem-locomo-hcache-j12-20260916-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --constraint=l20d
#SBATCH --exclude=gpu-node1,gpu-node2,gpu-node8
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/locomo_paper_astra_expansion_20260916/preparation/hcache/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/locomo_paper_astra_expansion_20260916/preparation/hcache/package/slurm-%j.err
set -euo pipefail
export QCOMEM_REPO_ROOT=/srv/encbank/qcomem_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/locomo_paper_hcache_20260916_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/locomo_paper_hcache_20260916_attempt1/xdg
export HF_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/locomo_paper_hcache_20260916_attempt1/hf
export TORCH_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/locomo_paper_hcache_20260916_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/locomo_paper_hcache_20260916_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QCOMEM_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/locomo_paper_astra_expansion_20260916/preparation/hcache/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/locomo_paper_astra_expansion_20260916/preparation/hcache/package/run_batch.py --plan /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/locomo_paper_astra_expansion_20260916/preparation/hcache/package/plan.json --expected-plan-sha256 717a8aa035327bb1e10e4a03314ca49d83a21cb5e770d12723a26767c6ac0c5d
