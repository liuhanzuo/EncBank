#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=qencbank-cluster-frozen-j12-qa5_4k100-v1-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --constraint=l20d
#SBATCH --exclude=gpu-node2,gpu-node8
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/frozen_babilong21_cluster_native_20260914/cells/qa5_4k/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/frozen_babilong21_cluster_native_20260914/cells/qa5_4k/package/slurm-%j.err
set -euo pipefail
export QENCBANK_REPO_ROOT=/srv/encbank/qencbank_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/task_cache/cluster_native_frozen_j12_qa5_4k_v1_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/cluster_native_frozen_j12_qa5_4k_v1_attempt1/xdg
export HF_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/cluster_native_frozen_j12_qa5_4k_v1_attempt1/hf
export TORCH_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/cluster_native_frozen_j12_qa5_4k_v1_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/task_cache/cluster_native_frozen_j12_qa5_4k_v1_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QENCBANK_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qencbank_runtime_20260911/python312/bin/python -B /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/frozen_babilong21_cluster_native_20260914/cells/qa5_4k/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u -B /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/frozen_babilong21_cluster_native_20260914/cells/qa5_4k/package/run_batch.py --plan /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/frozen_babilong21_cluster_native_20260914/cells/qa5_4k/package/plan.json --expected-plan-sha256 3f1b774eadbe3268dbd3c90490a33f3e1fb640884e8573e6cb7d7227bd6d136e
