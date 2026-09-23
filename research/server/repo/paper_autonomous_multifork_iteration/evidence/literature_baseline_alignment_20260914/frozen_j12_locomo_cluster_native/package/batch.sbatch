#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --job-name=qcomem-lit-frozen-j12-locomo1986-v1-cluster-native-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --constraint=l20d
#SBATCH --exclude=gpu-node2,gpu-node8
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_locomo_cluster_native/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_locomo_cluster_native/package/slurm-%j.err
set -euo pipefail
export QCOMEM_REPO_ROOT=/srv/encbank/qcomem_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_locomo1986_v1_attempt1_cluster_native/tmp
export XDG_CACHE_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_locomo1986_v1_attempt1_cluster_native/xdg
export HF_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_locomo1986_v1_attempt1_cluster_native/hf
export TORCH_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_locomo1986_v1_attempt1_cluster_native/torch
export TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/literature_frozen_j12_locomo1986_v1_attempt1_cluster_native/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QCOMEM_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_locomo_cluster_native/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_locomo_cluster_native/package/run_batch.py --plan /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/literature_baseline_alignment_20260914/frozen_j12_locomo_cluster_native/package/plan.json --expected-plan-sha256 9408b7aa0e81092f833a5dd64768cf06f209ab20fc857364eeac7b91432305e7
