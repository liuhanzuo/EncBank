#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=qcomem-kvdirect-admission-recovery-qa1_1k-v1-codex
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --constraint=l20d
#SBATCH --exclude=gpu-node1,gpu-node2,gpu-node8
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/kv_admission_recovery_20260914_2250/cells/qa1_1k/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/kv_admission_recovery_20260914_2250/cells/qa1_1k/package/slurm-%j.err
set -euo pipefail
export QCOMEM_REPO_ROOT=/srv/encbank/qcomem_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/kv_admission_recovery_20260914_2250_qa1_1k_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/kv_admission_recovery_20260914_2250_qa1_1k_attempt1/xdg
export HF_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/kv_admission_recovery_20260914_2250_qa1_1k_attempt1/hf
export TORCH_HOME=/srv/encbank/qcomem_align_codex_20260911/task_cache/kv_admission_recovery_20260914_2250_qa1_1k_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/task_cache/kv_admission_recovery_20260914_2250_qa1_1k_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QCOMEM_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/kv_admission_recovery_20260914_2250/cells/qa1_1k/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u -B /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/kv_admission_recovery_20260914_2250/cells/qa1_1k/package/run_batch.py --plan /srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/kv_admission_recovery_20260914_2250/cells/qa1_1k/package/plan.json --expected-plan-sha256 c9d49e9c5318adb93925444467b260b146f62e8c0ee1f28bf00a983269fdd6aa
