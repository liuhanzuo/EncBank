#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=liuhanzuo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --job-name=qencbank-align-codex-locomo-full6-36h
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --nodelist=gpu-node1
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_locomo_full_prediction_36h_20260913/package/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_locomo_full_prediction_36h_20260913/package/slurm-%j.err
set -euo pipefail
export QENCBANK_REPO_ROOT=/srv/encbank/qencbank_align_codex_20260911/repo
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/task_cache/locomo_full1986_six_36h_attempt1/tmp
export XDG_CACHE_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/locomo_full1986_six_36h_attempt1/xdg
export HF_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/locomo_full1986_six_36h_attempt1/hf
export TORCH_HOME=/srv/encbank/qencbank_align_codex_20260911/task_cache/locomo_full1986_six_36h_attempt1/torch
export TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/task_cache/locomo_full1986_six_36h_attempt1/triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
export PYTHONPATH=
# Preserve Slurm CUDA_VISIBLE_DEVICES; the native guard verifies its single physical UUID.
cd "$QENCBANK_REPO_ROOT"
on_exit() {
  code=$?
  trap - EXIT
  /srv/encbank/qencbank_runtime_20260911/python312/bin/python -B /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_locomo_full_prediction_36h_20260913/package/record_shell_exit.py "$code" || true
  exit "$code"
}
trap on_exit EXIT
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u -B /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_locomo_full_prediction_36h_20260913/package/run_batch.py --plan /srv/encbank/qencbank_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence/honly_locomo_full_prediction_36h_20260913/package/plan.json --expected-plan-sha256 270bfba1b259c6e2ca9a61795961ca95e0dd0d247791a7f0a2d0ba9bdc31e755
