#!/bin/bash
#SBATCH --job-name=qencbank-tb-encbank-codex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918/slurm-%j.err
set -euo pipefail
cd /srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918
export CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONPATH=/srv/encbank/encbank_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918/tmp_encbank TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918/triton_encbank CUDA_CACHE_PATH=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918/cuda_encbank HF_HOME=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u holder.py encbank
