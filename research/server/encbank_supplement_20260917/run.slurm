#!/bin/bash
#SBATCH --job-name=qencbank-agent27-dev6-codex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917/slurm-%j.err
set -euo pipefail
cd /srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917
export PYTHONPATH=/srv/encbank/encbank_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917/tmp TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917/triton CUDA_CACHE_PATH=/srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917/cuda_cache HF_HOME=/srv/encbank/qencbank_align_codex_20260911/encbank_supplement_20260917/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -X utf8 -u run_agent.py
