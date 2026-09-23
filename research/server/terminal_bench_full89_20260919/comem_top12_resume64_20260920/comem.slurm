#!/bin/bash
#SBATCH --job-name=qcomem-tb-k12-live-codex
#SBATCH --partition=gpu
#SBATCH --exclude=gpu-node4
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920/slurm-%j.err
set -euo pipefail
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920
export CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONPATH=/srv/encbank/comem_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920/tmp_dense TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920/triton_dense CUDA_CACHE_PATH=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920/cuda_dense HF_HOME=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/comem_top12_resume64_20260920/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u storage_preflight.py
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u holder.py
