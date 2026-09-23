#!/bin/bash
#SBATCH --job-name=midcache-hybrid-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/encbank/comem_new_backbones_20260915/smoke-%j.out
#SBATCH --error=/srv/encbank/comem_new_backbones_20260915/smoke-%j.err
set -euo pipefail
cd /srv/encbank/comem_new_backbones_20260915
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TRITON_CACHE_DIR="$PWD/cache/triton-smoke" CUDA_CACHE_PATH="$PWD/cache/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
/srv/encbank/Paper_Evolve/.venv/bin/python smoke.py --model /srv/encbank/models/Qwen3.5-9B-Base-text
