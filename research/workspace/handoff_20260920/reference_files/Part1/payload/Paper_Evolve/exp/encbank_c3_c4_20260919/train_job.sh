#!/bin/bash
set -euo pipefail
cd /srv/encbank/encbank_c3_c4_20260919
export PYTHONPATH="$PWD:/srv/encbank/encbank_infra_recheck_20260912/deps:/srv/encbank/encbank_followups_20260913/Encbank:/srv/encbank/encbank_frozen_j12_20260912"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TMPDIR="$PWD/cache/tmp-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$PWD/cache/triton-${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$PWD/cache/cuda-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u train.py --j "$1" --seed 42
