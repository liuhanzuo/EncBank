#!/bin/bash
set -euo pipefail
cd /srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918
export PYTHONPATH="$PWD:/srv/encbank/comem_infra_recheck_20260912/deps:/srv/encbank/comem_followups_20260913/COMem:/srv/encbank/comem_frozen_j12_20260912"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TMPDIR="$PWD/cache/tmp-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$PWD/cache/triton-${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$PWD/cache/cuda-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec /srv/encbank/Paper_Evolve/.venv/bin/python -u -B control/recovery_worker.py "$1"
