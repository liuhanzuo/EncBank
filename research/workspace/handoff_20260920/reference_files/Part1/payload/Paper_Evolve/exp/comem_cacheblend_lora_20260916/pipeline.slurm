#!/bin/bash
#SBATCH --job-name=midcache-cacheblend-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/srv/encbank/comem_cacheblend_lora_20260916/pipeline-%j.out
#SBATCH --error=/srv/encbank/comem_cacheblend_lora_20260916/pipeline-%j.err
set -euo pipefail
cd /srv/encbank/comem_cacheblend_lora_20260916
export PYTHONPATH="$PWD:/srv/encbank/comem_infra_recheck_20260912/deps:/srv/encbank/comem_followups_20260913/COMem"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TRITON_CACHE_DIR="$PWD/cache/triton-$SLURM_JOB_ID" CUDA_CACHE_PATH="$PWD/cache/cuda-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
PYTHON=/srv/encbank/Paper_Evolve/.venv/bin/python
"$PYTHON" -B test_trainable.py
if [[ -f training/last.pt ]]; then
  "$PYTHON" -B train_cacheblend.py --resume
else
  "$PYTHON" -B train_cacheblend.py
fi
"$PYTHON" -B evaluate.py
"$PYTHON" -B report.py
