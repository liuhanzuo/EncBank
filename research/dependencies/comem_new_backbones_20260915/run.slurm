#!/bin/bash
#SBATCH --job-name=midcache-new-backbones
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/srv/encbank/comem_new_backbones_20260915/pilot-%A_%a.out
#SBATCH --error=/srv/encbank/comem_new_backbones_20260915/pilot-%A_%a.err
set -euo pipefail
cd /srv/encbank/comem_new_backbones_20260915
export PYTHONPATH="/srv/encbank/comem_infra_recheck_20260912/deps:/srv/encbank/comem_followups_20260913/COMem:$PWD"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TRITON_CACHE_DIR="$PWD/cache/triton-$SLURM_ARRAY_TASK_ID" CUDA_CACHE_PATH="$PWD/cache/cuda-$SLURM_ARRAY_TASK_ID"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
MODELS=(Qwen3.5-9B Qwen3.8-27B)
MODEL="$PWD/models/${MODELS[$SLURM_ARRAY_TASK_ID]}"
test -f "$MODEL/download_complete.json"
test -f smoke_complete.json
/srv/encbank/Paper_Evolve/.venv/bin/python -B run_pilot.py --model "$MODEL" --steps 200 --n 10
