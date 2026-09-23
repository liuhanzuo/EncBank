#!/bin/bash
#SBATCH --job-name=cm-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --array=0-4%4
#SBATCH --output=logs/%A_%a.out
set -euo pipefail
cd /srv/encbank/comem_distillation_sweep_20260917
export PYTHONPATH="$PWD:/srv/encbank/comem_infra_recheck_20260912/deps:/srv/encbank/comem_followups_20260913/COMem"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TRITON_CACHE_DIR="$PWD/cache/triton-$SLURM_JOB_ID" CUDA_CACHE_PATH="$PWD/cache/cuda-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
/srv/encbank/Paper_Evolve/.venv/bin/python -u pipeline.py --arm "$SLURM_ARRAY_TASK_ID"
