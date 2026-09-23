#!/bin/bash
#SBATCH --job-name=midcache-depth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/srv/encbank/comem_depth_sweep_20260915/depth-%A_%a.out
#SBATCH --error=/srv/encbank/comem_depth_sweep_20260915/depth-%A_%a.err
set -euo pipefail
cd /srv/encbank/comem_depth_sweep_20260915
export PYTHONPATH="/srv/encbank/comem_infra_recheck_20260912/deps:/srv/encbank/comem_followups_20260913/COMem:$PWD/pilot_support:$PWD"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TRITON_CACHE_DIR="$PWD/cache/triton-$SLURM_ARRAY_TASK_ID" CUDA_CACHE_PATH="$PWD/cache/cuda-$SLURM_ARRAY_TASK_ID"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
test -f plan.json
/srv/encbank/Paper_Evolve/.venv/bin/python -B run_depth.py --index "$SLURM_ARRAY_TASK_ID"
/srv/encbank/Paper_Evolve/.venv/bin/python -B verify_depth.py --index "$SLURM_ARRAY_TASK_ID" --job "$SLURM_JOB_ID"
