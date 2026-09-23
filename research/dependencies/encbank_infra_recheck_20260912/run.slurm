#!/bin/bash
#SBATCH --job-name=encbank-infra-128k-recheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/encbank/encbank_infra_recheck_20260912/slurm-%j.out
#SBATCH --error=/srv/encbank/encbank_infra_recheck_20260912/slurm-%j.err
set -euo pipefail
TASK=/srv/encbank/encbank_infra_recheck_20260912
cd "$TASK"
mkdir -p "$TASK/cache/tmp" "$TASK/cache/hf" "$TASK/cache/xdg" "$TASK/cache/torch" "$TASK/cache/triton" "$TASK/cache/cuda"
export TMPDIR="$TASK/cache/tmp" TMP="$TASK/cache/tmp" TEMP="$TASK/cache/tmp"
export HF_HOME="$TASK/cache/hf" XDG_CACHE_HOME="$TASK/cache/xdg" TORCH_HOME="$TASK/cache/torch"
export TRITON_CACHE_DIR="$TASK/cache/triton" CUDA_CACHE_PATH="$TASK/cache/cuda"
export PYTHONPYCACHEPREFIX="$TASK/cache/pycache" PYTHONPATH="$TASK/deps:$TASK/Encbank"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
PYTHON=/srv/encbank/Paper_Evolve/.venv/bin/python
MODEL=/srv/encbank/encbank_sparse_slurm_20260912/models/Qwen3-8B
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv > gpu_inventory.csv
"$PYTHON" bench_remote.py --model "$MODEL" --process 0 --smoke
for PROCESS in 1 2 3; do
  "$PYTHON" bench_remote.py --model "$MODEL" --process "$PROCESS"
done
