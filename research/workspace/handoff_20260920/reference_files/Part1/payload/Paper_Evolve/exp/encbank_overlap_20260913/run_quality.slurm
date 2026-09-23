#!/bin/bash
#SBATCH --job-name=encbank-overlap-quality
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/srv/encbank/encbank_overlap_20260913/slurm-%A_%a.out
#SBATCH --error=/srv/encbank/encbank_overlap_20260913/slurm-%A_%a.err
set -euo pipefail
TASK=/srv/encbank/encbank_overlap_20260913
cd "$TASK"
export PYTHONPATH="/srv/encbank/encbank_infra_recheck_20260912/deps:$TASK/Encbank:$TASK"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HOME="$TASK/cache/hf" XDG_CACHE_HOME="$TASK/cache/xdg" TORCH_HOME="$TASK/cache/torch"
export TRITON_CACHE_DIR="$TASK/cache/triton" CUDA_CACHE_PATH="$TASK/cache/cuda"
export PYTHONPYCACHEPREFIX="$TASK/cache/pycache" TMPDIR="$TASK/cache/tmp"
mkdir -p "$TMPDIR"
PYTHON=/srv/encbank/Paper_Evolve/.venv/bin/python
MODEL=/srv/encbank/encbank_sparse_slurm_20260912/models/Qwen3-8B
ADAPTER=/srv/encbank/encbank_infra_recheck_20260912/adapter
if [[ "${1:-}" == "smoke" ]]; then
  "$PYTHON" -B run_quality.py --model "$MODEL" --adapter "$ADAPTER" --smoke
else
  "$PYTHON" -B run_quality.py --model "$MODEL" --adapter "$ADAPTER" --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
fi
