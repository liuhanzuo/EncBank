#!/bin/bash
#SBATCH --job-name=comem-followup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/srv/encbank/comem_followups_20260913/slurm-%A_%a.out
#SBATCH --error=/srv/encbank/comem_followups_20260913/slurm-%A_%a.err
set -euo pipefail
TASK=/srv/encbank/comem_followups_20260913
cd "$TASK"
export PYTHONPATH="/srv/encbank/comem_infra_recheck_20260912/deps:$TASK/COMem:$TASK"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HOME="$TASK/cache/hf" XDG_CACHE_HOME="$TASK/cache/xdg" TORCH_HOME="$TASK/cache/torch"
export TRITON_CACHE_DIR="$TASK/cache/triton" CUDA_CACHE_PATH="$TASK/cache/cuda"
export PYTHONPYCACHEPREFIX="$TASK/cache/pycache" TMPDIR="$TASK/cache/tmp"
mkdir -p "$TMPDIR"
PYTHON=/srv/encbank/Paper_Evolve/.venv/bin/python
MODEL=/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B
ADAPTER=/srv/encbank/comem_infra_recheck_20260912/adapter
MODE="${1:?diagnostic or kv}"
PHASE="${2:-smoke}"
if [[ "$MODE" == diagnostic ]]; then
  if [[ "$PHASE" == smoke ]]; then
    "$PYTHON" -B run_distillation.py --model "$MODEL" --adapter "$ADAPTER" --smoke
  else
    "$PYTHON" -B run_distillation.py --model "$MODEL" --adapter "$ADAPTER"
  fi
elif [[ "$MODE" == kv ]]; then
  if [[ "$PHASE" == smoke ]]; then
    "$PYTHON" -B run_kv_quality.py --model "$MODEL" --adapter "$ADAPTER" --smoke
  else
    "$PYTHON" -B run_kv_quality.py --model "$MODEL" --adapter "$ADAPTER" --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
  fi
elif [[ "$MODE" == clean ]]; then
  if [[ "$PHASE" == smoke ]]; then
    "$PYTHON" -B run_clean.py --model "$MODEL" --adapter "$ADAPTER" --smoke
  else
    "$PYTHON" -B run_clean.py --model "$MODEL" --adapter "$ADAPTER" --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
  fi
fi
