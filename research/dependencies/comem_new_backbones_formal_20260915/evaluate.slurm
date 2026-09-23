#!/bin/bash
#SBATCH --job-name=midcache-formal-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/srv/encbank/comem_new_backbones_formal_20260915/eval-%A_%a.out
#SBATCH --error=/srv/encbank/comem_new_backbones_formal_20260915/eval-%A_%a.err
set -euo pipefail
cd /srv/encbank/comem_new_backbones_formal_20260915
source environment.sh
/srv/encbank/Paper_Evolve/.venv/bin/python -B evaluate.py --model-index "$((SLURM_ARRAY_TASK_ID % 2))" --shard "$((SLURM_ARRAY_TASK_ID / 2))"
/srv/encbank/Paper_Evolve/.venv/bin/python -B verify.py --model-index "$((SLURM_ARRAY_TASK_ID % 2))" --shard "$((SLURM_ARRAY_TASK_ID / 2))"
