#!/bin/bash
#SBATCH --job-name=midcache-formal-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/srv/encbank/encbank_new_backbones_formal_20260915/train-%A_%a.out
#SBATCH --error=/srv/encbank/encbank_new_backbones_formal_20260915/train-%A_%a.err
set -euo pipefail
cd /srv/encbank/encbank_new_backbones_formal_20260915
source environment.sh
/srv/encbank/Paper_Evolve/.venv/bin/python -B train_formal.py --model-index "$SLURM_ARRAY_TASK_ID"
