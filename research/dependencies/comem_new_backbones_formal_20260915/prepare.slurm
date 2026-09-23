#!/bin/bash
#SBATCH --job-name=midcache-formal-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/srv/encbank/comem_new_backbones_formal_20260915/data-%A_%a.out
#SBATCH --error=/srv/encbank/comem_new_backbones_formal_20260915/data-%A_%a.err
set -euo pipefail
cd /srv/encbank/comem_new_backbones_formal_20260915
source environment.sh
/srv/encbank/Paper_Evolve/.venv/bin/python -B prepare_samples.py --model-index "$SLURM_ARRAY_TASK_ID"
