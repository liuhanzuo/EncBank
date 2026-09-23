#!/bin/bash
#SBATCH --job-name=midcache-model-download
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/srv/encbank/comem_new_backbones_20260915/download-%A_%a.out
#SBATCH --error=/srv/encbank/comem_new_backbones_20260915/download-%A_%a.err
set -euo pipefail
cd /srv/encbank/comem_new_backbones_20260915
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
/srv/encbank/Paper_Evolve/.venv/bin/python download_models.py
