#!/bin/bash
#SBATCH --job-name=midcache-release-gpu-slot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/srv/encbank/comem_cacheblend_lora_20260916/release-%j.out
#SBATCH --error=/srv/encbank/comem_cacheblend_lora_20260916/release-%j.err
set -euo pipefail
cd /srv/encbank/comem_cacheblend_lora_20260916
/srv/encbank/Paper_Evolve/.venv/bin/python -B release_slot.py
