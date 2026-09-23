#!/bin/bash
#SBATCH --job-name=qencbank-tb-no-docker-check
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/srv/encbank/qencbank_runtime_20260911/server_control_20260920/managed-integration-%j.log
set -euo pipefail
umask 077
ROOT=/srv/encbank/qencbank_runtime_20260911/server_control_20260920
export TMPDIR="$ROOT/tmp" APPTAINER_TMPDIR="$ROOT/tmp" APPTAINER_CACHEDIR="$ROOT/apptainer_cache"
export SINGULARITY_CACHEDIR="$ROOT/apptainer_cache" LITELLM_LOCAL_MODEL_COST_MAP=True
"$ROOT/harbor_env/bin/python" "$ROOT/managed_integration_probe.py"
