#!/bin/bash
#SBATCH --job-name=qencbank-tb-managed-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/srv/encbank/qencbank_runtime_20260911/server_control_20260920/managed-probe-%j.log
set -euo pipefail
umask 077
ROOT=/srv/encbank/qencbank_runtime_20260911/server_control_20260920
export TMPDIR="$ROOT/tmp" APPTAINER_TMPDIR="$ROOT/tmp" APPTAINER_CACHEDIR="$ROOT/apptainer_cache"
export SINGULARITY_CACHEDIR="$ROOT/apptainer_cache" LITELLM_LOCAL_MODEL_COST_MAP=True
"$ROOT/harbor_env/bin/python" "$ROOT/container_probe.py" --managed --task cancel-async-tasks --output "$ROOT/managed-$SLURM_JOB_ID-python"
"$ROOT/harbor_env/bin/python" "$ROOT/container_probe.py" --managed --task adaptive-rejection-sampler --output "$ROOT/managed-$SLURM_JOB_ID-ubuntu"
