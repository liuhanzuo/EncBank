#!/bin/bash
#SBATCH --job-name=qencbank-tb-container-candidate
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/srv/encbank/qencbank_runtime_20260911/server_control_20260920/candidate-%j.log
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=/srv/encbank/qencbank_runtime_20260911/server_control_20260920
export TMPDIR="$ROOT/tmp" APPTAINER_TMPDIR="$ROOT/tmp" APPTAINER_CACHEDIR="$ROOT/apptainer_cache"
export SINGULARITY_CACHEDIR="$ROOT/apptainer_cache" LITELLM_LOCAL_MODEL_COST_MAP=True
"$ROOT/harbor_env/bin/python" "$ROOT/container_probe.py" --candidate --task cancel-async-tasks --output "$ROOT/candidate-$SLURM_JOB_ID-python"
"$ROOT/harbor_env/bin/python" "$ROOT/container_probe.py" --candidate --task adaptive-rejection-sampler --output "$ROOT/candidate-$SLURM_JOB_ID-ubuntu"
