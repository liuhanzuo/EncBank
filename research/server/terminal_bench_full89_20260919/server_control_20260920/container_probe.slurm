#!/bin/bash
#SBATCH --job-name=qcomem-tb-container-check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/container-check-%j.log
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=/srv/encbank/qcomem_runtime_20260911/server_control_20260920
export TMPDIR="$ROOT/tmp"
export APPTAINER_TMPDIR="$ROOT/tmp"
export APPTAINER_CACHEDIR="$ROOT/apptainer_cache"
export SINGULARITY_CACHEDIR="$ROOT/apptainer_cache"
export LITELLM_LOCAL_MODEL_COST_MAP=True
hostname
id
docker version --format '{{json .Server}}' || true
dockerd-rootless-setuptool.sh check || true
"$ROOT/harbor_env/bin/python" "$ROOT/container_probe.py" --task cancel-async-tasks --output "$ROOT/qualification-$SLURM_JOB_ID-python"
