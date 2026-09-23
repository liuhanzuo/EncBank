#!/bin/bash
#SBATCH --job-name=qcomem-p8-recovery-qualification
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel8_recovery_r2_20260921/qualification-%j.log
set -euo pipefail
umask 077
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel8_recovery_r2_20260921
export PYTHONPATH=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel8_recovery_r2_20260921 LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
export TMPDIR=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/dense_parallel8_recovery_r2_20260921/tmp APPTAINER_CACHEDIR=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/apptainer_cache APPTAINER_TMPDIR=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/dense_parallel8_recovery_r2_20260921/tmp
mkdir -p "$TMPDIR"
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel8_recovery_r2_20260921/qualify_recovery.py
