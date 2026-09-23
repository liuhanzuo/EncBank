#!/bin/bash
#SBATCH --job-name=qcomem-dense-p4-node3-check
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel4_20260921/qualification-node3-%j.log
set -euo pipefail
export XDG_RUNTIME_DIR=/run/user/20021
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/20021/bus
umask 077
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel4_20260921
export PYTHONPATH=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel4_20260921 LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
export TMPDIR=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/dense_parallel4_20260921/tmp APPTAINER_CACHEDIR=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/apptainer_cache APPTAINER_TMPDIR=/srv/encbank/qcomem_runtime_20260911/server_control_20260920/dense_parallel4_20260921/tmp
mkdir -p "$TMPDIR"
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/qualify_dense_parallel4_node3.py
