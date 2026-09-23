#!/bin/bash
#SBATCH --job-name=qcomem-unbounded-envcheck
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_env_qualification_r3_20260921/qualification-%j.log
set -euo pipefail
umask 077
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_env_qualification_r3_20260921
export PYTHONPATH=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_env_qualification_r3_20260921 PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python -u qualify_all_environments.py
