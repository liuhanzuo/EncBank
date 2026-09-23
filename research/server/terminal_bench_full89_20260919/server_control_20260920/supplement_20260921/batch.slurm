#!/bin/bash
#SBATCH --job-name=qcomem-tb-supplement-20260921
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/slurm-%j.log
set -euo pipefail
umask 077
export LITELLM_LOCAL_MODEL_COST_MAP=True
/srv/encbank/qcomem_runtime_20260911/server_control_20260920/harbor_env/bin/python /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921.py run
