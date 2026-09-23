#!/bin/bash
#SBATCH --job-name=qencbank-tb-docker-supplement-r2
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node3
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/slurm-%j.log
set -euo pipefail
umask 077
/srv/encbank/qencbank_runtime_20260911/python312/bin/python /srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/remote_batch.py run
