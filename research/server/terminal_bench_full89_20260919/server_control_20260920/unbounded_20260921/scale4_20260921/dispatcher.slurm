#!/bin/bash
#SBATCH --job-name=qcomem-scale4-dispatch
#SBATCH --partition=gpu
#SBATCH --gres=none
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/scale4_20260921/dispatcher-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/scale4_20260921/dispatcher-%j.err
set -euo pipefail
umask 077
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/scale4_dispatch.py
