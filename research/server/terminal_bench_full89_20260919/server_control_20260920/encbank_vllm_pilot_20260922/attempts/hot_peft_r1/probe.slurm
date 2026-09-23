#!/bin/bash
#SBATCH --job-name=qencbank-kernel-hot-peft-r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --exclude=gpu-node1,gpu-node3,gpu-node6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922/attempts/hot_peft_r1/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922/attempts/hot_peft_r1/slurm-%j.err
set -euo pipefail
umask 077
cd /srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922/attempts/hot_peft_r1
export ENCBANK_ISOLATED_KERNEL_PILOT=1
export PYTHONPATH=/srv/encbank/encbank_infra_recheck_20260912/deps
export PATH=/srv/encbank/venvs/rllm/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922/attempts/hot_peft_r1/triton CUDA_CACHE_PATH=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/encbank_vllm_pilot_20260922/attempts/hot_peft_r1/cuda
unset TRITON_INTERPRET
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u run_probe.py
