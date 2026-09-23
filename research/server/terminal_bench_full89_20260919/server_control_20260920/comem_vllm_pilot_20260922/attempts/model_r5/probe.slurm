#!/bin/bash
#SBATCH --job-name=qcomem-kernel-model-r5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --exclude=gpu-node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/model_r5/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/model_r5/slurm-%j.err
set -euo pipefail
umask 077
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/model_r5
export COMEM_ISOLATED_KERNEL_PILOT=1
export PYTHONPATH=/srv/encbank/comem_infra_recheck_20260912/deps
export PATH=/srv/encbank/venvs/rllm/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/model_r5/triton CUDA_CACHE_PATH=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/model_r5/cuda
unset TRITON_INTERPRET
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u probe_supervisor.py
