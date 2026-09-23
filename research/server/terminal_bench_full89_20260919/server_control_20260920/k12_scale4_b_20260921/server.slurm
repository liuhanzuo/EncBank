#!/bin/bash
#SBATCH --job-name=qcomem-tb-k12-scale4-b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=0
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921/slurm-%j.err
set -euo pipefail
umask 077
export TMPDIR=/srv/encbank/qcomem_runtime_20260911/t89s4k12b
mkdir -p "$TMPDIR"
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921
export CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONPATH=/srv/encbank/comem_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921/tmp_dense TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921/triton_dense CUDA_CACHE_PATH=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921/cuda_dense HF_HOME=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/k12_scale4_b_20260921/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u storage_preflight.py
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u model_path_preflight.py
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u server_job.py
