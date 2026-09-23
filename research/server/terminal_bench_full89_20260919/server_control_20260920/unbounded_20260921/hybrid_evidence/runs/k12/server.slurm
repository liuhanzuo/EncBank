#!/bin/bash
#SBATCH --job-name=qencbank-tb-encbank-k12-server-r6
#SBATCH --nodelist=gpu-node1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921/slurm-%j.out
#SBATCH --error=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921/slurm-%j.err
set -euo pipefail
umask 077
export TMPDIR=/srv/encbank/qencbank_runtime_20260911/t89sp21k12
mkdir -p "$TMPDIR"
cd /srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921
export CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONPATH=/srv/encbank/encbank_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921/tmp_dense TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921/triton_dense CUDA_CACHE_PATH=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921/cuda_dense HF_HOME=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/supplement_20260921/k12_supplement_20260921/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u storage_preflight.py
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u model_path_preflight.py
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u server_job.py
