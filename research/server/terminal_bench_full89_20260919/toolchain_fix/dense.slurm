#!/bin/bash
#SBATCH --job-name=qcomem-tb-dense-full89-b2-toolchain-codex
#SBATCH --partition=gpu
#SBATCH --exclude=gpu-node4
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/toolchain_fix/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/toolchain_fix/slurm-%j.err
set -euo pipefail
cd /srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/toolchain_fix
export PATH="/srv/encbank/venvs/rllm/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda MAX_JOBS=2
export FLASHINFER_WORKSPACE_BASE="$PWD" TVM_FFI_CACHE_DIR="$PWD/tvm_ffi"
export CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qcomem_runtime_20260911/t89i4 VLLM_RPC_BASE_PATH=/srv/encbank/qcomem_runtime_20260911/t89i4 TRITON_CACHE_DIR="$PWD/triton" CUDA_CACHE_PATH="$PWD/cuda" HF_HOME="$PWD/hf_cache" XDG_CACHE_HOME="$PWD/cache" VLLM_CACHE_ROOT="$PWD/vllm_cache" VLLM_CONFIG_ROOT="$PWD/vllm_config" TORCHINDUCTOR_CACHE_DIR="$PWD/inductor" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u toolchain_guard.py
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u ipc_path_guard.py
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u storage_preflight.py
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -u holder.py
