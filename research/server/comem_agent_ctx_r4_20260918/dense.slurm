#!/bin/bash
#SBATCH --job-name=qcomem-agentmem-dense-r4-codex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918/slurm-%j.out
#SBATCH --error=/srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918/slurm-%j.err
set -euo pipefail
export CPATH=/srv/encbank/.cache/python/include/python3.12
cd /srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918
export PYTHONPATH=/srv/encbank/comem_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918/tmp_dense TRITON_CACHE_DIR=/srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918/triton_dense CUDA_CACHE_PATH=/srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918/cuda_cache_dense HF_HOME=/srv/encbank/qcomem_align_codex_20260911/comem_agent_ctx_r4_20260918/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qcomem_runtime_20260911/python312/bin/python -X utf8 -u run_agent.py dense
