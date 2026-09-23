#!/bin/bash
set -euo pipefail
umask 077
/srv/encbank/qencbank_runtime_20260911/python312/bin/python /srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/remote_batch.py verify k48
export TMPDIR=/srv/encbank/qencbank_runtime_20260911/t89dh21k48
mkdir -p "$TMPDIR"
cd /srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/k48
export CPATH=/srv/encbank/.cache/python/include/python3.12
export PYTHONPATH=/srv/encbank/encbank_infra_recheck_20260912/deps
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/k48/tmp_dense TRITON_CACHE_DIR=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/k48/triton_dense CUDA_CACHE_PATH=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/k48/cuda_dense HF_HOME=/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/docker_supplement_r2_20260921/k48/hf_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_HOME"
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u storage_preflight.py
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u model_path_preflight.py
/srv/encbank/qencbank_runtime_20260911/python312/bin/python -u holder.py
