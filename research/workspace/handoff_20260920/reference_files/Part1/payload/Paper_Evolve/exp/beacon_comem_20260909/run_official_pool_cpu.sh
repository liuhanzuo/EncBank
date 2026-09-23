#!/usr/bin/env bash
set -euo pipefail
root=/data/liuhanzuo/comem_v2_20260908
code="$root/workspace/exp/beacon_comem_20260909"
out="$code/data/activation_beacon_original/prepared"
mkdir -p "$out"
cd "$code"
if [[ -f "$out/cpu_prepare.pid" ]] && kill -0 "$(cat "$out/cpu_prepare.pid")" 2>/dev/null; then
  echo "CPU preparation already running at PID $(cat "$out/cpu_prepare.pid")"
  exit 1
fi
nohup env CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 "$root/venv/bin/python" -u prepare_official_pool.py --phase all --workers 2 --tokenizer "$root/models/Qwen3-8B" \
  > "$out/cpu_prepare.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$out/cpu_prepare.pid"
printf 'Started CPU-only full pool preparation: PID %s, log %s\n' "$pid" "$out/cpu_prepare.log"
