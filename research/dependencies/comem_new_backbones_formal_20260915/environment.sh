export PYTHONPATH="$PWD:/srv/encbank/comem_infra_recheck_20260912/deps:/srv/encbank/comem_followups_20260913/COMem:/srv/encbank/comem_frozen_j12_20260912"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export CPATH=/srv/encbank/.cache/python/include/python3.12
export TRITON_CACHE_DIR="$PWD/cache/triton-$SLURM_JOB_ID" CUDA_CACHE_PATH="$PWD/cache/cuda-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
