"""Prespecified controlled-depth and serving diagnostics, Qwen3-8B."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/encbank_serving_depth_control_20260918'
FOLLOW = Path('/srv/encbank/encbank_followups_20260913')
sys.path[:0] = [str(FOLLOW), str(FOLLOW/'Encbank')]
MODEL = '/srv/encbank/encbank_sparse_slurm_20260912/models/Qwen3-8B'
DATA = '/srv/encbank/encbank_new_backbones_20260915/data/pg19_train_64.jsonl'
TOKENS = '/srv/encbank/encbank_cacheblend_lora_20260916/training/pg19_tokens.u32'
ADAPTER = '/srv/encbank/encbank_infra_recheck_20260912/adapter'
LONGEVAL = Path('/srv/encbank/encbank_distillation_sweep_20260917/data/longeval_paired.jsonl.gz')
QASPER = FOLLOW/'data/kv_samples.jsonl.gz'
DEPTHS = [6,9,12,15,18,24]
SEEDS = [42,43,44]
ADAPTER_START = 24
STEPS = 4000
CONCURRENCIES = [1,2,4,8,16]
SOURCE_LENGTHS = [32768,131072]
MAX_BATCH = 4
REQUESTS = 384  # per process; three independent processes = 1,152 per point
OUTPUT_TOKENS = 32
CAP_BYTES = 120_000_000_000  # fixed total CUDA allocator cap, NOT activation-only

def tag(j,seed): return f'j{j}_s{seed}'
