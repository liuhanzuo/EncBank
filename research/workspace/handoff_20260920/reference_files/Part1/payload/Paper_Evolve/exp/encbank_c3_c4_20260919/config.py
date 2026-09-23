"""User-authorized minimal C3/C4, independent of the withdrawn larger sweep."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/encbank_c3_c4_20260919'
LOCAL=os.name=='nt'
FOLLOW=Path('F:/Paper_Evolve/exp/encbank_followups_20260913' if LOCAL else '/srv/encbank/encbank_followups_20260913')
sys.path[:0]=[str(FOLLOW),str(FOLLOW/'Encbank')]
MODEL='/srv/encbank/legacy_workspace/models/Qwen3-8B' if LOCAL else '/srv/encbank/encbank_sparse_slurm_20260912/models/Qwen3-8B'
DATA='/srv/encbank/encbank_new_backbones_20260915/data/pg19_train_64.jsonl'
TOKENS='/srv/encbank/encbank_cacheblend_lora_20260916/training/pg19_tokens.u32'
ADAPTER=Path('F:/Paper_Evolve/exp/encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final' if LOCAL else '/srv/encbank/encbank_infra_recheck_20260912/adapter')
DEPTHS=[6,12,18]
SEEDS=[42]
ADAPTER_START=18
STEPS=4000
CONCURRENCIES=[1,4]
SOURCE_LENGTHS=[32768]
MAX_BATCH=4
REQUESTS=1000
OUTPUT_TOKENS=32
CAP_BYTES=27*2**30
def tag(j,seed):return f'j{j}_s{seed}'
