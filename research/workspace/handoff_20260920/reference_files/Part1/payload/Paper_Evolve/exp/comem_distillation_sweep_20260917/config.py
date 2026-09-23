"""Qwen3-8B training-capacity sweep; all scientific choices fixed before evaluation."""
from pathlib import Path
import os, sys

ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/comem_distillation_sweep_20260917'
FOLLOW = ROOT.parent/'comem_followups_20260913' if os.name == 'nt' else Path('/srv/encbank/comem_followups_20260913')
sys.path.insert(0, str(FOLLOW/'COMem'))
sys.path.insert(0, str(FOLLOW))
MODEL = '/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B'
DATA = '/srv/encbank/comem_new_backbones_20260915/data/pg19_train_64.jsonl'
TOKEN_CACHE = '/srv/encbank/comem_cacheblend_lora_20260916/training/pg19_tokens.u32'
PRINCIPAL = '/srv/encbank/comem_infra_recheck_20260912/adapter'
SAMPLES = ROOT/'data/longeval_paired.jsonl.gz'
# Long/full arms start first; Slurm limits the five independent jobs to four GPUs.
ARMS = [
    dict(name='lora32_8000', mode='lora', rank=32, alpha=32, steps=8000, lr=1e-4),
    dict(name='full_4000_lr1e5', mode='full', steps=4000, lr=1e-5),
    dict(name='full_4000_lr1e4', mode='full', steps=4000, lr=1e-4),
    dict(name='lora128_4000', mode='lora', rank=128, alpha=128, steps=4000, lr=1e-4),
    dict(name='lora32_4000', mode='lora', rank=32, alpha=32, steps=4000, lr=1e-4),
]
