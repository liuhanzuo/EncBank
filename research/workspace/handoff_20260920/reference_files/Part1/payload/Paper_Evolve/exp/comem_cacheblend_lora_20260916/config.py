"""One fixed CacheBlend-style adaptation experiment, separate from prior runs."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/comem_cacheblend_lora_20260916'
if os.name=='nt':
    FOLLOW=ROOT.parent/'comem_followups_20260913'
else:
    FOLLOW=Path('/srv/encbank/comem_followups_20260913')
sys.path.insert(0,str(FOLLOW/'COMem'))
MODEL='/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B'
DATA='/srv/encbank/comem_new_backbones_20260915/data/pg19_train_64.jsonl'
PRINCIPAL='/srv/encbank/comem_infra_recheck_20260912/adapter'
SAMPLES=FOLLOW/'data/kv_samples.jsonl.gz'
STEPS=4000
RECIPE=dict(model='Qwen3-8B',j=12,rank=32,alpha=32,dropout=0,
    layers=list(range(12,36)),targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    steps=4000,window=4096,chunk=512,n_ctx=7,global_windows_per_step=1,
    training_tokens=16384000,skip_windows=8,seed=42,topk=64,lam=.6,
    loss='teacher-support-normalized bidirectional KL; query positions only',
    lr=1e-4,warmup=50,betas=[.9,.95],weight_decay=0,clip=1,
    precision='bf16 backbone, fp32 adapter master, bf16 autocast',
    ratio=.15,bootstrap_full_layers=2,selection='layer1 squared K+V difference, fixed selected set',
    student='independent full-depth chunk KV with differentiable writer and selective repair; rebuilt each step under current adapter',
    teacher='same causal full pack, adapter disabled',
    training_budget_note='Matched data/steps/trainable parameters, not identical FLOPs or GPU hours',
    evaluation='500 exact prior paired examples; no benchmark-based checkpoint/ratio selection',
    implementation='CacheBlend-style HF reference, not native CacheBlend engine')
