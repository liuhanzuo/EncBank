from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
OLD=Path('/srv/encbank/legacy_workspace/paper_autonomous_multifork_iteration/evidence/encbank_supplement_20260917')
FOLLOW=Path('F:/Paper_Evolve/exp/encbank_followups_20260913')
sys.path[:0]=[str(FOLLOW),str(FOLLOW/'Encbank')]
MODEL=Path('/srv/encbank/legacy_workspace/models/Qwen3-8B')
ADAPTER=Path('F:/Paper_Evolve/exp/encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final')
BGE=OLD/'bge_preparation/model'
SAMPLES=FOLLOW/'data/kv_samples.jsonl.gz'
QASPER=Path('F:/Paper_Evolve/exp/encbank_frozen_j12_20260912/data/longbench/qasper.jsonl')
FIX=Path('F:/Paper_Evolve/exp/encbank_v2_cache_reuse_20260909/fixtures')
SEED=20260918
KS=[6,8,10,12,14,16]
EXTENDED=[4,20,24]
REMOTE_OWNER='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
