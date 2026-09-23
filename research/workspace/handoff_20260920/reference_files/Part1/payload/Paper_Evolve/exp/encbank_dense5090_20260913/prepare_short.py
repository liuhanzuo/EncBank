"""Prepare short source lengths with the exact earlier local workload recipe."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'exp/encbank_infra_20260912'))
from bench_local import prepare_workloads
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/srv/encbank/legacy_workspace/models/Qwen3-8B',local_files_only=True)
for n in (8192,16384):
    path=HERE/'workloads'/f'{n}.json'
    assert not path.exists(),'Inspect existing inputs before regenerating'
    rows=prepare_workloads(tok,Path('/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl'),path,n)
    print(n,[r['book_index'] for r in rows],flush=True)
