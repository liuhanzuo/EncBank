"""Tokenizer/session isolation only. No CUDA context or model allocation."""
from pathlib import Path
import hashlib,importlib.metadata,json,os
from session import Session,native_ids,request_seed
from transformers import AutoTokenizer
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text())
tok=AutoTokenizer.from_pretrained(P['model'],local_files_only=True)
a=Session('fix-git');b=Session('log-summary-date-ranges')
ma=[dict(role='user',content='Inspect the repository and repair the broken branch.')]
mb=[dict(role='user',content='Parse these timestamps and summarize the logs.')]
rows=[]
for step in range(4):
    for s,ms in [(a,ma),(b,mb)]:
        full,static,chunks,query=s.split(tok,ms,step)
        assert full==native_ids(tok,ms) and sum(static,[])+sum(chunks,[])+query==full
        rows.append(dict(task=s.task,step=step,full_tokens=len(full),chunks=len(chunks),query=len(query),
            full_sha256=hashlib.sha256(json.dumps(full).encode()).hexdigest(),seed=request_seed(P,s.task,step)))
    assert a.static!=b.static and a.bank is not b.bank and a.hashes is not b.hashes
    ma.extend([dict(role='assistant',content='<think>Check A.</think>\n'+('tool command A\n'*350)),dict(role='user',content='Tool output A '+str(step))])
    mb.extend([dict(role='assistant',content='<think>Check B.</think>\n'+('tool command B\n'*350)),dict(role='user',content='Tool output B '+str(step))])
try:a.split(tok,ma,0)
except AssertionError:pass
else:raise AssertionError('Duplicate step accepted')
cp=Path(P['adapter_path']).resolve();cp.relative_to(Path('/srv/encbank').resolve())
assert hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']

from memory_selectors import iter_bm25_indices
import torch
corpus=[torch.tensor([7, i+100, 7, i+200]) for i in range(60)]
assert len(iter_bm25_indices(corpus,[7],P['top_k_chunks'],iter_hop_topk=4,iter_rounds=0))==P['top_k_chunks']
assert iter_bm25_indices(corpus,[999999],P['top_k_chunks'],iter_hop_topk=4,iter_rounds=0)==[]
assert len(iter_bm25_indices(corpus[:3],[7],P['top_k_chunks'],iter_hop_topk=4,iter_rounds=0))==3

out=dict(selector_cap_verified=P['top_k_chunks'],zero_overlap_not_filled=True,status='PASS',model_calls=0,cuda_context=False,session_isolation=True,chunk_reconstruction=True,
    duplicate_rejected=True,rows=rows,versions={x:importlib.metadata.version(x) for x in ['torch','transformers','tokenizers']},
    plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),adapter_sha256=P['adapter_sha256'])
(H/'cpu_preflight_remote.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
