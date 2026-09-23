"""Bounded CPU input checks before submitting full data/evaluation arrays."""
import ast, json, os, random
import torch
from common import ROOT, MODELS, tokenizer, dump
from eval import locomo, ruler
import run_accuracy as original
from comem.selectors import iter_bm25_indices

assert os.environ.get('PYTHONHASHSEED')=='0'
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
assert len(list(original.cells()))==47
records=locomo.build_locomo_samples(ROOT/'data/locomo10.json');assert len(records)==1986
assert sum(r['is_abstention'] for r in records)==446
ruler._ESSAY_PATH='/srv/encbank/comem_frozen_j12_20260912/data/pg19_essay.txt'
assert ruler._load_essay_words()
checks=[]
for cfg in MODELS:
 tok=tokenizer(cfg)
 tests=[('longbench','qasper',''),('babilong','qa1','0k'),('longeval','lines','8k'),
        ('ruler','niah_multikey_1','8k'),('ruler','variable_tracking','8k')]
 for b,t,l in tests:
  i,prompt,q,ans,budget,extra=next(original.samples(b,t,l,tok,0,1))
  ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
      add_generation_prompt=True,enable_thinking=False,return_dict=False)
  assert isinstance(ids,list) and ids
  chunks=[torch.tensor(ids[k:k+512],dtype=torch.long) for k in range(0,len(ids),512)]
  sel=iter_bm25_indices(chunks[:-1],tok.encode(q,add_special_tokens=False),12,iter_hop_topk=4,iter_rounds=0)
  assert len(sel)<=12 and len(sel)==len(set(sel))
  gold=' '.join(ans) if b=='ruler' else ans[0]
  assert original.score(b,t,gold,ans,extra)==1, (b,t,ans)
  checks.append(dict(model=cfg['name'],benchmark=b,task=t,length=l,tokens=len(ids),selected=len(sel),budget=budget))
dump(ROOT/'preflight.json',dict(passed=True,tests=checks,locomo_n=len(records),benchmark_cells=48))
print(json.dumps(dict(passed=True,cpu_input_checks=len(checks),locomo_n=len(records))))
print(json.dumps({'judge_environment_available':{k:bool(os.environ.get(k)) for k in
    ('OPENAI_BASE_URL','OPENAI_API_KEY','LOCOMO_JUDGE_BASE_URL','LOCOMO_JUDGE_API_KEY')}}))
