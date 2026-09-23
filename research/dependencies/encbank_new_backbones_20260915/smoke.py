import argparse, json, os, platform
from pathlib import Path
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_reader import HybridReader

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
torch.set_num_threads(2)
print(json.dumps({'gpu':torch.cuda.get_device_name(), 'memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
                  'host':platform.node(), 'torch':torch.__version__, 'transformers':transformers.__version__}),flush=True)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda').eval()
reader=HybridReader(model,round(.33*model.config.num_hidden_layers))
checks=reader.validate(tok)
params=reader.attach()
ids=tok.encode('A short document about a river. The river flows through a small town. The town has a bridge.',add_special_tokens=False)
segs=[ids[:1],ids[1:10],ids[10:]]
with torch.no_grad():
    before=reader.logits(reader.cache_hidden(segs)).clone()
with reader.adapter(False),torch.no_grad():
    disabled=reader.logits(reader.cache_hidden(segs)).clone()
assert torch.equal(before,disabled)
logits=reader.logits(reader.cache_hidden(segs,grad=True),last=3)
loss=logits.float().square().mean()
loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in params)
assert all(p.grad is None for name,p in model.named_parameters() if not p.requires_grad)
checks.update({'zero_adapter_exact':True,'suffix_gradient_finite_nonzero':True,'frozen_backbone_no_gradient':True,'j':reader.j})
Path('smoke_complete.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks),flush=True)
