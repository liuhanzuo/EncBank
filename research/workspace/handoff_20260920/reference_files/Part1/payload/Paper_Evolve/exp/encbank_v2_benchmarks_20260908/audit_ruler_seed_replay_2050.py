"""Reconstruct six sample-zero RULER inputs on CPU without model weights."""
import ast
import builtins
import json
import os
from pathlib import Path
import random
import re
import string
import sys
from transformers import AutoTokenizer
import torch

B=Path(__file__).resolve().parent
R=B.parent.parent
path=R/'Encbank/eval/ruler.py'
tree=ast.parse(path.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef)) and not (isinstance(n,ast.FunctionDef) and n.name=='main')]
tok=AutoTokenizer.from_pretrained('/srv/encbank/legacy_workspace/models/Qwen3-8B',local_files_only=True)
tasks=('niah_multikey_1','niah_single_2','variable_tracking')
oldseeds=(68736,15354,89905);newseeds=(4326,36690,37267)
result=[]
for draw,seedvalues,encoding in (('historical',oldseeds,'cp936'),('remote_baselines',newseeds,'utf-8')):
    def decoded_open(p,mode='r',**kwargs):
        if 'b' not in mode and 'encoding' not in kwargs:kwargs['encoding']=encoding
        return builtins.open(p,mode,**kwargs)
    ns={'os':os,'random':random,'re':re,'string':string,'Path':Path,'open':decoded_open}
    exec(compile(tree,str(path),'exec'),ns)
    ns['_ESSAY_PATH']=str(R/'exp/data/pg19_essay.txt')
    for task,base_seed in zip(tasks,seedvalues):
        icl=ns['_make_vt_icl'](random.Random(base_seed+777),4) if task=='variable_tracking' else None
        prompt,answers,gold=ns['_build_sample'](task,16384,tok,random.Random(base_seed*1000),icl)
        ntokens=len(tok.encode(prompt,add_special_tokens=True))
        if draw=='historical':
            source=R/'exp/results'/('s15c_ruler_vt_16k.json' if task=='variable_tracking' else 's15_ruler_j12_16k.json')
        else:
            source=B/'results/remote/outputs/trained_pub/full/ruler/pub'/f'{task}_16k.json'
        rs=json.loads(source.read_text(encoding='utf-8'))['rows'];saved=next(x for x in rs if x['task']==task and x['length']=='16k' and x['i']==0)
        item={'draw':draw,'task':task,'base_seed':base_seed,'essay_encoding':encoding,'answers':answers,'n_tokens':ntokens,'saved_answers':saved['answers'],'saved_n_tokens':saved['n_tokens'],'match':answers==saved['answers'] and ntokens==saved['n_tokens']}
        result.append(item);print(json.dumps(item),flush=True)
assert not torch.cuda.is_initialized()
(B/'heartbeat_ruler_seed_20260908_2050.json').write_text(json.dumps({'python':sys.version,'cuda_initialized':torch.cuda.is_initialized(),'cases':result},indent=2)+'\n',encoding='utf-8')
assert all(x['match'] for x in result)
