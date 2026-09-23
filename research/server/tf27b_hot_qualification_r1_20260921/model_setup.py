import hashlib, json, os
from pathlib import Path
import torch
from native_common import MODELS, load_model, load_state, tokenizer
from hybrid_reader import HybridReader
from runtime_identity import resolve_configs

def load(plan,adapter=True):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.backends.cuda.enable_cudnn_sdp(False)
    free,total=torch.cuda.mem_get_info()
    assert free/2**30>=250,('GPU not freshly available',free/2**30)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_gib']*2**30/total)
    identity,cfg=resolve_configs(plan,MODELS[1])
    tok=tokenizer(cfg);model=load_model(cfg);reader=HybridReader(model,21)
    if adapter:
        p=Path(plan['adapter_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==plan['adapter_sha256']
        saved=torch.load(p,map_location='cpu',weights_only=False)
        resolve_configs(plan,identity,saved);reader.attach();load_state(reader,saved,identity);del saved
    model.requires_grad_(False)
    stop=model.generation_config.eos_token_id
    stop=set(stop if isinstance(stop,list) else [stop if stop is not None else tok.eos_token_id])
    return model,reader,tok,stop

def tokens(tok,messages):
    return tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,
        enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=True)
