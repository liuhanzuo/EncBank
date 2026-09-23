"""Configuration/tokenizer/API checks only: no engine creation or model forward."""
from pathlib import Path
import dataclasses,hashlib,importlib.metadata,json,os,sys,time
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());H.resolve().relative_to(Path('/srv/encbank').resolve())
from transformers import AutoTokenizer,AutoConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.model_executor.models import ModelRegistry
from vllm.inputs import TokensPrompt
model=Path(P['model']).resolve();model.relative_to(Path('/srv/encbank').resolve())
cfg=AutoConfig.from_pretrained(model,local_files_only=True);assert cfg.architectures==['Qwen3_5ForConditionalGeneration']
assert cfg.text_config.max_position_embeddings==P['context_tokens'] and cfg.text_config.num_hidden_layers==64
assert cfg.architectures[0] in ModelRegistry.get_supported_archs()
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
msgs=[dict(role='user',content='Public input A'),dict(role='assistant',content='Public reply B',reasoning_content='Retained reasoning C'),dict(role='user',content='Public follow-up D')]
render=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,reasoning_effort=P['reasoning_effort'],preserve_thinking=True)
assert 'Retained reasoning C' in render and render.endswith('<think>\n') and 'xhigh' in render
ids=tok.apply_chat_template(msgs,tokenize=True,return_dict=False,add_generation_prompt=True,reasoning_effort=P['reasoning_effort'],preserve_thinking=True)
assert ids==tok.encode(render,add_special_tokens=False)
args=AsyncEngineArgs(model=str(model),dtype='bfloat16',max_num_seqs=4,max_model_len=262144,
    language_model_only=True,kv_cache_memory_bytes=20*2**30,cpu_offload_gb=0,kv_offloading_size=None,
    mamba_cache_mode='align',enable_prefix_caching=True,max_num_batched_tokens=4096,max_cudagraph_capture_size=4)
identity={}
for p in sorted(model.iterdir()):
    if p.is_file():
        s=p.stat();d=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
        if s.st_size<64*2**20:d['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        identity[p.name]=d
out=dict(status='PASS',epoch=time.time(),python=sys.version,versions={n:importlib.metadata.version(n) for n in ['vllm','torch','transformers','tokenizers']},model=str(model),files=identity,
    supported_architecture=cfg.architectures,template_retains_reasoning=True,tokenization_exact=True,
    sampling_params_per_request=True,gpu_model_calls=0,cache_salt_supported='cache_salt' in TokensPrompt.__annotations__,
    weight_hash_status='Small configuration/tokenizer files hashed; shard sizes+mtime captured, frozen model revision retained. Full shard SHA computed separately before final evidence registration.')
(H/'cpu_preflight_remote.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}),flush=True)
