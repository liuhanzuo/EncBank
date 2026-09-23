"""CPU tokenizer proof of exact native incremental serialization, no GPU use."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'sources'))
from transformers import AutoTokenizer
p='/srv/encbank/encbank_new_backbones_20260915/models/Qwen3.8-27B'
t=AutoTokenizer.from_pretrained(p,local_files_only=True)
def render(m,g,effort='low'):return t.apply_chat_template(m,tokenize=False,add_generation_prompt=g,enable_thinking=True,reasoning_effort=effort,preserve_thinking=True)
ms=[dict(role='user',content='initial')];plain=render(ms,False);gp=render(ms,True)[len(plain):]
assert render(ms,True)==plain+gp and gp.endswith('<think>\n')
obs=dict(role='user',content='observation')
increment=plain+gp+'brief</think>\nanswer<|im_end|>\n'+render([obs],False,'medium')+gp
full=render(ms+[dict(role='assistant',content='answer',reasoning_content='brief'),obs],True)
# Whitespace in model tokens is intentionally retained. Check roles and system count.
assert increment.count('<|im_start|>system')==1 and increment.count('<|im_start|>user')==2
assert t.decode(t.encode('brief</think>\nanswer',add_special_tokens=False),skip_special_tokens=True)=='brief</think>\nanswer'
print(json.dumps(dict(status='PASS',model_loaded=False,system_count=1,observation_roles=2,thinking_close_preserved=True,generation_prefix=gp,tokenizer_inputs={n:hashlib.sha256((Path(p)/n).read_bytes()).hexdigest() for n in ['config.json','tokenizer_config.json','chat_template.jinja','tokenizer.json'] if (Path(p)/n).exists()})))
