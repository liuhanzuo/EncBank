"""Exercise the actual service control flow on a tiny CPU hybrid model."""
from pathlib import Path
import copy,hashlib,json,os,time,types
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
from hybrid_reader import HybridReader
import service_loop as service
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());torch.set_num_threads(2);torch.manual_seed(1719)
cfg=copy.deepcopy(AutoConfig.from_pretrained(P['model'],local_files_only=True).text_config)
cfg.vocab_size=128;cfg.hidden_size=64;cfg.intermediate_size=128;cfg.num_hidden_layers=8
cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=32
cfg.linear_num_key_heads=2;cfg.linear_num_value_heads=4;cfg.linear_key_head_dim=8;cfg.linear_value_head_dim=8
cfg.layer_types=['linear_attention','linear_attention','linear_attention','full_attention']*2
cfg.rope_parameters['mrope_section']=[1,1,2];cfg.rope_parameters['partial_rotary_factor']=.25
cfg._attn_implementation='sdpa';cfg.tie_word_embeddings=False
model=Qwen3_5ForCausalLM(cfg).float().eval();reader=HybridReader(model,4)
assert all(p.device.type=='cpu' for p in model.parameters())
pressure={'reserved':170*2**30,'reclaims':0}
def empty_cache():pressure['reserved']=0;pressure['reclaims']+=1
class CPUProxy:
    cuda=types.SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
        max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,memory_allocated=lambda:0,memory_reserved=lambda:pressure['reserved'],empty_cache=empty_cache)
    def __getattr__(self,n):return getattr(torch,n)
    def tensor(self,*a,**kw):kw.pop('device',None);return torch.tensor(*a,**kw)
    def Generator(self,**kw):return torch.Generator(device='cpu')
class Tok:
    def apply_chat_template(self,messages,**kw):return [1,2,3]+([4,5] if kw['add_generation_prompt'] else [])
    def encode(self,*a,**kw):return [6]
    def decode(self,g,**kw):return ' '.join(map(str,g))
from live_mailbox import Mailbox
R=H/'cpu_service_control';assert not R.exists();R.mkdir()
transport=Mailbox(R/'mailbox',lambda:{});B=transport.path()
events=[];count=0;real_decode=service.decode_layers
def request(name,deadline=3600):
    q=dict(request_id=name,task_id=name,task=name,step=0,messages=[dict(role='user',content=name)],published_epoch=time.time(),deadline_epoch=time.time()+deadline)
    (B/(name+'.request.json')).write_immutable(q)
def decode(*args,**kwargs):
    global count
    z=real_decode(*args,**kwargs);count+=1
    if count==2:
        request('short');request('expired',-1);request('cancelled')
        (B/'cancelled.cancel.json').write_immutable({})
        (B/'long.release.json').write_immutable(dict(task_id='long'))
    if count==8:(B/'short.cancel.json').write_immutable({})
    return z
def event(kind,**kw):
    events.append(dict(event=kind,**kw))
    if kind=='batch_refill':
        assert not (B/'long.response.json').exists(),'Long request finished before refill'
        assert not (B/'long.released.json').exists(),'Active H bank released'
    if kind=='request_complete' and kw['request_id']=='long':
        (B/'short.release.json').write_immutable(dict(task_id='short'))
        (B/'stop.json').write_immutable({})
service.decode_layers=decode
request('long')
testplan=dict(P,tasks=['long','short','expired','cancelled'],context_tokens=100,max_new_tokens=16,batch_collect_seconds=0,refill_interval_seconds=0)
service.serve(testplan,R,B,model,reader,Tok(),set(),event,CPUProxy(),DynamicCache,lambda *a,**kw:[])
replies={n:json.loads((B/(n+'.response.json')).read_text()) for n in testplan['tasks']}
assert replies['long']['status']=='ok' and replies['long']['generated_tokens']==16
assert replies['short']['status']=='cancelled' and 0<replies['short']['generated_tokens']<16
assert replies['expired']['status']=='deadline' and replies['expired']['generated_tokens']==0
assert replies['cancelled']['status']=='cancelled' and replies['cancelled']['generated_tokens']==0
assert any(x['event']=='batch_refill' for x in events)
assert all(replies[n]['state_hashes_unchanged'] for n in ['long','short'])
assert (B/'long.released.json').exists() and (B/'short.released.json').exists()
assert any(x['event']=='allocator_cache_reclaim' for x in events)
out=dict(allocator_pressure_mock_only=True,actual_allocator_reclaim_branch_exercised=True,status='PASS',gpu_calls=0,model='tiny random8-layer FP32 CPU',actual_service_control_flow=True,
    refill_before_old_request_finishes=True,active_bank_release_protected=True,expiry_and_cancellation_preserved=True,
    service_sha256=hashlib.sha256((H/'service_loop.py').read_bytes()).hexdigest(),events=events,
    statuses={n:dict(status=r['status'],generated_tokens=r['generated_tokens']) for n,r in replies.items()})
(H/'service_cpu_check.json').write_text(json.dumps(out,indent=2)+'\n');transport.server.shutdown();transport.server.server_close();print(json.dumps(out))
