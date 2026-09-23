"""GPU functional prototype: hidden retrieval, growing active chunk, two rollovers."""
import datetime,gc,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;ARM=sys.argv[1];OUT=ROOT/ARM/'results'
def dump(name,obj):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def status(phase,**kw):
    row=dict(phase=phase,at=datetime.datetime.now().astimezone().isoformat(),**kw);dump('status.json',row);print(json.dumps(row),flush=True)
def main():
    assert os.environ.get('SLURM_JOB_ID') and not (OUT/'summary.json').exists()
    status('IMPORTS')
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from peft import PeftModel
    from unittest.mock import patch
    from engine import Reader
    from hot_buffer import HotKVBuffer
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(20260921)
    assert torch.cuda.device_count()==1
    free,total=torch.cuda.mem_get_info();assert free>100*2**30
    torch.cuda.set_per_process_memory_fraction(96*2**30/total)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    modelpath='/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B'
    adapter=Path('/srv/encbank/comem_infra_recheck_20260912/adapter')
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()=='1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13'
    status('LOAD_MODEL')
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(modelpath,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,str(adapter),autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);reader=Reader(model,12,tokenizer=tok)
    headpath=ROOT.parent/'hidden_reader_pilot_20260921/h12/heads.pt'
    version=hashlib.sha256(headpath.read_bytes()).hexdigest()
    weights=torch.load(headpath,map_location='cuda',weights_only=False)['weights']
    env=dict(job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),head_version=version,
        capacity_chunks=8,retrieval_topk=4,chunk_size=512,model_downloaded=False)
    dump('environment.json',env)
    def timed(fn):
        torch.cuda.synchronize();begin=time.perf_counter();out=fn();torch.cuda.synchronize()
        return out,(time.perf_counter()-begin)*1000
    def project(h):
        x=h.float();unit=(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16);out={}
        for l in range(12,36):
            layer=reader.layers[l]
            if l==12:
                u=layer.input_layernorm(h);raw=torch.cat([layer.self_attn.k_proj(u),layer.self_attn.v_proj(u)],-1)
            else:raw=F.linear(unit,*weights[l])
            k,v=raw.split(1024,-1)
            k=layer.self_attn.k_norm(k.reshape(1,-1,8,128)).transpose(1,2)
            v=v.reshape(1,-1,8,128).transpose(1,2)
            out[l]=(k.contiguous(),v.contiguous())
        return out
    def identity(h):return hashlib.sha256(h.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    def representation(h):return F.normalize(h.float().mean((0,1)),dim=0)
    bank={};embeddings={};events=[];parity=[]
    def add(h,name):
        hid=identity(h);bank[hid]=dict(h=h,name=name);embeddings[hid]=representation(h);return hid
    hot=HotKVBuffer(8,version)
    class Active:
        def __init__(self):self.native=DynamicCache(config=model.config);self.hidden=[];self.tokens=0
        def record(self,h):self.hidden.append(h.detach().clone());self.tokens+=h.shape[1]
    class DecodeCache:
        def __init__(self,memory,active):self.memory=memory;self.active=active
        def update(self,k,v,l,*a,**kw):
            nk,nv=self.active.native.update(k,v,l)
            mk,mv=self.memory[l]
            return torch.cat([mk,nk],2),torch.cat([mv,nv],2)
    def pack(parts):
        n=sum(next(iter(p.values()))[0].shape[2] for p in parts)
        pos=torch.arange(n,device='cuda')[None];cos,sin=reader.rotary_emb(bank[next(iter(bank))]['h'],position_ids=pos);out={}
        for l in range(12,36):
            k=torch.cat([part[l][0] for part in parts],2);v=torch.cat([part[l][1] for part in parts],2)
            out[l]=(k*cos[:,None]+rotate_half(k)*sin[:,None],v)
        return out
    def retrieve(query_h,phase):
        keys=list(bank);scores=torch.stack([embeddings[k] for k in keys])@representation(query_h)
        selected=[keys[i] for i in scores.topk(4).indices.tolist()]
        before=hot.snapshot()
        def warm():return [hot.get(k,lambda k=k:project(bank[k]['h'])) for k in selected]
        blocks,warm_ms=timed(warm)
        cold,cold_ms=timed(lambda:[project(bank[k]['h']) for k in selected])
        error=max(float((a[l][j]-b[l][j]).abs().max()) for a,b in zip(blocks,cold) for l in range(12,36) for j in [0,1])
        assert error==0,error
        memory,pack_ms=timed(lambda:pack([sink_projected]+blocks))
        cold_memory=pack([sink_projected]+cold)
        assert all(torch.equal(memory[l][j],cold_memory[l][j]) for l in memory for j in [0,1])
        parity.append(dict(phase=phase,kv_max_abs=error,packed_rope_equal=True))
        after=hot.snapshot()
        events.append(dict(phase=phase,selected=[bank[k]['name'] for k in selected],hits=after['hits']-before['hits'],
            misses=after['misses']-before['misses'],cached_fetch_ms=warm_ms,cold_projection_ms=cold_ms,pack_rope_ms=pack_ms,buffer=after))
        return memory,selected
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        pg=json.loads((ROOT/'vendor/workloads.json').read_text())[0]
        ids=torch.tensor([pg['source'][i*512:(i+1)*512] for i in range(16)],device='cuda')
        states=reader.write_chunk(ids)
        for i in range(16):add(states[i:i+1].clone(),'history_'+str(i))
        sink_projected=project(reader.write_chunk([model.config.bos_token_id]))
        active=Active();hot.begin(active)
        query=pg['queries'][0]['query'][:64];qh,bottom,qpos=reader.write_prefill(query)
        memory,selected=retrieve(qh,'initial')
        # Repeated selection verifies true hidden-cache hits before the trajectory.
        memory,selected=retrieve(qh,'repeat_same_hidden_query')
        top=DecodeCache(memory,active);n=2049
        hook=reader.layers[12].register_forward_pre_hook(lambda module,args:active.record(args[0]))
        pos=torch.arange(n,n+64,device='cuda')[None]
        mask=(torch.arange(n+64,device='cuda')[None,:]<=pos[0,:,None])[None,None]
        h=reader._run_layers(qh,slice(12,36),mask,pos,reader.rotary_emb(qh,position_ids=pos),past_key_values=top,use_cache=True)
        forced=torch.tensor(pg['source'][16*512:18*512],device='cuda');assert len(forced)==1024
        times=[];rollovers=[];torch.cuda.synchronize();begin=time.perf_counter()
        for step in range(1024):
            if active.tokens==512:
                completed_h=torch.cat(active.hidden,1);name='completed_current_'+str(len(rollovers))
                hid=add(completed_h,name);projected,project_ms=timed(lambda:project(completed_h))
                hot.complete(hid,projected)
                assert hot.key(hid) in hot.entries
                rollovers.append(dict(after_generated=step,name=name,tokens=active.tokens,project_ms=project_ms,inserted=True))
                active=Active();hot.begin(active)
                memory,selected=retrieve(completed_h[:,-32:],'rollover_'+str(step))
                rollovers[-1]['selected_again']=hid in selected
                rollovers[-1]['still_hot']=hot.key(hid) in hot.entries
                top=DecodeCache(memory,active)
                dump('rollovers.json',rollovers)
            start=time.perf_counter()
            logits=reader.decode_step(forced[step:step+1],bottom,top,qpos+step,n+active.tokens)
            torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            assert active.tokens<=512
            assert len(hot.entries)+1<=8
            if (step+1)%128==0:status('DECODE',generated=step+1,total=1024,buffer=hot.snapshot())
        torch.cuda.synchronize();wall=(time.perf_counter()-begin)*1000
        hook.remove()
        assert len(rollovers)==2 and active.tokens==64
        # Verify eviction order and active reservation without allocating a model.
        tiny=HotKVBuffer(3,'test');tiny.begin(object())
        tiny.get('a',lambda:'A');tiny.get('b',lambda:'B');tiny.get('a',lambda:'bad');tiny.get('c',lambda:'C')
        assert tiny.key('a') in tiny.entries and tiny.key('b') not in tiny.entries
        tiny.complete('current','C0');tiny.begin(object())
        assert tiny.key('current') in tiny.entries and len(tiny.entries)==2
        dump('events.json',events);dump('parity.json',parity)
        dump('summary.json',dict(complete=True,environment=env,generated_input_steps=1024,initial_query_tokens=64,
            chunk_rollovers=rollovers,active_final_tokens=active.tokens,buffer=hot.snapshot(),events=events,
            decode_step_median_ms=statistics.median(times),trajectory_wall_ms=wall,parity=parity,lru_active_eviction_test_passed=True,
            scope='Functional fixed-input GPU trajectory, hidden-mean cosine top4 retrieval, 16 initial chunks, eight hot slots including active. Not natural agent hit-rate or task quality. Active native KV/H12 accumulates; at rollover projected pre-RoPE KV replaces native entry once. Completed entries keyed by frozen hidden SHA plus head version; RoPE applied at packing.',
            memory_scope='Capacity bounds LRU entries plus active chunk, not total CUDA memory: selected attention pack, hidden bank, bottom/query cache, head/model weights and transition temporaries also exist.'))
    status('COMPLETE')
if __name__=='__main__':
    try:main()
    except BaseException:dump('failure.json',dict(traceback=traceback.format_exc()));raise
