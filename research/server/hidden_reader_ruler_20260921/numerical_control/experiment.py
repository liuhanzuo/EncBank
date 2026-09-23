"""Paired original Encbank / hidden-to-KV RULER quality evaluation, server only."""
import collections,datetime,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CELL=sys.argv[1];OUT=ROOT/CELL/'results'
def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(phase,**kw):
    obj=dict(cell=CELL,phase=phase,at=datetime.datetime.now().astimezone().isoformat(),**kw)
    dump('status.json',obj);print(json.dumps(obj),flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID') and not (OUT/'predictions.jsonl').exists()
    status('IMPORTS')
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from peft import PeftModel
    from unittest.mock import patch
    from engine import Reader
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(20260921)
    assert torch.cuda.device_count()==1
    free,total=torch.cuda.mem_get_info();assert free>100*2**30,(free,total)
    torch.cuda.set_per_process_memory_fraction(96*2**30/total)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    cfg=json.loads((ROOT/'configs.json').read_text())[CELL]
    inputpath=Path(cfg['input']);assert sha(inputpath)==cfg['input_sha256']
    cases=json.loads(inputpath.read_text());assert len(cases)==100
    checkpoints=json.loads((ROOT/'checkpoints.json').read_text())
    weights={}
    for arm,c in checkpoints['arms'].items():
        if c is None or cfg['control_only']:continue
        p=Path(c['path']);assert sha(p)==c['sha256']
        w=torch.load(p,map_location='cuda',weights_only=False)['weights']
        assert set(w)==set(range(12,36))-{12,16,20,24}
        weights[arm]={l:tuple(t.to(torch.bfloat16) for t in pair) for l,pair in w.items()}
    MODEL='/srv/encbank/encbank_sparse_slurm_20260912/models/Qwen3-8B'
    ADAPTER='/srv/encbank/encbank_infra_recheck_20260912/adapter'
    assert sha(Path(ADAPTER)/'adapter_model.safetensors')=='1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13'
    status('LOAD_MODEL')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):
        wrapper=PeftModel.from_pretrained(model,ADAPTER,autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False)
    reader=Reader(model,12,tokenizer=tok)
    assert model.config.num_hidden_layers==36 and model.config.hidden_size==4096
    assert model.config.num_key_value_heads==8 and model.config.head_dim==128
    assert not reader.write_sink and not reader.block_diagonal and reader.top_prepay_b==0
    dump('environment.json',dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        gpu=torch.cuda.get_device_name(0),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,transformers=transformers.__version__,input_sha256=sha(inputpath),
        checkpoints=checkpoints,model=MODEL,adapter=ADAPTER,dtype='BF16, original unmerged FP32 LoRA',
        allocator_cap_GiB=96,labels_loaded_by_worker=False,write_sink=False))
    depths=[12,16,20,24]
    def fresh():return DynamicCache(config=model.config)
    def capture(chunks):
        # Equal-length batching preserves independent WRITE and does not pad tails.
        chunks=[[151643]]+chunks
        groups=collections.defaultdict(list)
        for i,c in enumerate(chunks):groups[len(c)].append(i)
        result={d:[None]*len(chunks) for d in depths}
        for length,indices in groups.items():
            h=reader.write_chunk(torch.tensor([chunks[i] for i in indices],device='cuda'))
            for b,i in enumerate(indices):result[12][i]=h[b:b+1]
            if cfg['control_only']:continue
            p=torch.arange(length,device='cuda')[None].expand(len(indices),-1)
            mask=(p[0,None,:]<=p[0,:,None])[None,None];rope=reader.rotary_emb(h,position_ids=p)
            for l in range(12,24):
                h=reader._run_layers(h,slice(l,l+1),mask,p,rope,use_cache=False)
                if l+1 in depths:
                    for b,i in enumerate(indices):result[l+1][i]=h[b:b+1]
        return {d:torch.cat(pieces,1) for d,pieces in result.items() if pieces[0] is not None}
    def cache_from_hidden(hidden,heads):
        n=hidden[12].shape[1];positions=torch.arange(n,device='cuda')[None];cache=fresh()
        normalized={s:(h.float()*torch.rsqrt(h.float().square().mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16) for s,h in hidden.items()}
        for l in range(12,36):
            layer=reader.layers[l]
            if l in depths:
                u=layer.input_layernorm(hidden[l]);raw=torch.cat([layer.self_attn.k_proj(u),layer.self_attn.v_proj(u)],-1)
            else:
                source=max(d for d in depths if d<=l)
                raw=F.linear(normalized[source],*heads[l])
            k,v=raw.split(1024,-1)
            k=layer.self_attn.k_norm(k.reshape(1,-1,8,128)).transpose(1,2)
            v=v.reshape(1,-1,8,128).transpose(1,2)
            cos,sin=reader.rotary_emb(raw,position_ids=positions)
            k=k*cos[:,None]+rotate_half(k)*sin[:,None]
            cache.update(k.contiguous(),v.contiguous(),l)
        return cache
    def query_prefill(query,top,n):
        qh,bottom,qpos=reader.write_prefill(query)
        p=torch.arange(n,n+qpos,device='cuda')[None]
        mask=(torch.arange(n+qpos,device='cuda')[None,:]<=p[0,:,None])[None,None]
        h=reader._run_layers(qh,slice(12,36),mask,p,reader.rotary_emb(qh,position_ids=p),past_key_values=top,use_cache=True)
        return reader.lm_head(reader.norm(h[:,-1:])),bottom,top,qpos
    def split_true_cache(native,top,query,n,check=False):
        # Use true Encbank memory KV to qualify the student's cache/query path.
        c=fresh()
        for l in range(12,36):c.update(top.layers[l].keys[:,:,:n].clone(),top.layers[l].values[:,:,:n].clone(),l)
        actual,bottom,newtop,qpos=query_prefill(query,c,n)
        if check:
            prefix_exact=all(torch.equal(a[:,:,:n],b[:,:,:n]) for l in range(12,36) for a,b in
                [(top.layers[l].keys,newtop.layers[l].keys),(top.layers[l].values,newtop.layers[l].values)])
            assert prefix_exact and all(newtop.layers[l].keys.shape[2]==n+qpos for l in range(12,36))
            p=F.log_softmax(native.float(),-1);q=F.log_softmax(actual.float(),-1)
            kl=float((p.exp()*(p-q)).sum(-1).mean());same=bool((native.argmax(-1)==actual.argmax(-1)).all())
            row=dict(kl=kl,argmax_equal=same,max_abs=float((native.float()-actual.float()).abs().max()),
                memory_KV_bitwise_preserved=prefix_exact,cache_length_verified=True,
                scope='BF16 joint vs split-query numerical drift; task-level exact-KV split control measured on every item',
                qualification='KL<0.01 and max_abs<1 plus exact memory prefix and cache length; equality of native/split argmax is NOT assumed')
            dump('exact_cache_control.json',row)
            assert abs(kl)<.01 and row['max_abs']<1,row
        return actual,bottom,newtop,qpos
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        start=time.perf_counter()
        with (OUT/'predictions.jsonl').open('x') as output:
            for index,case in enumerate(cases):
                hidden=capture(case['selected_chunks']);n=hidden[12].shape[1]
                assert n==1+sum(map(len,case['selected_chunks']))
                query=case['query_token_ids'];row=dict(id=case['id'],index=index,
                    source_document_sha256=case['source_document_sha256'],source_document_tokens=case['source_document_tokens'],
                    selected_indices=case['selected_indices'],memory_tokens=n,query_tokens=len(query),arms={})
                arms=['encbank_split'] if cfg['control_only'] else ['encbank','encbank_split','kd256','kd_selected','kd2048']
                for arm in arms:
                    torch.cuda.synchronize();begin=time.perf_counter()
                    if arm in ['encbank','encbank_split']:
                        qh,bottom,qpos=reader.write_prefill(query)
                        logits,top,packed=reader.read_prefill(hidden[12][:,:1],[hidden[12][:,1:]],qh)
                        assert packed==n+qpos
                        if arm=='encbank_split':logits,bottom,top,qpos=split_true_cache(logits,top,query,n,check=index==0)
                    else:logits,bottom,top,qpos=query_prefill(query,cache_from_hidden(hidden,weights[arm]),n)
                    torch.cuda.synchronize();prefill_ms=(time.perf_counter()-begin)*1000
                    generated=[];stop='length';eos=case['eos_token_id'];cap=case['max_new_tokens']
                    for step in range(cap):
                        if step==0:logits[:,-1,eos]=-torch.inf
                        token=logits[:,-1].argmax(-1);value=int(token.item());generated.append(value)
                        if value==eos:stop='eos';break
                        if step+1<cap:logits=reader.decode_step(token,bottom,top,qpos+step,n+qpos+step)
                    torch.cuda.synchronize()
                    row['arms'][arm]=dict(token_ids=generated,prediction=tok.decode(generated,skip_special_tokens=True),
                        stop_reason=stop,generated_tokens=len(generated),prefill_ms=prefill_ms,total_ms=(time.perf_counter()-begin)*1000)
                    del logits,bottom,top
                output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush();del hidden
                if index%10==0 or index+1==len(cases):status('EVALUATE',completed=index+1,total=len(cases),elapsed_s=time.perf_counter()-start)
        dump('summary.json',dict(items=len(cases),generations=len(arms)*len(cases),elapsed_s=time.perf_counter()-start,
            predictions_sha256=sha(OUT/'predictions.jsonl'),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        status('COMPLETE',items=len(cases),generations=len(arms)*len(cases))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump('failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
