"""Query-dependent chunk selection pilot. No closed-loop agent claims."""
import datetime,gc,hashlib,json,os,platform,random,time
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from torch.nn.attention import sdpa_kernel,SDPBackend
from encbank import Encbank
from engine import Engine,sync
from adaptive import nucleus,fixed_topk,boundary_attention_mass
from common_benchmark import MODEL,ADAPTER,dump,Nvml
from queue_metrics import percentile
ROOT=Path(__file__).resolve().parent
ARMS=['iter_k12','iter_k48','bm25_k48']+[f'{family}_p{p}' for family in ['bm25','iter48_bm25','iter48_qk'] for p in [80,90,95]]+[f'iter48_qk_k{k}' for k in [12,24,32,48]]

@torch.inference_mode()
def validate_attention(engine):
    """Compare probe to actual HF eager attention with prefix KV and a causal mask."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    reader=engine.reader;layer=reader.layers[12];attn=layer.self_attn
    memory=engine.bank[:2].reshape(1,1024,-1)
    qh=reader.write_chunk(engine.docs[0]['queries'][0]['query'])
    predicted=boundary_attention_mass(reader,memory,qh)
    prefix=torch.cat([engine.sink,memory,qh[:,:-32]],dim=1)
    x=layer.input_layernorm(prefix);length=x.shape[1]
    key=attn.k_norm(attn.k_proj(x).view(1,length,-1,attn.head_dim)).transpose(1,2)
    value=attn.v_proj(x).view(1,length,-1,attn.head_dim).transpose(1,2)
    positions=torch.arange(length,device=x.device)[None,:]
    cos,sin=reader.rotary_emb(x,position_ids=positions)
    key=key*cos.unsqueeze(1)+rotate_half(key)*sin.unsqueeze(1)
    cache=DynamicCache(config=reader.config);cache.update(key,value,12)
    x=layer.input_layernorm(qh[:,-32:]);positions=torch.arange(length,length+32,device=x.device)[None,:]
    pe=reader.rotary_emb(x,position_ids=positions)
    blocked=torch.arange(length+32,device=x.device)[None,:]>positions[0,:,None]
    mask=torch.zeros(1,1,32,length+32,dtype=x.dtype,device=x.device).masked_fill(blocked[None,None],torch.finfo(x.dtype).min)
    old=reader.config._attn_implementation
    try:
        reader.config._attn_implementation='eager'
        _,weights=attn(x,position_embeddings=pe,attention_mask=mask,past_key_values=cache)
    finally:reader.config._attn_implementation=old
    assert weights is not None
    conditional=weights[:,:,:,1:1025].float();conditional/=conditional.sum(-1,keepdim=True).clamp_min(1e-30)
    expected=conditional.reshape(1,conditional.shape[1],32,2,512).sum(-1).mean((0,1,2))
    delta=predicted-expected
    result=dict(predicted=predicted.cpu().tolist(),hf_attention_reference=expected.cpu().tolist(),max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),scope='actual first-suffix attention; memory-conditional chunk mass; two candidate blocks')
    dump(ROOT/'correctness_progress.json',result)
    assert result['max_abs']<.01 and result['rms']<.005,result
    dump(ROOT/'correctness.json',dict(passed=True,check=result))

def select(engine,di,query_h,query_ids,arm):
    index=engine.indices[di]
    if arm.startswith('iter_k'):
        k=int(arm.split('_k')[1]);ids=index.select(query_ids[:32],k,2)
        return ids,dict(scope='iterative BM25 fixed budget',candidates=len(index.docs))
    if arm=='bm25_k48':
        ids=fixed_topk(index.scores(query_ids[:32]),48)
        return ids,dict(scope='single-pass BM25 fixed budget',candidates=len(index.docs))
    candidates=index.select(query_ids[:32],48,2) if arm.startswith('iter48_') else list(range(len(index.docs)))
    if 'qk' in arm:
        memory=engine.bank.index_select(0,torch.tensor([di*64+i for i in candidates],device='cuda')).reshape(1,len(candidates)*512,-1)
        scores=boundary_attention_mass(engine.reader,memory,query_h).cpu().tolist()
        scope='conditional attention mass inside iterative BM25-48 candidates, averaged across heads and final 32 query positions'
    else:
        all_scores=index.scores(query_ids[:32]);scores=[all_scores[i] for i in candidates]
        scope='linear-normalized nonnegative BM25 mass over '+('iterative BM25-48 candidates' if arm.startswith('iter48_') else 'all document chunks')
    if '_p' in arm:
        result=nucleus(scores,p=int(arm.rsplit('_p',1)[1])/100,max_chunks=48,min_chunks=4,ids=candidates)
        ids=result['selected']
    else:
        ids=fixed_topk(scores,k=int(arm.rsplit('_k',1)[1]),ids=candidates);result={}
    result.update(scope=scope,candidates=len(candidates),candidate_ids=candidates,scores=scores)
    return ids,result

@torch.inference_mode()
def request(engine,pick,arm):
    di,qi=pick;qids=engine.docs[di]['queries'][qi]['query'];reader=engine.reader
    start=sync();qh,bottom,qp=reader.write_prefill(qids);after_write=sync()
    selected,selection=select(engine,di,qh,qids,arm);after_selection=sync()
    assert 0<len(selected)<=48 and selected==sorted(set(selected))
    states=[engine.bank[di*64+i:di*64+i+1] for i in selected]
    logits,top,pack=reader.read_prefill(engine.sink,states,qh)
    assert pack==1+512*len(selected)+len(qids) and pack+32<=40960
    token=int(logits[:,-1].float().argmax(-1).item());tokens=[token];times=[time.perf_counter()]
    for n in range(1,32):
        logits=Encbank.decode_step(reader,token,bottom,top,qp+n-1,pack+n-1)
        token=int(logits[:,-1].float().argmax(-1).item());tokens.append(token);times.append(time.perf_counter())
    return dict(arm=arm,pick=pick,selected=selected,n_chunks=len(selected),selection=selection,generated_ids=tokens,start_time=start,after_query_write=after_write,after_selection=after_selection,
        query_write_ms=1000*(after_write-start),selector_ms=1000*(after_selection-after_write),ttft_ms=1000*(times[0]-start),e2e_ms=1000*(times[-1]-start),token_times=times)

def run_point(engine,arm,picks,uuid):
    out=ROOT/'results'/arm;out.mkdir(parents=True,exist_ok=False)
    dump(ROOT/'status.json',dict(phase='RUNNING',arm=arm))
    request(engine,picks[0],arm)
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    telemetry=Nvml(uuid);telemetry.start();rows=[]
    try:
        with (out/'requests.jsonl').open('w',encoding='utf-8') as f:
            for n,pick in enumerate(picks):
                r=request(engine,pick,arm);r['id']=n;rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                dump(out/'progress.json',dict(requests=n+1,target=len(picks)))
        peak=telemetry.stop();dump(out/'nvml_samples.json',telemetry.samples);telemetry=None
        result=dict(status='ok',arm=arm,requests=len(rows),mean_chunks=sum(r['n_chunks'] for r in rows)/len(rows),min_chunks=min(r['n_chunks'] for r in rows),max_chunks=max(r['n_chunks'] for r in rows),
            **{key:{f'p{p}':percentile([r[key] for r in rows],p) for p in [50,95]} for key in ['query_write_ms','selector_ms','ttft_ms','e2e_ms']},
            cap_limited_requests=sum(r['selection'].get('cap_limited',False) for r in rows),zero_mass_requests=sum(r['selection'].get('zero_mass',False) for r in rows),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),nvml_sampled_peak_bytes=peak)
        dump(out/'complete.json',result)
    except torch.OutOfMemoryError as exc:
        dump(out/'complete.json',dict(status='oom',arm=arm,completed_requests=len(rows),error=str(exc)))
    finally:
        if telemetry is not None:telemetry.stop()
        gc.collect();torch.cuda.empty_cache()

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(42)
    prop=torch.cuda.get_device_properties(0);assert (prop.major,prop.minor)==(10,3)
    torch.cuda.set_per_process_memory_fraction(128*2**30/prop.total_memory,0)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs:False
    uuid=str(prop.uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
    dump(ROOT/'environment.json',dict(name=prop.name,compute_capability=[prop.major,prop.minor],gpu_uuid=uuid,total_memory_bytes=prop.total_memory,
        torch=torch.__version__,transformers=transformers.__version__,node=platform.node(),job=os.environ['SLURM_JOB_ID'],model=MODEL,adapter=ADAPTER,adapter_sha256=hashlib.sha256((Path(ADAPTER)/'adapter_model.safetensors').read_bytes()).hexdigest()))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    from peft import PeftModel
    from unittest.mock import patch
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,ADAPTER,autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);tok.bos_token_id=model.config.bos_token_id
    docs=json.loads((ROOT/'workloads.json').read_text());picks=[(di,qi) for qi in [0,8,16,24] for di in range(3)]
    dump(ROOT/'access_patterns.json',dict(picks=picks,source_tokens=32768,query_tokens=512,output_tokens=32,max_chunks=48,min_chunks=4,probe_tokens=32,split_depth=12,arms=ARMS))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        engine=Engine(model,tok,docs,'h_gpu',1,k=48)
        dump(ROOT/'offline.json',dict(write_s=engine.write_s,index_s=engine.index_s,H_bytes=engine.hbytes))
        dump(ROOT/'status.json',dict(phase='CORRECTNESS'));validate_attention(engine)
        order=list(ARMS);random.Random(20260921).shuffle(order);dump(ROOT/'order.json',order)
        for arm in order:run_point(engine,arm,picks,uuid)
    dump(ROOT/'complete.json',dict(complete=True,points=len(ARMS),requests_per_point=len(picks),agent_success_measured=False))
    dump(ROOT/'status.json',dict(phase='COMPLETE',points=len(ARMS)))

if __name__=='__main__':
    try:main()
    except BaseException as exc:dump(ROOT/'failure.json',dict(error=repr(exc)));raise
