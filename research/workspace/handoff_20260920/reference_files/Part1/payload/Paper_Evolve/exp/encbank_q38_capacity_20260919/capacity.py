"""Qwen3.8-27B k48 capacity; distinguish true prefill and synthetic long-KV stress."""
import gc,hashlib,json,os,platform,time,traceback
from pathlib import Path
import torch,transformers
from transformers.cache_utils import DynamicCache
from common import MODELS,load_model,load_state
from hybrid_reader import HybridReader
from batch_cache import merge_caches,decode_layers
from io_utils import save
from telemetry import Nvml
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text())
def stamp():torch.cuda.synchronize();return time.perf_counter()
def cache_bytes(*caches):
    stores={}
    for cache in caches:
        for layer in cache.layers:
            for name in ['keys','values','conv_states','recurrent_states']:
                x=getattr(layer,name,None)
                for t in x.values() if isinstance(x,dict) else [x]:
                    if torch.is_tensor(t):
                        s=t.untyped_storage();stores[(str(t.device),s.data_ptr())]=s.nbytes()
    return sum(stores.values())
def extend_attention(cache,start,end,extra):
    if not extra:return
    for i in range(start,end):
        layer=cache.layers[i]
        if getattr(layer,'keys',None) is None:continue
        for attr in ['keys','values']:
            old=getattr(layer,attr)
            shape=list(old.shape);shape[-2]=extra
            setattr(layer,attr,torch.cat([old,old.new_zeros(shape)],dim=-2))

@torch.inference_mode()
def point(reader,inputs,source,n,prior):
    banks=[];lowers=[];uppers=[];qpos=[];upos=[];encoded=0
    start=stamp();Hbytes=0;bank_begin=start
    # Full independent GPU allocation per session, not shared expand()/views across sessions.
    for i in range(n):
        row=inputs[i%16];chunk=next(c for c in row['chunks'] if len(c)==512)
        h=reader.write(chunk);bank=h.repeat(source//512,1,1);banks.append(bank)
        Hbytes+=bank.numel()*bank.element_size();encoded+=len(chunk)
        del h
    assert len({b.untyped_storage().data_ptr() for b in banks})==n
    bank_seconds=stamp()-bank_begin;prefill_begin=stamp()
    for i in range(n):
        row=inputs[i%16];ql=P['query_lengths'][i%len(P['query_lengths'])]
        query=(row['query']*((ql+len(row['query'])-1)//len(row['query'])))[:ql]
        lower=DynamicCache(config=reader.config);upper=DynamicCache(config=reader.config)
        qh=reader.layers(reader.core.embed_tokens(reader.tensor(query)),0,reader.j,cache=lower)
        sink=reader.write(row['chunks'][0][:1])
        fixed=torch.cat([sink,banks[i][:48].reshape(1,48*512,-1)],dim=1)
        hidden=reader.layers(torch.cat([fixed,qh],dim=1),reader.j,reader.L,cache=upper)
        logits=reader.logits(hidden);assert bool(torch.isfinite(logits).all())
        del hidden,logits,qh,sink,fixed
        extend_attention(lower,0,reader.j,prior);extend_attention(upper,reader.j,reader.L,prior)
        lowers.append(lower);uppers.append(upper);qpos.append(ql+prior);upos.append(1+48*512+ql+prior)
    del lower,upper
    prefill_seconds=stamp()-prefill_begin
    individual_bytes=sum(cache_bytes(l,u) for l,u in zip(lowers,uppers))
    merge_begin=stamp()
    lower,lp=merge_caches(lowers,qpos,reader.config,0,reader.j,consume=True)
    upper,up=merge_caches(uppers,upos,reader.config,reader.j,reader.L,consume=True)
    del lowers,uppers
    merge_seconds=stamp()-merge_begin
    qp=torch.tensor(qpos,device='cuda');pp=torch.tensor(upos,device='cuda')
    times=[];decode_begin=stamp();finite=torch.ones((),device='cuda',dtype=torch.bool)
    for step in range(P['decode_positions']-1):
        tokens=torch.tensor([[inputs[i%16]['replay'][step%63]] for i in range(n)],device='cuda')
        hidden=reader.core.embed_tokens(tokens)
        hidden=decode_layers(reader,hidden,0,reader.j,lower,qp,lp)
        hidden=decode_layers(reader,hidden,reader.j,reader.L,upper,pp,up)
        logits=reader.logits(hidden);finite=finite & torch.isfinite(logits).all();qp+=1;pp+=1
        del logits,hidden,tokens
        times.append(stamp())
    assert bool(finite)
    duration=times[-1]-decode_begin
    return dict(status='ok',source_tokens=source,batch=n,prior_decode_attention_slots=prior,
        long_cache_is_synthetic=bool(prior),selected_chunks=48,query_lengths=qpos,
        hidden_bank_bytes=Hbytes,independent_cache_bytes=individual_bytes,merged_cache_bytes=cache_bytes(lower,upper),
        source_hidden_tokens_allocated=n*source,source_tokens_actually_encoded=encoded,bank_prepare_seconds=bank_seconds,
        prefill_and_cache_extension_seconds=prefill_seconds,cache_merge_seconds=merge_seconds,
        decode_seconds=duration,aggregate_decode_tps=n*(P['decode_positions']-1)/duration,
        mean_per_sequence_decode_tps=(P['decode_positions']-1)/duration,
        decode_step_seconds=[b-a for a,b in zip([decode_begin]+times,times)],
        point_seconds=stamp()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

def main():
    assert torch.cuda.device_count()==1 and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2);torch.manual_seed(P['seed']);torch.backends.cuda.enable_cudnn_sdp(False)
    free,total=torch.cuda.mem_get_info();assert free>P['admission_free_gib']*2**30
    torch.cuda.set_per_process_memory_fraction(P['device_cap_gib']*2**30/total)
    cfg=MODELS[1];assert cfg['path']==P['model'] and cfg['j']==P['j']
    cp=Path(P['adapter_path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']
    cold=time.monotonic();model=load_model(cfg);reader=HybridReader(model,cfg['j']);reader.attach()
    saved=torch.load(cp,map_location='cpu',weights_only=False);assert saved['step']==4000;load_state(reader,saved,cfg);del saved
    model.requires_grad_(False);dev=torch.cuda.get_device_properties(0)
    uuid=str(dev.uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
    save(ROOT/'environment.json',dict(gpu=dev.name,total_memory_bytes=dev.total_memory,gpu_uuid=uuid,cc=[dev.major,dev.minor],
        node=platform.node(),job=os.environ['SLURM_JOB_ID'],torch=torch.__version__,transformers=transformers.__version__,
        model=P['model'],j=P['j'],adapter_sha256=P['adapter_sha256'],model_load_seconds=time.monotonic()-cold,
        hidden_size=reader.config.hidden_size,layer_types=reader.config.layer_types))
    inputs=json.loads((ROOT/'inputs.json').read_text());results=[]
    for source in P['source_tokens']:
        for prior in P['history_decode_tokens']:
            blocked=False
            for n in P['batch_sizes']:
                key=f's{source}_past{prior}_b{n}'
                if blocked:
                    r=dict(status='skipped_after_oom',source_tokens=source,batch=n,prior_decode_attention_slots=prior)
                else:
                    gc.collect();torch.cuda.empty_cache();stamp();torch.cuda.reset_peak_memory_stats()
                    save(ROOT/'status.json',dict(phase='RUNNING',point=key,completed=len(results)))
                    telemetry=Nvml(uuid);telemetry.start()
                    try:r=point(reader,inputs,source,n,prior)
                    except torch.OutOfMemoryError as exc:
                        r=dict(status='oom',source_tokens=source,batch=n,prior_decode_attention_slots=prior,error=str(exc));blocked=True
                    finally:
                        peak=telemetry.stop()
                    r.update(nvml_sampled_peak_bytes=peak,nvml_period_ms=100)
                save(ROOT/'results'/(key+'.json'),r);results.append(r)
    save(ROOT/'complete.json',dict(points=len(results),results=results,scope=P['purpose']))
    save(ROOT/'status.json',dict(phase='COMPLETE',points=len(results)))
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'failure.json',dict(error=traceback.format_exc()));raise
