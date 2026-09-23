"""Isolated full CoMem teacher-forced decoder comparison; no benchmark attempts."""
import contextlib,copy,gc,hashlib,json,os,time,traceback
from pathlib import Path

H=Path(__file__).resolve().parent
def save(name,value):
    with (H/name).open('x') as f:json.dump(value,f,indent=2)

def main():
    from common import MODELS,load_model,load_state,tokenizer,torch
    from hybrid_reader import HybridReader
    from runtime_identity import resolve_configs
    from transformers.cache_utils import DynamicCache
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf
    from batch_cache import merge_caches,decode_layers
    from vllm_gdn_adapter import VllmGdnDecode
    from vendor_fused_recurrent import fused_recurrent_gated_delta_rule
    assert os.environ.get('COMEM_ISOLATED_KERNEL_PILOT')=='1'
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(4203);torch.backends.cuda.enable_cudnn_sdp(False)
    plan=json.loads((H/'baseline_plan.json').read_text());identity,cfg=resolve_configs(plan,MODELS[1])
    assert hashlib.sha256(Path(plan['adapter_path']).read_bytes()).hexdigest()==plan['adapter_sha256']
    started=time.time();model=load_model(cfg);reader=HybridReader(model,cfg['j']);reader.attach()
    ckpt=torch.load(plan['adapter_path'],map_location='cpu',weights_only=False)
    resolve_configs(plan,identity,ckpt);load_state(reader,ckpt,identity);del ckpt
    model.requires_grad_(False);tok=tokenizer(cfg)
    original=hf.torch_recurrent_gated_delta_rule
    replacement=VllmGdnDecode(kernel=fused_recurrent_gated_delta_rule)
    def hashed(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    save('model_ready.json',dict(epoch=time.time(),load_seconds=time.time()-started,job_id=os.environ['SLURM_JOB_ID'],
        hostname=os.uname().nodename,torch=torch.__version__,gpu=str(torch.cuda.get_device_properties(0)),
        model=cfg,adapter_sha256=plan['adapter_sha256'],allocated_bytes=torch.cuda.memory_allocated(),benchmark_attempts=0))
    words=tok.encode('The experiment keeps each document memory immutable. A query reads selected states and then generates an answer. ',add_special_tokens=False)
    segment=(words*100)[:512];query=(words*20)[:128]
    forced=(words*20)[:32]
    results=[]
    with torch.inference_mode():
        memory=reader.write(segment);h_sha=hashed(memory)
        def prepare(batch,chunks):
            ls=[];us=[]
            for row in range(batch):
                lower=DynamicCache(config=reader.config);upper=DynamicCache(config=reader.config)
                qh=reader.layers(reader.core.embed_tokens(reader.tensor(query)),0,reader.j,cache=lower)
                hidden=reader.layers(torch.cat([memory]*chunks+[qh],dim=1),reader.j,reader.L,cache=upper)
                ls.append(lower);us.append(upper)
            qlen=len(query);ulen=512*chunks+qlen
            lower,lp=merge_caches(ls,[qlen]*batch,reader.config,0,reader.j,consume=True)
            upper,up=merge_caches(us,[ulen]*batch,reader.config,reader.j,reader.L,consume=True)
            return lower,upper,lp,up,torch.full((batch,),qlen,device='cuda'),torch.full((batch,),ulen,device='cuda')
        def step(batch,cache,token):
            lower,upper,lp,up,qp,upos=cache
            h=reader.core.embed_tokens(torch.full((batch,1),token,device='cuda',dtype=torch.long))
            h=decode_layers(reader,h,0,reader.j,lower,qp,lp)
            h=decode_layers(reader,h,reader.j,reader.L,upper,upos,up)
            logits=reader.logits(h);qp+=1;upos+=1
            return logits
        for batch,chunks in [(1,4),(8,4),(8,12)]:
            hf.torch_recurrent_gated_delta_rule=original
            torch.cuda.synchronize();prep_start=time.perf_counter();cache=prepare(batch,chunks)
            torch.cuda.synchronize();prep_s=time.perf_counter()-prep_start
            initial=copy.deepcopy(cache)
            reference_logits=[];start=time.perf_counter()
            for token in forced:reference_logits.append(step(batch,cache,token))
            torch.cuda.synchronize();base_s=time.perf_counter()-start
            expected=torch.cat(reference_logits,dim=1);del reference_logits,cache
            hf.torch_recurrent_gated_delta_rule=replacement
            # Compile all kernel shapes on a scratch cache, excluded from timing.
            warm=copy.deepcopy(initial);step(batch,warm,forced[0]);torch.cuda.synchronize();del warm
            optimized=[];start=time.perf_counter()
            for token in forced:optimized.append(step(batch,initial,token))
            torch.cuda.synchronize();optimized_s=time.perf_counter()-start
            actual=torch.cat(optimized,dim=1);del optimized,initial
            delta=(actual.float()-expected.float()).abs()
            greedy_same=float((actual.argmax(-1)==expected.argmax(-1)).float().mean())
            # Report real differences before deciding the rollout gate.
            row=dict(batch=batch,history_chunks=chunks,history_tokens=512*chunks,query_tokens=len(query),
                teacher_forced_steps=len(forced),prefill_seconds=prep_s,baseline_decode_seconds=base_s,
                optimized_decode_seconds=optimized_s,baseline_tokens_per_second=batch*len(forced)/base_s,
                optimized_tokens_per_second=batch*len(forced)/optimized_s,speedup=base_s/optimized_s,
                max_logit_abs_error=float(delta.max()),mean_logit_abs_error=float(delta.mean()),
                greedy_token_agreement=greedy_same,H_unchanged=hashed(memory)==h_sha,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            row['numerical_gate_pass']=row['H_unchanged'] and greedy_same==1.0 and row['max_logit_abs_error']<=.125
            results.append(row);save(f'case_b{batch}_h{chunks}.json',row);print(json.dumps(row),flush=True)
            del actual,expected,delta;gc.collect();torch.cuda.empty_cache()
        # Attribute baseline operator costs on one fixed short workload.
        hf.torch_recurrent_gated_delta_rule=original
        cache=prepare(8,4)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for token in forced[:4]:step(8,cache,token)
            torch.cuda.synchronize()
        table=prof.key_averages().table(sort_by='self_device_time_total',row_limit=30)
        (H/'baseline_profile.txt').write_text(table)
        profiling=[dict(key=x.key,count=x.count,self_cpu_us=x.self_cpu_time_total,
            self_device_us=getattr(x,'self_device_time_total',None),device_us=getattr(x,'device_time_total',None)) for x in prof.key_averages()]
        save('baseline_profile.json',profiling)
        hf.torch_recurrent_gated_delta_rule=original
    save('model_probe_result.json',dict(status='PASS' if all(x['numerical_gate_pass'] for x in results) else 'NUMERICAL_REVIEW',
        cases=results,job_id=os.environ['SLURM_JOB_ID'],ended_epoch=time.time(),benchmark_attempts=0,
        production_approved=False,scope='Same model/adapter and teacher-forced tokens; GDN decode-only prototype, no sampler or full vLLM scheduler.'))

if __name__=='__main__':
    try:main()
    except BaseException:
        save('model_probe_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
