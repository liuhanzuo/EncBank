"""Qwen3-8B body-only hidden-to-KV pilot, server GPU only."""
import datetime,gc,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN_NAME=sys.argv[1]
ARM='h12'
OUT=ROOT/RUN_NAME/'results'
MODEL='/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B'
ADAPTER='/srv/encbank/comem_infra_recheck_20260912/adapter'
LAYERS=list(range(12,36))
REGS=[.001,.01,.1]

def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def status(phase,**kw):
    obj=dict(phase=phase,arm=ARM,at=datetime.datetime.now().astimezone().isoformat(),**kw)
    dump('status.json',obj);print(json.dumps(obj),flush=True)

def main():
    assert os.environ.get('SLURM_JOB_ID') and not (OUT/'summary.json').exists()
    status('IMPORTS')
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
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
    prop=torch.cuda.get_device_properties(0)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    sha=hashlib.sha256((Path(ADAPTER)/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert sha=='1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13'
    env=dict(arm=ARM,job=os.environ['SLURM_JOB_ID'],gpu=prop.name,gpu_uuid=str(prop.uuid),
        torch=torch.__version__,transformers=transformers.__version__,model=MODEL,adapter_sha256=sha,
        dtype='BF16 backbone and fitted heads, original unmerged FP32 LoRA',allocator_cap_GiB=96,
        cpu_affinity=sorted(os.sched_getaffinity(0)),model_downloaded=False)
    dump('environment.json',env);status('LOAD_MODEL')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):
        wrapper=PeftModel.from_pretrained(model,ADAPTER,autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False)
    assert model.config.num_hidden_layers==36 and model.config.hidden_size==4096
    assert model.config.num_key_value_heads==8 and model.config.head_dim==128
    reader=Reader(model,12,tokenizer=tok)
    dataset=json.loads((ROOT/'dataset.json').read_text())
    def fresh():return DynamicCache(config=model.config)
    def kv(cache,l):return cache.layers[l].keys,cache.layers[l].values
    def unit(h):
        x=h.float();return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16)
    def source(l):return 24 if ARM in ['dual','h24_late'] and l>=24 else 12
    active=([l for l in LAYERS if l!=12] if ARM=='h12' else
        [l for l in LAYERS if l not in [12,24]] if ARM=='dual' else
        list(range(25,36)) if ARM=='h24_late' else [])
    def timed(fn):
        torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
        return result,dict(wall_ms=(time.perf_counter()-begin)*1000,cuda_ms=a.elapsed_time(b))
    def raw_project(h,l):
        layer=reader.layers[l];u=layer.input_layernorm(h)
        return torch.cat([layer.self_attn.k_proj(u),layer.self_attn.v_proj(u)],dim=-1)
    def finish(raw,l,positions):
        k,v=raw.split(1024,-1)
        k=reader.layers[l].self_attn.k_norm(k.reshape(1,-1,8,128)).transpose(1,2)
        v=v.reshape(1,-1,8,128).transpose(1,2)
        cos,sin=reader.rotary_emb(raw,position_ids=positions)
        k=k*cos[:,None]+rotate_half(k)*sin[:,None]
        return k.contiguous(),v.contiguous()
    def capture(chunks):
        ids=torch.tensor(chunks,device='cuda');h=reader.write_chunk(ids).reshape(1,-1,4096)
        sink=reader.write_chunk([model.config.bos_token_id]);h=torch.cat([sink,h],dim=1)
        n=h.shape[1];p=torch.arange(n,device='cuda')[None]
        mask=(p[:,None,:]<=p[:,:,None])[:,None];rope=reader.rotary_emb(h,position_ids=p)
        cache=fresh();states={}
        for l in LAYERS:
            states[l]=h
            h=reader._run_layers(h,slice(l,l+1),mask,p,rope,past_key_values=cache,use_cache=True)
        return dict(states=states,kv={l:kv(cache,l) for l in LAYERS},n=n)
    def to_cpu(bundle):
        return dict(states={l:h.cpu() for l,h in bundle['states'].items()},
            kv={l:tuple(t.cpu() for t in pair) for l,pair in bundle['kv'].items()},n=bundle['n'])
    class Provider:
        def __init__(self,bundle,weights=None,teacher=False):
            self.n=bundle['n'];self.weights=weights;self.teacher=teacher;self.cached=None
            self.positions=torch.arange(self.n,device='cuda')[None]
            self.h={};self.exact={}
            if teacher:self.exact={l:tuple(t.to('cuda') for t in bundle['kv'][l]) for l in LAYERS}
            elif ARM=='exact':self.h={l:h.to('cuda') for l,h in bundle['states'].items()}
            else:
                bases=[12] if ARM=='h12' else [12,24] if ARM=='dual' else [24]
                self.h={l:bundle['states'][l].to('cuda') for l in bases}
                if ARM=='h24_late':self.exact={l:tuple(t.to('cuda') for t in bundle['kv'][l]) for l in range(12,24)}
        def project(self,l):
            if l in self.exact:return self.exact[l]
            if self.weights is None or l not in self.weights:
                raw=raw_project(self.h[l],l)
            else:
                w,b=self.weights[l];raw=F.linear(unit(self.h[source(l)]),w.to(torch.bfloat16),b.to(torch.bfloat16))
            return finish(raw,l,self.positions)
        def materialize(self):self.cached={l:self.project(l) for l in LAYERS};return self.cached
        def bytes(self):
            ts=list(self.h.values())+[t for pair in self.exact.values() for t in pair]
            if self.cached:ts += [t for pair in self.cached.values() for t in pair]
            unique={t.data_ptr():t for t in ts}
            return sum(t.numel()*t.element_size() for t in unique.values())
        def cache(self,on_demand=False):
            if on_demand:return MemoryCache(self)
            if self.cached is None:self.materialize()
            c=fresh()
            for l,pair in self.cached.items():c.update(*pair,l)
            return c
    class MemoryCache:
        def __init__(self,provider):self.provider=provider;self.local=fresh()
        def update(self,key_states,value_states,layer_idx,*args,**kwargs):
            k,v=self.local.update(key_states,value_states,layer_idx)
            mk,mv=self.provider.project(layer_idx)
            return torch.cat([mk,k],dim=2),torch.cat([mv,v],dim=2)
    def prefill(provider,query,on_demand=False):
        qh,bottom,qpos=reader.write_prefill(query)
        top=provider.cache(on_demand);n=provider.n;t=qh.shape[1]
        pos=torch.arange(n,n+t,device='cuda')[None]
        mask=(torch.arange(n+t,device='cuda')[None,:]<=pos[0,:,None])[None,None]
        hidden=reader._run_layers(qh,slice(12,36),mask,pos,reader.rotary_emb(qh,position_ids=pos),
            past_key_values=top,use_cache=True)
        logits=reader.lm_head(reader.norm(hidden))
        return logits,bottom,top,qpos
    def metrics(student,teacher,targets):
        p=F.log_softmax(teacher.float(),dim=-1);q=F.log_softmax(student.float(),dim=-1)
        return dict(kl=float((p.exp()*(p-q)).sum(-1).mean()),
            top1_agreement=float((student.argmax(-1)==teacher.argmax(-1)).float().mean()),
            student_nll=float(F.cross_entropy(student[0].float(),targets)),
            teacher_nll=float(F.cross_entropy(teacher[0].float(),targets)))
    def evaluate(case,weights,check=False):
        bundle=capture(case['memory']);teacher=Provider(bundle,teacher=True)
        reference,*_=prefill(teacher,case['continuation'][:128])
        student=Provider(bundle,weights);actual,*_=prefill(student,case['continuation'][:128])
        targets=torch.tensor(case['continuation'][1:129],device='cuda')
        out=metrics(actual,reference,targets)
        if check:
            uncached,*_=prefill(student,case['continuation'][:128],True)
            parity=metrics(uncached,actual,targets)
            out['resident_vs_on_demand']=parity
            assert abs(parity['kl'])<1e-4 and parity['top1_agreement']>.99,parity
        return out

    results={};selected=None;weights=None
    with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        # Exact H checkpoint control is performed in every arm before any fit.
        status('EXACT_CONTROL')
        bundle=capture(dataset['validation'][0]['memory']);diffs=[]
        positions=torch.arange(bundle['n'],device='cuda')[None]
        for l in LAYERS:
            restored=finish(raw_project(bundle['states'][l],l),l,positions)
            diff=max(float((x.float()-y.float()).abs().max()) for x,y in zip(restored,bundle['kv'][l]))
            rel=max(float((x.float()-y.float()).square().mean().sqrt()/(y.float().square().mean().sqrt()+1e-10)) for x,y in zip(restored,bundle['kv'][l]))
            diffs.append(dict(layer=l+1,max_abs=diff,relative_rms=rel))
            assert rel<1e-5,(l,diff,rel)
        results['exact_control']=diffs;dump('exact_control.json',diffs)
        del bundle,restored;gc.collect();torch.cuda.empty_cache()
        if False:
            status('COLLECT_FIT',documents=32)
            xs={s:[] for s in sorted({source(l) for l in active})};ys={l:[] for l in active}
            for i,case in enumerate(dataset['train']):
                bundle=capture(case['memory'])
                ix=torch.arange(1,bundle['n'],4,device='cuda')
                for s in xs:xs[s].append(unit(bundle['states'][s][:,ix]).reshape(-1,4096).cpu())
                for l in active:ys[l].append(raw_project(bundle['states'][l],l)[:,ix].reshape(-1,2048).cpu())
                del bundle
                if (i+1)%8==0:status('COLLECT_FIT',completed=i+1,total=32)
            X={s:torch.cat(rows).to('cuda').float() for s,rows in xs.items()};del xs
            Y={l:torch.cat(rows) for l,rows in ys.items()};del ys
            X={s:torch.cat([x,torch.ones((len(x),1),device='cuda')],dim=-1) for s,x in X.items()}
            G={s:x.T@x for s,x in X.items()}
            rhs={l:X[source(l)].T@Y[l].to('cuda').float() for l in active};del X,Y
            candidates=[];best=float('inf');fit_start=time.perf_counter()
            for reg in REGS:
                status('FIT_AND_VALIDATE',ridge=reg)
                factors={s:torch.linalg.cholesky(g+torch.eye(g.shape[0],device='cuda')*(reg*g.diag().mean())) for s,g in G.items()}
                candidate={}
                for l in active:
                    solution=torch.cholesky_solve(rhs[l],factors[source(l)])
                    candidate[l]=(solution[:-1].T.contiguous().to(torch.bfloat16),solution[-1].to(torch.bfloat16))
                val=[evaluate(case,candidate) for case in dataset['validation']]
                score=statistics.mean(v['kl'] for v in val)
                candidates.append(dict(ridge=reg,validation_kl=score,documents=val))
                dump('validation_selection.json',candidates)
                if score<best:best=score;weights=candidate;selected=reg
                del factors,candidate
            results['fit_seconds']=time.perf_counter()-fit_start;results['validation']=candidates
            del G,rhs,solution;gc.collect();torch.cuda.empty_cache()
            checkpoint=ROOT/ARM/'heads.pt'
            torch.save(dict(arm=ARM,ridge=selected,weights={l:tuple(t.cpu() for t in pair) for l,pair in weights.items()}),checkpoint)
            results['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        from distill import train_heads
        weights,training=train_heads({**globals(),**locals()},RUN_NAME,ROOT)
        results['distillation']=training
        checkpoint=ROOT/RUN_NAME/'heads.pt'
        torch.save(dict(arm=ARM,run=RUN_NAME,weights={l:tuple(t.cpu() for t in pair) for l,pair in weights.items()},training=training),checkpoint)
        results['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        status('TEST_HELDOUT',documents=8)
        test=[]
        for i,case in enumerate(dataset['test']):
            row=dict(id=case['id'],**evaluate(case,weights,check=i==0))
            test.append(row);dump('test.json',test)
            status('TEST_HELDOUT',completed=i+1,total=8,kl=row['kl'])
        results['test']=test
        # Short free generations are diagnostic examples, never task scores.
        generations=[]
        for case in dataset['test'][:2]:
            bundle=capture(case['memory']);entry=dict(id=case['id'])
            for name,teacher in [('teacher',True),('student',False)]:
                provider=Provider(bundle,weights,teacher=teacher)
                logits,bottom,top,qpos=prefill(provider,case['continuation'][:128]);token=logits[:,-1].argmax(-1)
                output=[int(token.item())]
                for step in range(31):
                    logits=reader.decode_step(token,bottom,top,qpos+step,provider.n+qpos+step)
                    token=logits[:,-1].argmax(-1);output.append(int(token.item()))
                entry[name]=dict(token_ids=output,text=tok.decode(output))
                del provider,logits,bottom,top,token
            generations.append(entry);del bundle
        dump('generations.json',generations)
        gc.collect();torch.cuda.empty_cache()
        # Runtime on the same saved PG19 workload used by the previous timing probes.
        pg=json.loads((ROOT/'vendor/workloads.json').read_text())[0]
        chosen=pg['queries'][0]['selected'];runtime=[]
        for count in [1,4,12]:
            status('RUNTIME',chunks=count)
            chunks=[pg['source'][i*512:(i+1)*512] for i in chosen[:count]]
            gpu_bundle=capture(chunks);cpu_bundle=to_cpu(gpu_bundle);del gpu_bundle
            gc.collect();torch.cuda.empty_cache()
            query=pg['queries'][0]['query'][:64]
            forced=torch.tensor(pg['queries'][0]['query'][64:96],device='cuda')
            for mode in ['native_KV','resident','on_demand']:
                provider=Provider(cpu_bundle,weights,teacher=mode=='native_KV')
                projection=[]
                if mode=='resident':
                    for repeat in range(7):
                        _,elapsed=timed(provider.materialize)
                        if repeat>=2:projection.append(elapsed)
                elif mode=='native_KV':provider.materialize()
                rows=[]
                for repeat in range(5):
                    torch.cuda.reset_peak_memory_stats()
                    (logits,bottom,top,qpos),prefill_time=timed(lambda:prefill(provider,query,mode=='on_demand'))
                    def decode():
                        last=None
                        for step in range(32):
                            last=reader.decode_step(forced[step:step+1],bottom,top,qpos+step,provider.n+qpos+step)
                        return last
                    last,elapsed=timed(decode)
                    if repeat>=2:rows.append(dict(decode_32=elapsed,ms_per_token=elapsed['wall_ms']/32,
                        query_prefill=prefill_time,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    del logits,bottom,top,last
                runtime.append(dict(chunks=count,mode=mode,history_persistent_bytes=provider.bytes(),
                    projection=projection,observations=rows,median_ms_per_token=statistics.median(r['ms_per_token'] for r in rows)))
                dump('runtime.json',runtime)
                del provider;gc.collect();torch.cuda.empty_cache()
            del cpu_bundle,forced
        results['runtime']=runtime
    results.update(complete=True,arm=ARM,run=RUN_NAME,environment=env,selected_ridge=selected,
        test_mean={key:statistics.mean(row[key] for row in results['test']) for key in ['kl','top1_agreement','student_nll','teacher_nll']},
        scope='H12 per-layer heads initialized from prior ridge and trained by logits KL; same32/4/8 document split; no agent task accuracy claim',
        h24_scope='H24 uses the fixed teacher joint-memory prefix; h24_late retains exact KV13..24; not a free reusable H24 across changing retrieval',
        runtime_scope='GPU-resident history, fixed ordered PG19 chunks, 64 query tokens then 32 teacher-forced decode steps; no eviction policy, transfers or live agent timing',
        source_manifest=json.loads((ROOT/'source_manifest.json').read_text()))
    dump('summary.json',results);status('COMPLETE',test_mean=results['test_mean'])

if __name__=='__main__':
    try:main()
    except BaseException:
        dump('failure.json',dict(traceback=traceback.format_exc()));raise
