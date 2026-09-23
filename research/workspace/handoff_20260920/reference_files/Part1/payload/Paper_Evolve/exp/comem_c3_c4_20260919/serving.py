"""Closed-loop serving-oriented GPU microbatch benchmark; raw request timestamps."""
import argparse,gc,hashlib,json,os,pickle,platform,random,subprocess,sys,time
import config
import common as original
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from comem import CoMem
from comem.selectors import iter_bm25_indices
from controlled_train_core import atomic_json
from queue_metrics import closed_loop,summarize
from bm25_index import BM25Index
from torch.nn.attention import sdpa_kernel,SDPBackend

class BatchedCoMem(CoMem):
    def _as_ids(self,token_ids):
        ids=torch.as_tensor(token_ids,device=self.device,dtype=torch.long)
        if ids.ndim==1:ids=ids[None,:]
        assert ids.ndim==2 and ids.shape[0]>=1
        return ids

    @torch.no_grad()
    def decode_step(self,token_id,bottom_cache,top_cache,q_local_pos,pack_pos):
        ids=torch.as_tensor(token_id,device=self.device,dtype=torch.long).reshape(-1,1)
        emb=self.embed_tokens(ids)
        if self.resume_j>0:
            p=torch.tensor([[q_local_pos]],device=self.device)
            emb=self._run_layers(emb,slice(0,self.resume_j),self._decode_attn_mask(q_local_pos+1),p,
                self.rotary_emb(emb,position_ids=p),past_key_values=bottom_cache,use_cache=True)
        p=torch.tensor([[pack_pos]],device=self.device)
        hidden=self._run_layers(emb,slice(self.resume_j,self.num_layers),self._decode_attn_mask(pack_pos+1),p,
            self.rotary_emb(emb,position_ids=p),past_key_values=top_cache,use_cache=True)
        return self.lm_head(self.norm(hidden))

def stamp():torch.cuda.synchronize();return time.perf_counter()

def load(adapter):
    if config.LOCAL:
        sys.path.insert(0,str(config.ROOT.parent))
        import gpu_gate
        # The user already authorized using this desktop when otherwise idle.
        # Keep cross-project locking and reject another Python GPU process.
        old=gpu_gate.HARD_IDLE_CEILING_GIB;gpu_gate.HARD_IDLE_CEILING_GIB=7.
        try:
            admission=gpu_gate.acquire_gpu(need_gb=24,cap_gb=config.CAP_BYTES/1e9,idle_slack_gb=7,poll=10,
                tag='comem-c3-c4-local-serving')
        finally:gpu_gate.HARD_IDLE_CEILING_GIB=old
        atomic_json(config.ROOT/'local_admission.json',admission)
        assert '5090' in torch.cuda.get_device_name()
    else:assert os.environ.get('SLURM_JOB_ID')
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(42)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *a,**kw:False
    prop=torch.cuda.get_device_properties(0)
    assert prop.total_memory>=config.CAP_BYTES
    torch.cuda.set_per_process_memory_fraction(config.CAP_BYTES/prop.total_memory,0)
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(config.MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    wrapper,model=original.attach(base,adapter)
    model.requires_grad_(False);tok.bos_token_id=model.config.bos_token_id
    assert tok.bos_token_id==151643
    return wrapper,model,tok

def gpu_metadata():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15)
    assert result.returncode==0
    return dict(gpu=torch.cuda.get_device_name(0),device=str(torch.cuda.get_device_properties(0)),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
        nvidia_smi=result.stdout,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        node=platform.node(),slurm_job=os.environ.get('SLURM_JOB_ID'),torch=torch.__version__,
        transformers=transformers.__version__,cuda=torch.version.cuda,cap_bytes=config.CAP_BYTES)

class Engine:
    def __init__(self,model,tok,j,docs,build_store=True):
        self.model,self.tok,self.j,self.docs=model,tok,j,docs
        self.reader=BatchedCoMem(model,j,tokenizer=tok)
        self.chunks=[list(torch.tensor(d['source'],dtype=torch.long).split(512)) for d in docs]
        t=time.perf_counter();self.indexes=[BM25Index(ch) for ch in self.chunks]
        self.index_build_s=time.perf_counter()-t
        self.index_serialized_bytes=len(pickle.dumps(self.indexes,protocol=5))
        self.stores=[];self.write_records=[]
        for d,chunks in (zip(docs,self.chunks) if build_store else []):
            t=stamp();states=[]
            for chunk in chunks:states.append(self.reader.write_chunk(chunk).cpu().pin_memory())
            sink=self.reader.write_chunk([tok.bos_token_id]).cpu().pin_memory()
            elapsed=stamp()-t
            self.stores.append((sink,states))
            self.write_records.append(dict(book_index=d['book_index'],j=j,source_tokens=len(d['source']),
                write_s=elapsed,cache_bytes=sum(s.numel()*s.element_size() for s in [sink,*states]),
                cpu_pinned=True))
        self.requests=[(d,q) for q in range(len(docs[0]['queries'])) for d in range(len(docs))]
        random.Random(18791).shuffle(self.requests)

    @torch.inference_mode()
    def infer(self,ids,method,count=config.OUTPUT_TOKENS,capture_logits=False):
        # Retrieval, fetch/CPU packing and H2D are inside this service interval.
        picks=[self.requests[i%len(self.requests)] for i in ids]
        chosen=[]
        if method!='dense':
            for di,qi in picks:
                q=self.docs[di]['queries'][qi]
                ix=self.indexes[di].select(q['query'][:32],12,2)
                assert ix==q['selected'] and len(ix)==12
                chosen.append(ix)
        if method=='comem':
            packed=torch.cat([torch.cat([self.stores[di][0],*[self.stores[di][1][ix] for ix in sel]],dim=1)
                for (di,qi),sel in zip(picks,chosen)],dim=0).pin_memory().to('cuda',non_blocking=True)
            query=torch.tensor([self.docs[di]['queries'][qi]['query'] for di,qi in picks]).pin_memory().to('cuda',non_blocking=True)
            q,bottom,qpos=self.reader.write_prefill(query)
            logits,top,ppos=self.reader.read_prefill(None,[packed],q)
        else:
            full=[]
            for n,(di,qi) in enumerate(picks):
                source=self.docs[di]['source'] if method=='dense' else torch.cat([self.chunks[di][ix] for ix in chosen[n]]).tolist()
                full.append([self.tok.bos_token_id]+source+self.docs[di]['queries'][qi]['query'])
            inputs=torch.tensor(full,dtype=torch.long).pin_memory().to('cuda',non_blocking=True)
            out=self.model(input_ids=inputs,use_cache=True,logits_to_keep=1)
            logits=out.logits;cache=out.past_key_values
        # First token is actually ready on CPU, not just its GPU logits.
        token=logits[:,-1].float().argmax(-1);generated=[token.cpu().tolist()];first=stamp()
        first_logits=logits[:,-1].float().cpu() if capture_logits else None
        for _ in range(1,count):
            if method=='comem':
                logits=self.reader.decode_step(token,bottom,top,qpos,ppos);qpos+=1;ppos+=1
            else:
                out=self.model(input_ids=token[:,None],past_key_values=cache,use_cache=True,logits_to_keep=1)
                logits=out.logits;cache=out.past_key_values
            token=logits[:,-1].float().argmax(-1);generated.append(token.cpu().tolist())
        ended=stamp()
        return first,ended,dict(request_templates=picks,generated_ids=list(map(list,zip(*generated))),
            first_logits=first_logits) if capture_logits else dict(request_templates=picks,generated_ids=list(map(list,zip(*generated))))

    def check(self,out):
        checks={}
        # Distinct real queries; compare single and batch first-logit distributions.
        # BF16 batching may perturb argmax when logits tie, so assess both, never demand exact generation equality.
        for method in ['comem','selected_replay']:
            batch=self.infer([0,1,2,3],method,count=2,capture_logits=True)[2]
            singles=[self.infer([i],method,count=2,capture_logits=True)[2] for i in [0,1,2,3]]
            ref=torch.cat([s['first_logits'] for s in singles]);actual=batch['first_logits']
            diff=(actual-ref).abs();rms=float(diff.square().mean().sqrt())
            top1=actual.argmax(-1).eq(ref.argmax(-1)).tolist()
            checks[method]=dict(max_abs=float(diff.max()),rms=rms,first_top1_equal=top1,
                scalar_tokens=[s['generated_ids'][0] for s in singles],batched_tokens=batch['generated_ids'])
            assert torch.isfinite(actual).all() and rms<.15 and float(diff.max())<1.0,checks[method]
        # Batched class at batch=1 must match original scalar helper, including cached decode.
        reader=CoMem(self.model,self.j,tokenizer=self.tok)
        d,q=self.requests[0];sel=self.docs[d]['queries'][q]['selected'];query=self.docs[d]['queries'][q]['query']
        sink=self.stores[d][0].cuda();states=[self.stores[d][1][i].cuda() for i in sel]
        # Use the original scalar methods, with the same fixed-output/no-EOS policy as timing.
        qh,bc,qp=reader.write_prefill(query);lg,tc,pp=reader.read_prefill(sink,states,qh)
        t=int(lg[0,-1].float().argmax());scalar=[t]
        lg=reader.decode_step(t,bc,tc,qp,pp);scalar.append(int(lg[0,-1].float().argmax()))
        recompute=reader.read_core(sink,states,reader.write_chunk(query+[t]),logits_tail=1)
        assert int(recompute[0,-1].float().argmax())==scalar[-1]
        actual=self.infer([0],'comem',count=2)[2]['generated_ids'][0]
        assert scalar==actual,(scalar,actual)
        checks['original_scalar_equal']=True
        checks['cached_recompute_equal']=True
        atomic_json(out,checks)
        return checks

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--rep',type=int,default=0);p.add_argument('--length',type=int,default=32768)
    p.add_argument('--j',type=int,default=12);p.add_argument('--seed',type=int);p.add_argument('--smoke',action='store_true')
    p.add_argument('--method',choices=['comem','selected_replay'],required=True)
    p.add_argument('--concurrency',type=int,choices=[1,4],required=True)
    a=p.parse_args();assert a.length in config.SOURCE_LENGTHS
    depth=a.seed is not None
    adapter=config.ROOT/'training'/config.tag(a.j,a.seed)/'final' if depth else config.ADAPTER
    root=config.ROOT/('latency_depth' if depth else 'serving')
    out=root/(config.tag(a.j,a.seed) if depth else f'rep{a.rep}')/str(a.length)/(a.method+'_c'+str(a.concurrency))
    if a.smoke:out=config.ROOT/'smoke_serving'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    wrapper,model,tok=load(adapter)
    meta=dict(**gpu_metadata(),j=a.j,adapter=str(adapter),output_tokens=32,batch_limit=4,
        smoke=a.smoke,source_tokens=a.length,closed_loop=True,network_included=False,
        tokenizer_included=False,document_write_included_in_online=False,precision='BF16 backbone / FP32 LoRA',
        adapter_sha256=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
        method=a.method,concurrency=a.concurrency,cpu_threads=2,
        scheduling='one model copy; FIFO closed-loop synchronous GPU microbatches, max_batch=4; no continuous batching',
        timing_origin='request submission to first/final token ID available on CPU; includes queue/retrieval/H2D/queryWrite/prefill/decode',
        index='same BM25 corpus statistics prepared offline; actual iterative query scoring online; no cached rankings used as retrieval',
        attention='fused SDPA with explicit repeat_kv; math disabled',rope_extension=False,
        workload='3 PG19 books x 32 overlapping continuation queries; no answer reuse')
    atomic_json(out/'metadata.json',meta)
    docs=json.loads((config.ROOT/'data'/f'serving_{a.length}.json').read_text())
    # Correctness uses SHORT distinct prompts to avoid consuming benchmark-scale Dense memory/time.
    short=[]
    for d in docs:
        source=d['source'][:6144];chunks=list(torch.tensor(source).split(512));queries=[]
        for q in d['queries'][:2]:
            query=q['query'][:32]
            ix=iter_bm25_indices(chunks,query[:32],12,iter_hop_topk=2,iter_rounds=0)
            assert len(ix)==12
            queries.append(dict(query=query,selected=ix))
        short.append(dict(book_index=d['book_index'],source=source,queries=queries))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        probe=Engine(model,tok,a.j,short);probe.check(out/'correctness.json');del probe
        gc.collect();torch.cuda.empty_cache()
        engine=Engine(model,tok,a.j,docs,build_store=a.method=='comem')
        atomic_json(out/'write.json',engine.write_records)
        import psutil
        atomic_json(out/'persistent_cpu.json',dict(process_rss_bytes=psutil.Process().memory_info().rss,
            H_pinned_bytes=sum(x['cache_bytes'] for x in engine.write_records),
            source_token_tensor_bytes=sum(c.numel()*c.element_size() for cs in engine.chunks for c in cs),
            bm25_serialized_bytes=engine.index_serialized_bytes,index_build_s=engine.index_build_s,
            serialized_index_is_not_live_RSS=True,request_templates=len(engine.requests)))
        cells=[(a.method,a.concurrency)]
        random.Random(5280+a.rep).shuffle(cells)
        for method,c in cells:
            dest=out/f'{method}_c{c}';dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():continue
            assert not (dest/'requests.jsonl').exists(),'Interrupted formal point; retain it and use a new explicit attempt.'
            # Each point starts with fresh queue and peak counters; failures never lower batch automatically.
            gc.collect();torch.cuda.empty_cache()
            phase='warmup'
            try:
                warm=16 if not a.smoke else 4
                closed_loop(lambda ix:engine.infer(ix,method),c,max(c,warm),config.MAX_BATCH)
                torch.cuda.reset_peak_memory_stats()
                phase='formal'
                count=4 if a.smoke else config.REQUESTS
                with (dest/'requests.jsonl').open('w') as stream:
                    def record_batch(batch,n):
                        for row in batch:stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        if n%20==0:atomic_json(out/'progress.json',dict(method=method,concurrency=c,requests=n,target=count))
                    records,span=closed_loop(lambda ix:engine.infer(ix,method),c,count,config.MAX_BATCH,on_batch=record_batch)
                result=dict(status='ok',method=method,concurrency=c,**summarize(records,span,32),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    failed_requests=0)
                assert result['peak_reserved_bytes']<=config.CAP_BYTES
            except torch.OutOfMemoryError as exc:
                partial=getattr(exc,'serving_progress',{})
                with (dest/'partial_requests.jsonl').open('w') as f:
                    for row in partial.get('completed',[]):f.write(json.dumps(row)+'\n')
                succeeded=len(partial.get('completed',[]));failed=partial.get('failed_batch_size',0)
                result=dict(status='OOM',method=method,concurrency=c,error=str(exc),
                    requests=None,p99=None,throughput=None,cap_bytes=config.CAP_BYTES,phase=phase,
                    completed_before_oom=succeeded,failed_batch=partial.get('failed_batch'),
                    attempted_request_failure_rate=failed/(failed+succeeded) if failed+succeeded else None,
                    point_censored_after_oom=True)
            atomic_json(dest/'complete.json',result)
            atomic_json(out/'progress.json',dict(last=method,concurrency=c,status=result['status']))
            print(json.dumps(result),flush=True)
        atomic_json(out/'complete.json',dict(complete=True,cells=len(cells),smoke=a.smoke))

if __name__=='__main__':main()
