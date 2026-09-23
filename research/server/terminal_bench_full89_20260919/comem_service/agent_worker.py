"""Shared native CoMem weights; task-local H banks and transient split caches."""
from pathlib import Path
import gc,hashlib,json,os,platform,subprocess,time,traceback
from persistence import dump
from session import Session,digest,request_seed
H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text());R=H/'run_comem';B=R/'mailbox'
def event(kind,**kw):
    with (R/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,epoch=time.time(),**kw))+'\n')
def main():
    H.resolve().relative_to(Path('/srv/encbank').resolve())
    event('imports_begin')
    from common import MODELS,load_model,load_state,tokenizer,torch
    from hybrid_reader import HybridReader
    from memory_selectors import iter_bm25_indices
    from transformers.cache_utils import DynamicCache
    import transformers
    def stamp():torch.cuda.synchronize();return time.perf_counter()
    def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def cache_bytes(c):
        seen=set();storage={}
        def walk(v):
            if id(v) in seen:return
            seen.add(id(v))
            if torch.is_tensor(v):
                s=v.untyped_storage();storage[(str(v.device),s.data_ptr())]=s.nbytes()
            elif isinstance(v,dict):
                for x in v.values():walk(x)
            elif isinstance(v,(tuple,list)):
                for x in v:walk(x)
            elif type(v).__module__.startswith('transformers.cache_utils'):walk(vars(v))
        walk(c);return sum(storage.values())
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(P['seed']);torch.backends.cuda.enable_cudnn_sdp(False)
    cfg=MODELS[1];assert cfg['j']==P['j'] and cfg['path']==P['model']
    admission=[]
    for n in range(3):
        free,total=torch.cuda.mem_get_info();assert free>=P['admission_free_gib']*2**30
        admission.append(dict(epoch=time.time(),free=free,total=total,gpus=subprocess.check_output(['nvidia-smi'],text=True),
            processes=subprocess.check_output(['ps','-u','liuhanzuo','-o','pid,ppid,etime,args'],text=True)))
        if n<2:time.sleep(5)
    torch.cuda.set_per_process_memory_fraction(P['device_cap_gib']*2**30/total)
    dump(R/'admission.json',dict(status='PASS',samples=admission))
    cold=time.monotonic();tok=tokenizer(cfg);event('model_load_begin')
    # Hash before allocation/copy; every formal launch uses the frozen adapter.
    cp=Path(P['adapter_path']).resolve();cp.relative_to(Path('/srv/encbank').resolve())
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==P['adapter_sha256']
    model=load_model(cfg);reader=HybridReader(model,cfg['j']);reader.attach()
    ckpt=torch.load(cp,map_location='cpu',weights_only=False);assert ckpt['step']==4000
    load_state(reader,ckpt,cfg);del ckpt;model.requires_grad_(False)
    assert all(p.device.type=='cuda' for p in model.parameters())
    stop=model.generation_config.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop or tok.eos_token_id])
    dev=torch.cuda.get_device_properties(0)
    dump(R/'worker_ready.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],hostname=platform.node(),gpu=dev.name,
        cc=[dev.major,dev.minor],uuid=str(getattr(dev,'uuid','unavailable')),torch=torch.__version__,transformers=transformers.__version__,
        cold_start_seconds=time.monotonic()-cold,model_weights_loaded_once=True,max_live_sessions=P['max_live_sessions'],
        max_simultaneous_gpu_generations=1,offload=False,plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
        sdpa_cudnn_enabled=torch.backends.cuda.cudnn_sdp_enabled(),serialization=P['serialization'],adapter_sha256=P['adapter_sha256']))
    sessions={};seen=set();released=set();last=time.monotonic()
    def releases():
        for path in B.glob('*.release.json'):
            if path.name in released:continue
            d=json.loads(path.read_text());sid=d['task_id'];s=sessions.pop(sid,None)
            if s is not None:
                size=sum(t.numel()*t.element_size() for t in s.bank.values());del s
            else:size=0
            released.add(path.name);gc.collect()
            dump(path.with_name(path.name.replace('.release.','.released.')),dict(task_id=sid,released_H_bytes=size,
                allocated_bytes=torch.cuda.memory_allocated(),remaining_sessions=list(sessions)))
            event('session_release',task_id=sid,released_H_bytes=size)
    with torch.inference_mode():
      while True:
        releases()
        pending=sorted((p for p in B.glob('*.request.json') if p.name not in seen),key=lambda p:p.stat().st_mtime_ns)
        if not pending:
            if (B/'stop.json').exists():break
            if time.monotonic()-last>14400:raise TimeoutError('No owner requests for four hours')
            time.sleep(.1);continue
        for path in pending:
            releases();q=json.loads(path.read_text());rid=q['request_id'];sid=q['task_id']
            began=stamp();cancel=B/(rid+'.cancel.json')
            reply=dict(request_id=rid,task=q['task'],task_id=sid,step=q['step'],arm='comem',status='error',
                queue_seconds=time.time()-q['published_epoch'])
            event('request_start',request_id=rid,task=q['task'],step=q['step'],live_sessions=len(sessions))
            if cancel.exists() or time.time()>=q['deadline_epoch']:
                reply.update(status='cancelled' if cancel.exists() else 'deadline',generated_tokens=0,text='')
                dump(B/(rid+'.response.json'),reply);seen.add(path.name);last=time.monotonic();continue
            assert q['task'] in P['tasks']
            if sid not in sessions:
                assert q['step']==0 and len(sessions)<P['max_live_sessions'],'Session not released or missing'
                sessions[sid]=Session(q['task'])
            s=sessions[sid];assert s.task==q['task']
            full,static,chunks,query=s.split(tok,q['messages'],q['step'])
            # Template, reasoning effort and full-history cap match the running Dense block.
            if len(full)>=P['context_tokens']:
                reply.update(status='context_limit',text='',prompt_tokens=len(full),generated_tokens=0)
                dump(B/(rid+'.response.json'),reply);seen.add(path.name);last=time.monotonic();continue
            selection_start=stamp()
            search=tok.encode(s.initial_question+'\n'+q['messages'][-1]['content'],add_special_tokens=False)
            ix=iter_bm25_indices([torch.tensor(c) for c in chunks],search,12,iter_hop_topk=4,iter_rounds=0) if chunks else []
            selection_end=stamp();fixed=static+[chunks[i] for i in ix];pack=sum(fixed,[])+query
            seed=request_seed(P,q['task'],q['step']);generator=torch.Generator(device='cuda');generator.manual_seed(seed)
            # Serial GPU dispatch: all native transient caches belong to this request only.
            for obj in [model,model.model]:
                if hasattr(obj,'rope_deltas'):obj.rope_deltas=None
            torch.cuda.reset_peak_memory_stats();write_start=stamp();new={};hashes={};encoded=reused=0
            # Recompute only a changed partial tail and genuinely new independent chunks.
            for c in static+chunks:
                k=digest(c)
                if k in new:continue
                if k in s.bank:
                    assert thash(s.bank[k])==s.hashes[k];new[k]=s.bank[k];hashes[k]=s.hashes[k];reused+=len(c)
                else:
                    t=reader.write(c).detach();new[k]=t;hashes[k]=thash(t);encoded+=len(c)
            s.bank=new;s.hashes=hashes;del new,hashes
            write_end=stamp();prefill=stamp();lower=DynamicCache(config=reader.config);upper=DynamicCache(config=reader.config)
            states=[s.bank[digest(c)] for c in fixed]
            qh=reader.layers(reader.core.embed_tokens(reader.tensor(query)),0,reader.j,cache=lower)
            hidden=reader.layers(torch.cat(states+[qh],dim=1),reader.j,reader.L,cache=upper)
            logits=reader.logits(hidden);position=len(pack);qposition=len(query);del hidden,qh,states
            generated=[];first=None;limit=min(P['max_new_tokens'],P['context_tokens']-len(full))
            for step in range(limit):
                if cancel.exists() or time.time()>=q['deadline_epoch']:
                    reply['status']='cancelled' if cancel.exists() else 'deadline';break
                scores=logits[0,-1].float()/P['temperature']
                if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Nonfinite CoMem logits; no sanitization or retry')
                values,indices=scores.topk(P['top_k']);cumulative=values.softmax(-1).cumsum(-1)
                mask=cumulative>P['top_p'];mask[1:]=mask[:-1].clone();mask[0]=False;values[mask]=-float('inf')
                token=int(indices[torch.multinomial(values.softmax(-1),1,generator=generator)].item());generated.append(token)
                if first is None:first=stamp()
                if token in stop or step==limit-1:break
                hidden=reader.core.embed_tokens(reader.tensor([token]))
                hidden=reader.layers(hidden,0,reader.j,cache=lower,offset=qposition);qposition+=1
                hidden=reader.layers(hidden,reader.j,reader.L,cache=upper,offset=position);position+=1
                logits=reader.logits(hidden);del hidden
            ended=stamp();first=first or ended
            state_ok=all(thash(t)==v.hashes[k] for v in sessions.values() for k,t in v.bank.items());assert state_ok
            reply.update(status='ok' if reply['status']=='error' else reply['status'],text=tok.decode(generated,skip_special_tokens=True),
                full_history_tokens=len(full),prompt_tokens=len(pack),prompt_ids=pack,full_prompt_sha256=digest(full),
                generated_ids=generated,generated_tokens=len(generated),seed=seed,hit_generation_cap=len(generated)==limit and (not generated or generated[-1] not in stop),
                archived_tokens=sum(map(len,chunks)),query_tokens=len(query),selected_history_chunks=ix,selected_pack_sha256=digest(pack),
                selection_seconds=selection_end-selection_start,write_seconds=write_end-write_start,newly_written_tokens=encoded,reused_written_tokens=reused,
                persistent_H_bytes=sum(t.numel()*t.element_size() for t in s.bank.values()),all_sessions_H_bytes=sum(t.numel()*t.element_size() for v in sessions.values() for t in v.bank.values()),
                active_native_cache_bytes=cache_bytes(lower)+cache_bytes(upper),raw_history_int64_bytes=len(full)*8,
                model_seconds=ended-prefill,ttft_seconds=first-prefill,request_seconds=ended-began,
                decode_tokens_per_second=max(0,len(generated)-1)/max(1e-9,ended-first),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),state_hashes_unchanged=state_ok)
            del lower,upper,logits,scores,values,indices,cumulative,mask,generator
            gc.collect();reply['post_request_allocated_bytes']=torch.cuda.memory_allocated()
            dump(B/(rid+'.response.json'),reply);seen.add(path.name);last=time.monotonic()
            event('request_complete',request_id=rid,status=reply['status'],generated_tokens=len(generated),seconds=time.monotonic()-began)
    sessions.clear();gc.collect();torch.cuda.empty_cache()
    dump(R/'worker_complete.json',dict(requests=len(seen),released_sessions=len(released),allocated_bytes=torch.cuda.memory_allocated(),actual_shutdown=True))
if __name__=='__main__':
    try:main()
    except BaseException:
        dump(R/'worker_failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
