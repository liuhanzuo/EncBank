"""Actual batch8 service: serial independent Write/prefill, ragged batched decode."""
from pathlib import Path
import gc,hashlib,json,time
from persistence import dump
from session import Session,digest,request_seed,native_ids
from unbounded_policy import output_allowance,deadline_expired,native_memory_admissible
from batch_cache import merge_caches,decode_layers,compact
from batch_cache_refill import join_cache_groups


def serve(P,R,B,model,reader,tok,stop,event,torch,DynamicCache,iter_bm25_indices):
    sessions={};seen=set();released=set();last=time.monotonic();batch_number=0
    def stamp():torch.cuda.synchronize();return time.perf_counter()
    def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def immutable():return all(thash(t)==s.hashes[k] for s in sessions.values() for k,t in s.bank.items())
    def state_bytes():return sum(t.numel()*t.element_size() for s in sessions.values() for t in s.bank.values())
    def cache_bytes(cache):
        stores={}
        for layer in cache.layers:
            for name in ['keys','values','conv_states','recurrent_states']:
                x=getattr(layer,name,None)
                for t in x.values() if isinstance(x,dict) else [x]:
                    if torch.is_tensor(t):
                        st=t.untyped_storage();stores[(str(t.device),st.data_ptr())]=st.nbytes()
        return sum(stores.values())
    def reclaim(where):
        reserved=torch.cuda.memory_reserved();allocated=torch.cuda.memory_allocated()
        if reserved>=P['allocator_reclaim_reserved_gib']*2**30 and reserved-allocated>=P['allocator_reclaim_gap_gib']*2**30:
            started=stamp();torch.cuda.empty_cache();ended=stamp()
            event('allocator_cache_reclaim',where=where,seconds=ended-started,
                allocated_before=allocated,reserved_before=reserved,
                allocated_after=torch.cuda.memory_allocated(),reserved_after=torch.cuda.memory_reserved())
    def releases(protected=()):
        for path in B.glob('*.release.json'):
            if path.name in released:continue
            sid=json.loads(path.read_text())['task_id']
            if sid in protected:continue
            s=sessions.pop(sid,None)
            size=sum(t.numel()*t.element_size() for t in s.bank.values()) if s is not None else 0
            del s;released.add(path.name);gc.collect()
            dump(path.with_name(path.name.replace('.release.','.released.')),dict(task_id=sid,released_H_bytes=size,
                allocated_bytes=torch.cuda.memory_allocated(),remaining_sessions=list(sessions)))
            event('session_release',task_id=sid,released_H_bytes=size)
    def cancelled(q):
        if (B/(q['request_id']+'.cancel.json')).exists():return 'cancelled'
        if deadline_expired(q):return 'deadline'
        return None
    def may_admit(path,rows,current_cache):
        q=json.loads(path.read_text())
        if cancelled(q):return True
        full=native_ids(tok,q['messages'])
        old=sessions.get(q['task_id'])
        old_h=sum(t.numel()*t.element_size() for t in old.bank.values()) if old else 0
        growth=max(0,len(full)*reader.config.hidden_size*2-old_h)
        allowed,projected=native_memory_admissible(torch.cuda.memory_allocated(),current_cache,growth,rows,P)
        if not allowed:
            event('memory_admission_wait',request_id=q['request_id'],projected_bytes=projected,active_rows=rows-1)
            if rows==1:raise MemoryError('Native memory capacity cannot fit one full-context generation with retained H; no token truncation or quality zero')
        return allowed
    def prepare(path):
        q=json.loads(path.read_text());rid=q['request_id'];sid=q['task_id'];began=stamp()
        reclaim('before_request_prefill')
        reply=dict(request_id=rid,task=q['task'],task_id=sid,step=q['step'],arm='encbank',status='error',
            queue_seconds=time.time()-q['published_epoch'])
        event('request_start',request_id=rid,task=q['task'],step=q['step'],live_sessions=len(sessions),batch_number=batch_number)
        status=cancelled(q)
        if status:
            reply.update(status=status,text='',generated_tokens=0);dump(B/(rid+'.response.json'),reply);seen.add(path.name);return None
        assert q['task'] in P['tasks']
        if sid not in sessions:
            assert q['step']==0 and len(sessions)<P['max_live_sessions'],'Session capacity or first step invalid'
            sessions[sid]=Session(q['task'])
        s=sessions[sid];assert s.task==q['task']
        full,static,chunks,query=s.split(tok,q['messages'],q['step'])
        selection_start=stamp();search=tok.encode(s.initial_question+'\n'+q['messages'][-1]['content'],add_special_tokens=False)
        ix=iter_bm25_indices([torch.tensor(c) for c in chunks],search,P['top_k_chunks'],iter_hop_topk=P['iter_hop_topk'],iter_rounds=P['iter_rounds']) if chunks else []
        selection_end=stamp();fixed=static+[chunks[i] for i in ix];pack=sum(fixed,[])+query
        if max(len(query),len(pack))>=P['context_tokens']:
            reply.update(status='context_limit',text='',prompt_tokens=len(pack),generated_tokens=0,actual_input_tokens=max(len(query),len(pack)))
            dump(B/(rid+'.response.json'),reply);seen.add(path.name);return None
        seed=request_seed(P,q['task'],q['step']);generator=torch.Generator(device='cuda');generator.manual_seed(seed)
        for obj in [model,model.model]:
            if hasattr(obj,'rope_deltas'):obj.rope_deltas=None
        write_start=stamp();new={};hashes={};encoded=reused=0
        for c in static+chunks:
            key=digest(c)
            if key in new:continue
            if key in s.bank:
                assert thash(s.bank[key])==s.hashes[key];new[key]=s.bank[key];hashes[key]=s.hashes[key];reused+=len(c)
            else:
                value=reader.write(c).detach();new[key]=value;hashes[key]=thash(value);encoded+=len(c)
        s.bank=new;s.hashes=hashes;write_end=stamp();prefill=stamp()
        lower=DynamicCache(config=reader.config);upper=DynamicCache(config=reader.config)
        qh=reader.layers(reader.core.embed_tokens(reader.tensor(query)),0,reader.j,cache=lower)
        states=[s.bank[digest(c)] for c in fixed]
        hidden=reader.layers(torch.cat(states+[qh],dim=1),reader.j,reader.L,cache=upper)
        logits=reader.logits(hidden);prefill_end=stamp()
        reply.update(full_history_tokens=len(full),prompt_tokens=len(pack),prompt_ids=pack,full_prompt_sha256=digest(full),seed=seed,
            archived_tokens=sum(map(len,chunks)),query_tokens=len(query),selected_history_chunks=ix,selected_pack_sha256=digest(pack),
            configured_top_k_chunks=P['top_k_chunks'],candidate_history_chunks=len(chunks),actual_selected_chunks=len(ix),selected_history_tokens=sum(len(chunks[i]) for i in ix),
            configured_auto_rounds=-(-P['top_k_chunks']//P['iter_hop_topk']),selection_algorithm='iterBM25 positive-overlap no fill',
            selection_seconds=selection_end-selection_start,write_seconds=write_end-write_start,newly_written_tokens=encoded,reused_written_tokens=reused,
            persistent_H_bytes=sum(t.numel()*t.element_size() for t in s.bank.values()),raw_history_int64_bytes=len(full)*8,
            independent_prefill_seconds=prefill_end-prefill,independent_prefill_cache_bytes=cache_bytes(lower)+cache_bytes(upper))
        return dict(path=path,q=q,reply=reply,lower=lower,upper=upper,logits=logits,qpos=len(query),upos=len(pack),
            generated=[],first=None,limit=output_allowance(P['context_tokens'],max(len(query),len(pack)),P['max_new_tokens']),generator=generator,
            began=began,prefill=prefill,prefill_end=prefill_end)
    def finish(row,status,shared_cache_bytes,batch_initial):
        ended=stamp();first=row['first'] or ended;g=row['generated'];reply=row['reply']
        assert immutable(),'Persistent H changed during Read'
        reply.update(status=status,text=tok.decode(g,skip_special_tokens=True),generated_ids=g,generated_tokens=len(g),
            hit_generation_cap=False,configured_generation_token_cap=None,hit_context_capacity=len(g)==row['limit'] and (not g or g[-1] not in stop),all_sessions_H_bytes=state_bytes(),
            shared_batch_cache_bytes=shared_cache_bytes,active_native_cache_bytes=None,batch_number=batch_number,batch_initial_size=row.get('admission_rows',batch_initial),cohort_initial_size=batch_initial,
            configured_decode_batch_size=P['decode_batch_size'],memory_scope='shared cohort peak; not per-request exclusive memory',
            model_seconds=ended-row['prefill'],ttft_seconds=first-row['prefill'],request_seconds=ended-row['began'],
            full_worker_ttft_seconds=first-row['began'],batch_prefill_wait_seconds=first-row['prefill_end'],
            decode_tokens_per_second=max(0,len(g)-1)/max(1e-9,ended-first),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            current_allocated_bytes=torch.cuda.memory_allocated(),current_reserved_bytes=torch.cuda.memory_reserved(),state_hashes_unchanged=True)
        dump(B/(row['q']['request_id']+'.response.json'),reply);seen.add(row['path'].name)
        event('request_complete',request_id=row['q']['request_id'],status=status,generated_tokens=len(g),seconds=ended-row['began'],batch_number=batch_number,batch_initial_size=row.get('admission_rows',batch_initial),cohort_initial_size=batch_initial)
    with torch.inference_mode():
        while True:
            releases()
            pending=sorted((p for p in B.glob('*.request.json') if p.name not in seen),key=lambda p:p.stat().st_mtime_ns)
            if not pending:
                if (B/'stop.json').exists():break
                time.sleep(.1);continue
            collect=time.monotonic()
            while len(pending)<P['decode_batch_size'] and time.monotonic()-collect<P['batch_collect_seconds']:
                time.sleep(.05);pending=sorted((p for p in B.glob('*.request.json') if p.name not in seen),key=lambda p:p.stat().st_mtime_ns)
            chosen=pending[:P['decode_batch_size']]
            assert len({json.loads(p.read_text())['task_id'] for p in chosen})==len(chosen)
            batch_number+=1;batch_begin=stamp();torch.cuda.reset_peak_memory_stats()
            active=[]
            for path in chosen:
                cached=sum(cache_bytes(r['lower'])+cache_bytes(r['upper']) for r in active)
                if not may_admit(path,len(active)+1,cached):break
                row=prepare(path)
                if row is not None:active.append(row)
            if not active:last=time.monotonic();continue
            n=len(active);event('batch_ready',batch_number=batch_number,size=n,request_ids=[r['q']['request_id'] for r in active],query_lengths=[r['qpos'] for r in active],upper_lengths=[r['upos'] for r in active])
            lower,lp=merge_caches([r['lower'] for r in active],[r['qpos'] for r in active],reader.config,0,reader.j,consume=True)
            upper,upad=merge_caches([r['upper'] for r in active],[r['upos'] for r in active],reader.config,reader.j,reader.L,consume=True)
            qp=torch.tensor([r['qpos'] for r in active],device='cuda');upos=torch.tensor([r['upos'] for r in active],device='cuda')
            logits=torch.cat([r['logits'] for r in active],dim=0)
            for r in active:
                r['admission_rows']=n
                del r['lower'],r['upper'],r['logits']
            last_refill_check=time.monotonic()
            generated_total=0;decode_steps=0;size_hist={}
            while active:
                keep=[];tokens=[];next_rows=[];bytes_now=cache_bytes(lower)+cache_bytes(upper)
                for index,row in enumerate(active):
                    status=cancelled(row['q'])
                    if not status:
                        scores=logits[index,-1].float()/P['temperature']
                        if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Nonfinite batch8 Encbank logits; no sanitization/retry')
                        values,indices=scores.topk(P['top_k']);cumulative=values.softmax(-1).cumsum(-1)
                        mask=cumulative>P['top_p'];mask[1:]=mask[:-1].clone();mask[0]=False;values[mask]=-float('inf')
                        token=int(indices[torch.multinomial(values.softmax(-1),1,generator=row['generator'])].item())
                        row['generated'].append(token);generated_total+=1
                        if row['first'] is None:row['first']=stamp()
                        if token in stop:status='ok'
                        elif len(row['generated'])>=row['limit']:status='context_limit'
                    if status:finish(row,status,bytes_now,n)
                    else:keep.append(index);tokens.append(token);next_rows.append(row)
                if not keep:break
                if len(keep)!=len(active):
                    idx=torch.tensor(keep,device='cuda');qp,lp=compact(lower,qp,lp,idx);upos,upad=compact(upper,upos,upad,idx)
                active=next_rows;hidden=reader.core.embed_tokens(torch.tensor(tokens,device='cuda')[:,None])
                hidden=decode_layers(reader,hidden,0,reader.j,lower,qp,lp)
                hidden=decode_layers(reader,hidden,reader.j,reader.L,upper,upos,upad)
                logits=reader.logits(hidden);del hidden;qp+=1;upos+=1;decode_steps+=1
                size_hist[len(active)]=size_hist.get(len(active),0)+1
                # Admit queued requests while older rows continue decoding. Preserve FIFO,
                # independent prefill/RNG/H banks, and true per-row positions.
                if len(active)<P['decode_batch_size'] and time.monotonic()-last_refill_check>=P.get('refill_interval_seconds',.5):
                    last_refill_check=time.monotonic()
                    releases({r['q']['task_id'] for r in active})
                    active_paths={r['path'].name for r in active}
                    waiting=sorted((p for p in B.glob('*.request.json') if p.name not in seen and p.name not in active_paths),key=lambda p:p.stat().st_mtime_ns)
                    extra=[]
                    for path in waiting[:P['decode_batch_size']-len(active)]:
                        cached=cache_bytes(lower)+cache_bytes(upper)+sum(cache_bytes(r['lower'])+cache_bytes(r['upper']) for r in extra)
                        if not may_admit(path,len(active)+len(extra)+1,cached):break
                        row=prepare(path)
                        if row is not None:extra.append(row)
                    if extra:
                        assert len(active)+len(extra)<=P['decode_batch_size']
                        assert len({r['q']['task_id'] for r in active+extra})==len(active)+len(extra)
                        nl,np=merge_caches([r['lower'] for r in extra],[r['qpos'] for r in extra],reader.config,0,reader.j,consume=True)
                        nu,nup=merge_caches([r['upper'] for r in extra],[r['upos'] for r in extra],reader.config,reader.j,reader.L,consume=True)
                        lower,lp=join_cache_groups([lower,nl],[lp,np],reader.config,0,reader.j,consume=True)
                        upper,upad=join_cache_groups([upper,nu],[upad,nup],reader.config,reader.j,reader.L,consume=True)
                        qp=torch.cat([qp,torch.tensor([r['qpos'] for r in extra],device='cuda')])
                        upos=torch.cat([upos,torch.tensor([r['upos'] for r in extra],device='cuda')])
                        logits=torch.cat([logits,*[r['logits'] for r in extra]],dim=0)
                        before=len(active)
                        for r in extra:
                            r['admission_rows']=before+len(extra)
                            del r['lower'],r['upper'],r['logits']
                        active.extend(extra)
                        event('batch_refill',batch_number=batch_number,before=before,added=len(extra),after=len(active),request_ids=[r['q']['request_id'] for r in extra])
                        del nl,np,nu,nup,extra
                        reclaim('after_refill_old_caches_released')

            del active,lower,upper,logits,qp,upos,lp,upad;gc.collect()
            event('batch_complete',batch_number=batch_number,initial_size=n,generated_tokens=generated_total,
                decode_forwards=decode_steps,batch_size_histogram=size_hist,seconds=stamp()-batch_begin,
                post_batch_allocated_bytes=torch.cuda.memory_allocated(),persistent_H_bytes=state_bytes())
            last=time.monotonic()
    sessions.clear();gc.collect();torch.cuda.empty_cache()
    dump(R/'worker_complete.json',dict(requests=len(seen),batches=batch_number,released_sessions=len(released),allocated_bytes=torch.cuda.memory_allocated(),actual_shutdown=True))
