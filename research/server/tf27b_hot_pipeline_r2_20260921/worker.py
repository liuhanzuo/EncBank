"""Shared Transformers model, independently seeded sessions, bounded hot pool."""
import gc,json,os,time,traceback
from collections import Counter
from pathlib import Path
import torch
from common import ROOT,PLAN as P,save,sha,verify_sources
from model_setup import load,tokens
from hybrid_hot import Session,Request,quantum,row_cache,bytes_cache,sync

OUT=ROOT/'worker';OUT.mkdir(exist_ok=True)
def event(kind,**kw):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,epoch=time.time(),**kw))+'\n')

@torch.no_grad()
def main():
    verify_sources();qroot=Path(P['qualification_root'])
    assert json.loads((qroot/'qualification_complete.json').read_text())['passed']
    assert json.loads((qroot/'parent_exit.json').read_text())['exit_code']==0
    # Numerical qualification applies only to the identical core implementation.
    for n in ['hybrid_hot.py','hybrid_reader.py','batch_cache.py','model_setup.py']:
        assert sha(ROOT/n)==sha(qroot/n),n
    model,r,tok,stop=load(P,adapter=P['arm']!='dense')
    sessions={};active=[];seen=set();released=set();hist=Counter();gpu_seconds=0;start=sync()
    device=str(torch.cuda.get_device_properties(0))
    save(OUT/'ready.json',dict(job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),gpu=device,torch=torch.__version__,
        arm=P['arm'],max_decode_batch=P['decode_batch_size'],max_tasks=P['task_concurrency'],base_allocated_gib=torch.cuda.memory_allocated()/2**30))
    for task in P['tasks']:save(ROOT/'pairs'/task/'ready.json',dict(shared_worker=True,job=os.environ['SLURM_JOB_ID']))
    def empty_inactive_dense():
        running={x.s.task for x in active}
        for task,s in sessions.items():
            if task not in running:s.dense_cache=None;s.dense_ids=[]
    def trim_hot():
        # Keep whole H banks; discard cheap-to-rebuild hot entries when headroom
        # shrinks. Actual capacity may therefore be below configured maximum.
        if P['arm']!='hot':return
        for task,s in sorted(sessions.items(),key=lambda x:getattr(x[1],'last_used',0)):
            while s.pool.items and torch.cuda.memory_allocated()/2**30>200:
                _,old=s.pool.items.popitem(last=False);s.pool.bytes-=old['bytes'];s.pool.stats['global_evictions']+=1;del old
    def finish(x):
        ended=sync();q=x.request;path=x.path;s=x.s
        # Do not retain a view of an entire old batch in an idle dense session.
        if s.arm=='dense' and x.status=='ok':
            s.dense_cache=row_cache(x.upper,r.config,0,r.L,0,0);s.dense_ids=list(x.ids)
        payload=dict(request_id=q['request_id'],request_sha256=sha(path),status=x.status,
            text=tok.decode(x.generated,skip_special_tokens=True),prompt_tokens=len(x.prompt_ids),prompt_ids=x.prompt_ids,
            generated_tokens=len(x.generated),generated_ids=x.generated,full_history_tokens=len(x.initial),
            total_seconds=ended-x.began,ttft_seconds=None if x.first is None else x.first-x.began,
            times=dict(x.time),events=x.events,h_bytes=s.hbytes(),hot_bytes=s.pool.bytes,hot_stats=dict(s.pool.stats),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            memory_scope='shared worker cumulative peak; request time includes scheduling of other requests',
            engine='transformers',arm=P['arm'],cached_prefix_tokens=x.events[0].get('cached_prefix_tokens',x.events[0].get('hit_chunks',0)*512))
        save(path.with_name(path.name.replace('.request.','.response.')),payload);seen.add(str(path))
        event('request_complete',task=s.task,request_id=q['request_id'],status=x.status,seconds=ended-x.began,
            generated_tokens=len(x.generated),hits=sum(e.get('hit_chunks',0) for e in x.events),
            promoted_hits=sum(e.get('promoted_hits',0) for e in x.events))
        x.upper=x.lower=x.logits=None;x.qh=[];s.last_used=time.time()
    while True:
        for task in P['tasks']:
            rel=ROOT/'mailbox'/task/'release.json'
            if rel.exists() and task not in released and all(x.s.task!=task for x in active):
                sessions.pop(task,None);released.add(task);gc.collect();event('session_release',task=task)
        pending=sorted((p for p in (ROOT/'mailbox').glob('*/*.request.json') if str(p) not in seen and all(x.path!=p for x in active)),key=lambda p:p.stat().st_mtime_ns)
        for path in pending:
            if len(active)>=P['decode_batch_size']:break
            q=json.loads(path.read_text());task=q['task']
            assert task in P['tasks'] and task not in released and all(x.s.task!=task for x in active)
            ids=tokens(tok,q['messages'])
            if len(ids)>=P['context_tokens'] and P['arm']=='dense':
                save(path.with_name(path.name.replace('.request.','.response.')),dict(status='context_limit',
                    request_sha256=sha(path),error='NATIVE_CONTEXT_CAPACITY',prompt_tokens=len(ids),generated_tokens=0));seen.add(str(path));continue
            if task not in sessions:sessions[task]=Session(r,P['arm'],P['hot_chunks'],task)
            s=sessions[task];trim_hot()
            current=torch.cuda.memory_allocated();active_bytes=sum(bytes_cache(x.upper)+bytes_cache(x.lower) for x in active)
            growth=max(0,len(ids)*10240-s.hbytes()) if P['arm']!='dense' else 0
            if P['arm']=='dense':
                # Strict admission reserves full native context, not a reply cap.
                projected=current-active_bytes+(len(active)+1)*16*2**30+32*2**30
                if projected>248*2**30:
                    empty_inactive_dense();gc.collect()
                    current=torch.cuda.memory_allocated();projected=current-active_bytes+(len(active)+1)*16*2**30+32*2**30
            else:
                projected=current+growth+(max(0,s.pool.budget-s.pool.bytes) if P['arm']=='hot' else 0)+2*2**30+32*2**30
            if projected>248*2**30:
                event('admission_wait',task=task,active=len(active),projected_gib=projected/2**30)
                if not active:raise MemoryError('Cannot admit one request without losing H history; physical capacity')
                break
            seed=P['seed']+int(__import__('hashlib').sha256((task+':'+str(q['step'])).encode()).hexdigest()[:8],16)
            x=Request(s,ids,seed%(2**31));x.request=q;x.path=path;active.append(x)
            event('request_start',task=task,request_id=q['request_id'],active=len(active),history_tokens=len(ids),
                prefill=x.time.get('prefill_seconds',0),h_bytes=s.hbytes(),hot_bytes=s.pool.bytes)
        if active:
            metric=quantum(active,stop,steps=32,temperature=P['temperature'],
                cancel=lambda x:x.path.with_name(x.path.name.replace('.request.','.cancel.')).exists())
            hist[str(metric['batch'])]+=metric['steps'];gpu_seconds+=metric['seconds']
            remain=[]
            for x in active:
                if x.status:finish(x)
                else:
                    if x.refresh_needed():
                        x.upper=x.lower=None;gc.collect();x.prefill()
                    remain.append(x)
            active=remain
            save(OUT/'status.json',dict(epoch=time.time(),completed=len(seen),active=len(active),sessions=len(sessions),
                batch_steps=dict(hist),decode_cohort_seconds=gpu_seconds,allocated_gib=torch.cuda.memory_allocated()/2**30,
                reserved_gib=torch.cuda.memory_reserved()/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                h_gib=sum(s.hbytes() for s in sessions.values())/2**30,hot_gib=sum(s.pool.bytes for s in sessions.values())/2**30))
        elif (ROOT/'worker'/'stop.json').exists():break
        else:time.sleep(.1)
    sessions.clear();active.clear();gc.collect();torch.cuda.empty_cache()
    save(OUT/'complete.json',dict(requests=len(seen),batch_steps=dict(hist),decode_cohort_seconds=gpu_seconds,
        elapsed_seconds=sync()-start,allocated_after_release_gib=torch.cuda.memory_allocated()/2**30))
if __name__=='__main__':
    try:main()
    except BaseException:
        failure=dict(error=traceback.format_exc(),epoch=time.time());save(OUT/'failure.json',failure)
        for task in P['tasks']:save(ROOT/'pairs'/task/'worker_failure.json',failure)
        raise
