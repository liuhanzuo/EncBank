"""Numerical/streaming qualification plus explicit SYNTHETIC memory stress.
No forced-token or synthetic test is scored as a Terminal-Bench attempt.
"""
import gc,json,os,time,traceback
from pathlib import Path
import torch
from common import ROOT,PLAN as P,save,verify_sources
from model_setup import load,tokens
from hybrid_hot import Session,Request,quantum,run,sync,bytes_cache,row_cache

def kl(a,b):
    x=a.float().log_softmax(-1);y=b.float().log_softmax(-1)
    return max(0,float((x.exp()*(x-y)).sum(-1).mean()))

@torch.no_grad()
def main():
    verify_sources();model,r,tok,stop=load(P)
    messages=json.loads((ROOT/'trace_messages.json').read_text())
    ids=tokens(tok,messages);ids=ids[:min(1600,len(ids))]
    assert len(ids)>600
    save(ROOT/'qualification_status.json',dict(phase='numerical',input_tokens=len(ids),epoch=time.time()))
    checks=[]
    # Dense split-prefill agrees with stock Transformers on the same prefix.
    with r.adapter(False):
        dense=Request(Session(r,'dense',0,'dense'),ids[:600],7)
        stock=model(input_ids=r.tensor(ids[:600]),use_cache=False,logits_to_keep=1).logits
        value=kl(stock,dense.logits);assert value<.01,value
        checks.append(dict(test='dense_vs_stock',kl=value))
        del dense,stock
    sessions={a:Session(r,a,24,'trace') for a in ['cold','hot']}
    cold=Request(sessions['cold'],ids,7);hot=Request(sessions['hot'],ids,7)
    value=kl(cold.logits,hot.logits);assert value<.01,value
    again=Request(sessions['hot'],ids,7)
    value2=kl(cold.logits,again.logits);assert value2<.01,value2
    assert again.events[-1]['hit_chunks']>0,again.events
    checks.append(dict(test='hot_prefix_repeat',cold_hot_kl=value,warm_kl=value2,event=again.events[-1]))
    del again
    # Change an early selected chunk. Its descendants must miss.
    changed=list(ids);changed[10]=int(tok.encode('changed',add_special_tokens=False)[0])
    altered=Request(sessions['hot'],changed,7)
    assert altered.events[-1]['hit_chunks']==0,altered.events
    checks.append(dict(test='changed_prefix_invalidates',event=altered.events[-1]));del altered
    del cold,hot,sessions;gc.collect();torch.cuda.empty_cache()
    # Gate the repaired batching path before the longer streaming/capacity runs.
    forced=(ids*4)[:1056];batch_checks=[]
    for arm,lens in [('hot',[700,719]),('hot',[719,700]),('hot',[700,700]),('dense',[700,719])]:
        with r.adapter(arm!='dense'):
            rows=[Request(Session(r,arm,24,str(i)),ids[:n],20+i) for i,n in enumerate(lens)]
            individual=[]
            for i,n in enumerate(lens):
                q=Request(Session(r,arm,24,str(i)),ids[:n],20+i)
                quantum([q],stop,steps=1,forced=forced);individual.append(q.logits.cpu());del q
            quantum(rows,stop,steps=1,forced=forced)
            values=[kl(a,x.logits.cpu()) for a,x in zip(individual,rows)]
            batch_checks.append(dict(arm=arm,lengths=lens,kl=values))
            save(ROOT/'batch_qualification_progress.json',batch_checks)
            assert max(values)<.02,(arm,lens,values)
            del rows;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'batch_qualification.json',dict(passed=True,rows=batch_checks))
    # Forced trace-token streaming exercises current-chunk promotion and two
    # retrieval boundaries, while all quality verdicts remain disabled.
    forced=(ids*4)[:1056];stream=[];refs=None
    for arm in ['cold','hot']:
        save(ROOT/'qualification_status.json',dict(phase='stream',arm=arm,epoch=time.time()))
        s=Session(r,arm,24,'stream');row=Request(s,ids[:700],11);logs=[];start=sync()
        while len(row.generated)<len(forced):
            quantum([row],stop,steps=min(32,len(forced)-len(row.generated)),forced=forced)
            if row.refresh_needed():
                row.lower=row.upper=None;row.prefill();logs.append(row.logits.cpu())
        elapsed=sync()-start
        if refs is None:refs=logs
        else:
            vals=[kl(a,b) for a,b in zip(refs,logs)];assert len(vals)==2 and max(vals)<.02,vals
            assert sum(x.get('promoted_hits',0) for x in row.events)>0,row.events
            checks.append(dict(test='stream_boundary_equivalence',kl=vals))
        stream.append(dict(arm=arm,seconds=elapsed,events=row.events,times=dict(row.time),peak_allocated=torch.cuda.max_memory_allocated()))
        del row,s;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'numerical_qualification.json',dict(passed=True,checks=checks,stream=stream))
    stress=[]
    # Reserve physically touched tensor storage with realistic tensor shapes.
    # These are synthetic capacity tests, not long-context accuracy or TB speed.
    for arm,batch,n,cap in [('dense',8,131072,0),('dense',16,131072,0),('dense',8,262144-512,0),
        ('hot',8,131072,24),('hot',16,131072,24),('hot',32,131072,24),('hot',24,262144,24)]:
        save(ROOT/'qualification_status.json',dict(phase='synthetic_memory',arm=arm,batch=batch,history=n,epoch=time.time()))
        rows=[];extra=[];workspace=None;torch.cuda.reset_peak_memory_stats();start=sync()
        try:
            for i in range(batch):
                row=Request(Session(r,arm,cap,'stress'+str(i)),ids[:700],100+i)
                if arm=='dense':
                    for layer in row.upper.layers:
                        if hasattr(layer,'keys') and layer.keys is not None:
                            shape=list(layer.keys.shape);shape[-2]=n
                            layer.keys=torch.zeros(shape,dtype=layer.keys.dtype,device='cuda')
                            layer.values=torch.zeros_like(layer.keys)
                    row.upos=n
                else:
                    # History bank and hot state tensors are additional to active
                    # decode caches. All buffers are touched, never merely reserved.
                    extra.append(torch.zeros((1,n,5120),dtype=torch.bfloat16,device='cuda'))
                    template=next(v for v in row.s.pool.items.values() if v['length']==512)
                    row.s.pool.items.clear();row.s.pool.bytes=0
                    for k in range(cap):
                        cache=row_cache(template['cache'],r.config,r.j,r.L,0,0)
                        size=bytes_cache(cache);row.s.pool.items[('stress',k)]=dict(cache=cache,length=512,bytes=size,origin='synthetic');row.s.pool.bytes+=size
                rows.append(row)
            # Explicitly touch 16GiB additional workspace; merge/activation peaks
            # are then measured on top, rather than claiming all 32GiB is free.
            workspace=torch.zeros((16*2**30,),dtype=torch.uint8,device='cuda')
            steady=torch.cuda.memory_allocated();q=quantum(rows,stop,steps=2,forced=forced)
            result=dict(arm=arm,batch=batch,history_tokens=n,hot_chunks=cap,passed=True,
                steady_allocated_gib=steady/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,quantum=q,seconds=sync()-start)
        except torch.OutOfMemoryError:
            result=dict(arm=arm,batch=batch,history_tokens=n,hot_chunks=cap,passed=False,error='CUDA_OOM',
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        stress.append(result);save(ROOT/'memory_stress.json',dict(synthetic=True,rows=stress))
        rows=[];extra=[];workspace=None
        row=template=cache=layer=None
        gc.collect();torch.cuda.empty_cache()
    save(ROOT/'qualification_complete.json',dict(passed=True,numerical=True,batch=True,stress=stress,
        gpu=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,epoch=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'qualification_failure.json',dict(error=traceback.format_exc(),epoch=time.time()));raise
