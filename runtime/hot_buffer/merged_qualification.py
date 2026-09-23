"""New-weight startup checks; does not claim equivalence to legacy FP32 LoRA."""
import gc,hashlib,time
import torch
from common import ROOT,save
from hybrid_hot import Session,Request,quantum,sync


@torch.no_grad()
def qualify(reader,tok,plan):
    started=sync()
    sample=tok.encode('Inspect the source, execute a deterministic test, and preserve the existing interface.\n',add_special_tokens=False)
    ids=(sample*100)[:1800]
    inputs=[ids[:700],ids[:719]]
    force=(sample*10)[:32]
    rows=[];checks=[]
    def kl(a,b):
        x,y=a.float().log_softmax(-1),b.float().log_softmax(-1)
        return float((x.exp()*(x-y)).sum(-1).max())
    for index,seq in enumerate(inputs):
        hot=Session(reader,'hot',24,'qualification_hot'+str(index))
        first=Request(hot,seq,11+index)
        first_logit=first.logits.clone()
        h_proof={key:hashlib.sha256(v[0].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for key,v in hot.bank.items()}
        del first
        reused=Request(hot,seq,11+index)
        cold=Request(Session(reader,'cold',0,'qualification_cold'+str(index)),seq,11+index)
        a,b=kl(first_logit,reused.logits),kl(cold.logits,reused.logits)
        assert a<=.02 and b<=.02,(a,b)
        assert reused.events[0]['hit_chunks']>=1
        for key,v in hot.bank.items():
            assert h_proof[key]==hashlib.sha256(v[0].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        quantum([cold],set(),steps=8,forced=force)
        checks.append(dict(input_tokens=len(seq),hot_reuse_kl=a,cold_hot_prefill_kl=b,hot_reused_chunks=reused.events[0]['hit_chunks'],h_unchanged=True))
        rows.append((reused,cold.logits.clone()))
        del cold,first_logit,hot
    batch=[row[0] for row in rows]
    quantum(batch,set(),steps=8,forced=force)
    for i,(x,expected) in enumerate(rows):
        value=kl(expected,x.logits);assert value<=.02,value
        checks[i]['cold_serial_hot_batch_decode_kl']=value
    del batch,rows,x,expected,reused
    gc.collect();torch.cuda.empty_cache()
    # Short actual 24-row dispatch sanity, not long-history or fragmentation qualification.
    capacity=[Request(Session(reader,'hot',24,'qualification24_'+str(i)),ids[:700+(i%8)*17],i+101) for i in range(24)]
    metric=quantum(capacity,set(),steps=2,forced=force)
    assert metric['batch']==24 and metric['steps']==2
    assert all(torch.isfinite(x.logits).all() for x in capacity)
    peak=torch.cuda.max_memory_allocated()/2**30
    del capacity
    gc.collect();torch.cuda.empty_cache()
    proof=dict(passed=True,epoch=time.time(),seconds=sync()-started,checks=checks,
        batch24=dict(batch=24,steps=2,finite_logits=True,peak_allocated_gib=peak),
        threshold_kl=.02,benchmark_model_calls=0,synthetic_model_forwards=True,
        legacy_fp32_equivalence_claimed=False,user_authorized_merged_weight_variant=True,
        limitations='Short startup checks only; prior legacy logits are not treated as bitwise equivalent, and dynamic long-run OOM remains possible.')
    save(ROOT/'worker'/'merged_startup_qualification.json',proof)
    return proof
