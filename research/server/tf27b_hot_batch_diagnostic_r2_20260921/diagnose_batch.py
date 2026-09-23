"""Localize failed batch equivalence without changing its acceptance threshold."""
import gc,json,time,traceback
import torch
from common import ROOT,PLAN as P,save,verify_sources
from model_setup import load,tokens
from hybrid_hot import Session,Request,run,sync,row_cache
from batch_cache import merge_caches

def error(a,b):
    a=a.float();b=b.float();d=(a-b).abs()
    return dict(max_abs=float(d.max()),rms=float(d.square().mean().sqrt()),rel_rms=float(d.square().mean().sqrt()/(a.square().mean().sqrt()+1e-12)))
def kl(a,b):
    x=a.float().log_softmax(-1);y=b.float().log_softmax(-1)
    return max(0,float((x.exp()*(x-y)).sum(-1).mean()))

@torch.no_grad()
def case(r,ids,lens,adapter,force_recurrent=False):
    with r.adapter(adapter):
        rows=[Request(Session(r,'hot',24,str(i)),ids[:n],20+i) for i,n in enumerate(lens)]
        baselines=[Request(Session(r,'hot',24,str(i)),ids[:n],20+i) for i,n in enumerate(lens)]
        pref=[kl(a.logits,b.logits) for a,b in zip(rows,baselines)]
        lo,lp=merge_caches([x.lower for x in rows],[x.qpos for x in rows],r.config,0,r.j,consume=True)
        up,upad=merge_caches([x.upper for x in rows],[x.upos for x in rows],r.config,r.j,r.L,consume=True)
        seq=[r.core.embed_tokens(r.tensor([ids[0]])) for _ in lens];batch=torch.cat(seq)
        logs=[]
        for i,block in enumerate(r.core.layers):
            low=i<r.j;cache=lo if low else up;pads=lp if low else upad
            pos=torch.tensor([x.qpos if low else x.upos for x in rows],device='cuda')[:,None]
            rope=r.core.rotary_emb(batch,pos[None].expand(3,-1,-1))
            size=cache.layers[i].keys.shape[-2]+1 if block.block_type=='full_attention' else None
            mask=(torch.arange(size,device='cuda')[None,:]>=pads[:,None])[:,None,None,:] if size else None
            batch=block(batch,position_embeddings=rope,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True)
            states=[]
            for b,row in enumerate(baselines):
                c=row.lower if low else row.upper;position=torch.tensor([[row.qpos if low else row.upos]],device='cuda')
                rotary=r.core.rotary_emb(seq[b],position[None].expand(3,-1,-1))
                length=c.layers[i].keys.shape[-2]+1 if size else None
                smask=torch.ones((1,1,1,length),dtype=torch.bool,device='cuda') if size else None
                seq[b]=block(seq[b],position_embeddings=rotary,attention_mask=smask,position_ids=position,past_key_values=c,use_cache=True)
                entry=dict(row=b,hidden=error(seq[b],batch[b:b+1]))
                if not size:
                    entry['recurrent']=error(c.layers[i].recurrent_states[0],cache.layers[i].recurrent_states[0][b:b+1])
                    entry['conv']=error(c.layers[i].conv_states[0],cache.layers[i].conv_states[0][b:b+1])
                states.append(entry)
            logs.append(dict(layer=i,type=block.block_type,rows=states))
        actual=r.logits(batch);expected=[r.logits(x) for x in seq]
        result=dict(lengths=lens,adapter=adapter,prefill_kl=pref,decode_kl=[kl(a,actual[i:i+1]) for i,a in enumerate(expected)],layers=logs)
        suffix='reduced' if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction else 'fullacc'
        result['bf16_reduced_precision_reduction']=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        save(ROOT/('diagnostic_'+str(int(adapter))+'_'+'_'.join(map(str,lens))+'_'+suffix+'.json'),result)
        return {k:v for k,v in result.items() if k!='layers'}

@torch.no_grad()
def main():
    verify_sources();model,r,tok,stop=load(P)
    ids=tokens(tok,json.loads((ROOT/'trace_messages.json').read_text()))
    results=[]
    for reduced in [True,False]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
        for adapter,lens in [(True,[700,719]),(False,[700,719])]:
            results.append(case(r,ids,lens,adapter));gc.collect();torch.cuda.empty_cache()
            save(ROOT/'diagnostic_summary.json',results)
    save(ROOT/'diagnostic_complete.json',dict(passed=True,epoch=time.time(),rows=results))
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'diagnostic_failure.json',dict(error=traceback.format_exc()));raise
