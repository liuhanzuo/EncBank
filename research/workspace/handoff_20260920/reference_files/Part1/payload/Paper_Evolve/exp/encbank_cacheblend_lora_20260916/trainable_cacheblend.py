"""Differentiable version of the previously evaluated two-bootstrap-layer control.

Unlike the inference-only implementation, gradients reach both the independent
cached K/V writer and selective repair through the same fresh suffix adapter.
Cache objects are local to each call, so checkpoint recomputation cannot append
to or overwrite a cache from a previous forward.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint
from encbank import Encbank
from encbank.cacheblend import CacheBlend

class LayerCache:
    def __init__(self,old=None,indices=None):
        self.old,self.indices=old,indices
        self.keys=self.values=None
    def update(self,k,v,*args,**kwargs):
        if self.old is None:self.keys,self.values=k,v
        else:
            self.keys=self.old[0].index_copy(2,self.indices,k.to(self.old[0].dtype))
            self.values=self.old[1].index_copy(2,self.indices,v.to(self.old[1].dtype))
        return self.keys,self.values

class TrainableBlend(CacheBlend):
    def __init__(self,model,j=12,grad_checkpoint=True):
        super().__init__(Encbank(model,0))
        self.j,self.grad_checkpoint=j,grad_checkpoint

    def layer(self,index,h,mask,pos,pe,old=None,indices=None):
        block=self.cm.layers[index]
        def forward(hidden,*prior):
            cache=LayerCache(tuple(prior) if prior else None,indices)
            output=block(hidden,attention_mask=mask,position_ids=pos,
                position_embeddings=pe,past_key_values=cache,use_cache=True)
            return self.cm._layer_out_hidden(output),cache.keys,cache.values
        args=(h,) if old is None else (h,*old)
        if torch.is_grad_enabled() and self.grad_checkpoint and index>=self.j:
            return checkpoint(forward,*args,use_reentrant=False)
        return forward(*args)

    def write(self,ids):
        x=self.cm._as_ids(ids);h=self.cm.embed_tokens(x)
        pos=torch.arange(x.shape[1],device=self.device).unsqueeze(0)
        mask,pe=self.cm._make_mask_and_rope(h,pos)
        layers=[]
        for l in range(self.num_layers):
            h,k,v=self.layer(l,h,mask,pos,pe);layers.append((k,v))
        return layers

    def assemble(self,segments):
        kvs=[];offsets=[];offset=0
        for ids in segments:
            kvs.append(self.write(ids));offsets.append(offset);offset+=len(ids)
        # The rotation primitive itself is differentiable; its inference wrapper
        # concat_kv_reindex is decorated no_grad and deliberately not called here.
        merged=[]
        for l in range(self.num_layers):
            merged.append((torch.cat([self._rotate_k_by_offset(kv[l][0],p) for kv,p in zip(kvs,offsets)],dim=2),
                           torch.cat([kv[l][1] for kv in kvs],dim=2)))
        pack=torch.cat([self.cm._as_ids(x).reshape(-1) for x in segments]).view(1,-1)
        return pack,merged

    def read_hidden(self,pack,merged,sink_len,query_len,ratio=.15):
        h=self.cm.embed_tokens(pack);n=pack.shape[1]
        pos=torch.arange(n,device=self.device).unsqueeze(0)
        mask,pe=self.cm._make_mask_and_rope(h,pos)
        fresh=[]
        for l in range(2):
            h,k,v=self.layer(l,h,mask,pos,pe);fresh.append((k,v))
        with torch.no_grad():
            fk,fv=fresh[1]
            dev=((fk.float()-merged[1][0].float()).square()+(fv.float()-merged[1][1].float()).square()).sum(dim=(1,3)).squeeze(0)
            start,end=int(sink_len),n-int(query_len)
            count=min(end-start,int(math.ceil(ratio*(end-start))))
            chosen=torch.zeros(n,dtype=torch.bool,device=self.device)
            chosen[:start]=True;chosen[end:]=True
            if count:chosen[torch.topk(dev[start:end],count).indices+start]=True
            ix=torch.nonzero(chosen,as_tuple=False).squeeze(1)
        h=h[:,ix];rp=pos[:,ix];rpe=self.cm.rotary_emb(h,position_ids=rp)
        sparse=self._sparse_attn_mask(torch.arange(n,device=self.device).view(1,n)<=ix.view(-1,1))
        for l in range(2,self.num_layers):
            h,k,v=self.layer(l,h,sparse,rp,rpe,merged[l],ix)
        assert torch.equal(ix[-query_len:],torch.arange(n-query_len,n,device=self.device))
        return self.cm.norm(h[:,-query_len:]),ix

    def hidden(self,segments,ratio=.15):
        pack,merged=self.assemble(segments)
        h,_=self.read_hidden(pack,merged,len(segments[0]),len(segments[-1]),ratio)
        return h
