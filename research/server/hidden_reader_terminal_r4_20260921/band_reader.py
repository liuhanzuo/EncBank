"""H12 -> joint causal blocks13..n -> independent causal blocks(n+1)..36."""
import collections
import torch
from transformers.cache_utils import DynamicCache
from engine import Reader

class JointBand:
    def __init__(self,model,tokenizer):
        self.model=model;self.reader=Reader(model,12,tokenizer=tokenizer)
        assert not self.reader.write_sink and not self.reader.block_diagonal
    def fresh(self):return DynamicCache(config=self.model.config)
    def write(self,chunks):
        pieces=[[151643]]+chunks;groups=collections.defaultdict(list);states=[None]*len(pieces)
        for i,piece in enumerate(pieces):groups[len(piece)].append(i)
        for length,indices in groups.items():
            h=self.reader.write_chunk(torch.tensor([pieces[i] for i in indices],device='cuda'))
            for b,i in enumerate(indices):states[i]=h[b:b+1]
        return states
    def mask_rope(self,h,p):
        mask=(torch.arange(h.shape[1],device=h.device)[None,:]<=torch.arange(h.shape[1],device=h.device)[:,None])[None,None]
        return mask,self.reader.rotary_emb(h,position_ids=p)
    def memory(self,states,n):
        assert 12<=n<=36
        h=torch.cat(states,1);lengths=[x.shape[1] for x in states];total=h.shape[1]
        positions=torch.arange(total,device='cuda')[None]
        top=self.fresh()
        if n>12:
            mask,rope=self.mask_rope(h,positions)
            h=self.reader._run_layers(h,slice(12,n),mask,positions,rope,past_key_values=top,use_cache=True)
        if n==36:return top
        parts=list(h.split(lengths,dim=1));position_parts=list(positions.split(lengths,dim=1))
        groups=collections.defaultdict(list)
        for i,length in enumerate(lengths):groups[length].append(i)
        collected={l:[None]*len(parts) for l in range(n,36)}
        for length,indices in groups.items():
            local=torch.cat([parts[i] for i in indices],dim=0)
            # Preserve packed positions; local attention is physically batched, not a dense block mask.
            p=torch.cat([position_parts[i] for i in indices],dim=0)
            mask,rope=self.mask_rope(local,p);cache=self.fresh()
            local=self.reader._run_layers(local,slice(n,36),mask,p,rope,past_key_values=cache,use_cache=True)
            for l in range(n,36):
                k,v=cache.layers[l].keys,cache.layers[l].values
                for b,i in enumerate(indices):collected[l][i]=(k[b:b+1],v[b:b+1])
        for l,pieces in collected.items():
            top.update(torch.cat([x[0] for x in pieces],dim=2),torch.cat([x[1] for x in pieces],dim=2),l)
        assert all(top.layers[l].keys.shape==(1,8,total,128) for l in range(12,36))
        return top
    def dense_mask_reference(self,states,n):
        h=torch.cat(states,1);lengths=[x.shape[1] for x in states];total=h.shape[1]
        positions=torch.arange(total,device='cuda')[None]
        block=torch.cat([torch.full((length,),i,device='cuda') for i,length in enumerate(lengths)])
        causal=(positions[0,None,:]<=positions[0,:,None])[None,None]
        within=causal & (block[None,:]==block[:,None])[None,None]
        rope=self.reader.rotary_emb(h,position_ids=positions);cache=self.fresh()
        for l in range(12,36):
            h=self.reader._run_layers(h,slice(l,l+1),causal if l<n else within,positions,rope,past_key_values=cache,use_cache=True)
        return cache
    def query(self,query,top,memory_tokens):
        h,bottom,qpos=self.reader.write_prefill(query)
        p=torch.arange(memory_tokens,memory_tokens+qpos,device='cuda')[None]
        mask=(torch.arange(memory_tokens+qpos,device='cuda')[None,:]<=p[0,:,None])[None,None]
        h=self.reader._run_layers(h,slice(12,36),mask,p,self.reader.rotary_emb(h,position_ids=p),past_key_values=top,use_cache=True)
        return self.reader.lm_head(self.reader.norm(h[:,-1:])),bottom,top,qpos
    def native(self,states,query):
        h,bottom,qpos=self.reader.write_prefill(query)
        logits,top,total=self.reader.read_prefill(states[0],states[1:],h)
        assert total==sum(x.shape[1] for x in states)+qpos
        return logits,bottom,top,qpos
