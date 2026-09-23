"""Bounded GPU KV reuse: exact causal prefixes or explicit approximate chunk reuse."""
from collections import OrderedDict,defaultdict,Counter
import hashlib,json
import torch

def token_key(tokens):return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
def nbytes(kv):return sum(t.numel()*t.element_size() for pair in kv for t in pair)

class Pool:
    def __init__(self,budget):self.budget=budget;self.bytes=0;self.items=OrderedDict();self.stats=Counter()
    def get(self,key):
        if key not in self.items:return None
        self.items.move_to_end(key);return self.items[key]
    def put(self,key,entry):
        size=nbytes(entry['kv']);entry=dict(entry,bytes=size)
        if size>self.budget:self.stats['oversize_rejections']+=1;return
        if key in self.items:self.bytes-=self.items.pop(key)['bytes']
        while self.bytes+size>self.budget:
            _,old=self.items.popitem(last=False);self.bytes-=old['bytes'];self.stats['evictions']+=1
        self.items[key]=entry;self.bytes+=size;self.stats['insertions']+=1
        assert self.bytes<=self.budget

class HotMemory:
    def __init__(self,band,policy='cold',depth=36,capacity=24,promote=False):
        assert policy in ['cold','exact','chunk'];assert depth in [24,36]
        assert not promote or policy=='chunk','Online KV and independent H12 have different semantics'
        self.band=band;self.reader=band.reader;self.policy=policy;self.depth=depth;self.promote=promote
        cfg=band.model.config;self.layers=list(range(12,36))
        scaling=getattr(cfg,'rope_scaling',None);assert scaling is None or scaling.get('rope_type',scaling.get('type','default'))=='default'
        self.block_bytes=2*24*cfg.num_key_value_heads*512*getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)*2
        self.pool=Pool(capacity*self.block_bytes+self.block_bytes//512)
        self.stats=Counter();self.last={}
    def key(self,parts,i):return (self.depth,tuple(parts[:i+1]) if self.policy=='exact' else parts[i])
    def extract(self,top,start,length,origin):
        return dict(kv=[(top.layers[l].keys[:,:,start:start+length].detach().clone(),top.layers[l].values[:,:,start:start+length].detach().clone()) for l in self.layers],
            position=start,tokens=length,origin=origin)
    def shifted(self,entry,start):
        delta=start-entry['position']
        if not delta:return entry['kv']
        inv=self.reader.rotary_emb.inv_freq.float();freq=delta*inv
        angles=torch.cat([freq,freq]);cos=angles.cos()[None,None,None];sin=angles.sin()[None,None,None]
        result=[]
        for k,v in entry['kv']:
            x=k.float();a,b=x.chunk(2,-1);rot=torch.cat([-b,a],-1)
            result.append(((x*cos+rot*sin).to(k.dtype),v))
        self.stats['rebased_entries']+=1
        return result
    def assemble(self,entries):
        top=self.band.fresh();offset=0;blocks=[]
        for entry in entries:blocks.append(self.shifted(entry,offset));offset+=entry['tokens']
        if blocks:
            for j,l in enumerate(self.layers):top.update(torch.cat([b[j][0] for b in blocks],2),torch.cat([b[j][1] for b in blocks],2),l)
        return top,offset
    def extend(self,states,top,past):
        """Only newly missing causal suffix is forwarded; prefix is already KV."""
        if not states:return top
        h=torch.cat(states,1);lengths=[x.shape[1] for x in states]
        p=torch.arange(past,past+h.shape[1],device=h.device)[None]
        mask=(torch.arange(past+h.shape[1],device=h.device)[None,:]<=p[0,:,None])[None,None]
        h=self.reader._run_layers(h,slice(12,self.depth),mask,p,self.reader.rotary_emb(h,position_ids=p),past_key_values=top,use_cache=True)
        if self.depth==36:return top
        parts=h.split(lengths,1);positions=p.split(lengths,1);groups=defaultdict(list);collected={l:[None]*len(parts) for l in range(self.depth,36)}
        for i,length in enumerate(lengths):groups[length].append(i)
        for _,indices in groups.items():
            local=torch.cat([parts[i] for i in indices]);pos=torch.cat([positions[i] for i in indices]);cache=self.band.fresh()
            mask,_=self.band.mask_rope(local,pos)
            self.reader._run_layers(local,slice(self.depth,36),mask,pos,self.reader.rotary_emb(local,position_ids=pos),past_key_values=cache,use_cache=True)
            for l in range(self.depth,36):
                for b,i in enumerate(indices):collected[l][i]=(cache.layers[l].keys[b:b+1],cache.layers[l].values[b:b+1])
        for l,parts in collected.items():top.update(torch.cat([x[0] for x in parts],2),torch.cat([x[1] for x in parts],2),l)
        return top
    def prefill(self,parts,states,query):
        self.last=Counter(requested_chunks=max(0,len(parts)-1));before=self.stats.copy()
        if not states:
            h,bottom,qpos=self.reader.write_prefill(query);logits,top,_=self.reader.read_prefill(None,[],h)
            return logits,bottom,top,qpos,0
        total=sum(h.shape[1] for h in states)
        if self.policy=='cold':
            self.last['rebuilt_chunks']=len(parts)-1
            if self.depth==36:
                logits,bottom,top,qpos=self.band.native(states,query)
            else:
                top=self.band.memory(states,self.depth);logits,bottom,top,qpos=self.band.query(query,top,total)
        elif self.policy=='exact':
            entries=[]
            for i in range(len(parts)):
                entry=self.pool.get(self.key(parts,i))
                if entry is None:break
                entries.append(entry)
            matched=len(entries);self.last['hit_chunks']=max(0,matched-1);self.last['rebuilt_chunks']=len(parts)-max(1,matched)
            top,past=self.assemble(entries)
            if self.depth==36:
                qh,bottom,qpos=self.reader.write_prefill(query)
                h=torch.cat(states[matched:]+[qh],1);p=torch.arange(past,total+qpos,device=h.device)[None]
                mask=(torch.arange(total+qpos,device=h.device)[None,:]<=p[0,:,None])[None,None]
                h=self.reader._run_layers(h,slice(12,36),mask,p,self.reader.rotary_emb(h,position_ids=p),past_key_values=top,use_cache=True)
                logits=self.reader.lm_head(self.reader.norm(h[:,-1:]))
            else:
                top=self.extend(states[matched:],top,past);logits,bottom,top,qpos=self.band.query(query,top,total)
            offset=0
            for i,state in enumerate(states):
                length=state.shape[1]
                if i>=matched:self.pool.put(self.key(parts,i),self.extract(top,offset,length,'reconstruction'))
                offset+=length
        else:
            hits=[self.pool.get(self.key(parts,i)) for i in range(len(parts))]
            self.last['hit_chunks']=sum(e is not None for e in hits[1:]);self.last['rebuilt_chunks']=sum(e is None for e in hits[1:])
            self.last['promoted_hits']=sum(e is not None and e['origin'].startswith('online') for e in hits[1:])
            entries=[];i=0;offset=0
            while i<len(parts):
                if hits[i] is not None:
                    entries.append(hits[i]);offset+=hits[i]['tokens'];i+=1;continue
                end=i+1
                while end<len(parts) and hits[end] is None:end+=1
                top,past=self.assemble(entries);assert past==offset
                top=self.extend(states[i:end],top,past)
                for j in range(i,end):
                    entry=self.extract(top,offset,states[j].shape[1],'reconstruction');entries.append(entry)
                    self.pool.put(self.key(parts,j),entry);offset+=entry['tokens']
                i=end
            top,past=self.assemble(entries);assert past==total
            logits,bottom,top,qpos=self.band.query(query,top,total)
        self.last['rebase_operations']=self.stats['rebased_entries']-before['rebased_entries']
        self.stats.update(self.last)
        return logits,bottom,top,qpos,total
    def promote_query(self,ids,anchor_len,archive_count,top,memory_tokens,origin):
        if not self.promote:return 0
        available=top.layers[12].keys.shape[2]-memory_tokens
        rest=ids[anchor_len:];count=0
        for i in range(archive_count,len(rest)//512):
            start=anchor_len+(i-archive_count)*512
            if start+512>available:break
            tokens=rest[i*512:(i+1)*512];part=(i,token_key(tokens))
            self.pool.put((self.depth,part),self.extract(top,memory_tokens+start,512,origin));count+=1
        self.stats['promotions']+=count
        return count
