"""H21 streaming memory and exact ordered-prefix KV/recurrent-state snapshots.

Online H21 is retained for newly completed chunks in BOTH cold/hot variants.
Hot entries never cross prefix identities. No additive/last-chunk approximation
is used for DeltaNet states. Numerical equivalence is qualified on GPU.
"""
import copy, hashlib, json, time
from collections import Counter, OrderedDict
import torch
from transformers.cache_utils import DynamicCache
from batch_cache import merge_caches, decode_layers
from memory_selectors import iter_bm25_indices

def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def sync():torch.cuda.synchronize();return time.perf_counter()
def bytes_cache(cache):
    stores={}
    if cache is None:return 0
    for layer in cache.layers:
        for key in ['keys','values','conv_states','recurrent_states']:
            v=getattr(layer,key,None)
            for t in v.values() if isinstance(v,dict) else [v]:
                if torch.is_tensor(t):
                    s=t.untyped_storage();stores[(str(t.device),s.data_ptr())]=s.nbytes()
    return sum(stores.values())

LINEAR_FIELDS=['number_of_states','is_conv_states_initialized','is_recurrent_states_initialized',
    'has_previous_state','conv_kernel_size','device','dtype','record_past']

def linear_copy(src,dst,row=0,clone=True):
    for name in LINEAR_FIELDS:setattr(dst,name,copy.copy(getattr(src,name)))
    for name in ['conv_states','recurrent_states']:
        setattr(dst,name,{k:(v[row:row+1].detach().clone() if clone else v[row:row+1]) for k,v in getattr(src,name).items()})

def row_cache(cache,cfg,start,end,row,pad,clone=True):
    dst=DynamicCache(config=cfg)
    for i in range(start,end):
        src,target=cache.layers[i],dst.layers[i]
        if cfg.layer_types[i]=='full_attention':
            k,v=src.keys[row:row+1,:,pad:],src.values[row:row+1,:,pad:]
            dst.update(k.clone() if clone else k,v.clone() if clone else v,i)
        else:linear_copy(src,target,row,clone)
    return dst

def run(reader,h,start,end,cache=None,offset=0):
    """Explicit bottom-right causal mask, including multi-token cache extension."""
    pos=torch.arange(offset,offset+h.shape[1],device=h.device)[None].expand(h.shape[0],-1)
    rotary=reader.core.rotary_emb(h,pos[None].expand(3,-1,-1))
    mask=(torch.arange(offset+h.shape[1],device=h.device)[None,:]<=pos[0,:,None])[None,None]
    for block in reader.core.layers[start:end]:
        h=block(h,position_embeddings=rotary,attention_mask=mask if block.block_type=='full_attention' else None,
            position_ids=pos,past_key_values=cache,use_cache=cache is not None)
    return h

class PrefixPool:
    def __init__(self,reader,capacity):
        self.r=reader;self.capacity=capacity;self.items=OrderedDict();self.bytes=0;self.stats=Counter()
        # Includes a separate one-token sink checkpoint in addition to full chunks.
        self.budget=int(capacity*120.5*2**20+99*2**20) if capacity else 0
    def get(self,key):
        if key not in self.items:return None
        self.items.move_to_end(key);return self.items[key]
    def put(self,key,cache,offset,length,row=0,pad=0,origin='rebuild'):
        if not self.capacity:return
        snap=DynamicCache(config=self.r.config)
        # Only this chunk's KV, but recurrent states after the WHOLE prefix.
        for i in range(self.r.j,self.r.L):
            src,layer=cache.layers[i],snap.layers[i]
            if self.r.config.layer_types[i]=='full_attention':
                snap.update(src.keys[row:row+1,:,pad+offset:pad+offset+length].clone(),
                    src.values[row:row+1,:,pad+offset:pad+offset+length].clone(),i)
            else:linear_copy(src,layer,row)
        size=bytes_cache(snap)
        if size>self.budget:raise MemoryError('Hot entry exceeds per-session byte budget')
        if key in self.items:self.bytes-=self.items.pop(key)['bytes']
        while self.bytes+size>self.budget:
            _,old=self.items.popitem(last=False);self.bytes-=old['bytes'];self.stats['evictions']+=1
        self.items[key]=dict(cache=snap,length=length,bytes=size,origin=origin);self.bytes+=size
        self.stats['insertions']+=1
    def restore(self,parts):
        entries=[]
        for i in range(len(parts)):
            x=self.get(tuple(parts[:i+1]))
            if x is None:break
            entries.append(x)
        out=DynamicCache(config=self.r.config)
        if entries:
            for i in range(self.r.j,self.r.L):
                if self.r.config.layer_types[i]=='full_attention':
                    out.update(torch.cat([e['cache'].layers[i].keys for e in entries],2),
                        torch.cat([e['cache'].layers[i].values for e in entries],2),i)
                else:linear_copy(entries[-1]['cache'].layers[i],out.layers[i])
        return out,len(entries),sum(e['length'] for e in entries),sum(e['origin']=='online' for e in entries[1:])

class Session:
    def __init__(self,reader,arm,capacity,task):
        self.r=reader;self.arm=arm;self.task=task;self.bank={};self.serial=0
        self.pool=PrefixPool(reader,capacity if arm=='hot' else 0)
        self.dense_cache=None;self.dense_ids=[]
    def key(self,index,ids):return (index,digest(ids))
    def put_h(self,key,h,origin):
        if key not in self.bank:
            self.serial+=1;self.bank[key]=(h.detach().clone(),(key,self.serial),origin)
        return self.bank[key]
    def hbytes(self):return sum(v[0].numel()*v[0].element_size() for v in self.bank.values())

class Request:
    def __init__(self,session,ids,seed,topk=12):
        self.s=session;self.r=session.r;self.ids=list(ids);self.initial=list(ids);self.generated=[]
        self.topk=topk;self.events=[];self.time=Counter();self.status=None;self.first=None
        self.generator=torch.Generator(device='cuda').manual_seed(seed)
        self.lower=self.upper=None;self.qpos=self.upos=0;self.logits=None
        self.qh=[];self.qtokens=[];self.qindex=0;self.parts=[];self.memory_tokens=0
        self.prompt_ids=[];self.began=sync();self.prefill()
    def prefill(self):
        start=sync();r=self.r;s=self.s
        if s.arm=='dense':
            reuse=s.dense_cache is not None and len(s.dense_ids)<len(self.ids) and self.ids[:len(s.dense_ids)]==s.dense_ids
            self.upper=s.dense_cache if reuse else DynamicCache(config=r.config)
            past=len(s.dense_ids) if reuse else 0;s.dense_cache=None;s.dense_ids=[]
            # Chunked prefill avoids constructing a quadratic full-prompt mask.
            for a in range(past,len(self.ids),512):
                h=run(r,r.core.embed_tokens(r.tensor(self.ids[a:a+512])),0,r.L,self.upper,a)
            self.logits=r.logits(h);self.upos=len(self.ids);self.prompt_ids=list(self.ids)
            self.events.append(dict(history_tokens=len(self.ids),cached_prefix_tokens=past))
            self.time['prefill_seconds']+=sync()-start;return
        assert len(self.ids)>=2
        n=(len(self.ids)-2)//512;chunks=[self.ids[1+i*512:1+(i+1)*512] for i in range(n)]
        sink=self.ids[:1];keys=[s.key(-1,sink)]+[s.key(i,x) for i,x in enumerate(chunks)]
        # Retain only current history identities, plus unarchived current online chunk.
        allowed=set(keys+[s.key(n,self.ids[1+n*512:1+(n+1)*512])])
        s.bank={k:v for k,v in s.bank.items() if k in allowed}
        write=sync()
        for key,tokens in zip(keys,[sink]+chunks):
            if key not in s.bank:s.put_h(key,run(r,r.core.embed_tokens(r.tensor(tokens)),0,r.j), 'independent')
        self.time['write_seconds']+=sync()-write
        retrieve=sync()
        search=self.initial[:2048]+self.ids[-2048:]
        selected=iter_bm25_indices([torch.tensor(x) for x in chunks],search,self.topk,iter_hop_topk=4,iter_rounds=0) if chunks else []
        # Same explicit recent-fill rule in cold and hot.
        selected=list(dict.fromkeys(selected))
        for i in reversed(range(n)):
            if len(selected)>=min(n,self.topk):break
            if i not in selected:selected.append(i)
        selected.sort();chosen=[keys[0]]+[keys[i+1] for i in selected]
        states=[s.bank[k][0] for k in chosen];self.parts=[s.bank[k][1] for k in chosen]
        self.time['retrieval_seconds']+=sync()-retrieve
        pf=sync();self.upper,matched,past,promoted=s.pool.restore(self.parts)
        offset=0
        for i,h in enumerate(states):
            if i>=matched:
                run(r,h,r.j,r.L,self.upper,offset)
                s.pool.put(tuple(self.parts[:i+1]),self.upper,offset,h.shape[1])
            offset+=h.shape[1]
        self.memory_tokens=offset;self.upos=offset
        tail=self.ids[1+n*512:];self.qindex=n;self.qtokens=list(tail)
        self.lower=DynamicCache(config=r.config);qh=run(r,r.core.embed_tokens(r.tensor(tail)),0,r.j,self.lower)
        self.qh=[qh.detach().clone()];self.qpos=len(tail);self.online_prefix_valid=True
        h=run(r,qh,r.j,r.L,self.upper,self.upos);self.upos+=len(tail);self.logits=r.logits(h)
        if len(tail)==512:self.promote(self.upper,0,0)
        event=dict(history_tokens=len(self.ids),archived_chunks=n,selected_chunks=selected,
            hit_chunks=max(0,matched-1),rebuilt_chunks=len(states)-max(1,matched),promoted_hits=promoted,
            hot_bytes=s.pool.bytes,h_bytes=s.hbytes(),generated=len(self.generated))
        self.events.append(event)
        if not self.prompt_ids:self.prompt_ids=sink+sum([chunks[i] for i in selected],[])+tail
        self.time['prefill_seconds']+=sync()-pf
        self.time['refresh_total_seconds']+=sync()-start
    def promote(self,upper,row,pad):
        # State captured exactly when this chunk ends, not at a later decode step.
        assert len(self.qtokens)==512
        began=sync();s=self.s;key=s.key(self.qindex,self.qtokens)
        h=torch.cat(self.qh,1);assert h.shape[1]==512
        value=s.put_h(key,h,'online');part=value[1]
        # If a serialized existing chunk already has a different H definition,
        # it must not use this online snapshot under that old identity.
        same=torch.equal(value[0],h)
        self.parts.append(part)
        self.online_prefix_valid=self.online_prefix_valid and same
        if self.online_prefix_valid:s.pool.put(tuple(self.parts),upper,self.upos-512,512,row,pad,'online')
        self.qindex+=1;self.qtokens=[];self.qh=[]
        self.time['promotion_seconds']+=sync()-began
    def append_h(self,h,token,upper,row,pad):
        self.qh.append(h.detach().clone());self.qtokens.append(token)
        if len(self.qtokens)==512:self.promote(upper,row,pad)
    def refresh_needed(self):return self.s.arm!='dense' and len(self.generated)>0 and len(self.generated)%512==0 and not self.status

def sample(logit,generator,temperature=1.0,top_p=.95,top_k=20):
    scores=logit[-1].float()
    if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Nonfinite logits')
    if not temperature:return int(scores.argmax().item())
    v,ix=(scores/temperature).topk(top_k);mask=v.softmax(-1).cumsum(-1)>top_p
    mask[1:]=mask[:-1].clone();mask[0]=False;v[mask]=-float('inf')
    return int(ix[torch.multinomial(v.softmax(-1),1,generator=generator)].item())

@torch.no_grad()
def quantum(rows,stop,context=262144,steps=32,forced=None,temperature=1.0,cancel=None):
    """Batched decode for BOTH dense and streaming CoMem; CPU row metadata only."""
    assert rows and len({x.s.arm=='dense' for x in rows})==1
    r=rows[0].r;dense=rows[0].s.arm=='dense';began=sync()
    up,upad=merge_caches([x.upper for x in rows],[x.upos for x in rows],r.config,0 if dense else r.j,r.L,consume=True)
    if not dense:lo,lpad=merge_caches([x.lower for x in rows],[x.qpos for x in rows],r.config,0,r.j,consume=True)
    logits=torch.cat([x.logits for x in rows]);upos=torch.tensor([x.upos for x in rows],device='cuda')
    qpos=torch.tensor([x.qpos for x in rows],device='cuda') if not dense else None
    for x in rows:x.upper=x.lower=x.logits=None
    count=0
    for step in range(steps):
        for x in rows:
            if cancel and cancel(x):x.status='cancelled'
            if x.upos>=context or (not dense and x.qpos>=context):x.status='context_limit'
        if any(x.status in ['cancelled','context_limit'] for x in rows):break
        tokens=[]
        for i,x in enumerate(rows):
            tok=forced[len(x.generated)] if forced is not None else sample(logits[i],x.generator,temperature)
            x.generated.append(tok);x.ids.append(tok);tokens.append(tok)
            if x.first is None:x.first=sync()
            if forced is None and tok in stop:x.status='ok'
        # Finished rows still consume this final token once, keeping dense prefix
        # cache and newly completed online chunks coherent. Cancelled/context rows
        # break before any forward to avoid an out-of-capacity access.
        h=r.core.embed_tokens(torch.tensor(tokens,device='cuda')[:,None])
        if dense:h=decode_layers(r,h,0,r.L,up,upos,upad)
        else:
            h=decode_layers(r,h,0,r.j,lo,qpos,lpad);online=h.detach().clone()
            h=decode_layers(r,h,r.j,r.L,up,upos,upad);qpos+=1
        upos+=1;logits=r.logits(h);count+=1
        for i,x in enumerate(rows):
            x.upos+=1
            if not dense:
                x.qpos+=1;x.append_h(online[i:i+1],tokens[i],up,i,int(upad[i]))
        if any(x.status or x.refresh_needed() for x in rows):break
    for i,x in enumerate(rows):
        x.upper=row_cache(up,r.config,0 if dense else r.j,r.L,i,int(upad[i]),clone=False)
        if not dense:x.lower=row_cache(lo,r.config,0,r.j,i,int(lpad[i]),clone=False)
        x.logits=logits[i:i+1].clone()
    del up
    if not dense:del lo
    elapsed=sync()-began
    for x in rows:x.time['decode_cohort_seconds']+=elapsed
    return dict(seconds=elapsed,batch=len(rows),steps=count,decoded_tokens=count*len(rows))
