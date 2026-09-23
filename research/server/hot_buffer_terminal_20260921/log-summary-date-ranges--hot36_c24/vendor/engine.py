"""Same-executor reference service: shared residual bank plus optional exact prefix KV."""
import collections,time
import torch
from transformers.cache_utils import DynamicCache
from encbank import Encbank
from bm25_index import BM25Index
from prefix_store import PrefixStore

METHODS=['raw','raw_kv','h_cpu','h_gpu','hybrid']
BUDGET=8*2**30

class Reader(Encbank):
    def _as_ids(self,ids):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        return ids[None,:] if ids.ndim==1 else ids
    @torch.no_grad()
    def decode_step(self,tokens,bottom,top,qpos,ppos):
        hidden=self.embed_tokens(tokens.reshape(-1,1))
        if self.resume_j:
            p=torch.tensor([[qpos]],device=self.device)
            hidden=self._run_layers(hidden,slice(0,self.resume_j),None,p,self.rotary_emb(hidden,position_ids=p),past_key_values=bottom,use_cache=True)
        p=torch.tensor([[ppos]],device=self.device)
        hidden=self._run_layers(hidden,slice(self.resume_j,self.num_layers),None,p,self.rotary_emb(hidden,position_ids=p),past_key_values=top,use_cache=True)
        return self.lm_head(self.norm(hidden))

def sync():torch.cuda.synchronize();return time.perf_counter()
def layer_kv(cache,layer):return cache.layers[layer].keys,cache.layers[layer].values

class Engine:
    @torch.inference_mode()
    def __init__(self,model,tok,docs,method,concurrency):
        assert method in METHODS
        self.model,self.tok,self.docs,self.method=model,tok,docs,method
        self.is_h=method in ['h_cpu','h_gpu','hybrid'];self.j=12 if self.is_h else 0
        self.reader=Reader(model,self.j,tokenizer=tok);self.layers=list(range(self.j,self.reader.num_layers))
        self.chunks=[list(torch.tensor(d['source']).split(512)) for d in docs]
        assert all(len(c)==64 and all(len(x)==512 for x in c) for c in self.chunks)
        t=time.perf_counter();self.indices=[BM25Index(c) for c in self.chunks];self.index_s=time.perf_counter()-t
        self.hbytes=0;self.write_s=0;self.host_buffer=None;self.bank=None;self.sink=None
        if self.is_h:
            t=sync();chunks=[c for doc in self.chunks for c in doc];states=[]
            for n in range(0,len(chunks),8):
                h=self.reader.write_chunk(torch.stack(chunks[n:n+8]))
                states.append(h.cpu() if method=='h_cpu' else h)
            self.bank=torch.cat(states,dim=0);self.sink=self.reader.write_chunk([tok.bos_token_id]);del states,h
            if method=='h_cpu':
                self.bank=self.bank.pin_memory();self.sink=self.sink.cpu().pin_memory()
                self.host_buffer=torch.empty((concurrency,6145,self.reader.hidden_size),dtype=self.bank.dtype,pin_memory=True)
            self.hbytes=(self.bank.numel()+self.sink.numel())*self.bank.element_size();self.write_s=sync()-t
        self.persistent_gpu_h=self.hbytes if method in ['h_gpu','hybrid'] else 0
        self.store=PrefixStore(BUDGET-self.persistent_gpu_h) if method in ['raw_kv','hybrid'] else None
        self.stream=torch.cuda.Stream() if method=='h_cpu' else None
        self.pending_copy=None
        self.stats=collections.Counter()
    def reset_stats(self):self.stats.clear()
    def layout(self,di,selected):return [(-1,-1)]+[(di,i) for i in selected]
    def restore(self,chains):
        cache=DynamicCache(config=self.model.config)
        if not chains[0]:return cache
        for offset,layer in enumerate(self.layers):
            kv=[]
            for component in [0,1]:
                per_request=[torch.cat([block['tensors'][offset][component] for block in chain],dim=2) for chain in chains]
                kv.append(torch.cat(per_request,dim=0))
            cache.update(kv[0],kv[1],layer)
        return cache
    def store_prefix(self,cache,layouts):
        if self.store is None:return
        for row,parts in enumerate(layouts):
            start=0
            for n,part in enumerate(parts):
                length=1 if n==0 else 512;end=start+length
                sample=layer_kv(cache,self.layers[0])[0]
                size=2*len(self.layers)*sample.shape[1]*length*sample.shape[3]*sample.element_size()
                def factory(row=row,start=start,end=end):
                    return [(k[row:row+1,:,start:end,:].detach().clone(),v[row:row+1,:,start:end,:].detach().clone()) for k,v in [layer_kv(cache,l) for l in self.layers]]
                self.store.put(parts[:n+1],factory,size,length);start=end
    def doc_hidden(self,picks,selected,matched):
        include_sink=matched==0;skip=max(0,matched-1);remaining=12-skip;batch=len(picks)
        if not self.is_h:
            tokens=[]
            for (di,qi),sel in zip(picks,selected):
                parts=([torch.tensor([self.tok.bos_token_id])] if include_sink else [])+[self.chunks[di][i] for i in sel[skip:]]
                tokens.append(torch.cat(parts) if parts else torch.empty(0,dtype=torch.long))
            return self.reader.embed_tokens(torch.stack(tokens).to('cuda'))
        ids=[di*64+i for (di,qi),sel in zip(picks,selected) for i in sel[skip:]]
        if self.method=='h_cpu':
            if self.pending_copy is not None:self.pending_copy.synchronize()
            length=remaining*512+int(include_sink);host=self.host_buffer[:batch,:length]
            for b,((di,qi),sel) in enumerate(zip(picks,selected)):
                cursor=0
                if include_sink:host[b,:1].copy_(self.sink[0]);cursor=1
                for i in sel[skip:]:host[b,cursor:cursor+512].copy_(self.bank[di*64+i]);cursor+=512
            with torch.cuda.stream(self.stream):
                result=host.to('cuda',non_blocking=True)
                self.pending_copy=torch.cuda.Event();self.pending_copy.record(self.stream)
            torch.cuda.current_stream().wait_stream(self.stream)
            result.record_stream(torch.cuda.current_stream())
            return result
        result=self.bank.index_select(0,torch.tensor(ids,device='cuda')).reshape(batch,remaining*512,-1) if remaining else self.bank.new_empty((batch,0,self.reader.hidden_size))
        return torch.cat([self.sink.expand(batch,-1,-1),result],dim=1) if include_sink else result
    def merge(self,groups,batch):
        if len(groups)==1:return groups[0][1],groups[0][2]
        cache=DynamicCache(config=self.model.config);logits=groups[0][2].new_empty((batch,1,groups[0][2].shape[-1]))
        for ids,c,lg in groups:logits.index_copy_(0,torch.tensor(ids,device='cuda'),lg)
        for layer in self.layers:
            ref=layer_kv(groups[0][1],layer)[0];shape=(batch,*ref.shape[1:]);k=ref.new_empty(shape);v=ref.new_empty(shape)
            for ids,c,lg in groups:
                ix=torch.tensor(ids,device='cuda');kk,vv=layer_kv(c,layer);k.index_copy_(0,ix,kk);v.index_copy_(0,ix,vv)
            cache.update(k,v,layer)
        return cache,logits
    @torch.inference_mode()
    def infer(self,picks,count=32,capture=False,forced_selected=None,forced_tokens=None):
        # All methods execute the actual online selector, including KV hits.
        t=time.perf_counter();chosen=[]
        for di,qi in picks:
            query=self.docs[di]['queries'][qi]
            sel=self.indices[di].select(query['query'][:32],12,2)
            assert sel==query['selected'] and len(sel)==12;chosen.append(sel)
        if forced_selected is not None:chosen=forced_selected
        retrieval_s=time.perf_counter()-t
        layouts=[self.layout(di,sel) for (di,qi),sel in zip(picks,chosen)]
        chains=[self.store.get_chain(parts) if self.store else [] for parts in layouts]
        matched=[sum(x['tokens'] for x in chain) for chain in chains]
        self.stats['requests']+=len(picks);self.stats['prefix_positions']+=6145*len(picks)
        self.stats['matched_positions']+=sum(matched);self.stats['full_hits']+=sum(n==6145 for n in matched)
        query=torch.tensor([self.docs[di]['queries'][qi]['query'] for di,qi in picks],device='cuda')
        qh,bottom,qpos=self.reader.write_prefill(query)
        bylength=collections.defaultdict(list)
        for row,chain in enumerate(chains):bylength[len(chain)].append(row)
        groups=[]
        for n,ids in bylength.items():
            selected=[chosen[i] for i in ids];subpicks=[picks[i] for i in ids]
            cache=self.restore([chains[i] for i in ids]);past=matched[ids[0]]
            doc=self.doc_hidden(subpicks,selected,n)
            hidden=torch.cat([doc,qh.index_select(0,torch.tensor(ids,device='cuda'))],dim=1)
            length=hidden.shape[1];positions=torch.arange(past,past+length,device='cuda')[None,:]
            causal=(torch.arange(past+length,device='cuda')[None,:]<=positions[0,:,None])[None,None,:,:]
            hidden=self.reader._run_layers(hidden,slice(self.j,self.reader.num_layers),causal,positions,self.reader.rotary_emb(hidden,position_ids=positions),past_key_values=cache,use_cache=True)
            logits=self.reader.lm_head(self.reader.norm(hidden[:,-1:,:]));groups.append((ids,cache,logits))
            del hidden,doc,causal
        top,logits=self.merge(groups,len(picks));del groups,cache,chains,qh
        self.store_prefix(top,layouts)
        first_logits=logits[:,-1].float().cpu() if capture else None
        traces=[first_logits] if capture else None
        token=logits[:,-1].float().argmax(-1);generated=[token.cpu().tolist()];arrivals=[time.perf_counter()]
        if forced_tokens is not None:token=torch.tensor([x[0] for x in forced_tokens],device='cuda')
        for n in range(1,count):
            logits=self.reader.decode_step(token,bottom,top,qpos+n-1,6657+n-1)
            token=logits[:,-1].float().argmax(-1);generated.append(token.cpu().tolist());arrivals.append(time.perf_counter())
            if capture:traces.append(logits[:,-1].float().cpu())
            if forced_tokens is not None:token=torch.tensor([x[n] for x in forced_tokens],device='cuda')
        detail=dict(templates=picks,selected=chosen,generated_ids=list(map(list,zip(*generated))),token_times=arrivals,
                    matched_positions=matched,retrieval_ms=retrieval_s*1000,persistent_gpu_h_bytes=self.persistent_gpu_h,
                    persistent_kv_bytes=self.store.bytes if self.store else 0,cache_evictions=self.store.evictions if self.store else 0)
        if capture:detail['first_logits']=first_logits;detail['logits_trace']=traces
        return arrivals[0],arrivals[-1],detail
