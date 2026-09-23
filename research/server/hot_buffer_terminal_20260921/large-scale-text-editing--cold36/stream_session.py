"""H12 cold bank plus hot memory; bounded active context refreshed every 512 generated tokens."""
import time
from collections import Counter
import torch
from hot_memory import HotMemory,token_key
from bm25_index import BM25Index
from layout import partition,fill_recent

def stamp():torch.cuda.synchronize();return time.perf_counter()

class StreamSession:
    def __init__(self,band,tok,policy='cold',depth=36,capacity=24,promote=False):
        self.band=band;self.reader=band.reader;self.tok=tok
        self.memory=HotMemory(band,policy,depth,capacity,promote);self.bank={};self.sink=None;self.anchor=None
    def prepare(self,ids):
        chunks,query=partition(ids,self.anchor,512,512);start=stamp()
        keys=[token_key(c) for c in chunks];missing={k:chunks[i] for i,k in enumerate(keys) if k not in self.bank}
        if (len(self.bank)+len(missing))*512*4096*2>8*2**30:raise RuntimeError('Explicit H12 bank 8GiB physical bound')
        todo=list(missing.items())
        for off in range(0,len(todo),8):
            group=todo[off:off+8];h=self.reader.write_chunk(torch.tensor([v for k,v in group],device='cuda'))
            for b,(key,_) in enumerate(group):self.bank[key]=h[b:b+1].detach().clone()
        if chunks and self.sink is None:self.sink=self.reader.write_chunk([151643])
        write=stamp()-start;start=time.perf_counter()
        selected=fill_recent(BM25Index(chunks).select(self.goal+ids[-2048:],12,4),len(chunks),12) if chunks else []
        retrieval=time.perf_counter()-start
        states=([self.sink]+[self.bank[keys[i]] for i in selected]) if selected else []
        parts=([(-1,'sink')]+[(i,keys[i]) for i in selected]) if selected else []
        if sum(h.shape[1] for h in states)+len(query)>=40960:raise RuntimeError('Native position capacity reached')
        start=stamp();logits,bottom,top,qpos,nt=self.memory.prefill(parts,states,query);prefill=stamp()-start
        event=dict(write_seconds=write,retrieval_seconds=retrieval,prefill_seconds=prefill,archived_chunks=len(chunks),
            selected_indices=selected,selected_chunks=len(selected),query_tokens=len(query),new_h12_chunks=len(missing),**self.memory.last)
        return logits,bottom,top,qpos,nt,len(chunks),event
    @torch.inference_mode()
    def run(self,messages,forced=None,cancelled=lambda:False,capture_logits=False):
        begin=stamp();torch.cuda.reset_peak_memory_stats()
        assert forced is None or len(forced)>0
        initial_stats=self.memory.stats.copy();initial_evictions=self.memory.pool.stats['evictions']
        ids=self.tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        if self.anchor is None:
            first=next(i for i,m in enumerate(messages) if m['role']=='user')
            self.anchor=self.tok.apply_chat_template(messages[:first+1],tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=False)
            self.goal=self.tok.encode(messages[first]['content'],add_special_tokens=False)[-512:]
        initial=len(ids);logits,bottom,top,qpos,nt,archive,event=self.prepare(ids);events=[event]
        first_logits=logits[0,-1].detach().float().cpu() if capture_logits else None
        boundary_logits=[]
        generated=[];decode_seconds=0.;promote_seconds=0.;ttft=None;reason=None
        eos=self.reader.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(self.tok.eos_token_id)
        while True:
            if cancelled():reason='cancelled';break
            token=int(logits[0,-1].argmax()) if forced is None else int(forced[len(generated)])
            generated.append(token);ids.append(token)
            if ttft is None:ttft=stamp()-begin
            if forced is None and token in eos:reason='eos';break
            if nt+qpos>=40960:reason='context_capacity';break
            start=stamp()
            logits=self.reader.decode_step(torch.tensor([token],device='cuda'),bottom,top,qpos,nt+qpos)
            qpos+=1;decode_seconds+=stamp()-start
            if len(generated)%512==0:
                start=stamp();self.memory.promote_query(ids,len(self.anchor),archive,top,nt,'online_decode');promote_seconds+=stamp()-start
                # Drop old active caches before building new ones; persistent pool entries own clones.
                del bottom,top,logits
                logits,bottom,top,qpos,nt,archive,event=self.prepare(ids);event['generated_boundary']=len(generated);events.append(event)
                if capture_logits:boundary_logits.append(logits[0,-1].detach().float().cpu())
            if forced is not None and len(generated)>=len(forced):reason='forced_replay';break
        start=stamp();self.memory.promote_query(ids,len(self.anchor),archive,top,nt,'online_decode' if len(generated)>1 else 'online_prefill');promote_seconds+=stamp()-start
        end=stamp();counts=Counter()
        for e in events:
            for k in ['requested_chunks','hit_chunks','rebuilt_chunks','promoted_hits']:counts[k]+=e.get(k,0)
        result=dict(text=self.tok.decode(generated,skip_special_tokens=True),generated_ids=generated,generated_tokens=len(generated),
            stop_reason=reason,logical_prompt_tokens=initial,logical_prompt_sha256=token_key(ids[:initial]),events=events,
            request_seconds=end-begin,ttft_seconds=ttft,decode_seconds=decode_seconds,promotion_seconds=promote_seconds,
            prefill_seconds=sum(e['prefill_seconds'] for e in events),write_seconds=sum(e['write_seconds'] for e in events),
            retrieval_seconds=sum(e['retrieval_seconds'] for e in events),cache_bytes=self.memory.pool.bytes,cache_budget=self.memory.pool.budget,
            evictions=self.memory.pool.stats['evictions']-initial_evictions,promotions=self.memory.stats['promotions']-initial_stats['promotions'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),**counts)
        if capture_logits:result.update(first_logits=first_logits,boundary_logits=boundary_logits)
        return result
