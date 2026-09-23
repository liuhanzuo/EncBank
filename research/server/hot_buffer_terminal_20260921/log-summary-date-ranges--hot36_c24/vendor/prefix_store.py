"""Bounded exact causal-prefix KV blocks, sharing only identical ordered prefixes."""
from collections import OrderedDict

class PrefixStore:
    def __init__(self,budget):
        self.budget=budget;self.bytes=0;self.items=OrderedDict();self.children={};self.evictions=0
    def get_chain(self,parts):
        chain=[]
        for n in range(1,len(parts)+1):
            key=tuple(parts[:n])
            if key not in self.items:break
            item=self.items[key];self.items.move_to_end(key);chain.append(item)
        return chain
    def put(self,key,tensors_factory,size,tokens):
        key=tuple(key)
        if key in self.items:self.items.move_to_end(key);return True
        if len(key)>1 and key[:-1] not in self.items:return False
        protected={key[:i] for i in range(1,len(key))}
        while self.bytes+size>self.budget:
            victim=next((k for k in self.items if self.children[k]==0 and k not in protected),None)
            if victim is None:return False
            old=self.items.pop(victim);self.bytes-=old['bytes'];self.children.pop(victim)
            if len(victim)>1:self.children[victim[:-1]]-=1
            self.evictions+=1
        if len(key)>1:self.children[key[:-1]]+=1
        self.items[key]=dict(tensors=tensors_factory(),bytes=size,tokens=tokens)
        self.children[key]=0;self.bytes+=size
        assert self.bytes<=self.budget
        return True
