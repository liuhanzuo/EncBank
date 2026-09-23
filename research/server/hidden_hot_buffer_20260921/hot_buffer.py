"""LRU by frozen hidden identity/head version, with one pinned active chunk."""
from collections import OrderedDict

class HotKVBuffer:
    def __init__(self,capacity,head_version):
        assert capacity>=2
        self.capacity=capacity;self.head_version=head_version
        self.entries=OrderedDict();self.active=None;self.hits=0;self.misses=0;self.evictions=0
    def key(self,hidden_id):return self.head_version,hidden_id
    def trim(self):
        limit=self.capacity-int(self.active is not None)
        while len(self.entries)>limit:self.entries.popitem(last=False);self.evictions+=1
        assert len(self.entries)+int(self.active is not None)<=self.capacity
    def begin(self,active):
        assert self.active is None
        self.active=active;self.trim()
    def get(self,hidden_id,factory):
        key=self.key(hidden_id)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
        self.misses+=1;value=factory()
        self.entries[key]=value;self.trim();return value
    def complete(self,hidden_id,projected):
        assert self.active is not None
        self.active=None;key=self.key(hidden_id)
        self.entries[key]=projected;self.entries.move_to_end(key);self.trim()
    def snapshot(self):
        return dict(capacity=self.capacity,completed_entries=len(self.entries),active_pinned=self.active is not None,
            hits=self.hits,misses=self.misses,evictions=self.evictions)
