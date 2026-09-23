"""Token-level append-only conversation layout; independent of model execution."""
def partition(ids, anchor_ids, recent=4096, chunk=512):
    assert ids[:len(anchor_ids)] == anchor_ids, 'First instruction changed'
    rest = ids[len(anchor_ids):]
    n = max(0, (len(rest)-recent)//chunk)
    archived = rest[:n*chunk]
    query = anchor_ids + rest[n*chunk:]
    assert anchor_ids + archived + query[len(anchor_ids):] == ids
    return [archived[i:i+chunk] for i in range(0,len(archived),chunk)], query

def validate_append(previous, current):
    assert len(current)>=len(previous) and current[:len(previous)]==previous, 'Archived history was edited; refusing stale hidden reuse'

def fill_recent(selected, n, k):
    filled=[]; seen=set(selected)
    for i in reversed(range(n)):
        if len(seen)>=min(n,k):break
        if i not in seen:seen.add(i);filled.append(i)
    return sorted(seen),sorted(filled)

def lcp(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n
