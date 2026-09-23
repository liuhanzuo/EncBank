"""Token-level original 8B agent layout, with a registered 512-token recent window."""
def partition(ids,anchor,recent=512,chunk=512):
    assert ids[:len(anchor)]==anchor,'Pinned initial prompt changed'
    rest=ids[len(anchor):];count=max(0,(len(rest)-recent)//chunk)
    archive=rest[:count*chunk];query=anchor+rest[count*chunk:]
    assert anchor+archive+query[len(anchor):]==ids
    return [archive[i:i+chunk] for i in range(0,len(archive),chunk)],query
def fill_recent(selected,n,k):
    result=set(selected)
    for i in reversed(range(n)):
        if len(result)>=min(n,k):break
        result.add(i)
    return sorted(result)
