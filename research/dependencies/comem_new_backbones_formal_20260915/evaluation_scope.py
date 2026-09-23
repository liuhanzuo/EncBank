"""Explicit active support, shared by generation, verification and aggregation."""
import json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read_scope():
    name=os.environ.get('MIDCACHE_SCOPE_FILE','evaluation_scope.json')
    assert name in ('evaluation_scope.json','locomo_scope.json')
    s=json.loads((ROOT/name).read_text())
    assert s['benchmarks'] and len(set(s['benchmarks']))==len(s['benchmarks'])
    assert set(s['benchmarks']).isdisjoint(s['deferred'])
    return s

def results_directory():
    name=os.environ.get('MIDCACHE_RESULTS_DIR','results')
    assert name in ('results','results_locomo')
    scope=read_scope()
    assert (name=='results_locomo') == (scope['benchmarks']==['locomo'])
    return name

def shard_count(spec, shard, scope):
    """Count retained rows under the original full-dataset round-robin sharding."""
    nshards=spec['shards'];assert 0<=shard<nshards
    cursor=0;selected=0
    for cell,n in spec['cells'].items():
        assert n>0
        if cell.split(':')[0] in scope['benchmarks']:
            first=cursor+(shard-cursor)%nshards
            if first<cursor+n:selected+=1+(cursor+n-1-first)//nshards
        cursor+=n
    assert cursor==spec['samples']
    return selected
