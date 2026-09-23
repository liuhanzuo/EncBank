"""Exact segment replay using real anchors, same positions and attention graph."""
import gc,random,statistics
from concurrent.futures import ThreadPoolExecutor

def measure_replay(reader,model,bundle,depths,count,timed):
    import torch
    from transformers.cache_utils import DynamicCache
    n=bundle['n'];p=torch.arange(n,device='cuda')[None]
    mask=(p[:,None,:]<=p[:,:,None])[:,None]
    anchors={s:bundle['states'][s].to('cuda') for s in depths}
    rope=reader.rotary_emb(anchors[12],position_ids=p)
    ends=depths[1:]+[36]
    streams=[torch.cuda.Stream() for _ in depths]
    default=torch.cuda.current_stream()
    pool=ThreadPoolExecutor(max_workers=len(depths),thread_name_prefix='shallow_replay')
    def run(h,lo,hi,cache):
        return reader._run_layers(h,slice(lo,hi),mask,p,rope,past_key_values=cache,use_cache=True)
    @torch.inference_mode()
    def branch(i,cache,start,done):
        with torch.cuda.device(0),torch.cuda.stream(streams[i]):
            streams[i].wait_event(start)
            h=run(anchors[depths[i]],depths[i],ends[i],cache)
            done.record(streams[i])
            return h
    def point(method):
        if method=='h12_full':
            cache=DynamicCache(config=model.config)
            h=run(anchors[12],12,36,cache)
            return cache
        caches=[DynamicCache(config=model.config) for _ in depths]
        if method=='segmented_serial':
            for i,s in enumerate(depths):h=run(anchors[s],s,ends[i],caches[i])
        else:
            start=torch.cuda.Event();start.record(default)
            done=[torch.cuda.Event() for _ in depths]
            futures=[pool.submit(branch,i,caches[i],start,done[i]) for i in range(len(depths))]
            hidden=[f.result() for f in futures]
            for event in done:default.wait_event(event)
        result=caches[0]
        for i in range(1,len(depths)):
            for l in range(depths[i],ends[i]):result.layers[l]=caches[i].layers[l]
        return result
    checks=[]
    for method in ['h12_full','segmented_serial','segmented_threads']:
        actual,_=timed(lambda:point(method));error=0.;refsum=0.;sq=0.
        for l in range(12,36):
            for x,ycpu in zip((actual.layers[l].keys,actual.layers[l].values),bundle['kv'][l]):
                y=ycpu.to('cuda');d=x.float()-y.float()
                error=max(error,float(d.abs().max()));sq+=float(d.square().sum());refsum+=float(y.float().square().sum())
        rel=(sq/max(refsum,1e-30))**.5
        assert rel<1e-5,(method,rel)
        checks.append(dict(method=method,max_abs=error,relative_rms=rel))
        del actual,x,y,d
    rows=[];rng=random.Random(20260921+count)
    for repeat in range(-2,5):
        methods=['h12_full','segmented_serial','segmented_threads'];rng.shuffle(methods)
        for method in methods:
            actual,elapsed=timed(lambda:point(method))
            if repeat>=0:rows.append(dict(method=method,repeat=repeat,**elapsed))
            del actual
    pool.shutdown();torch.cuda.synchronize()
    # Cost to create the extra anchors from H12 in the same joint context.
    def prepare():
        h=anchors[12];saved={};lo=12
        for s in depths[1:]:
            h=run(h,lo,s,None);saved[s]=h;lo=s
        return saved
    prep=[]
    for repeat in range(5):
        saved,elapsed=timed(prepare)
        if repeat>=2:prep.append(elapsed)
        del saved
    del anchors;gc.collect()
    return dict(chunks=count,depths=depths,checks=checks,observations=rows,
        median_ms={method:statistics.median(r['wall_ms'] for r in rows if r['method']==method)
            for method in ['h12_full','segmented_serial','segmented_threads']},
        extra_anchor_prepare_median_ms=statistics.median(r['wall_ms'] for r in prep),
        extra_anchor_prepare=prep,scope='Exact replay from fixed-context true hidden anchors. Preparation excluded from replay. Same single GPU; no layer beyond H24 cached.')
