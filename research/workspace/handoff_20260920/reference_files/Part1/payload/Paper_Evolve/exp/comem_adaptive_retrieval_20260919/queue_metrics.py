"""Real-timestamp closed-loop FIFO batching; no GPU assumptions."""
from collections import deque
import math, time

def percentile(xs,p):
    xs=sorted(xs);assert xs and 0<=p<=100
    z=(len(xs)-1)*p/100;lo=int(z);hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(z-lo)

def closed_loop(service, concurrency, count, max_batch, clock=time.perf_counter, on_batch=None):
    assert 0<concurrency<=count and max_batch>0
    origin=clock();pending=deque((i,origin) for i in range(concurrency))
    issued=concurrency;rows=[];batch_id=0
    while pending:
        batch=[pending.popleft() for _ in range(min(max_batch,len(pending)))]
        started=clock()
        try:
            first,ended,detail=service([i for i,_ in batch])
        except BaseException as exc:
            exc.serving_progress=dict(completed=rows,failed_batch=[i for i,_ in batch],
                failed_batch_size=len(batch),issued=issued,wall_s=clock()-origin)
            raise
        assert started<=first<=ended<=clock()
        returned=clock()
        for i,arrived in batch:
            rows.append(dict(id=i,batch=batch_id,batch_size=len(batch),arrival=arrived-origin,
                start=started-origin,first=first-origin,end=ended-origin,
                queue_s=started-arrived,ttft_s=first-arrived,e2e_s=ended-arrived,
                service_s=ended-started,detail=detail))
            if issued<count:
                pending.append((issued,returned));issued+=1
        if on_batch:on_batch(rows[-len(batch):],len(rows))
        batch_id+=1
    assert len(rows)==count and len({r['id'] for r in rows})==count
    return rows,max(r['end'] for r in rows)

def summarize(rows,span,output_tokens):
    assert rows and span>0 and all(math.isfinite(r['e2e_s']) for r in rows)
    return dict(requests=len(rows),batches=len({r['batch'] for r in rows}),wall_s=span,
        requests_per_s=len(rows)/span,output_tokens_per_s=len(rows)*output_tokens/span,
        **{field:{'p'+str(p):1000*percentile([r[field] for r in rows],p) for p in [50,95,99]}
           for field in ['queue_s','ttft_s','e2e_s','service_s']})
