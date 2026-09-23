"""Keep read-only health and host admission probes off the request dispatch loop."""
from concurrent.futures import ThreadPoolExecutor
import time

class ControlPlane:
    def __init__(self,ctl,host,owner,out,save,clock=time.monotonic):
        self.ctl=ctl;self.host=host;self.owner=owner;self.out=out;self.save=save;self.clock=clock
        self.pool=ThreadPoolExecutor(max_workers=2);self.status_future=None;self.host_future=None;self.host_row=None
        self.last_healthy=clock();self.next_status=clock()+20;self.next_host=0
    def tick(self):
        now=self.clock()
        if self.status_future is not None and self.status_future.done():
            future=self.status_future;self.status_future=None;self.next_status=now+20
            try:result=future.result()
            except Exception as exc:
                self.save(self.out/('status_probe_failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),no_inference_retry=True))
            else:
                if any(k in result for k in ['worker_failure.json','process_receipt.json','memory_cap_failure.json']):
                    raise RuntimeError('Remote worker terminal record: '+repr(result))
                if 'worker_ready.json' not in result:raise RuntimeError('Running service lost ready record')
                self.last_healthy=now
        if self.status_future is None and now>=self.next_status:
            self.status_future=self.pool.submit(self.ctl,'status')
        return now-self.last_healthy<=90
    def admit(self,row,healthy,enabled=True):
        now=self.clock()
        if self.host_future is not None and self.host_future.done():
            future=self.host_future;pending_row=self.host_row;self.host_future=None;self.host_row=None;self.next_host=now+2
            try:admitted,proof=future.result()
            except Exception as exc:
                admitted=False;proof=dict(reason='Host admission probe failed; no reservation granted',error=repr(exc));self.next_host=now+5
            self.save(self.out/'host_admission.json',proof)
            if admitted and (not healthy or not enabled):
                self.host.release(self.owner,pending_row['task']);admitted=False
            if admitted:return pending_row
        if row is not None and healthy and enabled and self.host_future is None and now>=self.next_host:
            self.host_row=row;self.host_future=self.pool.submit(self.host.acquire,self.owner,row['task'],row['memory_mb'])
        return None
    def close(self):
        # A reservation not yet delivered to start_task must not outlive this owner.
        self.pool.shutdown(wait=True)
        if self.host_future is not None:
            try:admitted,_=self.host_future.result()
            except Exception:admitted=False
            if admitted:self.host.release(self.owner,self.host_row['task'])
            self.host_future=None
