"""Tolerate read-only control transport failures while waiting for allocation/load."""
import time
def wait_for_worker(ctl,save,out,max_wait_seconds=168*3600,sleep=time.sleep,clock=time.monotonic):
    began=clock()
    while True:
        if clock()-began>max_wait_seconds:raise TimeoutError('Startup owner wait bound; no duplicate submission')
        try:status=ctl('status')
        except Exception as exc:
            save(out/'startup_status_read_failure.json',dict(error=repr(exc),epoch=time.time(),state='waiting_control_transport',no_model_retry=True,no_gpu_submission=True))
        else:
            if any(n in status for n in ['worker_failure.json','process_receipt.json','memory_cap_failure.json']):
                raise RuntimeError('Actual worker terminal record: '+repr(status))
            if 'worker_ready.json' in status:return status
            save(out/'startup_wait_status.json',dict(epoch=time.time(),state='waiting_gpu_worker',read_only_query_succeeded=True))
        sleep(20)
