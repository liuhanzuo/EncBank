"""Wait for C3 and complete C4 answers, then run same-5090 depth timing and report."""
import datetime,json,msvcrt,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(p,r):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def main():
    lock=(ROOT/'final_owner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        result=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'monitor.py')],capture_output=True,encoding='utf-8',timeout=125)
        assert result.returncode==0,result.stderr[-1000:]
        state=json.loads(result.stdout)
        assert not state['remote']['owner_failure'],state['remote']['owner_failure']
        if (ROOT/'local_failure.json').exists():raise RuntimeError('C3 failed; inspect retained run before resuming')
        local=json.loads((ROOT/'local_status.json').read_text()) if (ROOT/'local_status.json').exists() else {}
        ready=state['remote']['owner'] is not None and state['remote']['owner']['phase']=='COMPLETE'
        dump(ROOT/'final_owner_status.json',dict(phase='WAITING',quality_complete=ready,serving_complete=local.get('phase')=='COMPLETE',at=datetime.datetime.now().astimezone().isoformat()))
        if ready and local.get('phase')=='COMPLETE':break
        time.sleep(60)
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'collect_remote.py')],cwd=ROOT)
    assert result.returncode==0
    order=[6,12,18];random.Random(20260919).shuffle(order);dump(ROOT/'latency_order.json',dict(order=order,seed=20260919))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='0')
    for j in order:
        dest=ROOT/'latency'/('j%d_s42'%j);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'parent_exit.json').exists():
            assert json.loads((dest/'parent_exit.json').read_text())['returncode']==0;continue
        with (dest/'stdout.log').open('ab') as o,(dest/'stderr.log').open('ab') as e:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(ROOT/'latency.py'),'--j',str(j)],cwd=ROOT,env=env,stdout=o,stderr=e)
            dump(ROOT/'final_owner_status.json',dict(phase='LATENCY',j=j,child_pid=child.pid));code=child.wait()
        dump(dest/'parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=child.pid));assert code==0
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'report.py')],cwd=ROOT)
    assert result.returncode==0
    dump(ROOT/'final_owner_status.json',dict(phase='COMPLETE',at=datetime.datetime.now().astimezone().isoformat()))
if __name__=='__main__':
    try:main()
    except BaseException as exc:dump(ROOT/'final_owner_failure.json',dict(error=repr(exc)));raise
