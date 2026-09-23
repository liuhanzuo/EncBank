"""Sequential C3 methods on a single local GPU; exact child waits and output retention."""
import datetime,json,msvcrt,os,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(p,r):
    p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def main():
    lock=(ROOT/'local_owner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    cells=[('selected_replay',1),('encbank',1),('selected_replay',4),('encbank',4)]
    random.Random(20260919).shuffle(cells)
    dump(ROOT/'serving_order.json',dict(cells=cells,seed=20260919,requests_per_cell=1000,output_tokens=32))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='0')
    for method,c in cells:
        out=ROOT/'serving/rep0/32768'/(method+'_c'+str(c));out.mkdir(parents=True,exist_ok=True)
        if (out/'parent_exit.json').exists():
            old=json.loads((out/'parent_exit.json').read_text());assert old['returncode']==0 and (out/'complete.json').exists();continue
        cmd=[sys.executable,'-X','utf8','-B','-u',str(ROOT/'serving.py'),'--method',method,'--concurrency',str(c)]
        with (out/'stdout.log').open('ab') as stdout,(out/'stderr.log').open('ab') as stderr:
            child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=stdout,stderr=stderr)
            dump(ROOT/'local_status.json',dict(phase='RUNNING',method=method,concurrency=c,child_pid=child.pid,at=datetime.datetime.now().astimezone().isoformat()))
            code=child.wait()
        dump(out/'parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=child.pid))
        assert code==0,(method,c,code)
    dump(ROOT/'local_status.json',dict(phase='COMPLETE',cells=4,at=datetime.datetime.now().astimezone().isoformat()))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        dump(ROOT/'local_failure.json',dict(error=repr(exc),at=datetime.datetime.now().astimezone().isoformat()));raise
