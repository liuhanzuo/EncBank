"""Local observation only; never automatically re-submit GPU work."""
import datetime,json,msvcrt,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(name,v):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
    handle=(ROOT/'owner.lock').open('a+b');handle.write(b'0');handle.flush();handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'monitor.py')],capture_output=True,encoding='utf-8',timeout=55)
        if r.returncode:
            dump('owner_status.json',dict(phase='MONITOR_RETRY',error=r.stderr[-1800:],at=datetime.datetime.now().astimezone().isoformat()));time.sleep(60);continue
        state=json.loads((ROOT/'monitor.json').read_text());files=state['files'];wait=files.get('parent_exit.json')
        if files.get('failure.json') or (wait and wait['returncode']!=0):raise RuntimeError(dict(failure=files.get('failure.json'),wait=wait))
        row=next((l.split('|') for l in state['sacct'].splitlines() if l.startswith(state['job']+'|')),None)
        if row and row[1].split()[0] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']:raise RuntimeError(row)
        complete=sum('complete.json' in p for p in state['points'].values())
        dump('owner_status.json',dict(phase='WAITING',points=complete,at=datetime.datetime.now().astimezone().isoformat()))
        if wait and wait['actual_wait'] and wait['returncode']==0 and row and row[1:3]==['COMPLETED','0:0']:break
        time.sleep(60)
    for name in ['collect.py','report.py']:
        r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/name)],cwd=ROOT);assert r.returncode==0,name
    dump('owner_status.json',dict(phase='COMPLETE',at=datetime.datetime.now().astimezone().isoformat()))
if __name__=='__main__':
    try:main()
    except BaseException as exc:dump('owner_failure.json',dict(error=repr(exc)));raise
