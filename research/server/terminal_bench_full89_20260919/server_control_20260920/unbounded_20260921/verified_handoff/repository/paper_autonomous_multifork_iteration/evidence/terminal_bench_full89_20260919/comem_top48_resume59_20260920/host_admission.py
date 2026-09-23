"""Cross-owner RAM reservations for local Docker task worlds; no model offload."""
from pathlib import Path
import contextlib,errno,json,msvcrt,os,subprocess,time
ROOT=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/host_admission')
LEGACY=Path('/srv/encbank/legacy_workspace/paper_autonomous_multifork_iteration/evidence/terminal_bench_full89_20260919/toolchain_fix/execution/harbor_receipt.json')
@contextlib.contextmanager
def locked():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        deadline=time.monotonic()+120
        while True:
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError as exc:
                if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) or time.monotonic()>=deadline:raise
                time.sleep(.1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
def read():return json.loads((ROOT/'reservations.json').read_text()) if (ROOT/'reservations.json').exists() else {}
def save(d):
    p=ROOT/'reservations.json';t=p.with_name(p.name+'.tmp-'+str(os.getpid())+'-'+str(time.time_ns()));t.write_text(json.dumps(d,indent=2)+'\n')
    for attempt in range(10):
        try:t.replace(p);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.2)
def acquire(owner,task,memory_mb):
    # Slow WSL process startup must never hold the cross-owner metadata lock.
    for attempt in range(3):
        r=subprocess.run(['wsl','-d','Ubuntu','--exec','cat','/proc/meminfo'],capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30)
        assert r.returncode==0,r.stderr
        available=int(next(line.split()[1] for line in r.stdout.splitlines() if line.startswith('MemAvailable:')))/1024
        sampled=time.monotonic()
        with locked():
            if time.monotonic()-sampled>5:continue
            d=read();key=owner+':'+task;assert key not in d
            occupied=sum(x['memory_mb'] for x in d.values())
            legacy=0 if LEGACY.exists() else 16*1024
            proof=dict(epoch=time.time(),available_mb=available,owned_reserved_mb=occupied,legacy_reserved_mb=legacy,new_mb=memory_mb,safety_mb=2048,total_budget_mb=20480,sample_age_seconds=time.monotonic()-sampled)
            ok=occupied+legacy+memory_mb<=20480 and available>=occupied+legacy+memory_mb+2048
            if ok:d[key]=dict(owner=owner,task=task,memory_mb=memory_mb,owner_pid=os.getpid(),epoch=time.time());save(d)
            return ok,proof
    return False,dict(reason='fresh host memory sample unavailable after lock contention',epoch=time.time())

def release(owner,task):
    with locked():
        d=read();key=owner+':'+task;assert key in d;assert d[key]['owner_pid']==os.getpid();del d[key];save(d)
