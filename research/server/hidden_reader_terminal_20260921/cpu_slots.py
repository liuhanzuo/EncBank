"""CPU leases for this single server run; dead leases require explicit review."""
import fcntl,json,os
from pathlib import Path
from contextlib import contextmanager
H=Path(__file__).resolve().parent
@contextmanager
def locked():
    with (H/'container_cpu.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield
def acquire(name,count):
    with locked():
        f=H/'container_cpu_leases.json'
        leases=json.loads(f.read_text()) if f.exists() else {}
        assert name not in leases
        occupied={x for row in leases.values() for x in row['cpus']}
        free=sorted(set(os.sched_getaffinity(0))-occupied)
        assert len(free)>=count,'Insufficient allocated CPU slots for task'
        cpus=free[:count];leases[name]={'cpus':cpus,'owner_pid':os.getpid()}
        temp=f.with_suffix('.tmp');temp.write_text(json.dumps(leases));temp.replace(f)
        return cpus
def release(name):
    with locked():
        f=H/'container_cpu_leases.json'
        leases=json.loads(f.read_text()) if f.exists() else {}
        row=leases.pop(name,None)
        if row is not None: assert row['owner_pid']==os.getpid()
        temp=f.with_suffix('.tmp');temp.write_text(json.dumps(leases));temp.replace(f)
