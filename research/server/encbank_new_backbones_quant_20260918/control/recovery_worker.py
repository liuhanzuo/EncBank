"""Resume inspected infrastructure failures using unchanged scientific scripts."""
import datetime,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
from io_resilience import atomic_retry
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def write(path,value):
    def replace(p,v):
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
    return atomic_retry(replace,path,value)

def main():
    task_id=sys.argv[1]
    plan=json.loads((ROOT/'effective_plan.json').read_text())
    task=next(t for t in plan['tasks'] if t['id']==task_id)
    assert task['launcher']=='control/recovery.sh'
    out=ROOT/'runs'/task_id;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('SLURM_JOB_ID') and not (out/'parent_exit.json').exists()
    # Validate the same frozen source bytes without importing the model stack
    # into the CPU parent. The scientific child performs its own native checks.
    manifest=json.loads((ROOT/'package_manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if task['kind']=='training' and '--resume' in task['command']:
        info=task['resume_checkpoint']
        stat=(ROOT/'training'/task['model']/'last.pt').stat()
        assert info['step']==task['resume_step']
        assert stat.st_size==info['bytes'] and stat.st_mtime_ns==info['mtime_ns']
    wrapper=[str(ROOT/'control/durable_runner.py')] if task.get('io_wrapper') else []
    cmd=[sys.executable,'-B','-u',*wrapper,*task['command']]
    write(out/'start.json',dict(command=cmd,job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),
          recovery_from_job=task.get('recovery_from_job'),resume_step=task.get('resume_step'),
          resume_checkpoint=task.get('resume_checkpoint'),io_wrapper=bool(wrapper),
          at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (out/'child.stdout.log').open('ab') as stdout,(out/'child.stderr.log').open('ab') as stderr:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr);code=process.wait()
    complete=ROOT/task['complete']
    write(out/'parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=process.pid,
          completion_exists=complete.exists(),completion=str(complete),job=os.environ['SLURM_JOB_ID'],
          at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    raise SystemExit(code if code else (0 if complete.exists() else 2))

if __name__=='__main__':main()
