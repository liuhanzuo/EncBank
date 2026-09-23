"""Actual child wait and singleton, including infrastructure failures."""
import argparse, datetime, fcntl, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def main():
    task_id=sys.argv[1]
    plan=json.loads((ROOT/'plan.json').read_text());task=next(t for t in plan['tasks'] if t['id']==task_id)
    out=ROOT/'runs'/task_id;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('SLURM_JOB_ID')
    assert not (out/'parent_exit.json').exists(),'Existing attempt must be inspected; no automatic rerun'
    from binding import verify_code
    verify_code()
    at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    cmd=[sys.executable,'-B','-u',*task['command']]
    write(out/'start.json',dict(command=cmd,job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),at=at))
    with (out/'child.stdout.log').open('ab') as stdout,(out/'child.stderr.log').open('ab') as stderr:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        code=proc.wait()
    complete=ROOT/task['complete']
    write(out/'parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=proc.pid,
          completion_exists=complete.exists(),completion=str(complete),
          job=os.environ['SLURM_JOB_ID'],at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    raise SystemExit(code if code else (0 if complete.exists() else 2))

if __name__=='__main__':main()
