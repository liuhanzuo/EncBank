"""Read-only snapshot; lightweight enough to run without the ML environment."""
import json, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def read(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        return dict(read_error=str(exc))


def tail_record(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(0,2); size=f.tell();f.seek(max(0,size-16384));lines=f.read().decode('utf-8').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue


def main():
    rows=[]
    for run in sorted((ROOT/'runs').glob('*')) if (ROOT/'runs').exists() else []:
        row=dict(name=run.name,worker=read(run/'worker.json'),status=read(run/'training/status.json'),
            last_train=tail_record(run/'training/train.jsonl'),gradient_check=read(run/'training/gradient_check.json'),
            complete=read(run/'complete.json'),failure=read(run/'failure.json'))
        rows.append(row)
    evaluations={p.name:dict(progress=read(p/'progress.json'),complete=read(p/'complete.json'))
                 for p in (ROOT/'evaluation').glob('*')} if (ROOT/'evaluation').exists() else {}
    queue=subprocess.run(['squeue','-u','liuhanzuo','-o','%.18i %.24j %.10T %.8M %.5D %R'],capture_output=True,text=True,check=True).stdout
    print(json.dumps(dict(at=time.time(),runs=rows,evaluations=evaluations,queue=queue,submission=read(ROOT/'submission.json'))))


if __name__=='__main__':
    main()
