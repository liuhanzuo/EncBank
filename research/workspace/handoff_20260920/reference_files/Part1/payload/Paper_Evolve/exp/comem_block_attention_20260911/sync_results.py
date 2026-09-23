"""Read-only lightweight result sync; checkpoint weights stay on the remote host."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = "/data/liuhanzuo/comem_v2_20260908"
REMOTE_OUT = ROOT + "/outputs/sparse_comem_20260911"
MAX_TOTAL = 64 * 1024 * 1024
MAX_FILE = 16 * 1024 * 1024
TOP = {"queue.json", "launch.json", "controller.log", "SUMMARY.json", "REPORT.md", "SYNC_INFO.json",
       "launch_guard_retry1.json", "controller_guard_retry1.log"}
ARM_FILES = {"metadata.json", "status.json", "train.jsonl", "worker.log", "gpu_admission.json", "gpu_lease.json"}
# Keep the original failure and the explicitly authorized retry as separate runs.
STAGE_ARMS = {"smoke": ("D0", "A", "B", "D1", "D0_retry1"),
              "train": ("D0", "A", "B", "D1")}


def allowed_name(name):
    if "\\" in name or ":" in name or "\0" in name:
        return False
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or path.as_posix() != name or any(x in (".", "..") for x in parts):
        return False
    if len(parts) == 1:
        return name in TOP
    if len(parts) != 3 or parts[1] not in STAGE_ARMS.get(parts[0], ()):
        return False
    leaf = parts[2]
    if leaf in ARM_FILES:
        return True
    import re
    return bool(re.fullmatch(r"eval_step[0-9]+(?:\.records)?\.jsonl?", leaf))


def remote_source():
    return f"""
from pathlib import Path
from datetime import datetime, timezone
import io,json,sys,tarfile
root=Path({REMOTE_OUT!r})
top={sorted(TOP - {"SYNC_INFO.json"})!r}
stage_arms={STAGE_ARMS!r}
files=[]
for name in top:
 p=root/name
 if p.is_file() and not p.is_symlink(): files.append(p)
for stage,arms in stage_arms.items():
 for arm in arms:
  directory=root/stage/arm
  if not directory.is_dir() or directory.is_symlink(): continue
  for p in sorted(directory.iterdir()):
   if not p.is_file() or p.is_symlink(): continue
   n=p.name
   if n in {sorted(ARM_FILES)!r} or (n.startswith('eval_step') and p.suffix in ['.json','.jsonl']):
    files.append(p)
items=[]; total=0
for p in files:
 p.resolve().relative_to(root.resolve())
 size=p.stat().st_size
 if size>{MAX_FILE}: raise ValueError('Individual lightweight artifact exceeds bound: '+p.name)
 data=p.read_bytes()
 total+=len(data)
 if len(data)>{MAX_FILE} or total>{MAX_TOTAL}: raise ValueError('Lightweight sync exceeds bound')
 items.append((p.relative_to(root).as_posix(),data))
info={{'observed_utc':datetime.now(timezone.utc).isoformat(),'source':str(root),
       'checkpoint_weights_copied':False,'files':[n for n,_ in items],
       'consistency':'Individual atomic JSON files and live logs; global training state may advance during copy.'}}
items.append(('SYNC_INFO.json',json.dumps(info,indent=2).encode()))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,data in items:
  entry=tarfile.TarInfo(name); entry.size=len(data); entry.mode=0o600
  archive.addfile(entry,io.BytesIO(data))
"""


def extract_snapshot(payload, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not allowed_name(member.name) or member.name in seen:
                raise ValueError("Unexpected or unsafe artifact: " + member.name)
            if member.size < 0 or member.size > MAX_FILE:
                raise ValueError("Artifact too large")
            seen.add(member.name)
            total += member.size
            if total > MAX_TOTAL or len(seen) > 2000:
                raise ValueError("Snapshot exceeds bounds")
            data = archive.extractfile(member).read()
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.resolve().relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if "SYNC_INFO.json" not in seen:
        raise ValueError("Missing synchronization information")
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results/remote")
    args = parser.parse_args()
    command = [ROOT + "/venv/bin/python", "-c", remote_source()]
    result = subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12","longjing-1",
                             " ".join(shlex.quote(x) for x in command)],
                            check=True,capture_output=True,timeout=90)
    if len(result.stdout) > MAX_TOTAL:
        raise ValueError("Remote compressed payload exceeds bound")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = args.out / stamp
    names = extract_snapshot(result.stdout, destination)
    latest = {"snapshot":str(destination.resolve()),"file_count":len(names),
              "updated_utc":datetime.now(timezone.utc).isoformat(),
              "checkpoint_weights_copied":False}
    temp = args.out / "LATEST.tmp"
    temp.write_text(json.dumps(latest,indent=2)+"\n",encoding="utf-8")
    temp.replace(args.out / "LATEST.json")
    print(json.dumps(latest,indent=2))


if __name__ == "__main__":
    main()
