"""Read-only SSH synchronization of small SFT artifacts; never copy weights.

No model imports or task/process mutations. Remote Python streams a whitelisted
tar archive directly to this process. Local extraction rejects links, traversal,
duplicate entries, unexpected files, and excessive sizes. A completed immutable
snapshot is published through atomic LATEST.json only after validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = "/data/liuhanzuo/comem_v2_20260908/outputs/beacon_sft_20260909"
DEFAULT_OUT = HERE / "results/remote"
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MAX_FILES = 1000
ARM = re.compile(r"[A-Za-z0-9_-]+\Z")
EVAL = re.compile(r"eval_step[0-9]+(?:\.records(?:\.summary)?)?\.jsonl?\Z")
SMALL_NAMES = {"status.json", "metadata.json", "preparation.json", "train.jsonl", "smoke.log", "train.log"}


def allowed_name(name):
    if "\\" in name or ":" in name or "\x00" in name:
        return False
    parts = PurePosixPath(name).parts
    if PurePosixPath(name).as_posix() != name or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in parts):
        return False
    if len(parts) == 1:
        return name in {"queue.json", "controller.log", "SYNC_MANIFEST.json"}
    return (len(parts) == 2 and bool(ARM.fullmatch(parts[0]))
            and (parts[1] in SMALL_NAMES or bool(EVAL.fullmatch(parts[1]))))


def remote_script():
    # All archive members are bytes read from regular files inside one fixed task.
    return f'''
import datetime,io,json,os,pathlib,re,socket,stat,sys,tarfile
root=pathlib.Path({REMOTE_ROOT!r}).resolve()
max_file={MAX_FILE}; max_total={MAX_TOTAL}; max_files={MAX_FILES}
if not root.is_dir(): raise RuntimeError('SFT output directory does not exist')
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
names={{'status.json','metadata.json','preparation.json','train.jsonl','smoke.log','train.log'}}
eval_re=re.compile(r'eval_step[0-9]+(?:\\.records(?:\\.summary)?)?\\.jsonl?\\Z')
paths=[p for p in (root/'queue.json',root/'controller.log') if p.exists()]
for arm in sorted(root.iterdir()):
 if arm.is_symlink() or not arm.is_dir() or not re.fullmatch(r'[A-Za-z0-9_-]+',arm.name): continue
 for p in sorted(arm.iterdir()):
  if p.name in names or eval_re.fullmatch(p.name): paths.append(p)
if len(paths)>max_files: raise RuntimeError('Too many lightweight files')
files=[]; total=0; queue=None
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for p in paths:
  if p.is_symlink() or not stat.S_ISREG(p.stat().st_mode): raise RuntimeError('Nonregular lightweight artifact')
  p.resolve().relative_to(root)
  before=p.stat()
  if before.st_size>max_file: raise RuntimeError('Oversize artifact: '+p.name)
  with p.open('rb') as handle: data=handle.read(max_file+1)
  if len(data)>max_file: raise RuntimeError('Artifact grew beyond limit')
  after=p.stat(); total+=len(data)
  if total>max_total: raise RuntimeError('Snapshot exceeds byte limit')
  name=p.relative_to(root).as_posix()
  info=tarfile.TarInfo(name);info.size=len(data);info.mtime=int(after.st_mtime);info.mode=0o600
  archive.addfile(info,io.BytesIO(data))
  files.append({{'path':name,'bytes':len(data),'mtime_ns':after.st_mtime_ns,
                'changed_during_read':before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns}})
  if name=='queue.json': queue=json.loads(data)
 processes={{}}
 pids=[queue.get('pid')] if queue else []
 if queue: pids += [item.get('pid') for item in queue.get('jobs',{{}}).values()]
 for pid in pids:
  if not isinstance(pid,int) or pid<=0: continue
  base=pathlib.Path('/proc')/str(pid)
  try:
   state=(base/'stat').read_text().split(') ',1)[1].split()[0]
   command=(base/'cmdline').read_bytes().replace(b'\\x00',b' ').decode('utf-8','replace')
   processes[str(pid)]={{'exists':True,'state':state,'alive':state!='Z',
       'matches_sft_task':('/beacon_comem_20260909/' in command and ('train_sft.py' in command or 'remote_sft_queue.py' in command))}}
  except (FileNotFoundError,ProcessLookupError): processes[str(pid)]={{'exists':False,'alive':False}}
  except PermissionError: processes[str(pid)]={{'exists':True,'alive':None,'permission_denied':True}}
 manifest={{'schema_version':1,'source_host':socket.gethostname(),'source_root':str(root),
    'snapshot_started_utc':started,'snapshot_finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files':files,'total_file_bytes':total,'processes':processes,
    'checkpoints_copied':False,'models_loaded':False,'remote_mutations':False,
    'consistency':'each file captured independently; growing JSONL may have an incomplete trailing line'}}
 data=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
 info=tarfile.TarInfo('SYNC_MANIFEST.json');info.size=len(data);info.mode=0o600
 archive.addfile(info,io.BytesIO(data))
'''


def safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    seen, targets, total, contents = set(), set(), 0, []
    # Validate every header before creating even the first artifact file.
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not allowed_name(member.name) or member.name in seen:
                raise ValueError(f"Unsafe/unexpected archive member: {member.name!r}")
            if member.size < 0 or member.size > MAX_FILE:
                raise ValueError("Oversize archive member")
            seen.add(member.name); total += member.size
            if len(seen) > MAX_FILES+1 or total > MAX_TOTAL+MAX_FILE:
                raise ValueError("Oversize snapshot")
            target = destination.joinpath(*PurePosixPath(member.name).parts).resolve()
            target.relative_to(destination)
            canonical_target = str(target).casefold()
            if canonical_target in targets:
                raise ValueError("Duplicate extraction target")
            targets.add(canonical_target)
            contents.append((member, target))
        if "SYNC_MANIFEST.json" not in seen:
            raise ValueError("Missing snapshot manifest")
        for member, target in contents:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source:
                data = source.read(MAX_FILE+1)
            if len(data) != member.size:
                raise ValueError("Archive member length differs")
            target.write_bytes(data)
    manifest = json.loads((destination/"SYNC_MANIFEST.json").read_text(encoding="utf-8"))
    rows = manifest["files"]
    if len({r["path"] for r in rows}) != len(rows) or set(r["path"] for r in rows) != seen-{"SYNC_MANIFEST.json"}:
        raise ValueError("Manifest inventory differs")
    for row in rows:
        if not allowed_name(row["path"]) or (destination/row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("Manifest file size differs")
    if sum(r["bytes"] for r in rows) != manifest["total_file_bytes"]:
        raise ValueError("Manifest total differs")
    return manifest


def latest_snapshot(root=DEFAULT_OUT):
    root = Path(root).resolve()
    pointer = json.loads((root/"LATEST.json").read_text(encoding="utf-8"))
    relative = PurePosixPath(pointer["snapshot"])
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "snapshots" or not ARM.fullmatch(relative.parts[1]):
        raise ValueError("Unsafe snapshot pointer")
    target = root.joinpath(*relative.parts).resolve()
    target.relative_to(root)
    if not (target/"SYNC_MANIFEST.json").is_file():
        raise ValueError("Incomplete snapshot pointer")
    return target


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_snapshot(path):
    path = Path(path)
    manifest = read_json(path/"SYNC_MANIFEST.json")
    queue = read_json(path/"queue.json") if (path/"queue.json").exists() else {}
    arms = sorted(set(queue.get("jobs", {})) | {p.name for p in path.iterdir() if p.is_dir()})
    result = {"snapshot": str(path.resolve()), "source_host": manifest["source_host"],
        "snapshot_finished_utc": manifest["snapshot_finished_utc"], "queue_complete": queue.get("complete", False),
        "controller_pid": queue.get("pid"), "processes": manifest.get("processes", {}),
        "total_file_bytes": manifest["total_file_bytes"], "file_count": len(manifest["files"]),
        "checkpoints_copied": False, "arms": {}}
    for arm in arms:
        folder = path/arm
        state = read_json(folder/"status.json") if (folder/"status.json").exists() else {}
        job = queue.get("jobs", {}).get(arm, {})
        logs, issues, trailing = [], [], False
        if (folder/"train.jsonl").exists():
            data = (folder/"train.jsonl").read_bytes()
            trailing = bool(data and not data.endswith(b"\n"))
            for i, line in enumerate(data.splitlines()):
                if not line.strip(): continue
                try: logs.append(json.loads(line))
                except (ValueError, UnicodeDecodeError) as exc:
                    issues.append(f"train.jsonl line {i+1}: {type(exc).__name__}")
        evaluations = []
        for p in sorted(folder.glob("eval_step*.json")):
            match = re.fullmatch(r"eval_step([0-9]+)\.json", p.name)
            if not match: continue
            try:
                ev = read_json(p); rows = ev.get("records", []); summary = ev.get("summary", {})
                n, ce = summary.get("examples"), summary.get("answer_ce")
                ids = [row.get("id") for row in rows]
                valid = (isinstance(n, int) and n > 0 and n == len(rows) == len(set(ids))
                         and all(isinstance(i, str) for i in ids)
                         and isinstance(ce, (float, int)) and not isinstance(ce, bool) and math.isfinite(ce))
                evaluations.append({"step": int(match.group(1)), "examples": n, "records": len(rows),
                    "complete_by_counts": valid, "answer_ce": ce, "path": p.relative_to(path).as_posix()})
            except (ValueError, TypeError) as exc:
                issues.append(f"{p.name}: {type(exc).__name__}: {exc}")
        evaluations.sort(key=lambda item: item["step"])
        result["arms"][arm] = {"queue_phase": job.get("phase", "unknown"), "phase": state.get("phase", "not_started"),
            "step": state.get("step", 0), "latest_logged_step": max([r.get("step", 0) for r in logs], default=0),
            "target_steps": state.get("target_steps"), "complete": state.get("complete", False),
            "train_log_rows": len(logs), "incomplete_trailing_train_line": trailing,
            "pid": job.get("pid"), "gpu": job.get("gpu"), "evaluations": evaluations,
            "evaluated_examples_total_across_checkpoints": sum(e["records"] for e in evaluations if e["complete_by_counts"]),
            "issues": issues}
    return result


def sync(out=DEFAULT_OUT, host="longjing-1"):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:10]
    snapshots = out/"snapshots"; snapshots.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sync_", dir=out) as temporary:
        archive_path = Path(temporary)/"lightweight.tar.gz"
        with archive_path.open("wb") as target:
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host, "python3 -"],
                input=remote_script().encode("utf-8"), stdout=target, stderr=subprocess.PIPE, timeout=180)
        if proc.returncode:
            raise RuntimeError("Remote read-only export failed: "+proc.stderr.decode("utf-8", "replace")[-6000:])
        stage = Path(temporary)/"snapshot"
        manifest = safe_extract(archive_path, stage)
        summary = inspect_snapshot(stage)
        destination = snapshots/stamp
        stage.replace(destination)
    summary["snapshot"] = str(destination)
    (destination/"INSPECT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    pointer = {"schema_version": 1, "snapshot": "snapshots/"+stamp, "source_host": manifest["source_host"],
        "source_root": manifest["source_root"], "snapshot_finished_utc": manifest["snapshot_finished_utc"],
        "checkpoints_copied": False, "local_host": socket.gethostname()}
    temp_pointer = out/(".LATEST_"+stamp+".tmp")
    temp_pointer.write_text(json.dumps(pointer, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temp_pointer.replace(out/"LATEST.json")
    return summary


def self_test():
    def bundle(path, name, kind=tarfile.REGTYPE):
        with tarfile.open(path, "w:gz") as archive:
            m=tarfile.TarInfo(name); m.type=kind; m.linkname="../escape"; m.size=0
            archive.addfile(m, io.BytesIO())
    with tempfile.TemporaryDirectory(prefix="sft-sync-boundaries-") as folder:
        root=Path(folder)
        for i, (name, kind) in enumerate([("../escape",tarfile.REGTYPE),("/absolute",tarfile.REGTYPE),
                ("C:/escape",tarfile.REGTYPE),("beacon4/link",tarfile.SYMTYPE),
                ("beacon4/last.pt",tarfile.REGTYPE),("beacon4/../../escape",tarfile.REGTYPE),
                ("beacon4/./status.json",tarfile.REGTYPE),("beacon4//status.json",tarfile.REGTYPE)]):
            archive=root/f"bad{i}.tar.gz"; bundle(archive,name,kind)
            try: safe_extract(archive,root/f"out{i}")
            except ValueError: pass
            else: raise AssertionError(f"Accepted unsafe member {name}")
        if not allowed_name("beacon4/eval_step500.records.jsonl") or not allowed_name("beacon4/eval_step500.records.summary.json"):
            raise AssertionError("Rejected expected records")
    print("PASS: 8 unsafe archive cases rejected; expected evaluation record paths accepted")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--host",default="longjing-1")
    parser.add_argument("--inspect-only",action="store_true")
    parser.add_argument("--json",action="store_true",help="Print the complete lightweight inspection")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    result=inspect_snapshot(latest_snapshot(args.out)) if args.inspect_only else sync(args.out,args.host)
    if args.json:
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        print(f"{result['snapshot_finished_utc']} {result['source_host']}: {result['file_count']} files, {result['total_file_bytes']} bytes; no checkpoints")
        for arm,row in result["arms"].items():
            ev=", ".join(f"step{e['step']}:n{e['examples']}"+("" if e['complete_by_counts'] else " INVALID") for e in row["evaluations"]) or "none"
            print(f"{arm}: queue={row['queue_phase']} train={row['phase']} saved/logged={row['step']}/{row['latest_logged_step']} target={row['target_steps']} GPU={row['gpu']}; eval {ev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
