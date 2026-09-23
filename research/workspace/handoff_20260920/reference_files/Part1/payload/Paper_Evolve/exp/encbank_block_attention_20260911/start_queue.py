"""Start the authorized waiting controller after CPU semantics have passed."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/liuhanzuo/encbank_v2_20260908")
OUT = ROOT / "outputs/sparse_encbank_20260911"

def main():
    if sys.platform != "linux":
        raise RuntimeError("Launch the queue on the remote host only.")
    from remote_sparse_queue import read_json, process_identity
    receipt = read_json(HERE / "results/cpu_checks.json")
    reader_hash = hashlib.sha256((HERE / "sparse_reader.py").read_bytes()).hexdigest()
    if receipt.get("passed") is not True or receipt.get("reader_sha256") != reader_hash:
        raise RuntimeError("Current reader must pass validate_cpu.py before queuing GPU work.")
    data = HERE / "data/qasper_pilot"
    if not (data / "preparation.json").is_file():
        raise RuntimeError("Prepare the shared pilot data first.")
    OUT.mkdir(exist_ok=True)
    previous = read_json(OUT / "queue.json")
    current = process_identity(previous.get("pid")) if isinstance(previous.get("pid"), int) else None
    if current and current == previous.get("controller"):
        print(json.dumps({"event":"controller_already_alive","controller":current}))
        return
    command = [sys.executable, "-u", str(HERE / "remote_sparse_queue.py"),
               "--data-dir", str(data), "--out", str(OUT), "--allow-training"]
    env = {**os.environ, "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2",
           "OPENBLAS_NUM_THREADS":"2", "TOKENIZERS_PARALLELISM":"false"}
    with (OUT / "controller.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT / "workspace", env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = {"pid":process.pid,"command":command,
             "started_utc":datetime.now(timezone.utc).isoformat(),
             "log":str(OUT / "controller.log"), "cpu_checks":str(HERE / "results/cpu_checks.json")}
    (OUT / "launch.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(value), flush=True)

if __name__ == "__main__":
    main()

