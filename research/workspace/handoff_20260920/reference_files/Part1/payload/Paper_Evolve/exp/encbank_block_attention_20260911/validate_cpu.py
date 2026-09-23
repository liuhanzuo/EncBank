"""Run all required semantic checks remotely with CUDA hidden, and record outcome."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def main():
    if sys.platform != "linux":
        raise RuntimeError("Run CPU/Torch checks on the remote Linux host.")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "TOKENIZERS_PARALLELISM": "false"}
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", str(HERE),
         "-p", "test_sparse_reader.py", "-v"],
        [sys.executable, str(HERE / "remote_sparse_queue.py"), "--self-test"],
    ]
    outcomes = []
    for i, command in enumerate(commands):
        proc = subprocess.run(command, cwd=HERE, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=600)
        log = out / ("cpu_reader.log" if i == 0 else "cpu_queue.log")
        log.write_text(proc.stdout, encoding="utf-8")
        print(proc.stdout, flush=True)
        outcomes.append({"command": command, "returncode": proc.returncode, "log": str(log)})
    receipt = {"passed": all(x["returncode"] == 0 for x in outcomes), "checks": outcomes,
               "updated_utc": datetime.now(timezone.utc).isoformat(),
               "cuda_visible_devices": "", "kind": "tiny-real-Qwen3-CPU-semantics-not-quality",
               "reader_sha256": hashlib.sha256((HERE / "sparse_reader.py").read_bytes()).hexdigest()}
    target = out / "cpu_checks.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    tmp.replace(target)
    if not receipt["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

