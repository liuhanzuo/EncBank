"""CPU-only reproduction of Windows venv redirector ancestry (no GPU imports)."""
import json
from pathlib import Path
import subprocess
import sys

import psutil


def identity(process):
    return {"pid": process.pid, "name": process.name(), "exe": process.exe(),
            "created": process.create_time()}


if "--child" in sys.argv:
    current = psutil.Process()
    print(json.dumps({"worker": identity(current),
                      "ancestors": [identity(p) for p in current.parents()[:3]]}))
else:
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error = process.communicate(timeout=15)
    if process.returncode:
        raise RuntimeError(error)
    record = {"supervisor": identity(psutil.Process()), "popen_pid": process.pid,
              "child_exit_code": process.returncode, "observed": json.loads(output)}
    print(json.dumps(record, indent=2))
