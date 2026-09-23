"""Driver: run s17_reselect.py, then evaluate its consensus sets on RULER with s15.
Each child acquires the GPU gate itself; this driver holds nothing.  Launch detached."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "exp" / "results" / "s17_run.log"


def run(cmd):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"=== {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)


rc = run([PY, "exp/s17_reselect.py", "--cap-gb", "28"])
sets = json.loads((ROOT / "exp/results/s17_sets.json").read_text()) if rc == 0 else None
if sets:
    for name in ("S3", "S6"):
        layers = ",".join(str(l) for l in sets[name])
        run([PY, "exp/s15_ruler_lower.py", "--tasks", "niah_multikey_1,variable_tracking",
             "--lengths", "16k,32k", "--n", "50", "--arms", "fix_S", "--fix-layers", layers,
             "--cap-gb", "30", "--check", "0", "--out", f"exp/results/s17_ruler_{name}.json"])
with open(LOG, "a", encoding="utf-8") as f:
    f.write("=== ALL DONE\n")
