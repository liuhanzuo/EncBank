"""Four guarded diagnostic profiles, separate from all infrastructure timings."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "exp/comem_v2_benchmarks_20260908"))
from serving_local_bootstrap import bootstrap_lock
from infra_protocol import now, read_json, save_json
from infra_processes import identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", type=Path, default=ROOT / ".venv/Scripts/python.exe")
    ap.add_argument("--out", type=Path, default=HERE / "results/reader_profiles_20260912")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--skip-native", action="store_true", help="One bounded retry of D0/B after retaining an already complete native diagnostic")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commands = {}
    for variant, arms, modes in (("reference", ["D0"] if args.skip_native else ["NATIVE", "D0"], ["cold_hj"]),
                                 ("decode_v2", ["B"], ["cold_hj", "block_hot"])):
        commands[variant] = [str(args.python), "-X", "utf8", "-u", str(HERE / "launch_sparse_infra.py"),
            "--lengths", "4096", "--prompt-tokens", "64", "--generation-tokens", "4",
            "--arms", *arms, "--modes", *modes, "--repetitions", "1",
            "--reader-implementation", variant, "--profile-reader-only",
            "--out", str(args.out / variant)] + (["--run"] if args.run else [])
    if not args.run:
        save_json(args.out / "profile_plan.json", {"created_utc": now(), "commands": commands,
            "cells": 3 if args.skip_native else 4, "scope": "Actual operator/backend/allocator diagnosis; never eligible formal timing"})
        return 0
    with bootstrap_lock(args.out / "pipeline"):
        state = {**identity(os.getpid()), "status": "running", "started_utc": now(),
                 "commands": commands, "completed_variants": [], "timing_eligible": False}
        def update(**changes):
            state.update(changes, updated_utc=now())
            save_json(args.out / "pipeline_status.json", state)
        for variant, command in commands.items():
            if (HERE / "LOCAL_INFRA_PAUSED.json").exists():
                update(status="paused")
                return 0
            with (args.out / f"{variant}.stdout.log").open("a", encoding="utf-8") as out, \
                 (args.out / f"{variant}.stderr.log").open("a", encoding="utf-8") as err:
                child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                update(active_variant=variant, child=identity(child.pid))
                code = child.wait()
            status = read_json(args.out / variant / "status.json", {})
            if code or status.get("status") != "complete" or not status.get("jobs") or any(
                    j.get("status") != "complete" for j in status["jobs"].values()):
                update(status="attention_required", exit_code=code, child=None,
                       reason="Profile failure retained; inspect original shape and trace, no automatic retry")
                return 1
            state["completed_variants"].append(variant)
            update(child=None)
        update(status="complete", active_variant=None, completed_utc=now())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
