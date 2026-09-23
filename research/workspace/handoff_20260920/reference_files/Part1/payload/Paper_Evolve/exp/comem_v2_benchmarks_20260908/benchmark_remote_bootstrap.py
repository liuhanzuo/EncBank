"""Wait for model transfer, run isolated smoke jobs, then the complete benchmark queue."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path, payload):
    payload["updated_at"] = time.time()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--smoke-plan", type=Path, required=True)
    parser.add_argument("--full-plan", type=Path, required=True)
    parser.add_argument("--gpus", default="2")
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--wait-training-status", type=Path)
    parser.add_argument("--after-bootstrap-status", type=Path)
    args = parser.parse_args()
    args.status_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.status_dir / "bootstrap.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.status_dir / "status.json"
    state = {"pid": os.getpid(), "gpus": args.gpus, "started_at": time.time(),
             "status": "waiting_for_model", "full_plan": str(args.full_plan)}
    save(status_path, state)
    plan = json.loads(args.full_plan.read_text())
    marker = Path(plan["model"]) / "TRANSFER_COMPLETE.json"
    print(f"Waiting for {marker}", flush=True)
    while not marker.exists():
        time.sleep(30)
        save(status_path, state)
    # File-size readiness is about incomplete network transfers, not paper review.
    ready = json.loads(marker.read_text())
    state["model_transfer_marker"] = str(marker)
    state["model_transfer_ready"] = bool(ready)
    if args.wait_training_status:
        state["status"] = "waiting_for_training"
        while True:
            trained = json.loads(args.wait_training_status.read_text()) if args.wait_training_status.exists() else {}
            if trained.get("complete") and trained.get("step") == 4000:
                break
            save(status_path, state)
            time.sleep(30)
    if args.after_bootstrap_status:
        state["status"] = "waiting_for_previous_queue"
        while True:
            previous = json.loads(args.after_bootstrap_status.read_text()) if args.after_bootstrap_status.exists() else {}
            if previous.get("status") == "completed":
                break
            if previous.get("status") == "failed":
                state.update(status="failed", reason="Previous queue failed; inspect before proceeding")
                save(status_path, state)
                return 1
            save(status_path, state)
            time.sleep(30)
    for phase, path in (("smoke", args.smoke_plan), ("full", args.full_plan)):
        state.update(status="running", phase=phase)
        save(status_path, state)
        command = [sys.executable, str(HERE / "remote_queue.py"), "--plan", str(path), "--gpus", args.gpus]
        print(f"Starting {phase}: {command}", flush=True)
        with (args.status_dir / f"{phase}.log").open("a") as log:
            rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        phase_plan = json.loads(path.read_text())
        summary_path = args.status_dir / f"{phase}_summary.json"
        summary_rc = subprocess.call([sys.executable, str(HERE / "summarize_runs.py"),
            "--plan", str(path), "--out", str(summary_path)])
        summary = json.loads(summary_path.read_text()) if summary_rc == 0 else {}
        # Full reused RULER rows live locally and are merged by the local reporter.
        good = rc == 0 and summary_rc == 0 and summary.get("completed_jobs") == len(phase_plan["jobs"])
        state.update(last_exit_code=rc, summary=str(summary_path))
        if not good:
            state.update(status="failed", reason=f"{phase} queue/results incomplete")
            save(status_path, state)
            return 1
    state.update(status="completed", finished_at=time.time())
    save(status_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
