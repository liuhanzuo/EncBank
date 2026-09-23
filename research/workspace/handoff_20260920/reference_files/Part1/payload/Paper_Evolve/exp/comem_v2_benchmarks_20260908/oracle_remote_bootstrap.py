"""Run an independent support-evidence queue after matched RULER exits naturally.

This CPU coordinator never stops another process and never changes the main queue.
GPU admission remains in remote_queue.py (512 MiB, <=5% use, no compute PID).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save(path, payload):
    payload["updated_at"] = time.time()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def process_alive(pid):
    """Zombies have exited; a stale/reused live PID conservatively delays startup."""
    if not pid:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def predecessor_ready(status, summary, queue, alive=process_alive):
    if status.get("status") == "failed":
        raise RuntimeError("Matched RULER failed; inspect before oracle continuation")
    if status.get("status") != "completed":
        return False, "waiting_for_matched_ruler_completion"
    if (summary.get("planned_jobs") != 6 or summary.get("completed_jobs") != 6
            or summary.get("observed_predictions") != 300
            or summary.get("all_complete") is not True):
        raise RuntimeError("Matched RULER completion lacks its six-job/300-record summary")
    jobs = queue.get("jobs", {})
    if len(jobs) != 6 or any(v.get("status") != "completed" for v in jobs.values()):
        raise RuntimeError("Matched RULER full queue does not contain six completed jobs")
    if any(v.get("exit_code") != 0 for v in jobs.values()):
        raise RuntimeError("Matched RULER full queue contains a nonzero or absent exit code")
    pids = [status.get("pid"), queue.get("queue_pid")]
    pids.extend(v.get("pid") for v in jobs.values())
    if any(alive(pid) for pid in pids if pid):
        return False, "waiting_for_matched_ruler_process_exit"
    return True, "matched_ruler_complete_and_exited"


def validate_plans(smoke, full, main_plan):
    if smoke.get("mode") != "smoke" or full.get("mode") != "full":
        raise ValueError("Distinct smoke and full plans are required")
    if not smoke.get("jobs") or not full.get("jobs"):
        raise ValueError("Cannot launch an empty oracle plan")
    if smoke["state_dir"] == full["state_dir"]:
        raise ValueError("Smoke and full queue state must be separate")
    for plan in (smoke, full):
        if plan["state_dir"] == main_plan["state_dir"]:
            raise ValueError("Oracle cannot reuse the active main queue state")
        ids = [j["id"] for j in plan["jobs"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Oracle job IDs are not unique")
        outputs = {j["output"] for j in plan["jobs"]}
        if len(outputs) != len(ids):
            raise ValueError("Oracle jobs share an output path")
        if outputs & {j["output"] for j in main_plan["jobs"]}:
            raise ValueError("Oracle would overwrite existing main outputs")
        if set(ids) & {j["id"] for j in main_plan["jobs"]}:
            raise ValueError("Oracle IDs overlap existing main jobs")
        if plan["model"] != main_plan["model"]:
            raise ValueError("Oracle model must match the approved Qwen3-8B model")
    if {j["output"] for j in smoke["jobs"]} & {j["output"] for j in full["jobs"]}:
        raise ValueError("Smoke predictions must not enter full outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--smoke-plan", required=True, type=Path)
    parser.add_argument("--full-plan", required=True, type=Path)
    parser.add_argument("--main-plan", type=Path, default=HERE / "full_plan.json")
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--after-bootstrap-status", required=True, type=Path)
    parser.add_argument("--after-full-summary", required=True, type=Path)
    parser.add_argument("--after-queue-state", required=True, type=Path)
    parser.add_argument("--gpus", default="1", choices=["1"])
    args = parser.parse_args()
    smoke, full = read_json(args.smoke_plan), read_json(args.full_plan)
    validate_plans(smoke, full, read_json(args.main_plan))
    args.status_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.status_dir / "bootstrap.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.status_dir / "status.json"
    previous = read_json(status_path)
    if previous.get("status") == "completed":
        print("Oracle queue already completed; preserving all results", flush=True)
        return 0
    if previous:
        if any(process_alive(previous.get(k)) for k in ("pid", "queue_pid")):
            raise RuntimeError("Previous oracle coordinator remains alive; refusing duplication")
        history = args.status_dir / "history"
        history.mkdir(exist_ok=True)
        (history / f"status_before_resume_{time.time_ns()}.json").write_text(
            json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    state = {"pid": os.getpid(), "gpus": args.gpus, "started_at": time.time(),
             "status": "waiting_for_previous_queue", "full_plan": str(args.full_plan),
             "after_bootstrap_status": str(args.after_bootstrap_status),
             "admission": {"max_idle_mib": 512, "max_utilization_percent": 5,
                           "require_no_compute_pid": True}, "purpose": "accuracy_only"}
    save(status_path, state)
    try:
        while True:
            ready, reason = predecessor_ready(read_json(args.after_bootstrap_status),
                read_json(args.after_full_summary), read_json(args.after_queue_state))
            state["reason"] = reason
            save(status_path, state)
            if ready:
                break
            time.sleep(30)
        marker = Path(full["model"]) / "TRANSFER_COMPLETE.json"
        state["status"] = "waiting_for_model"
        while not marker.exists():
            save(status_path, state)
            time.sleep(30)
        if not read_json(marker):
            raise RuntimeError("Model transfer marker is empty")
        state["predecessor_verified_at"] = time.time()
        state["model_transfer_marker"] = str(marker)
        for phase, path, plan in (("smoke", args.smoke_plan, smoke),
                                  ("full", args.full_plan, full)):
            state.update(status="running", phase=phase)
            save(status_path, state)
            command = [sys.executable, "-u", str(HERE / "remote_queue.py"),
                       "--plan", str(path), "--gpus", args.gpus,
                       "--max-idle-mib", "512"]
            print(f"Starting {phase}: {command}", flush=True)
            with (args.status_dir / f"{phase}.log").open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["queue_pid"] = process.pid
                save(status_path, state)
                rc = process.wait()
            summary_path = args.status_dir / f"{phase}_summary.json"
            summary_rc = subprocess.call([sys.executable, str(HERE / "summarize_runs.py"),
                "--plan", str(path), "--out", str(summary_path)])
            summary = read_json(summary_path) if summary_rc == 0 else {}
            state.update(last_exit_code=rc, summary=str(summary_path))
            expected_n = sum(j["expected_n"] for j in plan["jobs"])
            if (rc != 0 or summary_rc != 0 or summary.get("all_complete") is not True
                    or summary.get("completed_jobs") != len(plan["jobs"])
                    or summary.get("observed_predictions") != expected_n):
                raise RuntimeError(f"Oracle {phase} queue/results incomplete; inspect retained attempts")
        state.update(status="completed", finished_at=time.time(), reason="all_oracle_jobs_verified")
        save(status_path, state)
        return 0
    except Exception as exc:
        state.update(status="failed", reason=str(exc), finished_at=time.time())
        save(status_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
