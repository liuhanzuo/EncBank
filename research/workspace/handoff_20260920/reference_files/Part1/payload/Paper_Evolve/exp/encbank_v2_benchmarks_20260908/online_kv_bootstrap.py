"""Queue direct online KV memory diagnostics after all local timed campaigns."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from qa_cost_runner import read_json, save_json, require
from serving_local_bootstrap import bootstrap_lock
from qa_cost_bootstrap import process_snapshot
from cacheblend_serving_bootstrap import dependencies_ready as prior_ready, verify_dependency_markers
from online_kv_diagnostic import HERE, ROOT, PROTOCOL, ARMS, THREAD_ENV, completed_attempt


def dependencies_ready():
    serving = read_json(HERE / "results/local/bootstrap_serving/status.json")
    qa = read_json(HERE / "results/local/bootstrap_qa_cost/status.json")
    cb = read_json(HERE / "results/local/bootstrap_cacheblend_serving/status.json")
    processes = process_snapshot()
    if not prior_ready(serving, qa, processes):
        return False
    if any("online_kv_diagnostic.py" in (p.get("CommandLine") or "") for p in processes):
        return False
    if cb.get("status") != "completed" or cb.get("completed_full_cells") != 24 or cb.get("completed_full_query_records") != 800:
        return False
    verify_dependency_markers(serving, qa)
    for arm_row in cb["jobs"].values():
        if arm_row["phase"] == "full":
            marker = read_json(Path(arm_row["result"]) / "COMPLETED.json")
            require(marker["status"] == "complete" and marker["cells"] == 12, "CB full marker missing")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=HERE / "results/local/bootstrap_online_kv")
    args = parser.parse_args(argv)
    plan = read_json(HERE / "online_kv_plan.json")
    require(plan["protocol"] == PROTOCOL and tuple(plan["methods"]) == ARMS and plan["planned_generations"] == 48 and plan["timing_eligible"] is False, "Diagnostic plan scope changed")
    with bootstrap_lock(args.state_dir):
        status = args.state_dir / "status.json"
        state = read_json(status) if status.exists() else {}
        state.update(pid=os.getpid(), protocol=PROTOCOL, required_gpu="NVIDIA GeForce RTX 5090",
            remote_timing_allowed=False, timing_eligible=False, target_diagnostic_generations=48,
            target_observed_cache_boundaries=96, jobs=state.get("jobs", {}), error=None,
            result_root=str(HERE / "results/local/online_kv"), strict_gpu_admission_gib=5.0)
        def update(**values):
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            state["completed_diagnostic_generations"] = sum(v["diagnostic_generations"] for v in state["jobs"].values() if v["status"] == "complete")
            state["completed_observed_cache_boundaries"] = sum(v["observed_cache_boundaries"] for v in state["jobs"].values() if v["status"] == "complete")
            save_json(status, state)
        try:
            while not dependencies_ready():
                update(status="waiting_for_all_local_timing_completion", child_pid=None, active_job=None)
                if not args.run:
                    return 0
                time.sleep(30)
            if not args.run:
                update(status="ready", child_pid=None, active_job=None)
                return 0
            update(status="running", dependencies_verified=True)
            for arm in ARMS:
                folder = HERE / "results/local/online_kv" / arm
                attempt = completed_attempt(folder, arm)
                if attempt is None:
                    attempts = folder / "attempts"
                    attempts.mkdir(parents=True, exist_ok=True)
                    number = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()] + [0])+1
                    attempt = attempts / f"{number:04d}"
                    attempt.mkdir()
                    command = [sys.executable, "-X", "utf8", "-u", str(HERE / "online_kv_diagnostic.py"), "--arm", arm, "--out", str(attempt.resolve())]
                    log = args.state_dir / "logs" / f"{arm}_{number:04d}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    update(status="waiting_for_gpu_or_running", active_job=arm, active_attempt=str(attempt), active_log=str(log), command=command)
                    with log.open("w", encoding="utf-8") as output:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                            env=dict(os.environ, **THREAD_ENV, CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1"),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        while child.poll() is None:
                            update(child_pid=child.pid)
                            time.sleep(15)
                    require(child.returncode == 0 and completed_attempt(folder, arm) == attempt, f"Incomplete/failed diagnostic {arm}; retained {attempt}; inspect {log}")
                marker = read_json(attempt / "COMPLETED.json")
                state["jobs"][arm] = {"status": "complete", "result": str(attempt), "diagnostic_generations": 8,
                    "observed_cache_boundaries": 16, "timing_eligible": False,
                    "historical_output_disagreement_cases": marker["historical_output_disagreement_cases"]}
                update(status="running", child_pid=None)
            update(status="completed", child_pid=None, active_job=None, error=None)
        except BaseException as exc:
            update(status="failed", error=repr(exc))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
