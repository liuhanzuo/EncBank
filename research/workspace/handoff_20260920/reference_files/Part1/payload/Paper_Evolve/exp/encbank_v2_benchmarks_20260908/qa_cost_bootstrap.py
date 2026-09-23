"""Serial QA cost queue after the existing serving campaign; CPU-only scheduler.

Each measured child owns the original gpu_gate. This scheduler never acquires a
GPU, changes the healthy serving scheduler, or starts remote timing. Failed or
partial attempts remain on disk; the runner reuses validated completed rows.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from qa_cost_runner import HERE, ROOT, VERSION, THREAD_ENV, read_json, save_json, require, load_job, reusable_rows, summarize
from serving_local_bootstrap import bootstrap_lock


def serving_dependency_ready(state, processes):
    """A completed status alone is insufficient while its measured child is alive."""
    if state.get("status") != "completed" or state.get("requested_campaign") != "all":
        return False
    full = [j for key, j in state.get("jobs", {}).items() if "/full/" in key]
    if len(full) != 10 or sum(j.get("cells", 0) for j in full if j.get("status") == "complete") != 120:
        return False
    # Check actual remaining measured children, including Windows venv launcher descendants.
    for p in processes:
        command = (p.get("CommandLine") or "").lower().replace("\\", "/")
        if "python" in (p.get("Name") or "").lower() and "serving_reuse.py" in command and "encbank_v2_benchmarks_20260908" in command:
            return False
    return True


def process_snapshot():
    command = "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout) if result.stdout.strip() else []
    return [value] if isinstance(value, dict) else value


def completed_attempt(folder, config, inputs):
    for attempt in sorted((folder / "attempts").glob("[0-9]*"), reverse=True):
        try:
            marker = read_json(attempt / "COMPLETED.json")
            if marker.get("status") != "complete" or marker.get("protocol") != VERSION or read_json(attempt / "config.json") != config:
                continue
            # Inspect this attempt, not another completed sibling's rows.
            from qa_cost_runner import validate_result
            rows = [json.loads(s) for s in (attempt / "measurements.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
            require(len(rows) == len(config["indices"])*config["repetitions"], "Incomplete row count")
            keys = {(r["index"], r["repetition"]) for r in rows}
            require(keys == {(i, rep) for i in config["indices"] for rep in range(config["repetitions"])}, "Repeated/missing row")
            for row in rows:
                validate_result(row, inputs[row["index"]], config)
            require(summarize(rows, config) == read_json(attempt / "summary.json"), "Summary differs from actual rows")
            return attempt
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, default=HERE / "results/protocol/qa_cost/task_plan.json")
    parser.add_argument("--state-dir", type=Path, default=HERE / "results/local/bootstrap_qa_cost")
    parser.add_argument("--result-root", type=Path, default=HERE / "results/local/qa_cost")
    parser.add_argument("--serving-status", type=Path, default=HERE / "results/local/bootstrap_serving/status.json")
    args = parser.parse_args(argv)
    require(platform.system() == "Windows", "Only the local Windows timing queue is allowed")
    plan = read_json(args.plan)
    require(len(plan["jobs"]) == 8 and sum(len(j["indices"])*j["timed_repetitions"] for j in plan["jobs"] if j["phase"] == "full") == 2400, "Full QA scope changed")
    with bootstrap_lock(args.state_dir):
        status_path = args.state_dir / "status.json"
        state = read_json(status_path) if status_path.exists() else {}
        state.update(pid=os.getpid(), protocol=VERSION, required_gpu="NVIDIA GeForce RTX 5090",
            remote_timing_allowed=False, result_root=str(args.result_root), plan=str(args.plan),
            strict_gpu_admission_gib=5, jobs=state.get("jobs", {}), error=None)
        def update(**values):
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            save_json(status_path, state)
        try:
            # Lightweight CPU contract checks only; no model import or CUDA initialization.
            for job in plan["jobs"]:
                load_job(args.plan, job["id"])
            verified = read_json(HERE / "results/protocol/qa_cost/CPU_VERIFIED.json")
            require(verified.get("passed") is True and verified.get("protocol") == VERSION, "CPU correctness checks must pass first")
            if not args.run:
                update(status="ready", gpu_experiment_started=False,
                    dependency_ready=serving_dependency_ready(read_json(args.serving_status), process_snapshot()))
                return 0
            while not serving_dependency_ready(read_json(args.serving_status), process_snapshot()):
                update(status="waiting_for_prior_serving_completion", child_pid=None, gpu_experiment_started=False)
                time.sleep(30)
            update(status="running", dependency_verified=True, dependency_verified_at=datetime.now(timezone.utc).isoformat())
            for planned in plan["jobs"]:
                job, inputs, config = load_job(args.plan, planned["id"])
                folder = args.result_root / job["phase"] / job["task"] / job["arm"]
                attempt = completed_attempt(folder, config, inputs)
                if attempt is None:
                    attempts = folder / "attempts"
                    attempts.mkdir(parents=True, exist_ok=True)
                    number = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()] + [0]) + 1
                    attempt = attempts / f"{number:04d}"
                    attempt.mkdir()
                    command = [sys.executable, "-X", "utf8", "-u", str(HERE / "qa_cost_runner.py"),
                        "--plan", str(args.plan.resolve()), "--job", job["id"], "--out", str(attempt.resolve())]
                    log = args.state_dir / "logs" / f"{job['phase']}_{job['task']}_{job['arm']}_{number:04d}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    update(status="waiting_for_gpu_or_running", active_job=job["id"], active_attempt=str(attempt),
                        active_log=str(log), command=command, gpu_experiment_started=False)
                    with log.open("w", encoding="utf-8") as stream:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            env=dict(os.environ, **THREAD_ENV, CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1"),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        while child.poll() is None:
                            update(child_pid=child.pid)
                            time.sleep(15)
                    require(child.returncode == 0 and completed_attempt(folder, config, inputs) == attempt,
                        f"QA job failed/incomplete: {job['id']}; retained {attempt}; inspect {log}")
                marker = read_json(attempt / "COMPLETED.json")
                state["jobs"][job["id"]] = {"status": "complete", "attempt": str(attempt),
                    "timed_generations": marker["expected_repetitions"], "unique_method_examples": marker["unique_method_examples"],
                    "new_timed_generations": marker["new_timed_generations"], "reused_repetitions": marker["reused_repetitions"],
                    "extra_smoke_reference_generations": marker["extra_smoke_reference_generations"], "warmup_generations": marker["warmup_generations"]}
                update(status="running", child_pid=None)
            update(status="completed", active_job=None, child_pid=None,
                full_timed_generations=2400, full_unique_method_examples=800, smoke_timed_generations=8,
                expected_additional_smoke_reference_generations=8, full_warmups=4)
        except BaseException as exc:
            update(status="failed", error=repr(exc))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
