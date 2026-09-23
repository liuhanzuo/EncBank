"""One registered whole-trace timing replacement; never interrupt a GPU job.

This CPU bootstrap only launches the unchanged run_capacity.py. That runner
retains the original shared GPU lock, double strict <5 GiB admission and 2/16
thread settings. Partial traces are retained, never resumed at a query boundary.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from launch_reuse import HERE, ROOT, bootstrap_lock, now
from run_reuse import save as save_json
from capacity_timing_policy import REGISTRY, remeasurement, verify_replacement_config
from local_execution_pause import require_unpaused


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def processes():
    command = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
               "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | "
               "Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                            encoding="utf-8", errors="strict", check=True)
    value = json.loads(result.stdout) if result.stdout.strip() else []
    return value if isinstance(value, list) else [value]


def same_process(expected, observed):
    return any(p.get("ProcessId") == expected["ProcessId"] and p.get("CreationDate") == expected["CreationDate"]
               and p.get("CommandLine") == expected["CommandLine"] for p in observed)


def active_attempt_processes(attempt, observed):
    needle = str(Path(attempt).resolve()).replace("\\", "/").lower()
    return [p for p in observed if "run_capacity.py" in (p.get("CommandLine") or "").lower()
            and needle in (p.get("CommandLine") or "").replace("\\", "/").lower()]


def command_for(config, attempt, python):
    if not (config["dataset"] == "locomo" and config["cohort"] == "locomo-00"
            and config["arm"] == "fix_all" and config["fraction"] == .25
            and config["smoke"] is False and len(config["ids"]) == len(set(config["ids"])) == 800):
        raise ValueError("This finite repair only permits the original LoCoMo 25% V2 800-request trace")
    return [str(python), "-X", "utf8", "-u", str(HERE/"run_capacity.py"),
            "--dataset", config["dataset"], "--fixtures", config["fixtures"],
            "--cohort", config["cohort"], "--fraction", str(config["fraction"]),
            "--arm", config["arm"], "--model", config["model"], "--out", str(attempt)]


def validate_finished(attempt, original, job_id):
    # Only the LoCoMo fixture and this 800-row result are read. No SCBench data,
    # scorer, model imports or tensor operations belong in the CPU validator.
    import aggregate_capacity as agg
    cfg = read_json(attempt/"config.json")
    if cfg != original:
        raise ValueError("Fresh attempt must reproduce the entire original config without changes")
    docs, rows = agg.load_fixture("locomo", Path(cfg["fixtures"]))
    jobs = agg.expected_jobs("locomo", docs, rows, cfg["fixtures"])
    job = next(j for j in jobs if j["id"] == job_id)
    checked = agg.validate_attempt(attempt, job, cfg["model"])
    return {"status": "complete", "validated_at": now(), "job_id": job_id,
            "replacement_attempt": attempt.name, "completed_queries": len(checked["records"]),
            "all_original_config_fields_equal": True, "raw_input_ids_order_pack_caps_verified": True,
            "hardware_and_double_gate_verified": True, "cold_complete_trace": True,
            "old_outputs_added_to_formal_denominator": False,
            "raw_records": str(attempt/"measurements.jsonl"), "hardware": read_json(attempt/"COMPLETED.json")["hardware"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE/"results/capacity_remeasure/plan.json")
    args = parser.parse_args(argv)
    require_unpaused(HERE)
    plan = read_json(args.plan)
    folder = args.plan.resolve().parent
    target = (HERE/"results/capacity"/plan["job_id"]).resolve()
    if plan["job_id"] != "locomo/full/locomo-00/025/fix_all":
        raise ValueError("Out-of-scope repair job")
    original_path = target/"attempts/0001"
    config = read_json(original_path/"config.json")
    registry = remeasurement(target)
    if registry is None or registry["replacement_attempt"] != plan["replacement_attempt"]:
        raise ValueError("Missing/mismatched explicit replacement registration")
    attempt = target/"attempts"/registry["replacement_attempt"]
    verify_replacement_config(attempt, config)
    command = command_for(config, attempt, plan["python"])
    with bootstrap_lock(folder):
        previous = read_json(folder/"status.json") if (folder/"status.json").exists() else {}
        state = {"status": "checking_dependencies", "pid": os.getpid(), "started_at": now(),
                 "job_id": plan["job_id"], "original_attempt": str(original_path), "replacement_attempt": str(attempt),
                 "expected_queries": 800, "completed_queries": 0, "child_pid": None,
                 "historical_child_pids": previous.get("historical_child_pids", []),
                 "gpu_policy": "unchanged run_capacity / gpu_gate: RTX5090, double strictly <5 GiB, no other model, CPU2/interop16"}
        def update(**values):
            state.update(values, updated_at=now())
            save_json(folder/"status.json", state)
            current = remeasurement(target)
            current.update(status=state["status"], updated_at=state["updated_at"], bootstrap_pid=os.getpid(),
                           child_pid=state.get("child_pid"), completed_queries=state["completed_queries"])
            save_json(target/REGISTRY, current)
        try:
            while True:
                require_unpaused(HERE)
                observed = processes()
                waiting = [p for p in plan["wait_for_processes"] if same_process(p, observed)]
                if not waiting:
                    break
                update(status="waiting_for_current_unit_natural_exit", waiting_pids=[p["ProcessId"] for p in waiting])
                time.sleep(30)
            update(waiting_pids=[], dependency_processes_naturally_exited=True)
            dependency = read_json(Path(plan["wait_for_attempt"])/"COMPLETED.json")
            if not (dependency.get("status") == "complete" and dependency.get("completed_queries") == 800):
                raise ValueError("The preserved prior unit exited without a full completion receipt")
            # Covers a bootstrap crash after Popen but before child_pid publication.
            live = active_attempt_processes(attempt, processes())
            child = None
            handles = []
            if not live and not (attempt/"COMPLETED.json").exists():
                if attempt.exists() and any(attempt.iterdir()):
                    raise ValueError("Partial replacement retained; explicitly register a new whole-trace attempt")
                attempt.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1")
                out = (attempt/"stdout.log").open("w", encoding="utf-8")
                err = (attempt/"stderr.log").open("w", encoding="utf-8")
                handles = [out, err]
                require_unpaused(HERE)
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                state["historical_child_pids"].append(child.pid)
                update(status="child_started_with_original_gate", child_pid=child.pid, command=command)
            try:
                while True:
                    live = active_attempt_processes(attempt, processes())
                    if child is not None and child.poll() is not None and not live:
                        if child.returncode:
                            raise RuntimeError(f"Fresh timing attempt exited {child.returncode}; evidence retained")
                        break
                    if child is None and not live:
                        break
                    snapshot = read_json(attempt/"status.json") if (attempt/"status.json").exists() else {}
                    update(status=snapshot.get("status", "waiting_for_replacement_child"),
                           completed_queries=snapshot.get("completed_queries", 0),
                           live_child_pids=[p["ProcessId"] for p in live])
                    time.sleep(30)
            finally:
                for handle in handles:
                    handle.close()
            update(status="validating_complete_trace", child_pid=None, live_child_pids=[])
            receipt = validate_finished(attempt, config, plan["job_id"])
            receipt["all_replacement_processes_naturally_exited"] = True
            save_json(attempt/"REMEASUREMENT_VALIDATED.json", receipt)
            update(status="complete", completed_queries=800, child_pid=None, finished_at=now(),
                   validation_receipt=str(attempt/"REMEASUREMENT_VALIDATED.json"))
            return 0
        except BaseException as exc:
            update(status="failed", error=repr(exc), finished_at=now())
            raise


if __name__ == "__main__":
    raise SystemExit(main())
