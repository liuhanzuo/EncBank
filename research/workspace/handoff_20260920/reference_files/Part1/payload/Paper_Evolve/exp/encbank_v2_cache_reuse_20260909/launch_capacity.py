"""Sequential Phase B queue; full runs wait for the LoCoMo Phase A queue."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from launch_reuse import HERE, ROOT, ARMS, bootstrap_lock, read_json, now
from run_capacity import CAPACITY_VERSION, cohorts, request_trace, reference_working_set
from run_reuse import load_bundle, save as save_json
from capacity_timing_policy import exclusion_reason, remeasurement, verify_replacement_config
from local_execution_pause import require_unpaused


def make_jobs(dataset, documents, queries, smoke_only=False):
    groups = cohorts(dataset, documents)
    smoke_groups = [next(iter(groups))] if dataset == "locomo" else [next(k for k in groups if k.startswith(t)) for t in ("scbench_qa_eng", "scbench_choice_eng")]
    rng = random.Random(20260909)
    jobs = []
    for smoke, keys in ((True, smoke_groups), (False, list(groups))):
        if smoke_only and not smoke:
            continue
        for key in keys:
            docs, rows = request_trace(groups[key], queries, dataset, smoke)
            working = reference_working_set(docs, rows, 147456)
            if working["full_depth_working_set_bytes"] > 32*1024**3:
                raise ValueError(f"Cohort {key} exceeds 32 GiB; must revise explicit protocol, not silently shrink inputs")
            for fraction in ((.25,) if smoke else (.25, .5, 1.0)):
                arms = list(ARMS)
                rng.shuffle(arms)
                for arm in arms:
                    if not smoke and arm == "j0" and fraction != 1.0:
                        continue
                    jobs.append({"id": f"{dataset}/{'smoke' if smoke else 'full'}/{key}/{int(fraction*100):03d}/{arm}",
                        "dataset": dataset, "cohort": key, "fraction": fraction, "arm": arm, "smoke": smoke,
                        "queries": len(rows), "ids": [r["id"] for r in rows],
                        "budget_bytes_expected": int(working["full_depth_working_set_bytes"]*fraction),
                        "j0_comparison": "one measurement shared across the three capacity columns; not three independent runs"})
    return jobs


def valid_complete(folder, job, model):
    for attempt in sorted((folder/"attempts").glob("*"), reverse=True):
        if exclusion_reason(attempt):
            continue
        if not (attempt/"COMPLETED.json").exists():
            continue
        state, cfg = read_json(attempt/"COMPLETED.json"), read_json(attempt/"config.json")
        try:
            verify_replacement_config(attempt, cfg)
        except ValueError:
            continue
        hw = state.get("hardware", {})
        gate = hw.get("gpu_admission", {})
        if (state.get("status") == "complete" and cfg.get("protocol") == CAPACITY_VERSION
            and all(cfg.get(k) == job[k] for k in ("arm", "cohort", "smoke", "dataset", "fraction", "ids"))
            and cfg.get("model") == model and cfg.get("cache_budget_bytes") == job["budget_bytes_expected"]
            and state.get("completed_queries") == job["queries"]
            and hw.get("device_name") == "NVIDIA GeForce RTX 5090" and hw.get("platform") == "Windows"
            and hw.get("torch_cpu_threads") == 2 and hw.get("torch_interop_threads") == 16
            and gate.get("initial_used_gib", 99) < 5 and gate.get("recheck_used_gib", 99) < 5
            and (not job["smoke"] or state.get("reference_checks_passed"))):
            return attempt
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=("locomo", "scbench"), default=["locomo", "scbench"])
    p.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    p.add_argument("--python", default=str(ROOT/".venv/Scripts/python.exe"))
    p.add_argument("--run", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args(argv)
    require_unpaused(HERE)
    folder = HERE/"results/capacity_queue"
    with bootstrap_lock(folder):
        state = {"status": "preparing", "pid": os.getpid(), "started_at": now(), "jobs": {},
                 "datasets": args.datasets, "smoke_only": args.smoke_only,
                 "gpu_policy": "original local RTX5090 strict double <5 GiB gate; no concurrent model processes"}
        def update(**kwargs):
            state.update(kwargs, updated_at=now())
            save_json(folder/"status.json", state)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1")
        plans = []
        for dataset in args.datasets:
            fixtures = HERE/"fixtures/scbench" if dataset == "scbench" else HERE/"fixtures"
            _, _, documents, queries = load_bundle(dataset, fixtures)
            jobs = make_jobs(dataset, documents, queries, args.smoke_only)
            for job in jobs:
                job["fixtures"] = str(fixtures)
            plans.extend(jobs)
            save_json(folder/f"{dataset}_plan.json", jobs)
            del documents, queries
        if args.run and not args.smoke_only:
            for _ in range(2880):
                ast = read_json(HERE/"results/queue/status.json")
                full = [v for v in ast.get("jobs", {}).values() if not v.get("smoke")]
                if ast.get("status") == "complete" and len(full) == 40 and all(v["status"] == "complete" for v in full):
                    break
                update(status="waiting_for_phase_A", phase_A_status=ast.get("status"), phase_A_complete_full_jobs=sum(v["status"] == "complete" for v in full))
                if ast.get("status") == "failed":
                    raise RuntimeError("Phase A failed; preserve outputs and repair before proceeding")
                time.sleep(30)
            else:
                raise RuntimeError("Phase A did not complete in 24 hours")
        for job in plans:
            require_unpaused(HERE)
            target = HERE/"results/capacity"/job["id"]
            done = valid_complete(target, job, args.model)
            if done:
                state["jobs"][job["id"]] = {"status": "complete", "attempt": str(done), "queries": job["queries"], "smoke": job["smoke"]}
                update()
                continue
            if not args.run:
                state["jobs"][job["id"]] = {"status": "pending", "queries": job["queries"], "smoke": job["smoke"]}
                continue
            # An independently dispatched complete-trace replacement owns this
            # job. A restarted primary queue must never allocate a duplicate.
            while remeasurement(target) is not None:
                repair = remeasurement(target)
                if repair.get("status") == "failed":
                    raise RuntimeError("Registered timing replacement failed; preserve evidence and repair explicitly")
                done = valid_complete(target, job, args.model)
                if done:
                    break
                update(status="waiting_for_registered_remeasurement", active_job=job["id"], child_pid=None)
                time.sleep(30)
            if done:
                state["jobs"][job["id"]] = {"status": "complete", "attempt": str(done), "queries": job["queries"], "smoke": job["smoke"]}
                update()
                continue
            attempts = target/"attempts"
            numbers = [int(d.name) for d in attempts.glob("*") if d.is_dir() and d.name.isdigit()]
            attempt = attempts/f"{max(numbers, default=0)+1:04d}"
            attempt.mkdir(parents=True, exist_ok=False)
            command = [args.python, "-X", "utf8", "-u", str(HERE/"run_capacity.py"), "--dataset", job["dataset"],
                       "--fixtures", job["fixtures"], "--cohort", job["cohort"], "--fraction", str(job["fraction"]),
                       "--arm", job["arm"], "--model", args.model, "--out", str(attempt)]
            if job["smoke"]:
                command.append("--smoke")
            with (attempt/"stdout.log").open("w", encoding="utf-8") as out, (attempt/"stderr.log").open("w", encoding="utf-8") as err:
                require_unpaused(HERE)
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
                state["jobs"][job["id"]] = {"status": "running", "attempt": str(attempt), "queries": job["queries"], "smoke": job["smoke"]}
                update(status="running", active_job=job["id"], child_pid=child.pid, active_attempt=str(attempt), command=command)
                while child.poll() is None:
                    time.sleep(15)
                    if (attempt/"status.json").exists():
                        try:
                            snapshot = read_json(attempt/"status.json")
                            state["jobs"][job["id"]].update(status=snapshot["status"], completed_queries=snapshot["completed_queries"])
                            update()
                        except (OSError, ValueError):
                            pass
            done = valid_complete(target, job, args.model)
            if child.returncode or not done:
                update(status="failed", exit_code=child.returncode, child_pid=None, error="Capacity child failed; existing attempts preserved")
                return 1
            state["jobs"][job["id"]] = {"status": "complete", "attempt": str(done), "queries": job["queries"], "smoke": job["smoke"]}
            update(child_pid=None)
        update(status="complete" if args.run else "ready", active_job=None, child_pid=None, finished_at=now(),
               full_queries=sum(v["queries"] for v in state["jobs"].values() if v["status"] == "complete" and not v["smoke"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
