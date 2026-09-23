"""Independent serial CacheBlend serving scheduler; each child owns gpu_gate.

Waits for QA 2400 and original serving 120 plus actual model-child exit. Runs an
8-query verified smoke before the two full jobs (24 cells / 800 query records).
Only complete protocol-valid attempts are reused; failures stay on disk.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

from serving_local_bootstrap import bootstrap_lock, read_json, save_json
from qa_cost_bootstrap import process_snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROTOCOL = "cacheblend-serving-reuse-v1"
REQUIRED_GPU = "NVIDIA GeForce RTX 5090"
VARIANT = "qwen3_layer1_v_full_two_layer_bootstrap"
THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
TOTAL_FIELDS = ("retrieval_s", "load_s", "load_bytes", "transfer_s", "transfer_bytes",
    "rotate_prepare_s", "read_prefill_s", "ttft_s", "decode_s", "total_s", "generated_tokens", "decode_steps", "tokenization_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def load_plans():
    jobs = []
    for phase, lengths, qs, count in (("smoke", [1024], [1, 2], 8), ("full", [32768, 131072], [1, 10, 100], 24)):
        path = HERE / f"cacheblend_serving_{phase}_plan.json"
        plan = read_json(path)
        require(plan["phase"] == phase and plan["lengths"] == lengths and plan["query_counts"] == qs, "Plan lengths/query counts changed")
        require(plan["arm"] == "cacheblend16" and plan["variant"] == VARIANT and plan["recompute_ratio"] == .16, "Unsupported CacheBlend variant")
        require(plan["fixed_generation_lengths"] == [16, 128] and plan["tiers"] == ["cpu", "disk"] and not plan["natural_eos"], "Fixed-G/CPU/disk protocol differs")
        require(plan["environment"] == THREAD_ENV and plan["torch_cpu_threads"] == 2 and plan["torch_interop_threads"] == 16, "Thread protocol differs")
        require(plan["gate"]["used_gib_limit"] == 5 and plan["gate"]["initial_and_locked_recheck"] and plan["gate"]["comparison"] == "strictly_less_than", "Strict original gate required")
        require(plan["planned_cells"] == count and len(plan["jobs"]) == len(lengths), "Plan scope differs")
        for raw in plan["jobs"]:
            length = raw["context_tokens"]
            require(length in lengths, "Unplanned context length")
            folder = Path(raw["output"]).parent.parent
            expected_folder = HERE / "results/local/cacheblend_serving_reuse" / phase / f"cacheblend16_{length}"
            require(folder.resolve() == expected_folder.resolve(), "Output folder outside named campaign")
            jobs.append({"id": f"{phase}/cacheblend16_{length}", "phase": phase, "context_tokens": length,
                "cells": 4*len(qs), "query_records": 4*max(qs), "query_counts": qs,
                "folder": folder, "plan_path": path, "plan": plan})
    return jobs


def expected_config(job):
    plan = job["plan"]
    return {"model": str(Path(plan["model"]).resolve()), "context_file": str(Path(plan["context_file"]).resolve()),
        "lengths": [job["context_tokens"]], "query_counts": job["query_counts"], "generation_lengths": [16, 128],
        "arms": ["cacheblend16"], "tiers": ["cpu", "disk"], "j": 12, "chunk_size": 512, "topk": 12,
        "device": "cuda:0", "dtype": "bfloat16", "allow_eos": False, "cpu_test_only": False,
        "adapter_ckpt": None, "adapter_completion_marker": None, "seed": 42,
        "gpu_idle_slack_gb": 5.0, "gpu_cap_gb": 28.0}


def hardware_valid(hw):
    require(hw["timing_eligible"] and hw["device_name"] == REQUIRED_GPU and hw["platform"] == "Windows" and hw["hostname"] == socket.gethostname(), "Wrong hardware")
    require(hw["torch_cpu_threads"] == 2 and hw["torch_interop_threads"] == 16, "Wrong Torch threads")
    require((hw["omp_num_threads"], hw["mkl_num_threads"], hw["tokenizers_parallelism"]) == ("2", "2", "false"), "Wrong thread environment")
    gate = hw["gpu_admission"]
    require(gate["initial_used_gib"] < 5 and gate["recheck_used_gib"] < 5 and not gate["other_python_compute_processes"], "Invalid dual GPU admission")
    require(gate["comparison"] == "strictly_less_than", "Wrong gate comparison")


def verify_attempt(attempt, job):
    marker, config = read_json(attempt / "COMPLETED.json"), read_json(attempt / "config.json")
    require(marker["status"] == "complete" and marker["cells"] == job["cells"] and marker["adapter"] is None, "Missing completed scope")
    hardware = marker["hardware"]
    hardware_valid(hardware)
    require(all(config.get(k) == value for k, value in expected_config(job).items()), "Config changed")
    length = job["context_tokens"]
    store = read_json(attempt / f"store_{length}_cacheblend16/store.json")
    signature = store["signature"]
    require(signature["cacheblend_variant"] == VARIANT and signature["recompute_ratio"] == .16 and signature["bootstrap_full_layers"] == 2 and signature["chunk_write_sink"] is False, "Cache identity changed")
    require(store["n_tokens"] == length and store["chunk_size"] == 512, "Stored document scope changed")
    write = read_json(attempt / f"write_{length}_cacheblend16.json")
    require(write["capture_calls"] == math.ceil(length/512)+1 and write["n_document_tokens"] == length, "Document not captured once per chunk plus sink")
    all_queries, mapping = [], {}
    for tier in ("cpu", "disk"):
        for g in (16, 128):
            file = attempt / f"queries_{length}_cacheblend16_{tier}_g{g}.jsonl"
            queries = [json.loads(s) for s in file.read_text(encoding="utf-8").splitlines() if s.strip()]
            require([r["id"] for r in queries] == list(range(max(job["query_counts"]))), "Missing/repeated/reordered query rows")
            for row in queries:
                require(row["hardware"] == hardware and row["fixed_generation_length"] and len(row["generated_ids"]) == row["generated_tokens"] == g and row["decode_steps"] == g-1, "Fixed-G or hardware mismatch")
                require(row["capture_calls"] == row["document_capture_calls"] == row["query_capture_calls"] == 0, "A reuse query recaptured chunks")
                require(all(math.isfinite(row[k]) and row[k] >= 0 for k in TOTAL_FIELDS), "Invalid timing/counter")
                require(row["total_s"] >= row["ttft_s"] and row["peak_allocated_bytes"] >= row["incremental_peak_bytes"] >= 0, "Invalid time/memory")
                cb = row["cacheblend"]
                require(cb["cacheblend_variant"] == VARIANT and cb["recompute_ratio"] == .16 and cb["bootstrap_full_layers"] == 2, "Different read variant")
                require(cb["n_recompute_ctx"] == math.floor(.16*cb["n_context_tokens"]) and cb["pack_len"] == row["read_tokens"], "Wrong contextual recomputation count")
                positions = cb["selected_positions"]
                suffix = range(row["read_tokens"]-row["query_tokens"], row["read_tokens"])
                require(positions == sorted(set(positions)) and len(positions) == 1+cb["n_recompute_ctx"]+row["query_tokens"] and positions[0] == 0 and set(suffix).issubset(positions), "Missing sink/query selected positions")
                if job["phase"] == "full":
                    require(row["timing_eligible"] is True and not row["instrumentation_diagnostic"] and "fresh_reference" not in row, "Diagnostic cannot be full timing")
                else:
                    require(row["timing_eligible"] is False and row["instrumentation_diagnostic"] is True, "Smoke must remain timing-ineligible")
            mapping[tier, g] = queries
            all_queries.extend(queries)
    summaries = read_json(attempt / "summary.json")
    keys = {(r["tier"], r["G"], r["Q"]) for r in summaries}
    require(len(summaries) == job["cells"] and keys == {(t, g, q) for t in ("cpu", "disk") for g in (16, 128) for q in job["query_counts"]}, "Incomplete matrix cells")
    for cell in summaries:
        require(cell["arm"] == "cacheblend16" and cell["context_tokens"] == length and cell["hardware"] == hardware and cell["fixed_generation_length"] and cell["write"] == write, "Summary protocol differs")
        rows = mapping[cell["tier"], cell["G"]][:cell["Q"]]
        for key in TOTAL_FIELDS:
            require(close(cell["query_totals"][key], sum(r[key] for r in rows)), f"Q-prefix {key} mismatch")
        require(close(cell["end_to_end_total_s"], write["write_total_s"]+cell["startup"]["startup_load_s"]+cell["query_totals"]["total_s"]), "End-to-end arithmetic differs")
        require(cell["peak_allocated_bytes"] == max(write["write_peak_allocated_bytes"], max(r["peak_allocated_bytes"] for r in rows)), "Peak arithmetic differs")
    if job["phase"] == "smoke":
        fresh = read_json(attempt / "FRESH_REFERENCE_CHECK.json")
        require(fresh["complete"] and fresh["verified_queries"] == fresh["extra_reference_sequences"] == len(fresh["checks"]) == 8 and fresh["timing_eligible"] is False, "Fresh reference scope missing")
        for row, check in zip(all_queries, fresh["checks"]):
            require(check == row["fresh_reference"] and check["generated_ids"] == row["generated_ids"] and check["selected_indices"] == row["selected_indices"], "Reference identity differs")
            require(check["equal_ids"] and check["equal_selected_positions"] and check["persistent_cpu_versions_and_disk_metadata_unchanged"] and check["reuse_capture_calls"] == 0 and check["extra_reference_sequences"] == 1 and check["reference_outside_recorded_query_timing"], "Fresh reference check failed")
    require(len(all_queries) == job["query_records"], "Query record total differs")
    return {"status": "complete", "phase": job["phase"], "context_tokens": length,
        "cells": job["cells"], "query_records": len(all_queries), "result": str(attempt),
        "fresh_reference_sequences": 8 if job["phase"] == "smoke" else 0,
        "timing_eligible": job["phase"] == "full"}


def completed_attempt(job):
    for attempt in sorted((job["folder"] / "attempts").glob("[0-9]*"), reverse=True):
        try:
            return attempt, verify_attempt(attempt, job)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, None


def dependencies_ready(serving, qa, processes):
    if serving.get("status") != "completed" or qa.get("status") != "completed" or qa.get("full_timed_generations") != 2400:
        return False
    original = [v for k, v in serving.get("jobs", {}).items() if "/full/" in k]
    quality_cost = [v for k, v in qa.get("jobs", {}).items() if "/full/" in k]
    if len(original) != 10 or sum(j.get("cells", 0) for j in original if j.get("status") == "complete") != 120:
        return False
    if len(quality_cost) != 4 or sum(j.get("timed_generations", 0) for j in quality_cost if j.get("status") == "complete") != 2400:
        return False
    for process in processes:
        cmd = (process.get("CommandLine") or "").lower()
        if any(name in cmd for name in ("qa_cost_runner.py", "serving_reuse.py", "cacheblend_serving_driver.py")):
            return False
    return True


def verify_dependency_markers(serving, qa):
    for key, row in serving["jobs"].items():
        if "/full/" in key:
            marker = read_json(Path(row["result"]) / "COMPLETED.json")
            require(marker["status"] == "complete" and marker["cells"] == row["cells"], "Original serving receipt missing")
    for key, row in qa["jobs"].items():
        if "/full/" in key:
            marker = read_json(Path(row["attempt"]) / "COMPLETED.json")
            require(marker["status"] == "complete" and marker["expected_repetitions"] == 600 and marker["unique_method_examples"] == 200, "QA full receipt missing")


def child_command(job, attempt):
    plan = job["plan"]
    command = [sys.executable, "-X", "utf8", "-u", str(HERE / "cacheblend_serving_driver.py"),
        "--model", str(Path(plan["model"]).resolve()), "--context-file", str(Path(plan["context_file"]).resolve()),
        "--out", str(attempt.resolve()), "--arms", "cacheblend16", "--lengths", str(job["context_tokens"]),
        "--query-counts", *map(str, job["query_counts"]), "--generation-lengths", "16", "128", "--tiers", "cpu", "disk",
        "--gpu-idle-slack-gb", "5.0", "--gpu-wait-seconds", "86400"]
    if job["phase"] == "smoke":
        command.append("--verify-fresh")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=HERE / "results/local/bootstrap_cacheblend_serving")
    args = parser.parse_args(argv)
    require(platform.system() == "Windows", "Only local Windows scheduling is allowed")
    jobs = load_plans()
    with bootstrap_lock(args.state_dir):
        state_path = args.state_dir / "status.json"
        state = read_json(state_path) if state_path.exists() else {}
        state.update(pid=os.getpid(), protocol=PROTOCOL, required_gpu=REQUIRED_GPU, remote_timing_allowed=False,
            jobs=state.get("jobs", {}), error=None, target_full_cells=24, target_full_query_records=800,
            target_smoke_cells=8, target_smoke_query_records=8, target_extra_smoke_reference_sequences=8,
            result_root=str(HERE / "results/local/cacheblend_serving_reuse"), strict_gpu_admission_gib=5.0)
        def update(**values):
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            for phase in ("full", "smoke"):
                done = [j for j in state["jobs"].values() if j["status"] == "complete" and j["phase"] == phase]
                state[f"completed_{phase}_cells"] = sum(j["cells"] for j in done)
                state[f"completed_{phase}_query_records"] = sum(j["query_records"] for j in done)
            state["extra_smoke_reference_sequences"] = sum(j["fresh_reference_sequences"] for j in state["jobs"].values() if j["status"] == "complete")
            save_json(state_path, state)
        try:
            serving_path = HERE / "results/local/bootstrap_serving/status.json"
            qa_path = HERE / "results/local/bootstrap_qa_cost/status.json"
            while True:
                serving, qa = read_json(serving_path), read_json(qa_path)
                ready = dependencies_ready(serving, qa, process_snapshot())
                if ready:
                    verify_dependency_markers(serving, qa)
                if not args.run:
                    update(status="ready", dependency_ready=ready, child_pid=None, active_job=None)
                    return 0
                if ready:
                    break
                update(status="waiting_for_prior_local_campaigns", child_pid=None, active_job=None)
                time.sleep(30)
            update(status="running", dependency_ready=True, dependencies_verified_at=datetime.now(timezone.utc).isoformat())
            for job in jobs:
                attempt, record = completed_attempt(job)
                if attempt is None:
                    attempts = job["folder"] / "attempts"
                    attempts.mkdir(parents=True, exist_ok=True)
                    number = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()] + [0])+1
                    attempt = attempts / f"{number:04d}"
                    attempt.mkdir()
                    command = child_command(job, attempt)
                    log = args.state_dir / "logs" / f"{job['phase']}_cacheblend16_{job['context_tokens']}_{number:04d}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    update(status="waiting_for_gpu_or_running", active_job=job["id"], active_attempt=str(attempt),
                        active_log=str(log), command=command, child_gpu_admitted=False)
                    with log.open("w", encoding="utf-8") as output:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                            env=dict(os.environ, **THREAD_ENV, CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1"),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        while child.poll() is None:
                            cfg = read_json(attempt / "config.json") if (attempt / "config.json").exists() else {}
                            update(child_pid=child.pid, child_gpu_admitted=bool(cfg.get("hardware", {}).get("gpu_admission")))
                            time.sleep(15)
                    require(child.returncode == 0, f"Child failed: {job['id']}; retained {attempt}; see {log}")
                    record = verify_attempt(attempt, job)
                state["jobs"][job["id"]] = record
                update(status="running", child_pid=None)
            update(status="completed", child_pid=None, active_job=None, error=None)
        except BaseException as exc:
            update(status="failed", error=repr(exc))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
