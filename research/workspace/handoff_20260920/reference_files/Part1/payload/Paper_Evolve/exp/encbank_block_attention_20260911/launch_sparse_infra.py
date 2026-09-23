"""CPU-only sequential supervisor for guarded local sparse infrastructure cells.

Default is a reviewable dry plan. --run executes independent worker processes;
OOM/cap failures retain the original shape and do not trigger shorter retries.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "exp/encbank_v2_benchmarks_20260908"))
from serving_local_bootstrap import bootstrap_lock
from infra_protocol import (VERSION, GIB, GPU_NAME, INCREMENTAL_CAP_BYTES, digest,
    make_jobs, model_processes, now, read_json, render_report, save_json, snapshot, validate_shape)
from infra_processes import identity, identity_alive, validate_worker, terminate_owned_tree, MONITOR_POLICY_VERSION
from shared_state_protocol import validate_shared_mode
from same_math_protocol import validate_same_math_mode


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", type=Path, default=ROOT / ".venv/Scripts/python.exe")
    ap.add_argument("--model", type=Path, default=Path("/srv/encbank/legacy_workspace/models/Qwen3-8B"))
    ap.add_argument("--adapter", type=Path, default=ROOT / "exp/encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final/adapter.pt")
    ap.add_argument("--adapter-kind", choices=("strong", "trained"), default="strong")
    ap.add_argument("--reader-implementation", choices=("reference", "decode_v2", "backend_v3"), default="reference")
    ap.add_argument("--validate-optimized-decode", action="store_true")
    ap.add_argument("--validate-backend", action="store_true")
    ap.add_argument("--backend-model-parity", action="store_true")
    ap.add_argument("--profile-reader-only", action="store_true")
    ap.add_argument("--shared-state-diagnostic-only", action="store_true")
    ap.add_argument("--shared-state-cpu-receipt", type=Path)
    ap.add_argument("--same-math-diagnostic-only", action="store_true")
    ap.add_argument("--same-math-cpu-receipt", type=Path)
    ap.add_argument("--checkpoint-map", type=Path, help="Optional JSON arm -> local adapter path; used with --adapter-kind trained")
    ap.add_argument("--out", type=Path, default=HERE / "results/infra_pilot")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--arms", nargs="+", choices=("D0", "A", "B", "D1", "NATIVE", "FULL"), default=["D0", "A", "B", "D1"])
    ap.add_argument("--modes", nargs="+", choices=("cold_hj", "block_hot"), default=["cold_hj", "block_hot"])
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--generation-tokens", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--rho", type=float, default=.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.)
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--reuse-requests", type=int, default=0,
                    help="Run a separate fixed-pack stream with persistent cross-request caches")
    ap.add_argument("--reuse-quality-receipt", type=Path)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--gpu-wait-seconds", type=int, default=86400)
    ap.add_argument("--sample-seconds", type=float, default=1.)
    ap.add_argument("--retry-failed", action="store_true", help="Explicit new attempts at the SAME original shape")
    ap.add_argument("--max-jobs", type=int)
    ap.add_argument("--run", action="store_true")
    return ap


def heartbeat(path):
    save_json(path, {**identity(os.getpid()), "unix_s": time.time(), "updated_at": now()})


def stop_owned_child(child):
    if hasattr(child, "_infra_root_create_time"):
        terminate_owned_tree(child)
    elif child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def monitor_worker(child, attempt, beat_path, sample_seconds):
    """A monitor failure must finalize its receipt and stop only its own worker."""
    try:
        return _monitor_worker(child, attempt, beat_path, sample_seconds)
    except Exception as exc:
        stop_error = None
        try:
            stop_owned_child(child)
        except Exception as stop_exc:
            stop_error = f"{type(stop_exc).__name__}: {stop_exc}"
        monitor = read_json(attempt / "monitor.json", {})
        monitor.update(status="monitor_failed", error=f"{type(exc).__name__}: {exc}",
                       finished_at=now(), exit_code=child.poll(),
                       supervisor_pid=os.getpid(), child_pid=child.pid,
                       ownership_stop_error=stop_error)
        save_json(attempt / "monitor.json", monitor)
        if stop_error is not None:
            raise RuntimeError("Owned child exit could not be confirmed; queue stopped") from exc
        return monitor


def _monitor_worker(child, attempt, beat_path, sample_seconds):
    """Monitor the complete CUDA lifetime, including model load and cache build."""
    monitor = {"status": "watching", "child_pid": child.pid, "supervisor_pid": os.getpid(),
               "monitor_policy_version": MONITOR_POLICY_VERSION,
               "launcher_sha256": digest(HERE / "launch_sparse_infra.py"),
               "started_at": now(), "cap_bytes": INCREMENTAL_CAP_BYTES,
               "peak_incremental_gpu_bytes": None, "peak_process_gpu_bytes": None,
               "process_memory_available_samples": 0, "samples": 0, "interference": [],
               "sample_period_requested_s": sample_seconds,
               "peak_scope": "sampled maxima, not an upper bound on instantaneous allocations",
               "baseline_is_fixed_before_cuda": True}
    baseline = None
    failure = None
    previous_sample_at = None
    max_gap = 0.
    owned_pids = {child.pid}
    actual_worker_pid = child.pid
    worker_validated = False
    with (attempt / "telemetry.jsonl").open("w", encoding="utf-8") as log:
        while child.poll() is None:
            heartbeat(beat_path)
            if (HERE / "LOCAL_INFRA_PAUSED.json").exists():
                failure = ("paused", "Local infrared queue paused by the user marker")
                stop_owned_child(child)
                break
            progress = read_json(attempt / "progress.json", {})
            lease = progress.get("supervisor_lease")
            if lease and not worker_validated:
                try:
                    owned_pids = validate_worker(child.pid, child._infra_root_create_time, os.getpid(),
                                                lease, HERE / "infra_sparse.py", attempt)
                    actual_worker_pid = lease["worker"]["pid"]
                    child._infra_worker_lease = lease
                    worker_validated = True
                    monitor["worker_lease"] = lease
                    monitor["actual_worker_pid"] = actual_worker_pid
                except Exception as exc:
                    failure = ("invalid_worker_identity", str(exc))
                    stop_owned_child(child)
                    break
            if baseline is None and progress.get("baseline_snapshot"):
                if hasattr(child, "_infra_root_create_time") and not worker_validated:
                    failure = ("invalid_worker_identity", "CUDA admission preceded a validated owned-worker handshake")
                    stop_owned_child(child)
                    break
                baseline = progress["baseline_snapshot"]
                monitor["baseline_snapshot"] = baseline
                monitor["baseline_gpu_used_bytes"] = baseline["used_bytes"]
                if not baseline["used_bytes"] < 5 * GIB:
                    failure = ("invalid_admission", "Admission baseline is not strictly below 5 GiB")
                    stop_owned_child(child)
                    break
            if baseline is not None:
                try:
                    sample = snapshot()
                    if sample["name"] != GPU_NAME or sample["uuid"] != baseline["uuid"]:
                        raise RuntimeError("GPU identity changed during the run")
                    sample_at = time.monotonic()
                    if previous_sample_at is not None:
                        max_gap = max(max_gap, sample_at - previous_sample_at)
                    previous_sample_at = sample_at
                    own = [p for p in sample["processes"] if p["pid"] in owned_pids]
                    # A known redirector value does not make an unknown actual
                    # worker allocation known. Preserve partial N/A as unknown.
                    own_values = [p["used_bytes"] for p in own]
                    process_used = sum(own_values) if own_values and all(v is not None for v in own_values) else None
                    signed_delta = sample["used_bytes"] - baseline["used_bytes"]
                    delta = max(0, signed_delta)
                    monitor["samples"] += 1
                    monitor["peak_incremental_gpu_bytes"] = max(monitor["peak_incremental_gpu_bytes"] or 0, delta)
                    if process_used is not None:
                        monitor["process_memory_available_samples"] += 1
                        monitor["peak_process_gpu_bytes"] = max(monitor["peak_process_gpu_bytes"] or 0, process_used)
                    foreign_models = [p for p in model_processes(sample) if p["pid"] not in owned_pids]
                    row = {"timestamp": sample["timestamp"], "gpu_used_bytes": sample["used_bytes"],
                           "signed_increment_bytes": signed_delta, "own_process_bytes": process_used,
                           "processes": sample["processes"], "foreign_model_rows": foreign_models}
                    log.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    if foreign_models:
                        monitor["interference"].extend(foreign_models)
                        failure = ("invalid_interference", "A foreign model/compute process appeared; only the owned worker is stopped")
                    elif delta > INCREMENTAL_CAP_BYTES or (process_used is not None and process_used > INCREMENTAL_CAP_BYTES):
                        failure = ("incremental_cap_exceeded", "Sampled task GPU increase/process occupancy exceeded 28 GiB")
                    if failure:
                        stop_owned_child(child)
                        break
                    monitor["last_snapshot"] = sample
                    monitor["maximum_observed_sample_gap_s"] = max_gap
                    save_json(attempt / "monitor.json", monitor)
                except Exception as exc:
                    failure = ("monitor_failed", str(exc))
                    stop_owned_child(child)
                    break
            time.sleep(sample_seconds)
    # poll/wait observes actual process exit before this queue can launch another.
    exit_code = child.wait()
    lease = getattr(child, "_infra_worker_lease", None)
    if lease and identity_alive(lease["worker"]):
        failure = ("worker_outlived_redirector", "Actual GPU worker outlived the owned process root")
        stop_owned_child(child)
    monitor.update(finished_at=now(), exit_code=exit_code, maximum_observed_sample_gap_s=max_gap)
    if failure:
        monitor.update(status=failure[0], error=failure[1])
    elif baseline is None:
        monitor.update(status="not_admitted", error="Worker exited before CUDA admission")
    elif monitor["samples"] == 0:
        monitor.update(status="unobserved", error="No post-admission GPU samples")
    else:
        monitor["status"] = "complete"
    save_json(attempt / "monitor.json", monitor)
    return monitor


def expected_identity(args, job, adapter):
    if args.reader_implementation != "reference" and job["arm"] in ("NATIVE", "FULL"):
        raise ValueError("Decode optimization is only for sparse wrapper arms")
    shared_cpu = validate_shared_mode(args)
    same_math_cpu = validate_same_math_mode(args)
    expected = {"protocol": VERSION, "model": str(args.model.resolve()),
            "model_config_sha256": digest(args.model / "config.json"),
            "adapter": str(adapter.resolve()), "adapter_sha256": digest(adapter),
            "reader_sha256": digest(HERE / "sparse_reader.py"),
            "reader_implementation": args.reader_implementation,
            "optimized_reader_sha256": digest(HERE / "optimized_sparse_reader.py") if args.reader_implementation in ("decode_v2", "backend_v3") else None,
            "backend_reader_sha256": digest(HERE / "backend_sparse_reader.py") if args.reader_implementation == "backend_v3" else None,
            "backend_gpu_validation": args.validate_backend,
            "backend_validation_sha256": {name: digest(HERE / name) for name in ("backend_sparse_reader.py", "validate_backend_cpu.py", "test_backend_sparse_reader.py")} if args.validate_backend else None,
            "backend_model_parity": args.backend_model_parity,
            "backend_parity_sha256": digest(HERE / "backend_parity.py") if args.backend_model_parity else None,
            "shared_state_diagnostic_only": args.shared_state_diagnostic_only,
            "shared_state_sources_sha256": shared_cpu["source_sha256"] if shared_cpu else None,
            "shared_state_cpu_receipt_sha256": digest(args.shared_state_cpu_receipt) if shared_cpu else None,
            "shared_state_protocol_sha256": digest(HERE / "shared_state_protocol.py"),
            "same_math_diagnostic_only": args.same_math_diagnostic_only,
            "same_math_sources_sha256": same_math_cpu["source_sha256"] if same_math_cpu else None,
            "same_math_cpu_receipt_sha256": digest(args.same_math_cpu_receipt) if same_math_cpu else None,
            "same_math_protocol_sha256": digest(HERE / "same_math_protocol.py"),
            "optimized_gpu_validation": args.validate_optimized_decode,
            "profile_reader_only": args.profile_reader_only,
            "profiler_sha256": digest(HERE / "reader_profiler.py") if args.profile_reader_only else None,
            "optimized_validation_sha256": {name: digest(HERE / name) for name in ("validate_optimized_cpu.py", "test_optimized_sparse_reader.py")} if args.validate_optimized_decode else None,
            "infra_worker_sha256": digest(HERE / "infra_sparse.py"),
            "infra_protocol_sha256": digest(HERE / "infra_protocol.py"),
            "infra_processes_sha256": digest(HERE / "infra_processes.py"),
            "infra_launcher_sha256": digest(HERE / "launch_sparse_infra.py"),
            "monitor_policy_version": MONITOR_POLICY_VERSION,
            "encbank_sha256": digest(ROOT / "Encbank/encbank/model.py"),
            "native_reader_sha256": digest(HERE / "native_infra_readers.py") if job["arm"] in ("NATIVE", "FULL") else None,
            "adapter_kind": args.adapter_kind, "arm": job["arm"], "cache_mode": job["cache_mode"],
            "document_tokens": job["document_tokens"], "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens, "chunk_size": args.chunk_size,
            "j": args.j, "m": args.m, "rho": args.rho, "rank": args.rank,
            "alpha": args.alpha, "seed": args.seed, "repetitions": args.repetitions}
    if getattr(args, "reuse_requests", 0):
        from reuse_protocol import validate_reuse_mode
        per_job = argparse.Namespace(**vars(args))
        per_job.arm, per_job.cache_mode = job["arm"], job["cache_mode"]
        per_job.document_tokens, per_job.adapter = job["document_tokens"], adapter
        expected["reuse"] = validate_reuse_mode(per_job)
    return expected


def find_prior(folder, expected):
    for attempt in sorted((folder / "attempts").glob("*"), reverse=True):
        result, mon = read_json(attempt / "result.json", {}), read_json(attempt / "monitor.json", {})
        if result.get("identity") != expected:
            continue
        state = result.get("status", "failed")
        if mon.get("status") not in ("complete", "not_admitted"):
            state = mon.get("status", "failed")
        eligible = (result.get("timing_eligible")
            or (expected.get("profile_reader_only") and result.get("profiler_receipt"))
            or (expected.get("shared_state_diagnostic_only") and result.get("shared_state_receipt", {}).get("status") == "complete"
                and result.get("shared_state_receipt", {}).get("protocol_checks_passed"))
            or (expected.get("same_math_diagnostic_only") and result.get("same_math_receipt", {}).get("status") == "complete"
                and result.get("same_math_receipt", {}).get("passed") is True))
        if state == "complete" and (not eligible or mon.get("status") != "complete"
                                     or mon.get("exit_code") != 0):
            state = "invalid_incomplete_receipt"
        return {"status": state, "attempt": str(attempt), "result": result, "monitor": mon}
    return None


def main(argv=None):
    args = parser().parse_args(argv)
    if args.repetitions < 1 or args.reuse_requests < 0 or not .1 <= args.sample_seconds <= 5:
        raise ValueError("Use positive repetitions and a 0.1--5 second monitoring period")
    if args.profile_reader_only and (args.repetitions != 1 or args.generation_tokens != 4):
        raise ValueError("Bounded profiling requires repetitions=1 and generation-tokens=4")
    if args.reader_implementation == "backend_v3" and not args.validate_backend:
        raise ValueError("The new backend requires guarded tiny GPU validation")
    if args.backend_model_parity and (not args.profile_reader_only or args.reader_implementation != "backend_v3"):
        raise ValueError("8B backend parity is diagnostic-only and requires backend_v3")
    validate_shared_mode(args)
    validate_same_math_mode(args)
    if len(set(args.arms)) != len(args.arms) or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("Duplicate methods/shapes are not allowed")
    jobs = make_jobs(args.lengths, args.prompt_tokens, args.generation_tokens, args.arms, args.modes)
    if not jobs:
        raise ValueError("No valid arm/cache-mode combinations selected")
    if args.reuse_requests:
        # A/B must use their resident independent-block cache in this stream.
        # Other paths retain h_j (or FULL token IDs) and use their legal reuse.
        jobs = [job for job in jobs if job["cache_mode"] ==
                ("block_hot" if job["arm"] in ("A", "B") else "cold_hj")]
        if not jobs:
            raise ValueError("No valid persistent-cache stream cells selected")
        for job in jobs:
            job["id"] += f"_reuse{args.reuse_requests}"
    adapters = read_json(args.checkpoint_map, {}) if args.checkpoint_map else {}
    if args.checkpoint_map and not adapters:
        raise ValueError("Checkpoint map must be a nonempty JSON object")
    config = read_json(args.model / "config.json")
    if config is None:
        raise FileNotFoundError(args.model / "config.json")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    args.out.mkdir(parents=True, exist_ok=True)
    state = {"protocol": VERSION, "status": "ready", "pid": os.getpid(), "started_at": now(), "jobs": {},
             "reader_implementation": args.reader_implementation,
             "profile_reader_only": args.profile_reader_only,
             "shared_state_diagnostic_only": args.shared_state_diagnostic_only,
             "same_math_diagnostic_only": args.same_math_diagnostic_only,
             "reuse_requests": args.reuse_requests,
             "model": str(args.model), "quality_comparison": False,
             "resource_policy": "local 5090 only; original locked strict <5GiB admission; +28GiB task budget incl. weights",
             "pause_marker": str(HERE / "LOCAL_INFRA_PAUSED.json")}
    save_json(args.out / "plan.json", jobs)
    def update(**changes):
        state.update(changes, updated_at=now())
        save_json(args.out / "status.json", state)
        if args.reuse_requests:
            lines = ["# Fixed-pack cache-reuse queue", "",
                     "A separate request stream; legacy repetition-cell results are not reused.", "",
                     "| Method | Document tokens | Queries | Status |", "|---|---:|---:|---|"]
            for job in jobs:
                cell = state["jobs"].get(job["id"], {})
                lines.append(f"| {job['arm']} | {job['document_tokens']} | {args.reuse_requests} | {cell.get('status', 'pending')} |")
            (args.out / "REUSE_QUEUE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif not (args.profile_reader_only or args.shared_state_diagnostic_only or args.same_math_diagnostic_only):
            render_report(args.out, jobs, state["jobs"])
    # Default dry planning uses no NVIDIA/CUDA calls and starts no subprocess.
    if not args.run:
        for job in jobs:
            adapter = Path(adapters.get(job["arm"], args.adapter))
            try:
                validate_shape(config, job["document_tokens"], args.prompt_tokens,
                               args.generation_tokens, args.j, args.m, args.chunk_size)
                expected = expected_identity(args, job, adapter)
                state["jobs"][job["id"]] = find_prior(args.out / job["id"], expected) or {"status": "pending"}
            except Exception as exc:
                waiting = args.reuse_requests and str(exc).startswith("waiting_quality:")
                state["jobs"][job["id"]] = {"status": "waiting_quality" if waiting else "invalid_configuration", "error": str(exc)}
        if args.reuse_requests and any(cell["status"] == "waiting_quality" for cell in state["jobs"].values()):
            update(status="waiting_quality")
        else:
            update()
        return 0
    if platform.system() != "Windows":
        raise ValueError("This supervisor only runs on the authorized local Windows host")
    with bootstrap_lock(args.out / "supervisor"):
        beat = args.out / "supervisor_heartbeat.json"
        launched = 0
        for job in jobs:
            if (HERE / "LOCAL_INFRA_PAUSED.json").exists():
                update(status="paused")
                return 0
            adapter = Path(adapters.get(job["arm"], args.adapter))
            try:
                validate_shape(config, job["document_tokens"], args.prompt_tokens,
                               args.generation_tokens, args.j, args.m, args.chunk_size)
                expected = expected_identity(args, job, adapter)
            except Exception as exc:
                waiting = args.reuse_requests and str(exc).startswith("waiting_quality:")
                state["jobs"][job["id"]] = {"status": "waiting_quality" if waiting else "invalid_configuration", "error": str(exc)}
                if waiting:
                    update(status="waiting_quality", active_job=None, child_pid=None,
                           error="Remote trained-path quality is not ready; no GPU worker started for this cell")
                    return 0
                update()
                continue
            target = args.out / job["id"]
            prior = find_prior(target, expected)
            if prior and (prior["status"] == "complete" or not args.retry_failed):
                state["jobs"][job["id"]] = prior
                update()
                continue
            if args.max_jobs is not None and launched >= args.max_jobs:
                state["jobs"][job["id"]] = {"status": "pending"}
                continue
            attempts = target / "attempts"
            numbers = [int(p.name) for p in attempts.glob("*") if p.is_dir() and p.name.isdigit()]
            attempt = attempts / f"{max(numbers, default=0)+1:04d}"
            attempt.mkdir(parents=True)
            command = [str(args.python), "-X", "utf8", "-u", str(HERE / "infra_sparse.py"),
                "--model", str(args.model.resolve()), "--adapter", str(adapter.resolve()),
                "--adapter-kind", args.adapter_kind, "--arm", job["arm"], "--cache-mode", job["cache_mode"],
                "--reader-implementation", args.reader_implementation,
                "--document-tokens", str(job["document_tokens"]), "--prompt-tokens", str(args.prompt_tokens),
                "--generation-tokens", str(args.generation_tokens), "--chunk-size", str(args.chunk_size),
                "--j", str(args.j), "--m", str(args.m), "--rho", str(args.rho),
                "--rank", str(args.rank), "--alpha", str(args.alpha), "--seed", str(args.seed),
                "--repetitions", str(args.repetitions), "--gpu-wait-seconds", str(args.gpu_wait_seconds),
                "--out", str(attempt.resolve()), "--supervisor-pid", str(os.getpid()),
                "--supervisor-heartbeat", str(beat.resolve())]
            if args.validate_optimized_decode:
                command.append("--validate-optimized-decode")
            if args.reuse_requests:
                command.extend(["--reuse-requests", str(args.reuse_requests),
                                "--reuse-quality-receipt", str(args.reuse_quality_receipt.resolve())])
            if args.validate_backend:
                command.append("--validate-backend")
            if args.backend_model_parity:
                command.append("--backend-model-parity")
            if args.profile_reader_only:
                command.append("--profile-reader-only")
            if args.shared_state_diagnostic_only:
                command.extend(["--shared-state-diagnostic-only", "--shared-state-cpu-receipt",
                                str(args.shared_state_cpu_receipt.resolve())])
            if args.same_math_diagnostic_only:
                command.extend(["--same-math-diagnostic-only", "--same-math-cpu-receipt",
                                str(args.same_math_cpu_receipt.resolve())])
            heartbeat(beat)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            with (attempt / "stdout.log").open("w", encoding="utf-8") as out, (attempt / "stderr.log").open("w", encoding="utf-8") as err:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                child._infra_root_create_time = identity(child.pid)["create_time"]
                state["jobs"][job["id"]] = {"status": "running", "attempt": str(attempt), "child_pid": child.pid}
                update(status="running", active_job=job["id"], child_pid=child.pid)
                mon = monitor_worker(child, attempt, beat, args.sample_seconds)
            launched += 1
            result = read_json(attempt / "result.json", {})
            status = result.get("status", "failed")
            if mon["status"] not in ("complete", "not_admitted"):
                status = mon["status"]
            if status == "complete" and (mon["status"] != "complete" or mon.get("exit_code") != 0
                                          or result.get("identity") != expected):
                status = "invalid_incomplete_receipt"
            state["jobs"][job["id"]] = {"status": status, "attempt": str(attempt), "result": result, "monitor": mon}
            update(child_pid=None)
            if status not in ("complete", "oom", "incremental_cap_exceeded"):
                update(status="attention_required", error="Worker failure retained; no automatic recipe/input change")
                return 1
        pending = any(v["status"] == "pending" for v in state["jobs"].values())
        update(status="partial" if pending else "complete", active_job=None, child_pid=None, finished_at=now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
