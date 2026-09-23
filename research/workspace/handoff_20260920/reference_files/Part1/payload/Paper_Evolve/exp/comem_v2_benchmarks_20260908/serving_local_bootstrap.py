"""Resume the serving-cost queue on the local RTX 5090, without competing GPU jobs.

The default invocation only checks readiness. Add --run to execute. Base smoke is
1024 tokens, Q=1/2, G=4 (16 cells); full is 32k/128k, Q=1/10/100, G=16/128
(96 cells). Each arm/document runs in one process and writes its document once.
Completed jobs are reused only with the same protocol and eligible hardware.
Unfinished attempts are retained, and a fresh attempt is used on restart.

--campaign all runs the base queue first, then waits for the synchronized final
4000-step adapter. --campaign pub_lora can instead add that queue in a later run.
Each child acquires the shared exp/gpu_gate.py lock and a 28 GB allocator cap.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOCAL = HERE / "results/local"
REQUIRED_GPU = "NVIDIA GeForce RTX 5090"
BASE_ARMS = ("fix_all", "pub", "pub_sink", "j0")
THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
THREAD_HARDWARE = {"torch_cpu_threads": 2, "torch_interop_threads": 16,
                   "omp_num_threads": "2", "mkl_num_threads": "2", "tokenizers_parallelism": "false"}
PROTOCOL = {
    "smoke": {"lengths": [1024], "query_counts": [1, 2], "generation_lengths": [4]},
    "full": {"lengths": [32768, 131072], "query_counts": [1, 10, 100], "generation_lengths": [16, 128]},
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_process_record(pid):
    """Inspect one explicit local PID during a scheduler-only admission migration."""
    command = (f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | "
               "Select-Object ProcessId,CommandLine,CreationDate | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def wait_for_existing_job(pid, state, update):
    """Let the already admitted measured child finish before resuming its queue."""
    process = local_process_record(pid)
    if process is None:
        return
    expected_attempt = state.get("active_attempt")
    command = process.get("CommandLine") or ""
    if not expected_attempt or "serving_reuse.py" not in command or expected_attempt not in command:
        raise ValueError("--resume-after-pid must identify the recorded local serving attempt")
    identity = process["CreationDate"]
    while process and process.get("CreationDate") == identity:
        update(status="waiting_for_current_job", preserved_child_pid=pid,
               prior_admission_retained=True)
        time.sleep(15)
        process = local_process_record(pid)
    update(preserved_child_pid=None, child_pid=None)


@contextmanager
def bootstrap_lock(folder):
    """An OS-held lock also prevents duplicate bootstraps while waiting for the GPU."""
    import msvcrt
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "bootstrap.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another local serving bootstrap is already running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def jobs_for(campaign, phases, result_root):
    arms = BASE_ARMS if campaign == "base" else ("pub_lora",)
    root = Path(result_root) if campaign == "base" else Path(result_root) / "pub_lora"
    jobs = []
    for phase in phases:
        protocol = PROTOCOL[phase]
        for length in protocol["lengths"]:
            for arm in arms:
                jobs.append({"id": f"{campaign}/{phase}/{arm}_{length}", "phase": phase,
                    "arm": arm, "length": length, "query_counts": protocol["query_counts"],
                    "generation_lengths": protocol["generation_lengths"], "tiers": ["cpu", "disk"],
                    "cells": 2 * len(protocol["query_counts"]) * len(protocol["generation_lengths"]),
                    "folder": root / phase / f"{arm}_{length}"})
    return jobs


def expected_config(args, job):
    return {"model": str(args.model.resolve()), "context_file": str(args.context_file.resolve()),
            "lengths": [job["length"]], "query_counts": job["query_counts"],
            "generation_lengths": job["generation_lengths"], "arms": [job["arm"]],
            "tiers": job["tiers"], "j": 12, "chunk_size": 512, "topk": 12,
            "dtype": "bfloat16", "device": "cuda:0", "allow_eos": False, "seed": 42,
            "adapter_ckpt": str(args.adapter_ckpt.resolve()) if job["arm"] == "pub_lora" else None}


def completed_attempt(args, job):
    attempts = job["folder"] / "attempts"
    for attempt in sorted(attempts.glob("*"), reverse=True):
        try:
            if (attempt / "INVALIDATED.json").exists():
                continue
            marker = read_json(attempt / "COMPLETED.json")
            config = read_json(attempt / "config.json")
            rows = read_json(attempt / "summary.json")
            hardware = marker["hardware"]
            # The first completed smoke predates thread provenance. It verifies
            # functionality only and is never included in the full cost table.
            legacy_smoke = job["phase"] == "smoke" and all(k not in hardware for k in THREAD_HARDWARE)
            if (marker.get("status") != "complete" or marker.get("cells") != job["cells"]
                    or not hardware.get("timing_eligible") or hardware.get("device_name") != REQUIRED_GPU
                    or hardware.get("platform") != "Windows" or hardware.get("hostname") != socket.gethostname()
                    or (not legacy_smoke and any(hardware.get(k) != v for k, v in THREAD_HARDWARE.items()))
                    or any(config.get(k) != v for k, v in expected_config(args, job).items())):
                continue
            expected = {(job["arm"], job["length"], tier, q, g) for tier in job["tiers"]
                        for q in job["query_counts"] for g in job["generation_lengths"]}
            actual = {(r["arm"], r["context_tokens"], r["tier"], r["Q"], r["G"]) for r in rows}
            if len(rows) != job["cells"] or actual != expected or any(
                    r.get("hardware") != hardware or not r.get("fixed_generation_length")
                    or r["query_totals"]["generated_tokens"] != r["Q"] * r["G"] for r in rows):
                continue
            if job["arm"] == "pub_lora" and (marker.get("adapter", {}).get("step") != 4000
                    or marker["adapter"].get("load_mode") != "peft_unmerged"):
                continue
            for tier in job["tiers"]:
                for g in job["generation_lengths"]:
                    records = (attempt / f'queries_{job["length"]}_{job["arm"]}_{tier}_g{g}.jsonl').read_text().splitlines()
                    if len(records) != max(job["query_counts"]):
                        raise ValueError("Incomplete query records")
            return attempt
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def child_command(args, job, attempt):
    command = [str(args.python), str(HERE / "serving_reuse.py"), "--model", str(args.model.resolve()),
               "--context-file", str(args.context_file.resolve()), "--out", str(attempt),
               "--lengths", str(job["length"]), "--query-counts", *map(str, job["query_counts"]),
               "--generation-lengths", *map(str, job["generation_lengths"]),
               "--arms", job["arm"], "--tiers", *job["tiers"],
               "--gpu-idle-slack-gb", str(args.gpu_idle_slack_gb),
               "--gpu-wait-seconds", str(args.gpu_wait_seconds)]
    if job["arm"] == "pub_lora":
        command += ["--adapter-ckpt", str(args.adapter_ckpt.resolve()),
                    "--adapter-completion-marker", str(args.adapter_completion_marker.resolve())]
    return command


def child_environment():
    """Do not inherit timing-sensitive thread defaults from a restarted scheduler."""
    return dict(os.environ, **THREAD_ENV, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")


def preflight(args):
    if platform.system() != "Windows":
        raise ValueError("This bootstrap is restricted to the local Windows RTX 5090")
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if not inventory or inventory[0].split(",")[1].strip() != REQUIRED_GPU:
        raise ValueError(f"Local GPU 0 is not {REQUIRED_GPU}: {inventory}")
    config = read_json(args.model / "config.json")
    index = read_json(args.model / "model.safetensors.index.json")
    files = sorted(set(index["weight_map"].values()))
    if config.get("model_type") != "qwen3" or any(not (args.model / f).is_file() for f in files):
        raise ValueError("Local Qwen3 model or weight shards are missing")
    if not args.context_file.is_file():
        raise FileNotFoundError(args.context_file)
    # Import validation cannot initialize CUDA or load the 8B model.
    env = dict(child_environment(), CUDA_VISIBLE_DEVICES="-1")
    probe = subprocess.run([str(args.python), "-c", "import json,sys,torch,transformers,peft; "
        "from serving_reuse import ReusableReader; from transformers import Qwen3ForCausalLM; "
        "print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),"
        "transformers=transformers.__version__,peft=peft.__version__,torch_cpu_threads=torch.get_num_threads(),"
        "torch_interop_threads=torch.get_num_interop_threads())))"], cwd=HERE,
        env=env, capture_output=True, text=True, timeout=180)
    if probe.returncode:
        raise RuntimeError(f"Local import preflight failed:\n{probe.stderr[-6000:]}")
    return {"hostname": socket.gethostname(), "platform": platform.system(),
            "gpu_inventory": inventory, "environment": json.loads(probe.stdout.strip().splitlines()[-1]),
            "model": str(args.model.resolve()), "model_shards": {f: (args.model / f).stat().st_size for f in files},
            "context_file": str(args.context_file.resolve()), "gpu_experiment_started": False}


def run_queue(args, state, update):
    campaigns = ("base", "pub_lora") if args.campaign == "all" else (args.campaign,)
    for campaign in campaigns:
        if campaign == "pub_lora":
            while not args.adapter_completion_marker.is_file():
                update(status="waiting_for_final_adapter", campaign=campaign,
                       required_checkpoint_step=4000, adapter_marker=str(args.adapter_completion_marker))
                time.sleep(30)
            from serving_reuse import validate_adapter_checkpoint
            state["adapter"] = validate_adapter_checkpoint(args.adapter_ckpt, args.adapter_completion_marker)
        jobs = jobs_for(campaign, args.phases, args.result_root)
        for phase in args.phases:
            phase_rows = []
            phase_jobs = [j for j in jobs if j["phase"] == phase]
            for job in phase_jobs:
                attempt = completed_attempt(args, job)
                if attempt is None:
                    attempts = job["folder"] / "attempts"
                    attempts.mkdir(parents=True, exist_ok=True)
                    number = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()] + [0]) + 1
                    attempt = attempts / f"{number:04d}"
                    attempt.mkdir()
                    command = child_command(args, job, attempt)
                    log = args.state_dir / "logs" / f'{campaign}_{phase}_{job["arm"]}_{job["length"]}_{number:04d}.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    update(status="waiting_for_gpu_or_running", campaign=campaign, phase=phase,
                           active_job=job["id"], active_attempt=str(attempt), active_log=str(log), command=command,
                           child_thread_environment=THREAD_ENV)
                    with log.open("w", encoding="utf-8") as stream:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            env=child_environment(),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        update(child_pid=child.pid)
                        while child.poll() is None:
                            time.sleep(15)
                            update(child_pid=child.pid)
                    if child.returncode != 0 or completed_attempt(args, job) != attempt:
                        raise RuntimeError(f"Serving job failed or has incomplete output: {job['id']}; see {log}")
                state.setdefault("jobs", {})[job["id"]] = {"status": "complete", "cells": job["cells"], "result": str(attempt)}
                phase_rows += read_json(attempt / "summary.json")
                update(status="running", campaign=campaign, phase=phase, child_pid=None)
                save_json(phase_jobs[0]["folder"].parent / "summary.json", phase_rows)
            phase_folder = phase_jobs[0]["folder"].parent
            save_json(phase_folder / "COMPLETED.json", {"status": "complete", "cells": len(phase_rows),
                      "campaign": campaign, "phase": phase, "required_gpu": REQUIRED_GPU})
        state.setdefault("campaigns", {})[campaign] = {"status": "complete", "phases": args.phases,
            "cells": sum(j["cells"] for j in jobs)}
        update(status="running", campaign=campaign)
    update(status="completed", child_pid=None, active_job=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute after preflight; otherwise only prepare/check readiness")
    parser.add_argument("--campaign", choices=["base", "pub_lora", "all"], default="base")
    parser.add_argument("--phases", nargs="+", choices=["smoke", "full"], default=["smoke", "full"])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, default=Path("/srv/encbank/legacy_workspace/models/Qwen3-8B"))
    parser.add_argument("--context-file", type=Path, default=ROOT / "exp/data/pg19_essay.txt")
    parser.add_argument("--state-dir", type=Path, default=LOCAL / "bootstrap_serving")
    parser.add_argument("--result-root", type=Path, default=LOCAL / "serving_reuse")
    parser.add_argument("--adapter-ckpt", type=Path, default=HERE / "checkpoints/8b_j12_pub_4k/final")
    parser.add_argument("--adapter-completion-marker", type=Path)
    parser.add_argument("--gpu-idle-slack-gb", type=float, default=5.0)
    parser.add_argument("--gpu-wait-seconds", type=int, default=24*3600)
    parser.add_argument("--resume-after-pid", type=int,
                        help="During scheduler migration, first let the recorded active serving process finish")
    args = parser.parse_args(argv)
    if not math.isfinite(args.gpu_idle_slack_gb) or not 0 < args.gpu_idle_slack_gb <= 5:
        parser.error("--gpu-idle-slack-gb must be > 0 and <= 5 GiB; admission uses strict <")
    if args.phases != sorted(set(args.phases), key=("smoke", "full").index):
        parser.error("Phases must be unique and smoke must precede full")
    args.adapter_completion_marker = args.adapter_completion_marker or args.adapter_ckpt.parent / "TRANSFER_COMPLETE.json"
    with bootstrap_lock(args.state_dir):
        path = args.state_dir / "status.json"
        state = read_json(path) if path.exists() else {}
        state.update(pid=os.getpid(), requested_campaign=args.campaign, requested_phases=args.phases,
                     required_gpu=REQUIRED_GPU, result_root=str(args.result_root), remote_timing_allowed=False,
                     gpu_idle_slack_gib=args.gpu_idle_slack_gb, gpu_admission_comparison="strictly_less_than")
        def update(**values):
            state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            save_json(path, state)
        try:
            if args.resume_after_pid:
                if not args.run:
                    raise ValueError("--resume-after-pid requires --run")
                wait_for_existing_job(args.resume_after_pid, state, update)
            update(status="preflight", error=None)
            state["preflight"] = preflight(args)
            if not args.run:
                update(status="ready", gpu_experiment_started=False,
                       adapter_ready=args.adapter_completion_marker.is_file())
                print(json.dumps(state, indent=2), flush=True)
                return 0
            update(gpu_experiment_started=True)
            run_queue(args, state, update)
        except BaseException as exc:
            update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
