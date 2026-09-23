"""Start or inspect only the independent oracle-support accuracy continuation."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path("/data/liuhanzuo/comem_v2_20260908")
HERE = ROOT / "workspace/exp/comem_v2_benchmarks_20260908"


def main():
    import fcntl
    from oracle_remote_bootstrap import process_alive, read_json, validate_plans
    folder = ROOT / "outputs/bootstrap_oracle_support"
    folder.mkdir(parents=True, exist_ok=True)
    guard = (folder / "launch.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    smoke_path = HERE / "oracle_support_smoke_plan.json"
    full_path = HERE / "oracle_support_full_plan.json"
    validate_plans(read_json(smoke_path), read_json(full_path), read_json(HERE / "full_plan.json"))
    status = read_json(folder / "status.json")
    if status.get("status") == "completed":
        print(json.dumps({"name": "oracle_support", "status": "already_completed"}), flush=True)
        return 0
    for pid in [status.get("pid"), read_json(folder / "launch.json").get("pid")]:
        if process_alive(pid):
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"oracle_remote_bootstrap.py" not in command_line:
                raise RuntimeError(f"Recorded PID {pid} is live with an unexpected command; inspect")
            print(json.dumps({"name": "oracle_support", "status": "already_running",
                              "pid": pid, "phase": status.get("phase")}), flush=True)
            return 0
    if process_alive(status.get("queue_pid")):
        raise RuntimeError("Recorded oracle queue is still alive without its wrapper; inspect")
    for path in (smoke_path, full_path):
        plan = read_json(path)
        queue = read_json(Path(plan["state_dir"]) / "state.json")
        if process_alive(queue.get("queue_pid")):
            raise RuntimeError("An existing oracle phase coordinator is alive; do not duplicate")
        for job_id, job in queue.get("jobs", {}).items():
            if job.get("status") == "running" and process_alive(job.get("pid")):
                raise RuntimeError(f"Oracle model child {job_id} is still alive; do not duplicate")
    command = [sys.executable, "-u", str(HERE / "oracle_remote_bootstrap.py"),
        "--smoke-plan", str(smoke_path), "--full-plan", str(full_path),
        "--gpus", "1", "--status-dir", str(folder),
        "--after-bootstrap-status", str(ROOT / "outputs/bootstrap_ruler_matched16k/status.json"),
        "--after-full-summary", str(ROOT / "outputs/bootstrap_ruler_matched16k/full_summary.json"),
        "--after-queue-state", str(ROOT / "outputs/queue_ruler_matched16k/full/state.json")]
    with (folder / "bootstrap.log").open("a") as log, open(os.devnull) as devnull:
        process = subprocess.Popen(command, cwd=HERE, stdin=devnull, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                 "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ""})
    launched = {"name": "oracle_support", "status": "launched", "pid": process.pid,
                "gpu": 1, "launched_at": time.time(), "command": command,
                "log": str(folder / "bootstrap.log")}
    (folder / "launch.json").write_text(json.dumps(launched, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launched), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
