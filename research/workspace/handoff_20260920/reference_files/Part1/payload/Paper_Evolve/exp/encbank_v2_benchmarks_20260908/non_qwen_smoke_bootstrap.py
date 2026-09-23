"""One finite GPU1 smoke after BABI16 completes/exits. No formal follow-on job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from non_qwen_smoke_driver import REMOTE, HERE, alive, predecessor_complete, write, now
from non_qwen_download import ALLOWED_ROOT

def completion_ok(path):
    marker = path / "non_qwen_SMOKE_COMPLETE.json"
    if not marker.exists():
        return False
    result = json.loads(marker.read_text())
    return (result.get("status") == "complete" and result.get("actual_rows") == 70
            and result.get("actual_extra_references") == 4 and result.get("all_raw_reference_ids_equal") is True
            and result.get("all_generated_step_logits_passed") is True
            and len(list((path / "records").glob("*.json"))) == 70
            and len(list((path / "references").glob("*.json"))) == 4)

def idle_enough(reading):
    return (reading["memory_mib"] <= 512 and reading["utilization"] <= 5
            and reading["compute_pids"] == [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["protocol"] == "non-qwen-smollm2-finite-smoke-v1"
    assert plan["gpu"] == 1 and plan["target_generations"] == 70 and plan["extra_references"] == 4
    assert plan["formal_benchmark_auto_start"] is False
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "bootstrap is CPU only"
    assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    out = Path(plan["bootstrap_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "bootstrap.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statuspath = out / "status.json"
    prior = json.loads(statuspath.read_text()) if statuspath.exists() else {}
    if alive(prior.get("child_pid")):
        raise RuntimeError("an existing smoke child is alive")
    if prior.get("status") == "completed":
        assert completion_ok(Path(prior["result_dir"]))
        return 0
    if prior.get("status") == "failed":
        raise RuntimeError("failed smoke is retained; inspect before explicitly choosing a new bootstrap attempt")
    resultdir = Path(plan["result_dir"])
    assert not resultdir.exists(), "preserve real smoke attempts"
    cpuready = json.loads(Path(plan["cpu_ready"]).read_text())
    assert cpuready["ready_for_root_coordinated_smoke"] and cpuready["all_350_fixtures_validated"]
    assert cpuready["native_gates_first"] and cpuready["all_generated_step_logit_gate_required"]
    assert cpuready["driver_entry_version"] == "native-first-all-step-logits-v2"
    state = {"protocol": plan["protocol"], "pid": os.getpid(), "status": "waiting", "gpu": 1,
             "plan": str(args.plan), "result_dir": str(resultdir), "child_pid": None,
             "target_generations": 70, "extra_references": 4, "accuracy_only": True,
             "formal_benchmark_auto_start": False, "started_utc": now()}
    def save():
        state["updated_utc"] = now()
        write(statuspath, state)
    from remote_queue import gpu_idle
    save()
    try:
        while True:
            try:
                predecessor = predecessor_complete()
            except (AssertionError, FileNotFoundError, KeyError) as exc:
                state.update(status="waiting", reason="BABI16_receipts_or_live_processes", detail=str(exc))
                save()
                time.sleep(10)
                continue
            idle, reading = gpu_idle(1, 512)
            assert idle == idle_enough(reading)
            state.update(predecessor=predecessor, gpu_check=reading, reason="GPU1_idle_admission")
            save()
            if idle:
                # Fresh reread immediately before dispatch; child also checks the
                # complete predecessor and locks/rechecks GPU1 before loading.
                predecessor_complete()
                idle2, reading2 = gpu_idle(1, 512)
                state["pre_dispatch_gpu_check"] = reading2
                if idle2:
                    break
            time.sleep(10)
        argv = [sys.executable, "-u", str(HERE / "non_qwen_smoke_driver.py"), "--execute-gpu-smoke",
                "--model-dir", str(ALLOWED_ROOT), "--fixtures", plan["fixtures"], "--out", str(resultdir)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with (out / "child.log").open("x") as log:
            child = subprocess.Popen(argv, cwd=HERE.parents[1], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="running", child_pid=child.pid, reason=None, argv=argv)
            save()
            code = child.wait()
        state.update(child_exit_code=code, child_pid=None)
        if code != 0 or not completion_ok(resultdir):
            raise RuntimeError("real-weight smoke failed/incomplete; preserve evidence and exit")
        state.update(status="completed", completed_generations=70, completed_extra_references=4,
                     completed_utc=now(), after_completion="exit; no formal benchmark auto-start")
        save()
        return 0
    except Exception:
        state.update(status="failed", error=traceback.format_exc(), failed_utc=now())
        save()
        raise

if __name__ == "__main__":
    raise SystemExit(main())
