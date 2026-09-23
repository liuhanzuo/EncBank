"""One finite GPU1 formal job; verified smoke70 reuse +1680 new generations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from non_qwen_formal_driver import REMOTE, HERE, FORMAL, released_smoke, alive, write, now
from non_qwen_smoke_bootstrap import idle_enough

def complete():
    path = FORMAL / "non_qwen_FORMAL_COMPLETE.json"
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    return (r.get("status") == "complete" and r.get("formal_rows") == 1750 and r.get("verified_smoke_reused") == 70
            and r.get("formal_generations") == 1680 and r.get("completed_cells") == 20
            and len(list((FORMAL / "records").glob("*.json"))) == 1750)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--resume-failed", action="store_true")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["protocol"] == "non-qwen-smollm2-formal-v1" and plan["gpu"] == 1
    assert plan["expected_rows"] == 1750 and plan["smoke_reused"] == 70 and plan["new_generations"] == 1680
    assert plan["automatic_followup"] is False and Path(plan["result_dir"]) == FORMAL
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    out = Path(plan["bootstrap_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "bootstrap.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    assert not alive(prior.get("child_pid")), "prior formal child still alive"
    if prior.get("status") == "completed":
        assert complete()
        return 0
    if prior.get("status") == "failed" and not args.resume_failed:
        raise RuntimeError("inspect retained failure before an explicitly requested resume")
    if prior:
        history = out / "history"
        history.mkdir(exist_ok=True)
        number = 1
        while (history / f"status{number:04d}.json").exists():
            number += 1
        write(history / f"status{number:04d}.json", prior)
    cpuready = json.loads((FORMAL / "non_qwen_FORMAL_CPU_READY.json").read_text())
    assert cpuready["all_350_fixtures_validated"] and cpuready["all_70_smoke_predictions_redecoded_rescored"]
    state = {"protocol": plan["protocol"], "pid": os.getpid(), "status": "waiting", "gpu": 1,
             "plan": str(args.plan), "result_dir": str(FORMAL), "child_pid": None,
             "expected_rows": 1750, "smoke_reused": 70, "new_generations": 1680,
             "accuracy_only": True, "timing_eligible": False, "automatic_followup": False, "started_utc": now()}
    def save():
        state["updated_utc"] = now()
        write(out / "status.json", state)
    from remote_queue import gpu_idle
    save()
    try:
        while True:
            try:
                state["predecessor"] = released_smoke()
            except (AssertionError, FileNotFoundError, KeyError) as exc:
                state.update(reason="smoke_complete_receipts_and_all_historical_PIDs_exit", detail=str(exc))
                save()
                time.sleep(10)
                continue
            idle, reading = gpu_idle(1, 512)
            assert idle == idle_enough(reading)
            state.update(gpu_check=reading, reason="GPU1_idle_admission")
            save()
            if idle:
                released_smoke()
                idle2, reading2 = gpu_idle(1, 512)
                state["pre_dispatch_gpu_check"] = reading2
                if idle2:
                    break
            time.sleep(10)
        log_number = 1
        while (out / f"child{log_number:04d}.log").exists():
            log_number += 1
        logpath = out / f"child{log_number:04d}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        argv = [sys.executable, "-u", str(HERE / "non_qwen_formal_driver.py"), "--execute",
                "--fixtures", plan["fixtures"], "--out", str(FORMAL)]
        with logpath.open("x") as log:
            child = subprocess.Popen(argv, cwd=HERE.parents[1], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="running", child_pid=child.pid, child_log=str(logpath), reason=None, argv=argv)
            save()
            code = child.wait()
        state.update(child_exit_code=code, child_pid=None)
        if code != 0 or not complete():
            raise RuntimeError("formal run incomplete/failed; retain attempt/raw and verified records for explicit resume")
        state.update(status="completed", completed_rows=1750, completed_utc=now())
        save()
        return 0
    except Exception:
        state.update(status="failed", error=traceback.format_exc(), failed_utc=now())
        save()
        raise

if __name__ == "__main__":
    raise SystemExit(main())
