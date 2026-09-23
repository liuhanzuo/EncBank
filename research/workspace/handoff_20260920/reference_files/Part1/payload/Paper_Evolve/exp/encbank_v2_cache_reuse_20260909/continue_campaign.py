"""Continue verified smoke -> measured capacity jobs and generate result reports."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import time
from launch_reuse import HERE, ROOT, bootstrap_lock, read_json, now
from run_reuse import save as save_json
from local_execution_pause import require_unpaused


def main():
    require_unpaused(HERE)
    folder = HERE/"results/campaign"
    with bootstrap_lock(folder):
        previous = read_json(folder/"status.json")
        state = {"status": "watching", "pid": os.getpid(), "started_at": now(),
                 "phase_A_full_expected": 6116, "phase_B_full_expected": 13800,
                 "old_20260908_campaign": "complete; never restarted by this continuation"}
        if (previous.get("phase_A_report")
                and read_json(HERE/"results/queue/status.json").get("status") == "complete"
                and (HERE/"results/locomo/aggregate/aggregate.json").is_file()):
            state["phase_A_report"] = previous["phase_A_report"]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1")
        python = str(ROOT/".venv/Scripts/python.exe")
        child = None
        handles = []
        def update(**kw):
            state.update(kw, updated_at=now())
            save_json(folder/"status.json", state)
        def report(script, key):
            if state.get(key):
                return True
            if not (HERE/script).exists():
                return False
            result = subprocess.run([python, "-X", "utf8", str(HERE/script)], cwd=ROOT, env=env,
                                    capture_output=True, text=True, encoding="utf-8")
            (folder/(key+".stdout.log")).write_text(result.stdout, encoding="utf-8")
            (folder/(key+".stderr.log")).write_text(result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{script} failed; inspect report logs")
            update(**{key: now()})
            return True
        try:
            for _ in range(5760):
                require_unpaused(HERE)
                a = read_json(HERE/"results/queue/status.json")
                b = read_json(HERE/"results/capacity_queue/status.json")
                if a.get("status") == "failed" or b.get("status") == "failed":
                    raise RuntimeError("A measured queue failed; inspect its actual logs, preserve completed attempts, repair before resuming")
                if child is not None and child.poll() not in (None, 0):
                    raise RuntimeError("Capacity continuation exited with an error")
                update(phase_A_status=a.get("status"), phase_B_status=b.get("status"),
                       phase_A_active=a.get("active_job"), phase_B_active=b.get("active_job"))
                a_complete = a.get("status") == "complete" and a.get("full_queries") == 6116
                if a_complete:
                    report("aggregate_reuse.py", "phase_A_report")
                smoke_done = (b.get("status") == "complete" and b.get("smoke_only") is True
                              and len(b.get("jobs", {})) == 12
                              and all(v.get("status") == "complete" for v in b["jobs"].values()))
                if smoke_done and child is None:
                    out = (folder/"capacity.full.stdout.log").open("w", encoding="utf-8")
                    err = (folder/"capacity.full.stderr.log").open("w", encoding="utf-8")
                    handles.extend((out, err))
                    command = [python, "-X", "utf8", "-u", str(HERE/"launch_capacity.py"), "--datasets", "locomo", "scbench", "--run"]
                    require_unpaused(HERE)
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    update(status="capacity_full_queued_after_smoke", capacity_launcher_pid=child.pid,
                           capacity_command=command, capacity_queue_policy="full capacity waits for all 40 phase A jobs")
                b_complete = b.get("status") == "complete" and b.get("full_queries") == 13800 and b.get("smoke_only") is False
                if a_complete and b_complete and report("aggregate_capacity.py", "phase_B_report"):
                    update(status="measurements_and_reports_complete_pending_paper_writeback", finished_at=now())
                    return 0
                time.sleep(30)
            raise RuntimeError("Campaign continuation exceeded 48 hours; inspect jobs before resuming")
        except BaseException as exc:
            update(status="failed", error=repr(exc), finished_at=now())
            raise
        finally:
            for handle in handles:
                handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
