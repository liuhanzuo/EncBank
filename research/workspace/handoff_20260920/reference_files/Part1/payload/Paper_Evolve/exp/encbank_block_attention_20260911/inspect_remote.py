"""Compact read-only inspection for the hourly continuation."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from remote_sparse_queue import ROOT, read_json, process_identity

def main():
    out = ROOT / "outputs/sparse_encbank_20260911"
    queue = read_json(out / "queue.json")
    actual = process_identity(queue.get("pid")) if isinstance(queue.get("pid"), int) else None
    jobs = {}
    for name, job in queue.get("jobs", {}).items():
        status = read_json(Path(job["out"]) / "status.json")
        item = {"phase":job["phase"], "gpu":job.get("gpu"), "pid":job.get("pid"),
                "step":status.get("step"), "target_steps":job.get("steps"),
                "loss":status.get("loss"), "error":job.get("error") or status.get("error"),
                "output":job["out"]}
        if item["pid"]:
            item["process_alive"] = process_identity(item["pid"]) == job.get("process")
        jobs[name] = item
    gpus = [{"index":g.get("index"),"name":g.get("name"),"used_mib":g.get("used_mib"),
             "eligible":g.get("eligible"),"process_count":len(g.get("processes",[])),
             "inventory_error":g.get("inventory_error")} for g in queue.get("gpus",[])]
    result = {"observed_utc":datetime.now(timezone.utc).isoformat(),
              "queue_updated_utc":queue.get("updated_utc"), "reason":queue.get("reason"),
              "controller_pid":queue.get("pid"), "controller_alive":bool(actual and actual == queue.get("controller")),
              "allow_training":queue.get("allow_training"),"complete":queue.get("complete"),
              "gpus":gpus,"jobs":jobs}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()

