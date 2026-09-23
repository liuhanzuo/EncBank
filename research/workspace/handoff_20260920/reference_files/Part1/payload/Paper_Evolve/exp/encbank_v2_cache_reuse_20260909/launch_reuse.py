"""Durable sequential queue for the authorized real-query cache benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / "exp/encbank_v2_benchmarks_20260908"
sys.path.insert(0, str(OLD))
from serving_local_bootstrap import bootstrap_lock, read_json, save_json
ARMS = ("j0", "prefix", "fix_all", "cacheblend16")


def now():
    return datetime.now(timezone.utc).isoformat()


def valid_complete(folder, job, model="/srv/encbank/legacy_workspace/models/Qwen3-8B"):
    for attempt in sorted((folder / "attempts").glob("*"), reverse=True):
        if not (attempt / "COMPLETED.json").exists():
            continue
        state = read_json(attempt / "COMPLETED.json")
        cfg = read_json(attempt / "config.json")
        hw = state.get("hardware", {})
        admission = hw.get("gpu_admission", {})
        if (state.get("status") == "complete" and cfg.get("arm") == job["arm"]
            and cfg.get("document") == job["document"] and cfg.get("smoke") == job["smoke"]
            and cfg.get("protocol") == "real-question-reuse-v1" and cfg.get("dataset") == job["dataset"]
            and cfg.get("model") == model and cfg.get("ids") == job["ids"]
            and cfg.get("cache_budget_bytes") == job["budget_bytes_expected"]
            and cfg.get("dtype") == "bfloat16" and cfg.get("j") == 12
            and cfg.get("query_order_seed") == 20260909 and cfg.get("chunk_size") == 512 and cfg.get("topk") == 12
            and state.get("completed_queries") == job["queries"]
            and hw.get("device_name") == "NVIDIA GeForce RTX 5090"
            and hw.get("platform") == "Windows" and hw.get("torch_cpu_threads") == 2 and hw.get("torch_interop_threads") == 16
            and admission.get("initial_used_gib", 99) < 5 and admission.get("recheck_used_gib", 99) < 5
            and (not job["smoke"] or state.get("reference_checks_passed"))):
            return attempt, state
    return None, None


def make_jobs(dataset, documents, queries):
    # Fixed data-independent method rotation avoids always timing V2 last.
    rng = random.Random(20260909)
    docs = list(documents.values()) if isinstance(documents, dict) else list(documents)
    smoke_docs = docs[:2]
    jobs = []
    for smoke, chosen in ((True, smoke_docs), (False, docs)):
        for doc in chosen:
            arms = list(ARMS)
            rng.shuffle(arms)
            query_rows = sorted((r for r in queries if r["document_id"] == doc["document_id"]), key=lambda r: r["stream_index"])
            count = len(query_rows)
            for arm in arms:
                jobs.append({"id": f"{dataset}/{'smoke' if smoke else 'full'}/{doc['document_id']}/{arm}",
                    "dataset": dataset, "document": doc["document_id"], "arm": arm, "smoke": smoke,
                    "queries": min(3, count) if smoke else count,
                    "ids": [r["id"] for r in (query_rows[:3] if smoke else query_rows)],
                    "budget_bytes_expected": (doc["context_tokens"]+1)*147456})
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=("locomo",), default=["locomo"],
                   help="Phase A full-store CPU jobs are admitted for LoCoMo; SCBench requires the finite-capacity runner")
    p.add_argument("--python", default=str(ROOT / ".venv/Scripts/python.exe"))
    p.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    p.add_argument("--run", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args(argv)
    folder = HERE / "results/queue"
    with bootstrap_lock(folder):
        state = {"status": "preparing", "pid": os.getpid(), "started_at": now(), "jobs": {},
                 "datasets": args.datasets, "gpu_policy": "local RTX5090 only, original double <5 GiB gate",
                 "answer_cache": False, "phase_B_capacity": "pending implementation; not included in phase A completion"}
        def update(**changes):
            state.update(changes, updated_at=now())
            save_json(folder / "status.json", state)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1")
        for dataset in args.datasets:
            fixtures = HERE / "fixtures" / "scbench" if dataset == "scbench" else HERE / "fixtures"
            waited = 0
            while not (fixtures / "manifest.json").exists():
                update(status="waiting_for_fixtures", active_dataset=dataset, fixture_path=str(fixtures))
                if not args.run:
                    return 0
                if waited >= 86400:
                    raise RuntimeError("Prepared fixtures did not arrive within 24 hours")
                time.sleep(30)
                waited += 30
            # Import CPU-only workload utilities in a separate interpreter to keep
            # the queue's own process lightweight and GPU-free.
            code = "import json; from run_reuse import load_bundle; _,m,d,q=load_bundle(" + repr(dataset) + "," + repr(str(fixtures)) + "); print(json.dumps({'documents':[{'document_id':r['document_id'],'context_tokens':len(r['context_ids'])} for r in d],'queries':[{'id':r['id'],'stream_index':r['stream_index'],'document_id':r['document_id']} for r in q]}))"
            probe = subprocess.run([args.python, "-X", "utf8", "-c", code], cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8")
            if probe.returncode:
                update(status="failed", error=probe.stderr)
                raise RuntimeError(probe.stderr)
            payload = json.loads(probe.stdout)
            jobs = make_jobs(dataset, payload["documents"], payload["queries"])
            if args.smoke_only:
                jobs = [j for j in jobs if j["smoke"]]
            save_json(folder / f"{dataset}_plan.json", jobs)
            for job in jobs:
                target = HERE / "results" / job["id"]
                completed, receipt = valid_complete(target, job, args.model)
                if completed:
                    state["jobs"][job["id"]] = {"status": "complete", "attempt": str(completed), "queries": job["queries"], "smoke": job["smoke"]}
                    update()
                    continue
                if not args.run:
                    state["jobs"][job["id"]] = {"status": "pending", "queries": job["queries"], "smoke": job["smoke"]}
                    continue
                attempts = target / "attempts"
                numbers = [int(d.name) for d in attempts.glob("*") if d.is_dir() and d.name.isdigit()]
                attempt = attempts / f"{max(numbers, default=0)+1:04d}"
                attempt.mkdir(parents=True, exist_ok=False)
                command = [args.python, "-X", "utf8", "-u", str(HERE / "run_reuse.py"), "--dataset", dataset,
                           "--fixtures", str(fixtures), "--document", job["document"], "--arm", job["arm"],
                           "--model", args.model, "--out", str(attempt)]
                if job["smoke"]:
                    command.append("--smoke")
                with (attempt / "stdout.log").open("w", encoding="utf-8") as out, (attempt / "stderr.log").open("w", encoding="utf-8") as err:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
                    state["jobs"][job["id"]] = {"status": "running", "attempt": str(attempt), "queries": job["queries"], "smoke": job["smoke"]}
                    update(status="running", active_job=job["id"], child_pid=child.pid, active_attempt=str(attempt), command=command)
                    while child.poll() is None:
                        time.sleep(15)
                        if (attempt / "status.json").exists():
                            try:
                                snapshot = read_json(attempt / "status.json")
                                state["jobs"][job["id"]].update(status=snapshot["status"], completed_queries=snapshot["completed_queries"])
                                update()
                            except (OSError, ValueError):
                                pass
                completed, receipt = valid_complete(target, job, args.model)
                if child.returncode or not completed:
                    update(status="failed", exit_code=child.returncode, error="Measured child failed or did not produce a valid completion receipt", child_pid=None)
                    return 1
                state["jobs"][job["id"]] = {"status": "complete", "attempt": str(completed), "queries": job["queries"], "smoke": job["smoke"]}
                update(child_pid=None)
        update(status="complete" if args.run else "ready", active_job=None, child_pid=None, finished_at=now(),
               full_queries=sum(v["queries"] for v in state["jobs"].values() if v["status"] == "complete" and not v["smoke"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
