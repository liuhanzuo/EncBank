"""Additional strong-baseline comparisons on all six QA tasks and three RULER cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from build_run_plan import make_job, specifications, cell, path_at


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--baseline", choices=["trained_pub", "cacheblend16"], required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-format", choices=["peft", "flat"], default="peft")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-file", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline == "trained_pub" and not args.adapter:
        parser.error("--trained_pub requires an explicit --adapter")
    mode = "smoke" if args.smoke else "full"
    arm = "pub" if args.baseline == "trained_pub" else "cacheblend16"
    base = path_at(args.output_root, mode)
    specs = specifications(args.workspace_root, args.smoke)
    jobs = [make_job(args, "longbench", arm, spec, base) for spec in specs["longbench"]]
    tasks = ["niah_multikey_1"] if args.smoke else ["niah_single_2", "niah_multikey_1", "variable_tracking"]
    for task in tasks:
        count = 2 if args.smoke else 50
        spec = {"label": f"{task}_16k", "task": task, "length": "16k", "limit": count,
                "cells": [cell(task, range(count), "16k", "recall")]}
        jobs.append(make_job(args, "ruler", arm, spec, base))
    if args.adapter:
        for job in jobs:
            if job["benchmark"] == "longbench":
                job["argv"] += ["--adapter", args.adapter]
            elif args.adapter_format == "flat":
                job["argv"] += ["--adapter-pt", args.adapter, "--lora-fp32-accum"]
            else:
                job["argv"] += ["--adapter", args.adapter]
    plan = {"schema_version": 1, "cwd": args.workspace_root, "model": args.model,
        "state_dir": path_at(args.state_dir, mode), "output_root": base,
        "mode": mode, "arms": [arm], "baseline": args.baseline, "adapter": args.adapter,
        "adapter_format": args.adapter_format if args.adapter else None,
        "env": {"COMEM_REMOTE_QUEUE": "1", "PAPER_EVOLVE_ALLOW_8B": "1"},
        "jobs": jobs, "reused_results": [],
        "counts": {"jobs": len(jobs), "new_predictions": sum(job["expected_n"] for job in jobs),
                   "reusable_predictions": 0, "total_predictions": sum(job["expected_n"] for job in jobs)}}
    args.plan_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(args.plan_file), **plan["counts"]}))


if __name__ == "__main__":
    main()
