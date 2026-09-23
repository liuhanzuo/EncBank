"""Build the bounded six-benchmark CoMem queue plan; never start model work."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
DEFAULT_ARMS = ("fix_all", "j0", "pub", "pub_sink", "cbos")
BENCHMARK_ORDER = ("longbench", "babilong", "locomo", "infinitebench", "longeval", "ruler")
LB_COUNTS = OrderedDict(qasper=200, hotpotqa=200, musique=200, narrativeqa=200,
                        **{"2wikimqa": 200}, multifieldqa_en=150)
BABI_TASKS = ("qa1", "qa2", "qa3", "qa5")
BABI_LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k")
NIAH_TASKS = ("niah_single_2", "niah_multikey_1")
LONGEVAL_LENGTHS = ("4k", "8k", "16k", "32k", "64k", "128k")
IB_COUNTS = OrderedDict(longbook_qa_eng=351, longbook_choice_eng=229)


def path_at(base, *parts):
    return str(PurePosixPath(str(base).replace("\\", "/"), *parts))


def cell(task, indices, length=None, metric="accuracy"):
    return {"key": task + ("/" + length if length else ""), "task": task,
            "length": length, "metric": metric, "indices": list(indices),
            "expected_n": len(indices)}


def locomo_categories(workspace):
    remote = Path(path_at(workspace, "exp/comem_v2_benchmarks_20260908/data/locomo/locomo10.json"))
    source = remote if remote.is_file() else HERE / "data/locomo/locomo10.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    categories = [int(qa["category"]) for conversation in data for qa in conversation["qa"]]
    expected = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
    if {c: categories.count(c) for c in expected} != expected or len(categories) != 1986:
        raise ValueError("LoCoMo source is not the expected full 1986-QA release")
    return categories


def specifications(workspace, smoke):
    """Each spec is one short task/cell or stride shard, preserving source IDs."""
    specs = {b: [] for b in BENCHMARK_ORDER}
    if smoke:
        specs["longeval"] = [{"label": "4k", "cells": [cell("lines", range(2), "4k")],
                               "length": "4k", "limit": 2}]
        specs["ruler"] = [{"label": "niah_single_2_4k", "cells": [cell("niah_single_2", range(2), "4k", "recall")],
                            "task": "niah_single_2", "length": "4k", "limit": 2}]
        specs["babilong"] = [{"label": "qa1_0k", "cells": [cell("qa1", range(2), "0k")],
                               "task": "qa1", "length": "0k", "limit": 2}]
    else:
        for length in LONGEVAL_LENGTHS:
            specs["longeval"].append({"label": length, "cells": [cell("lines", range(50), length)],
                                       "length": length, "limit": 50})
        missing = [(task, "4k") for task in (*NIAH_TASKS, "variable_tracking")]
        missing += [("variable_tracking", length) for length in ("8k", "64k")]
        missing += [(task, "128k") for task in (*NIAH_TASKS, "variable_tracking")]
        missing += [(task, "256k") for task in NIAH_TASKS]
        for task, length in missing:
            specs["ruler"].append({"label": task + "_" + length, "cells": [cell(task, range(50), length, "recall")],
                                    "task": task, "length": length, "limit": 50})
        # Visit every reasoning task before increasing the context length.
        for length in BABI_LENGTHS:
            for task in BABI_TASKS:
                specs["babilong"].append({"label": task + "_" + length, "cells": [cell(task, range(100), length)],
                                          "task": task, "length": length, "limit": 100})
    for task, count in list(LB_COUNTS.items())[:1 if smoke else None]:
        shards, limit = (1, 2) if smoke else (2, count)
        for shard in range(shards):
            specs["longbench"].append({"label": f"{task}_s{shard}of{shards}",
                "cells": [cell(task, range(limit)[shard::shards], metric="f1")],
                "task": task, "num_shards": shards, "shard_index": shard,
                "limit": 2 if smoke else -1})
    categories = locomo_categories(workspace)
    shards, limit = (1, 2) if smoke else (10, len(categories))
    for shard in range(shards):
        indices = list(range(limit))[shard::shards]
        cells = [cell("category_" + str(category), [i for i in indices if categories[i] == category],
                      metric="accuracy" if category == 5 else "f1") for category in range(1, 6)]
        specs["locomo"].append({"label": f"all_s{shard}of{shards}",
            "cells": [c for c in cells if c["expected_n"]], "num_shards": shards,
            "shard_index": shard, "limit": 2 if smoke else -1})
    for task, count in list(IB_COUNTS.items())[:1 if smoke else None]:
        shards, limit = (1, 2) if smoke else (4, count)
        for shard in range(shards):
            specs["infinitebench"].append({"label": f"{task}_s{shard}of{shards}",
                "cells": [cell(task, range(limit)[shard::shards], metric="f1" if "qa_" in task else "accuracy")],
                "task": task, "num_shards": shards, "shard_index": shard,
                "limit": 2 if smoke else -1})
    return specs


def reused_ruler(arms):
    entries = []
    def add(source, tasks, lengths, source_arms):
        selected = [arm for arm in arms if arm in source_arms]
        if selected:
            entries.append({"relative_path": source, "benchmark": "ruler", "arms": selected,
                "cells": [cell(task, range(50), length, "recall") for task in tasks for length in lengths]})
    core = ("pub", "pub_sink", "fix_all", "j0")
    for length in ("16k", "32k"):
        add(f"exp/results/s15_ruler_j12_{length}.json", NIAH_TASKS, [length], core)
    for length in ("8k", "64k"):
        add(f"exp/results/s15c_ruler_j12_{length}.json", NIAH_TASKS, [length], core)
    for length in ("16k", "32k"):
        add(f"exp/results/s15c_ruler_vt_{length}.json", ["variable_tracking"], [length], core)
    add("exp/results/s15e_ruler_cbos_niah.json", NIAH_TASKS, ["16k", "32k"], ["cbos"])
    add("exp/results/s15e_ruler_cbos_vt.json", ["variable_tracking"], ["16k", "32k"], ["cbos"])
    add("exp/comem_v2_benchmarks_20260908/results/ruler_cbos_8k_64k.json", NIAH_TASKS, ["8k", "64k"], ["cbos"])
    return entries


def make_job(args, benchmark, arm, spec, base):
    root = args.workspace_root
    entry_root = path_at(root, "exp/comem_v2_benchmarks_20260908")
    job_id = f"{benchmark}__{arm}__{spec['label']}"
    relative_output = path_at(benchmark, arm, spec["label"])
    output = path_at(base, relative_output)
    if benchmark == "ruler":
        relative_output += ".json"
        output += ".json"
        argv = [path_at(entry_root, "run_ruler_remote.py"), "--model", args.model,
            "--j", "12", "--arms", arm, "--tasks", spec["task"], "--lengths", spec["length"],
            "--n", str(spec["limit"]), "--topk", "12", "--selector", "auto",
            "--iter-hop-topk", "4", "--max-new-tokens", "48", "--seed", "42",
            "--essay", path_at(root, "exp/data/pg19_essay.txt"), "--check", "0",
            "--cap-gb", "23", "--need-gb", "22", "--out", output]
        layout = "ruler"
    elif benchmark in {"longbench", "locomo"}:
        argv = [path_at(entry_root, "official_qa_driver.py"), "--benchmark", benchmark,
            "--arm", arm, "--out", output, "--model", args.model, "--j", "12",
            "--selector", "bm25", "--topk", "12", "--chunk-size", "512",
            "--device", "cuda:0", "--max-samples", str(spec["limit"]),
            "--num-shards", str(spec["num_shards"]), "--shard-index", str(spec["shard_index"])]
        if benchmark == "longbench":
            argv += ["--tasks", spec["task"], "--data-dir", path_at(entry_root, "data/longbench")]
        else:
            argv += ["--locomo-data", path_at(entry_root, "data/locomo/locomo10.json")]
        layout = "official_qa"
    else:
        argv = [path_at(entry_root, "run_driver.py"), "--benchmark", benchmark,
            "--arm", arm, "--out", output, "--", "--model", args.model, "--j", "12",
            "--baseline", "none", "--selector", "bm25", "--topk", "4" if benchmark == "babilong" else "12",
            "--chunk_size", "512", "--sink_tokens", "bos", "--device", "cuda:0",
            "--dtype", "bfloat16", "--attn_impl", "sdpa"]
        if benchmark == "longeval":
            argv += ["--lengths", spec["length"], "--num_samples", str(spec["limit"]),
                     "--seed", "1234", "--max_new_tokens", "16"]
        elif benchmark == "babilong":
            argv += ["--tasks", spec["task"], "--lengths", spec["length"],
                     "--limit", str(spec["limit"]), "--max_new_tokens", "20"]
        else:
            argv += ["--tasks", spec["task"], "--data_dir", path_at(entry_root, "data/infinitebench"),
                     "--prompt_style", "yarn-mistral", "--max_samples", str(spec["limit"]),
                     "--num_shards", str(spec["num_shards"]), "--shard_index", str(spec["shard_index"])]
        layout = "wrapper"
    return {"id": job_id, "argv": argv, "depends_on": [], "benchmark": benchmark,
            "arm": arm, "output": output, "relative_output": relative_output,
            "layout": layout, "cells": spec["cells"],
            "expected_n": sum(c["expected_n"] for c in spec["cells"])}


def build_plan(args):
    arms = list(dict.fromkeys(args.arms.replace(",", " ").split()))
    if not arms or any(arm not in DEFAULT_ARMS for arm in arms):
        raise ValueError(f"--arms must select from {DEFAULT_ARMS}")
    mode = "smoke" if args.smoke else "full"
    base = path_at(args.output_root, mode)
    specs = specifications(args.workspace_root, args.smoke)
    priorities = [arm for arm in ("fix_all", "j0") if arm in arms]
    remaining = [arm for arm in arms if arm not in priorities]
    jobs, prior_ids = [], {b: [] for b in BENCHMARK_ORDER}
    # Each round exposes all six families. Large BABILong/LoCoMo runs do not
    # monopolize the queue before other benchmarks receive their first samples.
    for phase_arms in (priorities, remaining):
        for index in range(max(map(len, specs.values()))):
            for arm in phase_arms:
                for benchmark in BENCHMARK_ORDER:
                    if index >= len(specs[benchmark]):
                        continue
                    job = make_job(args, benchmark, arm, specs[benchmark][index], base)
                    if arm in priorities:
                        prior_ids[benchmark].append(job["id"])
                    else:
                        job["depends_on"] = list(prior_ids[benchmark])
                    jobs.append(job)
    # The visibility ablation uses exactly the natural-QA protocol and packs.
    # It has no reusable RULER rows and is not silently treated as a full sixth arm.
    if args.with_fix_none:
        for spec in specs["longbench"]:
            if spec["task"] in {"qasper", "hotpotqa"}:
                jobs.append(make_job(args, "longbench", "fix_none", spec, base))
    selected = set(args.benchmarks.replace(",", " ").split()) if args.benchmarks else set(BENCHMARK_ORDER)
    if not selected.issubset(BENCHMARK_ORDER):
        raise ValueError(f"Unknown benchmarks: {selected - set(BENCHMARK_ORDER)}")
    jobs = [job for job in jobs if job["benchmark"] in selected]
    reuse = [] if args.smoke or "ruler" not in selected else reused_ruler(arms)
    reuse_n = sum(sum(c["expected_n"] for c in row["cells"]) * len(row["arms"]) for row in reuse)
    plan = {"schema_version": 1, "cwd": str(args.workspace_root).replace("\\", "/"),
        "workspace_root": str(args.workspace_root).replace("\\", "/"), "model": args.model,
        "state_dir": path_at(args.state_dir, mode), "output_root": base,
        "mode": mode, "arms": arms, "env": {"COMEM_REMOTE_QUEUE": "1", "PAPER_EVOLVE_ALLOW_8B": "1"},
        "jobs": jobs, "reused_results": reuse,
        "counts": {"jobs": len(jobs), "new_predictions": sum(j["expected_n"] for j in jobs),
                   "reusable_predictions": reuse_n,
                   "total_predictions": sum(j["expected_n"] for j in jobs) + reuse_n},
        "notes": ["Smoke and full runs use separate output/state subdirectories.",
                  "All torch devices are cuda:0 inside a queue-assigned CUDA_VISIBLE_DEVICES.",
                  "Reused RULER results remain local; summarize with --reuse-root pointing at the original workspace.",
                  "No trained baseline dependencies are added by this builder."]}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--with-fix-none", action="store_true")
    parser.add_argument("--benchmarks", default="", help="Optional comma-separated family subset")
    parser.add_argument("--plan-file", type=Path, help="Otherwise print JSON to stdout")
    args = parser.parse_args()
    plan = build_plan(args)
    serialized = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.plan_file:
        args.plan_file.parent.mkdir(parents=True, exist_ok=True)
        args.plan_file.write_text(serialized, encoding="utf-8")
        print(json.dumps({"plan": str(args.plan_file), **plan["counts"]}))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
