"""Lightweight stdlib-only audit/rescore and complete-task QA cost writeback."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import string
import time

from qa_cost_runner import HERE, VERSION, TIME_FIELDS, load_job, read_json, save_json, summarize, utc_now, validate_result
from qa_cost_bootstrap import completed_attempt


def official_scorer():
    path = HERE / "protocol_sources/longbench/metrics.py"
    names = {"normalize_answer", "f1_score", "qa_f1_score"}
    functions = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["qa_f1_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "heartbeat_qa_timing_20260908_2304_writeback.json")
    args = parser.parse_args()
    started = time.perf_counter()
    score_fn = official_scorer()
    result = {"protocol": VERSION, "audited_at": utc_now(), "timing_device": "local RTX 5090",
        "aggregation": "Each sample has three raw repetitions; take the within-sample median of each cost field, then mean across all 200 samples. F1 is mean actual locally generated official F1 over all repetitions; outputs are separately checked for repeat equality.",
        "peak_definition": "Maximum synchronized Torch max_memory_allocated across all 600 raw repetitions, converted by 2**30 to GiB; resident model included; no outlier removal.",
        "timing_boundary": "Memory-resident official marked prompt -> chat template/tokenizer/BM25 -> input H2D -> selected-chunk capture/write fresh per query -> native lower-query/upper-pack prefills -> natural-EOS decode -> output text decode. Excludes source-file IO, model load, explicit warmup, equality validation, official scoring and output-file serialization.",
        "not_write_once": True, "not_steady_cached_read": True, "fixed_generation_length": False,
        "local_accuracy_paired_to_local_timing": True, "source_file_io_in_total": False,
        "task_caps": {"qasper": 128, "hotpotqa": 32}, "complete_tasks": {}, "complete_unpaired_jobs": {},
        "pending_jobs": [], "known_limits": [
            "Sequential method jobs, not randomized or interleaved repetitions; shared machine activity can affect wall-time tails.",
            "Phase/decode-forward hooks synchronize CUDA. These are consistently instrumented wall times, not an optimized production serving benchmark.",
            "Different methods may produce different natural answer lengths; separate actual token counts/forward calls and local F1 are required for interpreting decode/total costs.",
            "Local/remote output differences are observations; their cause has not been established and remote F1 must not be substituted for local scored outputs.",
            "Peak allocated memory includes the resident model. Live selected read-state logical bytes are not persistent serialized storage.",
            "The selected chunks are recaptured for every query; the whole-document write-once reuse experiment is separate."]}
    all_jobs, raw_by_task = {}, {}
    for task in ("qasper", "hotpotqa"):
        rows_by_arm = {}
        for arm in ("fix_all", "j0"):
            job_id = f"qa_cost/full/{task}/{arm}"
            job, inputs, config = load_job(HERE / "results/protocol/qa_cost/task_plan.json", job_id)
            folder = HERE / "results/local/qa_cost/full" / task / arm
            attempt = completed_attempt(folder, config, inputs)
            if attempt is None:
                result["pending_jobs"].append(job_id)
                continue
            rows = [json.loads(s) for s in (attempt / "measurements.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
            assert len(rows) == 600
            group = defaultdict(list)
            for row in rows:
                source = inputs[row["index"]]
                validate_result(row, source, config)
                actual_score = max(score_fn(row["prediction"], str(answer)) for answer in source["answers"])
                assert math.isclose(actual_score, row["score"], rel_tol=0, abs_tol=1e-14)
                assert row["scored_prediction"] == row["prediction"]
                assert row["input_contract"] == config["input"]
                assert row["timings"]["total_from_marked_prompt_s"] == row["timings"]["cpu_prompt_tokenize_bm25_s"] + row["timings"]["h2d_s"] + row["timings"]["native_adapter_generation_s"]
                group[row["index"]].append(row)
            assert set(group) == set(range(200))
            assert all({r["repetition"] for r in rows_for_sample} == {0, 1, 2} for rows_for_sample in group.values())
            summary = summarize(rows, config)
            assert summary == read_json(attempt / "summary.json")
            for_mem = lambda name: [r["memory"][name] for r in rows]
            output_means = [statistics.mean(r["generated_tokens"] for r in g) for g in group.values()]
            forward_means = [statistics.mean(r["actual_decode_forward_calls"] for r in g) for g in group.values()]
            changed = sorted(i for i, g in group.items() if any(not r["historical_prediction_equal"] for r in g))
            score_changed = sorted(i for i, g in group.items() if any(not math.isclose(r["score"], r["historical_score"], rel_tol=0, abs_tol=1e-14) for r in g))
            metrics = {"job_id": job_id, "attempt": str(attempt), "unique_samples": 200, "repetitions_per_sample": 3,
                "raw_timed_repetitions": 600, "official_rescored_repetitions": 600,
                "local_f1_percent": statistics.mean(r["score"] for r in rows)*100,
                "historical_remote_f1_percent_for_disagreement_only": statistics.mean(inputs[i]["reference_outputs"][arm]["official_score"] for i in range(200))*100,
                "latency_s_sample_median_then_dataset_aggregates": summary["sample_median_latency_aggregates"],
                "peak_allocated_gib_max_all_repetitions": max(for_mem("peak_allocated_bytes"))/(2**30),
                "incremental_peak_allocated_gib_max_all_repetitions": max(for_mem("incremental_peak_allocated_bytes"))/(2**30),
                "peak_reserved_gib_max_all_repetitions": max(for_mem("peak_reserved_bytes"))/(2**30),
                "resident_model_baseline_gib_range": [min(for_mem("resident_model_baseline_allocated_bytes"))/(2**30), max(for_mem("resident_model_baseline_allocated_bytes"))/(2**30)],
                "generated_returned_token_mean_per_sample": statistics.mean(output_means),
                "generated_returned_token_range_all_repetitions": [min(r["generated_tokens"] for r in rows), max(r["generated_tokens"] for r in rows)],
                "decode_forward_calls_mean_per_sample": statistics.mean(forward_means),
                "cap_exhausted_repetitions": sum(not r["terminated_by_eos"] for r in rows),
                "cap_exhausted_samples": sum(any(not r["terminated_by_eos"] for r in g) for g in group.values()),
                "eos_terminated_repetitions": sum(r["terminated_by_eos"] for r in rows),
                "output_repeat_disagreement_samples": summary["output_repeat_disagreement_samples"],
                "local_remote_text_disagreement_samples": len(changed), "local_remote_text_disagreement_indices": changed,
                "local_remote_f1_disagreement_samples": len(score_changed), "local_remote_f1_disagreement_indices": score_changed,
                "max_read_pack_tokens": max(r["pack"]["read_pack_tokens"] for r in rows),
                "min_read_pack_tokens": min(r["pack"]["read_pack_tokens"] for r in rows),
                "raw_max_total_from_marked_prompt_s": max(r["timings"]["total_from_marked_prompt_s"] for r in rows),
                "raw_max_decode_s": max(r["timings"]["decode_s"] for r in rows),
                "gate_admissions": list({json.dumps(r["hardware"]["gpu_admission"], sort_keys=True): r["hardware"]["gpu_admission"] for r in rows}.values()),
                "marker": read_json(attempt / "COMPLETED.json"), "outlier_exclusions": 0}
            means = {k: v["mean"] for k, v in metrics["latency_s_sample_median_then_dataset_aggregates"].items()}
            metrics["main_table_ready"] = {"local_f1_percent": metrics["local_f1_percent"],
                "selected_write_ms": means["selected_capture_write_s"]*1000,
                "upper_prefill_ms": means["upper_pack_prefill_s"]*1000,
                "ttft_ms": means["ttft_from_marked_prompt_s"]*1000,
                "total_s": means["total_from_marked_prompt_s"],
                "peak_gib": metrics["peak_allocated_gib_max_all_repetitions"]}
            all_jobs[job_id] = metrics
            rows_by_arm[arm] = {r["index"]: r for r in rows if r["repetition"] == 0}
        if set(rows_by_arm) == {"fix_all", "j0"}:
            for i in range(200):
                assert rows_by_arm["fix_all"][i]["id"] == rows_by_arm["j0"][i]["id"]
                assert rows_by_arm["fix_all"][i]["pack"] == rows_by_arm["j0"][i]["pack"]
            arms = {arm: all_jobs[f"qa_cost/full/{task}/{arm}"] for arm in ("fix_all", "j0")}
            a, b = arms["fix_all"]["main_table_ready"], arms["j0"]["main_table_ready"]
            result["complete_tasks"][task] = {"matched_ids_and_packs": True, "methods": arms,
                "v2_minus_j0_local_f1_pp": a["local_f1_percent"]-b["local_f1_percent"],
                "v2_over_j0_total_latency_ratio_of_dataset_means": a["total_s"]/b["total_s"],
                "v2_over_j0_ttft_ratio_of_dataset_means": a["ttft_ms"]/b["ttft_ms"],
                "v2_over_j0_task_peak_memory_ratio": a["peak_gib"]/b["peak_gib"]}
        else:
            for arm in rows_by_arm:
                result["complete_unpaired_jobs"][f"qa_cost/full/{task}/{arm}"] = all_jobs[f"qa_cost/full/{task}/{arm}"]
    result["complete_full_jobs"] = len(all_jobs)
    result["official_rescored_complete_repetitions"] = 600*len(all_jobs)
    result["audit_cpu_wall_s"] = time.perf_counter()-started
    result["audit_imported_torch_or_loaded_model"] = False
    save_json(args.out, result)
    print(json.dumps({"out": str(args.out), "complete_tasks": list(result["complete_tasks"]),
        "complete_full_jobs": result["complete_full_jobs"], "official_rescored_repetitions": result["official_rescored_complete_repetitions"],
        "audit_cpu_wall_s": result["audit_cpu_wall_s"],
        "table": {task: {arm: m["main_table_ready"] for arm, m in values["methods"].items()} for task, values in result["complete_tasks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
