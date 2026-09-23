"""Bounded completed-B audit: small receipts plus the ten LoCoMo traces.

No model, tokenization, scorer, SCBench document read, A aggregation or dispatch.
The canonical 08:40 full report is retained unchanged.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import aggregate_capacity as agg
from capacity_timing_policy import exclusion_reason, unresolved_remeasurements

HERE = Path(__file__).resolve().parent
LOCAL = timezone(timedelta(hours=8))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def stamp(text):
    value = datetime.fromisoformat(text)
    return value.replace(tzinfo=LOCAL) if value.tzinfo is None else value


def main():
    start = time.perf_counter()
    report_path = HERE/"results/reports/capacity/capacity_summary.json"
    full = read(report_path)
    queue = read(HERE/"results/capacity_queue/status.json")
    campaign = read(HERE/"results/campaign/status.json")
    plans = [j for task in ("locomo", "scbench") for j in
             read(HERE/f"results/capacity_queue/{task}_plan.json") if not j["smoke"]]
    plan = {j["id"]: j for j in plans}
    require(len(plan) == len(plans) == 340 and sum(j["queries"] for j in plans) == 13800,
            "Canonical full scope differs")
    require(full["all_full_workloads_complete"] and full["status"] == "complete"
            and full["progress"]["valid_complete_jobs"] == 340
            and full["progress"]["unique_measured_outputs"] == 13800
            and full["progress"]["paired_capacity_points"] == 102,
            "Incomplete canonical report")
    require(queue["status"] == "complete" and queue["full_queries"] == 13800
            and queue["child_pid"] is None and queue["active_job"] is None,
            "Queue not naturally complete")
    require(campaign["status"] == "measurements_and_reports_complete_pending_paper_writeback"
            and campaign["phase_A_report"] == "2026-09-09T12:20:28.425122+00:00",
            "Campaign/A report inheritance differs")
    require(not unresolved_remeasurements(HERE/"results/capacity"), "Timing repair incomplete")
    accepted = {a["job_id"]: a for a in full["valid_attempts"]}
    require(set(accepted) == set(plan) and len(full["valid_attempts"]) == 340,
            "Accepted report scope differs from canonical jobs")
    intervals, gate_values, hardware, counts = [], [], set(), Counter()
    for key, result in accepted.items():
        path, expected = Path(result["path"]), plan[key]
        require(path.parent.parent == HERE/"results/capacity"/key, "Accepted path outside canonical job")
        require(exclusion_reason(path) is None, "Excluded attempt accepted")
        marker, config = read(path/"COMPLETED.json"), read(path/"config.json")
        require(marker["status"] == "complete" and marker["smoke"] is False
                and marker["timing_eligible"] is True, "Invalid complete receipt")
        require(marker["completed_queries"] == marker["expected_queries"] == marker["new_generations"]
                == expected["queries"] and marker["extra_reference_generations"] == 0,
                "Receipt denominator differs")
        for name in ("arm", "cohort", "dataset", "smoke", "fraction", "ids"):
            require(config[name] == expected[name], f"Canonical config mismatch: {name}")
        require(config["cache_budget_bytes"] == expected["budget_bytes_expected"], "Budget differs")
        require(config["model"] == agg.MODEL and config["dtype"] == "bfloat16"
                and config["j"] == 12 and config["topk"] == 12 and config["chunk_size"] == 512,
                "Model/read setup differs")
        contract = agg.hardware_contract(marker["hardware"])
        hardware.add(json.dumps(contract, sort_keys=True))
        gate = marker["hardware"]["gpu_admission"]
        gate_values.extend([gate["initial_used_gib"], gate["recheck_used_gib"]])
        admission, finish = stamp(gate["admitted_at"]), stamp(marker["finished_at"])
        require(finish >= admission, "Finish precedes admission")
        intervals.append({"job_id": key, "admission": admission.isoformat(), "finish": finish.isoformat()})
        counts[config["dataset"]] += expected["queries"]
        require(Path(queue["jobs"][key]["attempt"]) == path, "Final queue/report attempt differs")
    require(len(hardware) == 1, "Across-job hardware/software/thread provenance differs")
    intervals.sort(key=lambda x: x["admission"])
    for earlier, later in zip(intervals, intervals[1:]):
        require(stamp(earlier["finish"]) <= stamp(later["admission"]),
                f"Accepted jobs overlap: {earlier['job_id']} -> {later['job_id']}")
    require(stamp(queue["finished_at"]) <= stamp(full["generated_at"]),
            "Final aggregate ran before the GPU queue completed")

    # Only LoCoMo's 4.75 MB document fixture is parsed, never SCBench's 349 MB.
    docs, queries = agg.load_fixture("locomo", HERE/"fixtures")
    jobs = agg.expected_jobs("locomo", docs, queries, HERE/"fixtures")
    checked, summaries, compacts = {}, {}, {}
    for job in jobs:
        path = Path(accepted[job["id"]]["path"])
        item = agg.validate_attempt(path, job)
        summary = agg.summarize([item])
        require(summary == full["complete_job_summaries"][job["id"]], "LoCoMo summary/raw mismatch")
        checked[job["id"]] = item
        summaries[job["id"]] = summary
        compacts[job["id"]] = [(r["id"], r["generated_ids"], r["prediction"], r["score"])
                               for r in item["records"]]
    base_key = "locomo/full/locomo-00/100/j0"
    identity = []
    for fraction in (.25, .5, 1.):
        key = f"locomo/full/locomo-00/{int(fraction*100):03d}/prefix"
        for index, (prefix, j0) in enumerate(zip(compacts[key], compacts[base_key])):
            require(prefix == j0, f"Prefix/j0 raw output mismatch: {fraction} row {index}")
        identity.append({"comparison": f"prefix{int(fraction*100)} vs j0", "equal_raw_ID_text_score_records": 800})
    for arm in ("fix_all", "cacheblend16"):
        base = compacts[f"locomo/full/locomo-00/025/{arm}"]
        for fraction in (.5, 1.):
            other = compacts[f"locomo/full/locomo-00/{int(fraction*100):03d}/{arm}"]
            require(base == other, f"Across-budget raw output mismatch: {arm} {fraction}")
            identity.append({"comparison": f"{arm}25 vs {int(fraction*100)}", "equal_raw_ID_text_score_records": 800})
    for fraction in (.25, .5, 1.):
        group = {arm: checked[base_key if arm == "j0" else f"locomo/full/locomo-00/{int(100*fraction):03d}/{arm}"]
                 for arm in agg.ARMS}
        require(agg.paired(group) == next(x for x in full["capacity_points"]
                    if x["dataset"] == "locomo" and x["fraction"] == fraction)["summaries"],
                "Four-arm LoCoMo pair differs from canonical report")
    event50 = [r["stats"]["capacity_cache"]["events"] for r in checked["locomo/full/locomo-00/050/fix_all"]["records"]]
    event100 = [r["stats"]["capacity_cache"]["events"] for r in checked["locomo/full/locomo-00/100/fix_all"]["records"]]
    require(event50 == event100, "V2 50/100 cache event lists differ")
    v2old = HERE/"results/capacity/locomo/full/locomo-00/025/fix_all/attempts/0001"
    require(exclusion_reason(v2old) is not None, "Contaminated attempt not excluded")
    require(Path(accepted["locomo/full/locomo-00/025/fix_all"]["path"]).name == "0002", "Wrong V2 replacement")
    prefixold = HERE/"results/capacity/locomo/full/locomo-00/025/prefix/attempts/0001"
    require(not (prefixold/"COMPLETED.json").exists(), "Old prefix partial incorrectly completed")
    with (prefixold/"measurements.jsonl").open(encoding="utf-8") as stream:
        require(sum(bool(line.strip()) for line in stream) == 209, "Old prefix partial changed")
    require(Path(accepted["locomo/full/locomo-00/025/prefix"]["path"]).name == "0002", "Wrong fresh prefix")

    quality = []
    costs = []
    for arm in agg.ARMS:
        s = summaries[base_key if arm == "j0" else f"locomo/full/locomo-00/025/{arm}"]
        quality.append({"arm": arm, "n": s["n"], "quality_by_category": s["quality_by_task"],
                        "mean_generated_tokens": s["generated_tokens"]/s["n"]})
    for point in (p for p in full["capacity_points"] if p["dataset"] == "locomo"):
        for arm in agg.ARMS:
            if arm == "j0" and point["fraction"] != 1:
                continue
            s = point["summaries"][arm]
            costs.append({"arm": arm, "fraction": None if arm == "j0" else point["fraction"],
                          "absolute_budget_bytes": None if arm == "j0" else point["cache_budget_bytes"],
                          "n": s["n"], "actual_token_hit_fraction": s["actual_token_hit_fraction"],
                          "cumulative_prepare_plus_query_s": s["cumulative_prepare_plus_query_s"],
                          "ttft_mean_s": s["ttft_mean_s"], "ttft_p50_s": s["ttft_p50_s"], "ttft_p95_s": s["ttft_p95_s"],
                          "maximum_persistent_cpu_representation_bytes": s["maximum_persistent_cpu_representation_bytes"],
                          "maximum_total_gpu_allocated_bytes": s["maximum_total_gpu_allocated_bytes"],
                          "maximum_incremental_gpu_allocated_bytes": s["maximum_incremental_gpu_allocated_bytes"],
                          "vs_j0": point["vs_j0"].get(arm),
                          "fixed_document_prepare_s": s["fixed_document_prepare_s"]})
    result = {"status": "passed", "at_local": datetime.now(LOCAL).isoformat(),
              "audit_cpu_seconds": time.perf_counter()-start, "canonical_report": str(report_path),
              "canonical_report_generated_at": full["generated_at"],
              "complete_receipts": 340, "complete_queries": 13800,
              "by_dataset_outputs": dict(counts), "cohorts": 34, "matched_capacity_points": 102,
              "all_receipt_gates_threads_and_config_match": True,
              "admission_used_gib_range": [min(gate_values), max(gate_values)],
              "accepted_admission_finish_intervals_do_not_overlap": True,
              "all_formal_report_inputs_unchanged": True,
              "locomo_raw_records_revalidated": 8000, "locomo_output_identity_checks": identity,
              "v2_50_100_cache_event_check": {"equal_records": 800,
                  "events_per_trace": [sum(map(len, event50)), sum(map(len, event100))],
                  "exact_event_lists_equal": True, "excludes_timing_fields_and_budget_values": True},
              "locomo_quality": quality, "locomo_costs": costs,
              "timing_boundary": full["timing_boundary"], "memory_boundary": full["memory_boundary"],
              "statistics_boundary": full["statistics_boundary"],
              "validation_boundary": "340 small receipts/configs checked against canonical plans and selected attempts; 10 LoCoMo traces independently revalidated against existing fixtures and canonical summaries. No scorer/model rerun and no SCBench document parse. SCBench choice quality format diagnosis is delegated separately.",
              "intervals": intervals}
    destination = HERE/"heartbeat_capacity_20260910_0842_writeback.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "at_local", "audit_cpu_seconds", "complete_receipts",
                    "complete_queries", "locomo_raw_records_revalidated", "admission_used_gib_range")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
