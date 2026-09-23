"""Validate and describe Phase B capacity traces, using only stdlib CPU code.

Reconstruct the full protocol from fixtures, never infer its scope from a smoke
launcher plan. No model/scorer runs, pooling of partial attempts, significance
tests, or independent-query speed confidence intervals are performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from capacity_timing_policy import exclusion_reason, unresolved_remeasurements, verify_replacement_config

HERE = Path(__file__).resolve().parent
ARMS = ("j0", "prefix", "fix_all", "cacheblend16")
FRACTIONS = (.25, .5, 1.0)
SEED = 20260909
VERSION = "real-question-capacity-v1"
MODEL = "/srv/encbank/legacy_workspace/models/Qwen3-8B"
BPT = 147456
METRICS = {"category_1": "official_LoCoMo_F1", "category_2": "official_LoCoMo_F1",
           "category_3": "official_LoCoMo_F1", "category_4": "official_LoCoMo_F1",
           "scbench_qa_eng": "official_case_insensitive_substring_accuracy",
           "scbench_choice_eng": "official_multiple_choice_accuracy"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def number(value, name, integer=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{name}: invalid nonnegative number")
    require(not integer or type(value) is int, f"{name}: not integer")
    return value


def close(a, b, name):
    require(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7), f"{name}: {a} != {b}")


def normalized_path(path):
    return str(path).replace("\\", "/").rstrip("/").casefold()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def iter_json_array(path):
    """Read documents one at a time; SCBench has 24.6M raw token IDs."""
    decoder, buffer, end, first = json.JSONDecoder(), "", False, True
    with Path(path).open(encoding="utf-8-sig") as stream:
        while True:
            buffer = buffer.lstrip()
            if not buffer and not end:
                chunk = stream.read(1024*1024)
                buffer += chunk
                end = not chunk
                continue
            if first:
                require(buffer.startswith("["), "Expected document JSON array")
                buffer, first = buffer[1:], False
                continue
            if buffer.startswith("]"):
                require(not (buffer[1:]+stream.read()).strip(), "Trailing document data")
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            try:
                item, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                require(not end, "Truncated/invalid document array")
                chunk = stream.read(1024*1024)
                buffer += chunk
                end = not chunk
                continue
            yield item
            buffer = buffer[offset:]


def load_fixture(dataset, path):
    path = Path(path).resolve()
    manifest = read_json(path/"manifest.json")
    if dataset == "locomo":
        require(manifest["protocol"] == "locomo-real-multiquery-reuse-v1", "Wrong LoCoMo fixture")
        expected_docs, expected_queries = 10, 1529
    else:
        require(manifest["retrieval_adapted"] and manifest["complete_two_task_sources"], "Incomplete SCBench fixture")
        expected_docs, expected_queries = 127, 580
    # Raw IDs need not remain resident for aggregation: pack lengths are exact
    # functions of token count and ordered selected indices. The runner itself
    # asserts its encoded full context equals these IDs before model requests.
    docs = []
    for d in iter_json_array(path/"documents.json"):
        count = len(d["context_ids"])
        require(count == d["context_tokens"], "Fixture context size differs")
        docs.append({"document_id": d["document_id"], "context_tokens": count})
    queries = [json.loads(s) for s in (path/"queries.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    require(len(docs) == expected_docs and len(queries) == expected_queries, "Full fixture scope differs")
    require(len({d["document_id"] for d in docs}) == len(docs), "Duplicate document")
    require(len({q["id"] for q in queries}) == len(queries), "Duplicate query")
    dm = {d["document_id"]: d for d in docs}
    for q in queries:
        count = dm[q["document_id"]]["context_tokens"]
        selected = q["selected_indices"]
        require(len(selected) == len(set(selected)) and all(type(i) is int and 0 <= i < math.ceil(count/512) for i in selected), "Invalid fixture selected indices")
        require(selected == q["pack"]["selected_indices"], "Fixture pack differs")
        pack_tokens = 1+len(q["query_ids"])+sum(min(512, count-i*512) for i in selected)
        require(pack_tokens == q["pack"]["read_pack_tokens"], "Fixture raw-token pack length differs")
        require(q["sample"]["task"] in METRICS, "Unexpected score task")
    return docs, queries


def cohorts(dataset, documents):
    if dataset == "locomo":
        return {"locomo-00": documents}
    groups = {}
    for task in ("scbench_qa_eng", "scbench_choice_eng"):
        docs = [d for d in documents if d["document_id"].startswith(task+"/")]
        for start in range(0, len(docs), 4):
            groups[f"{task}-{start//4:02d}"] = docs[start:start+4]
    require(sum(map(len, groups.values())) == len(documents), "Unassigned cohort document")
    return groups


def trace(dataset, documents, queries):
    grouped = {d["document_id"]: sorted((q for q in queries if q["document_id"] == d["document_id"]),
                                      key=lambda q: q["stream_index"]) for d in documents}
    for key, rows in grouped.items():
        require([r["stream_index"] for r in rows] == list(range(len(rows))), "Fixture stream index differs")
        if dataset == "locomo":
            require(len(rows) >= 80, "LoCoMo needs 80 unique requests per document")
            grouped[key] = rows[:80]
    order = list(grouped)
    random.Random(SEED).shuffle(order)
    rows = [grouped[key][i] for i in range(max(map(len, grouped.values()))) for key in order if i < len(grouped[key])]
    require(len(rows) == len({r["id"] for r in rows}), "Repeated trace question")
    return rows


def working_set(documents, rows):
    used = {d["document_id"]: set() for d in documents}
    dm = {d["document_id"]: d for d in documents}
    for row in rows:
        used[row["document_id"]].update(row["selected_indices"])
    tokens = 1+sum(min(512, dm[key]["context_tokens"]-i*512) for key, indices in used.items() for i in indices)
    return {"scope": "predeclared trace referenced chunk union plus one shared sink; not the full source corpus",
            "full_depth_kv_bytes_per_token": BPT, "referenced_tokens_with_sink": tokens,
            "referenced_chunks": sum(map(len, used.values())), "full_depth_working_set_bytes": tokens*BPT,
            "whole_documents_full_depth_bytes": (1+sum(d["context_tokens"] for d in documents))*BPT}


def expected_jobs(dataset, documents, queries, fixture_path):
    jobs = []
    for cohort, docs in cohorts(dataset, documents).items():
        rows = trace(dataset, docs, queries)
        working = working_set(docs, rows)
        require(working["full_depth_working_set_bytes"] <= 32*1024**3, "Cohort exceeds protocol guard")
        for fraction in FRACTIONS:
            for arm in ARMS:
                if arm == "j0" and fraction != 1:
                    continue
                jobs.append({"id": f"{dataset}/full/{cohort}/{int(fraction*100):03d}/{arm}",
                             "dataset": dataset, "cohort": cohort, "fraction": fraction, "arm": arm,
                             "rows": rows, "documents": docs, "fixtures": str(Path(fixture_path).resolve()),
                             "working_set": working, "budget": int(working["full_depth_working_set_bytes"]*fraction)})
    return jobs


def hardware_contract(hw):
    require(hw["timing_eligible"] is True and hw["device_name"] == "NVIDIA GeForce RTX 5090"
            and hw["device"] == "cuda:0" and hw["platform"] == "Windows", "Wrong timing hardware")
    for key, expected in (("torch_cpu_threads", 2), ("torch_interop_threads", 16), ("omp_num_threads", "2"),
                          ("mkl_num_threads", "2"), ("tokenizers_parallelism", "false")):
        require(hw[key] == expected, f"Thread setting differs: {key}")
    gate = hw["gpu_admission"]
    threshold = number(gate["effective_idle_slack_gib"], "gate threshold")
    require(0 < threshold <= 5 and gate["comparison"] == "strictly_less_than", "Wrong admission policy")
    for key in ("initial_used_gib", "recheck_used_gib"):
        require(number(gate[key], key) < threshold, "GPU was not idle at both gates")
    require(gate["other_python_compute_processes"] == [], "Concurrent model admission")
    require(bool(hw.get("hostname")), "Missing hostname")
    return {k: hw.get(k) for k in ("hostname", "device", "device_name", "device_uuid", "platform", "torch",
            "transformers", "cuda_runtime", "torch_cpu_threads", "torch_interop_threads", "omp_num_threads",
            "mkl_num_threads", "tokenizers_parallelism")}


def validate_attempt(path, job, model=MODEL):
    path = Path(path)
    require(exclusion_reason(path) is None, exclusion_reason(path))
    marker, cfg, store = (read_json(path/name) for name in ("COMPLETED.json", "config.json", "store_cost.json"))
    verify_replacement_config(path, cfg)
    records = [json.loads(s) for s in (path/"measurements.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    require(marker["status"] == "complete" and cfg["protocol"] == VERSION, "Incomplete/wrong protocol")
    require(cfg["smoke"] is False and marker["smoke"] is False and marker["timing_eligible"] is True, "Smoke is not formal timing")
    for key in ("dataset", "cohort", "arm"):
        require(cfg[key] == marker[key] == job[key], f"Wrong {key}")
    require(cfg["fraction"] == job["fraction"] and cfg["cache_budget_bytes"] == marker["cache_budget_bytes"] == job["budget"], "Budget differs from fixed trace")
    require(cfg["working_set"] == job["working_set"], "Working-set bytes differ")
    require(cfg["ids"] == [q["id"] for q in job["rows"]], "Config input ID/order differs")
    require(cfg["documents"] == [d["document_id"] for d in job["documents"]], "Cohort documents differ")
    require(normalized_path(cfg["fixtures"]) == normalized_path(job["fixtures"]), "Fixture path differs")
    require(normalized_path(cfg["model"]) == normalized_path(model) and not cfg.get("adapter"), "Model/adapter differs")
    require(cfg["dtype"] == "bfloat16" and cfg["chunk_size"] == 512 and cfg["topk"] == 12 and cfg["j"] == 12, "Model/read protocol differs")
    require(cfg["query_order_seed"] == SEED and cfg["answer_cache"] is False and cfg["source_context_truncation"] == "none", "Input/generation protocol differs")
    require(cfg["j0_budget_independent"] == (job["arm"] == "j0"), "Wrong j0 budget contract")
    require(marker["expected_queries"] == marker["completed_queries"] == marker["new_generations"] == len(records) == len(job["rows"]), "Incomplete query denominator")
    require(marker["extra_reference_generations"] == 0, "Reference work in formal measurement")
    hw = hardware_contract(marker["hardware"])
    require(set(store["document_prepare_s"]) == set(cfg["documents"]), "Document preparation differs")
    for value in store["document_prepare_s"].values():
        number(value, "document prepare")
    close(number(store["fixed_cost_s"], "fixed cost"), sum(store["document_prepare_s"].values()), "B fixed cost boundary")
    require(store["initial_persistent_kv_bytes"] == 0 and store["serialized_kv_bytes"] == 0, "Not a cold CPU-only cache")
    require(store["raw_token_bytes"] == sum(d["context_tokens"]*8 for d in job["documents"]), "Raw token bytes differ")
    for i, (row, ref) in enumerate(zip(records, job["rows"])):
        require(row["id"] == ref["id"] and row["document_id"] == ref["document_id"] and row["arm"] == job["arm"], "Record input identity differs")
        require(row["trace_index"] == i and row["stream_index"] == ref["stream_index"] and row["task"] == ref["sample"]["task"], "Record order/task differs")
        require(row["timing_eligible"] is True and row["smoke_reference_equal"] is None, "Ineligible record")
        require(row["query_ids"] == ref["query_ids"] and row["query_tokens"] == len(ref["query_ids"]), "Raw query token IDs differ")
        require(row["selected_indices"] == ref["selected_indices"], "Ordered selected pack differs")
        require(row["max_new_tokens"] == ref["sample"]["max_new_tokens"], "Generation cap differs")
        ids = row["generated_ids"]
        require(type(ids) is list and all(type(t) is int and t >= 0 for t in ids), "Invalid generated IDs")
        require(0 < len(ids) == row["generated_tokens"] <= row["max_new_tokens"], "Invalid output count")
        require(type(row["terminated_by_eos"]) is bool and row["terminated_by_eos"] == (len(ids) < row["max_new_tokens"]), "EOS accounting differs")
        require(number(row["score"], "score") <= 1 and isinstance(row["prediction"], str) and isinstance(row["scored_prediction"], str), "Invalid scored output")
        s, t = row["stats"], row["timings"]
        require(s["document_id"] == row["document_id"] and s["selected_indices"] == ref["selected_indices"]
                and s["read_tokens"] == ref["pack"]["read_pack_tokens"] and s["query_tokens"] == len(ref["query_ids"]), "Measured raw-token pack differs")
        require(s["generated_tokens"] == len(ids) and s["fixed_generation_length"] is False
                and s["decode_steps"] == len(ids)-1+int(row["terminated_by_eos"]), "Measured decode accounting differs")
        for key in ("total_s", "ttft_s", "decode_s", "load_s", "transfer_s", "transfer_bytes", "cache_fill_s",
                    "cache_fill_transfer_bytes", "peak_allocated_bytes", "incremental_peak_bytes", "cache_resident_bytes",
                    "evicted_nodes", "evicted_bytes", "capture_calls", "cache_hit_context_tokens", "cache_miss_context_tokens"):
            number(s[key], key)
        context = s["read_tokens"]-s["query_tokens"]
        require(s["cache_hit_context_tokens"]+s["cache_miss_context_tokens"] == context, "Hit/miss denominator differs")
        require(s["cache_budget_bytes"] == job["budget"] and s["cache_resident_bytes"] <= job["budget"], "Resident cache exceeds shared budget")
        require(s["incremental_peak_bytes"] <= s["peak_allocated_bytes"], "Incremental GPU peak exceeds total peak")
        if job["arm"] == "j0":
            require(s["cache_lookup_applicable"] is False and s["cache_resident_bytes"] == s["cache_hit_context_tokens"] == s["capture_calls"] == 0, "j0 unexpectedly reuses persistent KV")
        else:
            require(s["cache_lookup_applicable"] is True, "Cache lookup disabled")
        if job["arm"] in ("fix_all", "cacheblend16"):
            cap = s["capacity_cache"]
            events = cap["events"]
            expected_events = [("sink", "__shared_sink__", -1, 1)]
            doc_tokens = next(d["context_tokens"] for d in job["documents"] if d["document_id"] == ref["document_id"])
            expected_events += [("chunk", ref["document_id"], index, min(512, doc_tokens-index*512)) for index in ref["selected_indices"]]
            require([(e["kind"], e["document_id"], e["chunk_index"], e["tokens"]) for e in events] == expected_events,
                    "Actual cache request events differ from selected pack")
            for event in events:
                require(event["occurrences"] == 1 and event["status"] in ("hit", "miss_admitted", "miss_bypass"), "Invalid cache event")
                number(event["tensor_bytes"], "actual cache tensor bytes", integer=True)
            require(sum(e["tokens"] for e in events if e["status"] == "hit") == s["cache_hit_context_tokens"], "Hit tokens differ from actual cache events")
            require(sum(e["tokens"] for e in events if e["kind"] == "sink" and e["status"] == "hit") == s["cache_hit_sink_tokens"], "Sink hit differs from actual cache event")
            require(sum(e["tensor_bytes"] for e in events if e["status"] != "hit") == cap["miss_tensor_bytes"], "Miss bytes differ from actual cache events")
            require(sum(e["tensor_bytes"] for e in events if e["status"] == "miss_bypass") == cap["bypass_tensor_bytes"], "Bypass bytes differ from actual cache events")
            require(cap["released"] is True and cap["persistent_bytes_after"] == s["cache_resident_bytes"], "Unreleased/inconsistent LRU lease")
            close(cap["miss_tensor_bytes"], s["cache_fill_transfer_bytes"], "D2H tensor payload bytes")
            close(cap["evicted_tensor_bytes"], s["evicted_bytes"], "Eviction bytes")
            close(cap["bypass_tensor_bytes"], s["cache_bypass_bytes"], "Bypass bytes")
            close(cap["build_s"]+cap["cpu_copy_s"], s["cache_fill_s"], "Cache fill timing")
            require(cap["build_calls"] == s["capture_calls"] == s["document_capture_calls"]+s["sink_capture_calls"], "Capture calls differ")
        for key in ("document_bind_s", "query_tokenization_s", "external_retrieval_s", "text_decode_s", "ttft_s", "query_total_s"):
            number(t[key], key)
        extra = t["document_bind_s"]+t["query_tokenization_s"]+t["external_retrieval_s"]
        close(t["ttft_s"], extra+s["ttft_s"], "B TTFT boundary")
        close(t["query_total_s"], extra+s["total_s"]+t["text_decode_s"], "B query total boundary")
        require(s["total_s"] >= s["ttft_s"] and t["query_total_s"] >= t["ttft_s"], "Total precedes first token")
    contract = {"model": normalized_path(cfg["model"]), "dtype": cfg["dtype"], "cohort": cfg["cohort"],
                "documents": cfg["documents"], "hardware": hw, "working_set": cfg["working_set"],
                "input_digest": digest([{k: q[k] for k in ("id", "document_id", "query_ids", "selected_indices", "sample")} for q in job["rows"]])}
    return {"job_id": job["id"], "path": str(path.resolve()), "arm": job["arm"], "dataset": job["dataset"],
            "cohort": job["cohort"], "fraction": job["fraction"], "budget_bytes": job["budget"], "contract": contract,
            "records": records, "store": store, "completed_at": marker.get("finished_at")}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values)-1)*p
    a, b = math.floor(x), math.ceil(x)
    return values[a] if a == b else values[a]*(b-x)+values[b]*(x-a)


def summarize(attempts):
    """Pool only caller-selected complete traces; retain actual denominators."""
    require(bool(attempts), "Cannot summarize no measurements")
    rows = [r for a in attempts for r in a["records"]]
    stats = [r["stats"] for r in rows]
    arm = attempts[0]["arm"]
    require(all(a["arm"] == arm for a in attempts), "Mixed arm summary")
    quality = {}
    for task in sorted({r["task"] for r in rows}):
        values = [r["score"] for r in rows if r["task"] == task]
        quality[task] = {"metric": METRICS[task], "n": len(values), "score_sum": sum(values),
                         "mean_score": statistics.mean(values), "score_percent": 100*statistics.mean(values)}
    total_context = sum(s["cache_hit_context_tokens"]+s["cache_miss_context_tokens"] for s in stats)
    sink_hit = sum(int(s["cache_hit_context_tokens"] > 0) if arm == "prefix" else s.get("cache_hit_sink_tokens", 0) for s in stats)
    hit = sum(s["cache_hit_context_tokens"] for s in stats)
    fixed = sum(a["store"]["fixed_cost_s"] for a in attempts)
    online = sum(r["timings"]["query_total_s"] for r in rows)
    ttft = [r["timings"]["ttft_s"] for r in rows]
    def optional_sum(key):
        return sum(s[key] for s in stats) if all(key in s for s in stats) else None
    def optional_max(key):
        return max(s[key] for s in stats) if all(key in s for s in stats) else None
    return {"n": len(rows), "cohorts": [a["cohort"] for a in attempts], "quality_by_task": quality,
        "fixed_document_prepare_s": fixed, "query_total_sum_s": online, "cumulative_prepare_plus_query_s": fixed+online,
        "ttft_mean_s": statistics.mean(ttft), "ttft_p50_s": percentile(ttft, .5), "ttft_p95_s": percentile(ttft, .95),
        "query_total_mean_s": online/len(rows), "generated_tokens": sum(r["generated_tokens"] for r in rows),
        "eos_terminated_queries": sum(r["terminated_by_eos"] for r in rows),
        "cap_exhausted_queries": sum(not r["terminated_by_eos"] for r in rows),
        "decode_forward_calls": sum(s["decode_steps"] for s in stats),
        "cache_lookup_applicable": arm != "j0", "context_tokens_requested_with_sink": total_context,
        "context_tokens_hit_with_sink": hit, "context_tokens_miss_with_sink": total_context-hit,
        "document_tokens_hit": hit-sink_hit, "document_tokens_miss": total_context-len(rows)-(hit-sink_hit),
        "sink_tokens_hit": sink_hit, "sink_tokens_miss": len(rows)-sink_hit,
        "actual_token_hit_fraction": hit/total_context if arm != "j0" else None,
        "cache_evicted_nodes": sum(s["evicted_nodes"] for s in stats), "cache_evicted_tensor_bytes": sum(s["evicted_bytes"] for s in stats),
        "cache_bypass_tokens": optional_sum("cache_bypass_tokens"), "cache_bypass_tensor_bytes": optional_sum("cache_bypass_bytes"),
        "capture_calls": sum(s["capture_calls"] for s in stats), "document_capture_calls": optional_sum("document_capture_calls"),
        "sink_capture_calls": optional_sum("sink_capture_calls"), "query_capture_calls": optional_sum("query_capture_calls"),
        "maximum_persistent_cpu_representation_bytes": max(s["cache_resident_bytes"] for s in stats),
        "final_persistent_cpu_bytes_by_cohort": {a["cohort"]: a["records"][-1]["stats"]["cache_resident_bytes"] for a in attempts},
        "maximum_active_cpu_payload_bytes": optional_max("cache_active_payload_bytes"),
        "maximum_total_gpu_allocated_bytes": max(s["peak_allocated_bytes"] for s in stats),
        "maximum_incremental_gpu_allocated_bytes": max(s["incremental_peak_bytes"] for s in stats),
        "raw_cpu_token_bytes_by_cohort": {a["cohort"]: a["store"]["raw_token_bytes"] for a in attempts},
        "h2d_bytes_including_raw_ids": sum(s["transfer_bytes"] for s in stats), "h2d_s": sum(s["transfer_s"] for s in stats),
        "cache_fill_d2h_tensor_bytes": sum(s["cache_fill_transfer_bytes"] for s in stats),
        "cache_fill_s_inclusive": sum(s["cache_fill_s"] for s in stats), "cache_capture_s_subphase": optional_sum("cache_capture_s"),
        "cache_build_s_subphase": optional_sum("cache_build_s"), "cache_cpu_copy_s_subphase": optional_sum("cache_cpu_copy_s"),
        "load_s_inclusive": sum(s["load_s"] for s in stats), "reader_wrapper_s": optional_sum("reader_wrapper_s"),
        "document_bind_s": sum(r["timings"]["document_bind_s"] for r in rows),
        "query_tokenization_s": sum(r["timings"]["query_tokenization_s"] for r in rows),
        "external_retrieval_s": sum(r["timings"]["external_retrieval_s"] for r in rows),
        "read_prefill_s": optional_sum("read_prefill_s"), "decode_s": sum(s["decode_s"] for s in stats),
        "text_decode_s": sum(r["timings"]["text_decode_s"] for r in rows)}


def paired(attempts):
    require(set(attempts) == set(ARMS), "Need all four arms for a capacity point")
    base = attempts["j0"]
    budgets = set()
    for arm, a in attempts.items():
        require(a["contract"] == base["contract"], f"Cross-arm input/model/hardware mismatch: {arm}")
        if arm != "j0":
            budgets.add(a["budget_bytes"])
    require(len(budgets) == 1, "Non-j0 absolute budgets differ")
    require(base["fraction"] == 1, "j0 must be the one measured 100% baseline")
    summaries = {arm: summarize([a]) for arm, a in attempts.items()}
    return summaries


def compare(summary, reference):
    require(summary["n"] == reference["n"] and summary["cohorts"] == reference["cohorts"], "Unpaired comparison denominator")
    diffs = {}
    for task, q in summary["quality_by_task"].items():
        r = reference["quality_by_task"][task]
        require(q["n"] == r["n"], "Task denominator differs")
        diffs[task] = 100*(q["mean_score"]-r["mean_score"])
    total, base = summary["cumulative_prepare_plus_query_s"], reference["cumulative_prepare_plus_query_s"]
    return {"quality_delta_percentage_points": diffs, "cumulative_seconds_delta": total-base,
            "cumulative_time_ratio_to_j0": total/base if base else None,
            "speedup_j0_over_arm": base/total if total else None,
            "ttft_mean_ratio_to_j0": summary["ttft_mean_s"]/reference["ttft_mean_s"] if reference["ttft_mean_s"] else None,
            "generated_tokens_delta": summary["generated_tokens"]-reference["generated_tokens"]}


def aggregate(jobs, results_root, model=MODEL):
    require(bool(jobs), "Empty expected campaign cannot be complete")
    valid, progress, rejected = {}, [], []
    for job in jobs:
        folder = Path(results_root)/job["id"]
        attempts = sorted((folder/"attempts").glob("*"), reverse=True)
        chosen = None
        for path in attempts:
            if not (path/"COMPLETED.json").exists():
                continue
            try:
                chosen = validate_attempt(path, job, model)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append({"job_id": job["id"], "attempt": str(path), "error": str(exc)})
                continue
            break
        if chosen:
            valid[job["id"]] = chosen
        else:
            status = {}
            if attempts:
                try:
                    status = read_json(attempts[0]/"status.json")
                except (OSError, ValueError):
                    pass
            progress.append({"job_id": job["id"], "status": status.get("status", "not_started"),
                             "observed_queries_unvalidated": status.get("completed_queries", 0), "expected_queries": len(job["rows"])})
    groups = sorted({(j["dataset"], j["cohort"]) for j in jobs})
    points, pairing_errors = [], []
    for dataset, cohort in groups:
        baseid = f"{dataset}/full/{cohort}/100/j0"
        for fraction in FRACTIONS:
            ids = {arm: baseid if arm == "j0" else f"{dataset}/full/{cohort}/{int(100*fraction):03d}/{arm}" for arm in ARMS}
            if not all(key in valid for key in ids.values()):
                continue
            attempts = {arm: valid[key] for arm, key in ids.items()}
            try:
                summaries = paired(attempts)
            except ValueError as exc:
                pairing_errors.append({"dataset": dataset, "cohort": cohort, "fraction": fraction, "error": str(exc)})
                continue
            points.append({"dataset": dataset, "cohort": cohort, "fraction": fraction,
                           "cache_budget_bytes": attempts["fix_all"]["budget_bytes"],
                           "working_set": attempts["j0"]["contract"]["working_set"], "measurement_job_ids": ids,
                           "j0_shared_measurement": True, "summaries": summaries,
                           "vs_j0": {arm: compare(summaries[arm], summaries["j0"]) for arm in ARMS if arm != "j0"}})
    cohort_complete = {(d, c) for d, c in groups if sum(p["dataset"] == d and p["cohort"] == c for p in points) == 3}
    tasks = sorted({(j["dataset"], r["sample"]["task"]) for j in jobs for r in j["rows"]})
    task_summaries = []
    # Only all-four-arm matched complete cohort points enter cross-cohort
    # descriptive pools. Each fraction reuses j0 by reference, not new outputs.
    for dataset in sorted({d for d, _ in groups}):
        expected_cohorts = [c for d, c in groups if d == dataset]
        for fraction in FRACTIONS:
            selected = [p for p in points if p["dataset"] == dataset and p["fraction"] == fraction]
            if not selected:
                continue
            summaries = {arm: summarize([valid[p["measurement_job_ids"][arm]] for p in selected]) for arm in ARMS}
            task_summaries.append({"dataset": dataset, "fraction": fraction, "paired_cohorts": len(selected),
                "expected_cohorts": len(expected_cohorts), "scope_complete": len(selected) == len(expected_cohorts),
                "summaries": summaries, "vs_j0": {arm: compare(summaries[arm], summaries["j0"]) for arm in ARMS if arm != "j0"}})
    pending_repairs = unresolved_remeasurements(results_root)
    all_complete = len(valid) == len(jobs) and len(cohort_complete) == len(groups) and not pairing_errors and not pending_repairs
    return {"schema_version": 1, "protocol": VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all_complete else "partial", "all_full_workloads_complete": all_complete,
        "scope": {"expected_cohorts": len(groups), "expected_jobs": len(jobs),
                  "expected_unique_measured_outputs": sum(len(j["rows"]) for j in jobs),
                  "expected_j0_measurements": len(groups), "datasets_tasks": tasks,
                  "fixed_cohort_budgets": [{"dataset": j["dataset"], "cohort": j["cohort"],
                      "documents": len(j["documents"]), "trace_queries": len(j["rows"]),
                      "working_set": j["working_set"],
                      "budget_bytes": {f"{int(f*100):03d}": int(j["working_set"]["full_depth_working_set_bytes"]*f) for f in FRACTIONS}}
                      for j in jobs if j["arm"] == "j0"]},
        "progress": {"valid_complete_jobs": len(valid), "valid_complete_cohorts": len(cohort_complete),
                     "unique_measured_outputs": sum(len(a["records"]) for a in valid.values()),
                     "unique_j0_outputs": sum(len(a["records"]) for a in valid.values() if a["arm"] == "j0"),
                     "paired_capacity_points": len(points), "pending_jobs": progress},
        "j0_accounting": "one measured 100% trace per cohort; referenced at 25/50/100%, never counted three times",
        "timing_attempt_policy": "Preserved attempts with external CPU contamination are excluded. A registered full-trace replacement is the only eligible attempt for that job; partial replacements never fall back to the contaminated trace.",
        "pending_timing_remeasurements": pending_repairs,
        "timing_boundary": {"fixed": "per-cohort document tokenization and prepared CPU raw chunks only; no initial KV write",
            "query": "document bind + query tokenization + external BM25 + reader total + output text decode",
            "reader": "on-demand lookup/capture/D2H/LRU included in load_s; wrapper release/bookkeeping included in total_s; prefix fill separately inside its total",
            "nested_components": "capture/build/CPU copy/fill/load are overlapping subphases, not additive extras to query total",
            "excluded": "model load, warmup, source file I/O, score calculation, output JSON I/O, GC outside the measured request",
            "interpretation": "natural EOS and observed answer lengths; descriptive service time, not equal-length decode throughput"},
        "validation_boundary": "Raw records independently match fixed IDs/order/query token IDs/ordered selected chunks/caps and pack lengths. Full context token equality is the runner's per-document assertion against the named fixture; attempts do not archive a separate context token copy or model weight fingerprint. Scores are recorded official-wrapper outputs, not independently rescored here.",
        "memory_boundary": "CPU persistent representation uses actual tensor bytes (V2 includes h_j and lower KV). Raw token IDs/metadata, active bypass payload, and GPU allocated peak are separate. Prefix bypass/active CPU payload is not recorded and remains null. GPU peak includes model and active tensors; incremental peak is relative to per-query allocated baseline, not direct online-KV inventory. Prefix/j0 miss prefill is not a chunk capture call.",
        "statistics_boundary": "No confidence intervals or significance tests. LoCoMo is one continuous 800-request trace. SCBench cohorts retain their predeclared membership; two task metrics are never averaged into a cross-task quality score.",
        "valid_attempts": [{k: a[k] for k in ("job_id", "path", "arm", "dataset", "cohort", "fraction", "budget_bytes", "completed_at")} for a in valid.values()],
        "complete_job_summaries": {key: summarize([a]) for key, a in valid.items()},
        "capacity_points": points, "dataset_summaries": task_summaries,
        "rejected_complete_attempts": rejected, "pairing_errors": pairing_errors}


def render_report(report):
    p, scope = report["progress"], report["scope"]
    lines = ["# 多文档等字节缓存容量实验", "",
        f"状态：{'全部正式测量通过完整性校验' if report['all_full_workloads_complete'] else '部分进度；尚未完成全量实验'}。",
        f"完整有效任务 {p['valid_complete_jobs']}/{scope['expected_jobs']}；四臂三个容量点全齐的 cohort {p['valid_complete_cohorts']}/{scope['expected_cohorts']}；唯一实测输出 {p['unique_measured_outputs']}/{scope['expected_unique_measured_outputs']}。",
        "", "j0 每个 cohort 仅实测一次，在 25%/50%/100% 三列共享。表中重复展示不增加实测次数或样本量。LoCoMo 是固定的 10 文档交错 800 请求；SCBench 保留 33 个预定 cohort，QA 与选择题分别报告官方准确率。", "",
        "容量分母是该 cohort 固定请求流所引用 chunk 的并集加一个共享 sink 的全深 KV 字节数；同时保留全文字节参照，不能称为全文缓存容量百分比。", "",
        "固定成本仅包含每 cohort 的文档分词与 CPU 原始块准备。逐问成本包含文档绑定、问题分词、BM25、reader total 和输出文本解码。按需构建、D2H 与 LRU 已在 load 内；wrapper 已在 total 内，子阶段不重复相加。模型加载、预热、源文件读取、评分、结果写盘及计时段外 GC 排除。", "",
        "质量使用正式 runner 保存的官方 wrapper 分数，此汇总器不重跑评分器。原始 ID、顺序、query token IDs、检索块顺序、读取长度与生成 cap 均核对 fixture。全文 token 一致性由 runner 入场断言保证；结果未另存一份全文 token 或模型权重指纹。", "",
        "CPU 常驻张量、原始 tokens、活跃旁路 payload、GPU 总峰值及 GPU 增量峰值分列；V2 张量含 h_j 与下层 KV。GPU 增量不是直接 online-KV 字节。prefix 未记录的旁路/CPU 活跃 payload 显示为 null。自然 EOS 下答案长度可能不同，不解释为等长解码吞吐。", "",
        "不提供显著性结论或请求独立速度置信区间。部分 cohort 的均值仅代表所列已配对完整子集。", ""]
    lines.extend(["已知外部CPU并发的原始attempt保留但不计入正式结果。登记重测的单元仅接受指定新attempt的完整trace；新trace尚未完整时不回退受污染结果，也不把两次输出加进正式分母。", ""])
    if not report["dataset_summaries"]:
        lines.append("目前没有四臂完整配对的正式容量点；不填性能或质量数值。")
    for item in report["dataset_summaries"]:
        lines.extend([f"## {item['dataset']} · {int(item['fraction']*100)}% · {item['paired_cohorts']}/{item['expected_cohorts']} cohort", "",
                      "| 方法 | n | 质量（按任务） | 实际 token 命中率 | 常驻峰值 GiB | GPU总/增量峰值 GiB | TTFT均值/p50/p95 秒 | 准备+请求 秒 | 相对j0时间 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"])
        for arm in ARMS:
            s = item["summaries"][arm]
            quality = "; ".join(f"{t}: {q['score_percent']:.4f} (n={q['n']})" +
                (f" Δ{item['vs_j0'][arm]['quality_delta_percentage_points'][t]:+.4f}pp" if arm != "j0" else "")
                for t, q in s["quality_by_task"].items())
            hit = "不适用" if s["actual_token_hit_fraction"] is None else f"{100*s['actual_token_hit_fraction']:.2f}%"
            ratio = "共享对照" if arm == "j0" else f"{item['vs_j0'][arm]['cumulative_time_ratio_to_j0']:.4f}"
            lines.append(f"| {arm} | {s['n']} | {quality} | {hit} | {s['maximum_persistent_cpu_representation_bytes']/1024**3:.4f} | {s['maximum_total_gpu_allocated_bytes']/1024**3:.4f}/{s['maximum_incremental_gpu_allocated_bytes']/1024**3:.4f} | {s['ttft_mean_s']:.4f}/{s['ttft_p50_s']:.4f}/{s['ttft_p95_s']:.4f} | {s['cumulative_prepare_plus_query_s']:.4f} | {ratio} |")
        lines.extend(["", "| 方法 | 文档token命中/缺失 | sink命中/缺失 | 淘汰 GiB | 旁路 GiB | capture次数 | D2H/H2D GiB |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for arm in ARMS:
            s = item["summaries"][arm]
            bypass = "未记录" if s["cache_bypass_tensor_bytes"] is None else f"{s['cache_bypass_tensor_bytes']/1024**3:.4f}"
            lines.append(f"| {arm} | {s['document_tokens_hit']}/{s['document_tokens_miss']} | {s['sink_tokens_hit']}/{s['sink_tokens_miss']} | {s['cache_evicted_tensor_bytes']/1024**3:.4f} | {bypass} | {s['capture_calls']} | {s['cache_fill_d2h_tensor_bytes']/1024**3:.4f}/{s['h2d_bytes_including_raw_ids']/1024**3:.4f} |")
        lines.extend(["", "j0 的缺失 token 表示需重算的输入量，缓存查找不适用。逐 cohort 的完整字节预算、全文参照、输出长度及精确分量见 capacity_summary.json。", ""])
    lines.extend([f"被拒绝的完整标记 attempt：{len(report['rejected_complete_attempts'])}；跨臂配对错误：{len(report['pairing_errors'])}。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results/capacity")
    parser.add_argument("--fixtures", type=Path, default=HERE/"fixtures")
    parser.add_argument("--out", type=Path, default=HERE/"results/reports/capacity")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)
    jobs = []
    for dataset in ("locomo", "scbench"):
        fixture = args.fixtures/("scbench" if dataset == "scbench" else "")
        docs, queries = load_fixture(dataset, fixture)
        jobs.extend(expected_jobs(dataset, docs, queries, fixture))
    require(len(jobs) == 340 and sum(len(j["rows"]) for j in jobs) == 13800, "Full predeclared campaign scope differs")
    report = aggregate(jobs, args.results, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, text in (("capacity_summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                           ("CAPACITY_REPORT.md", render_report(report))):
        temp = args.out/(filename+".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(args.out/filename)
    print(json.dumps({"status": report["status"], "progress": {k: v for k, v in report["progress"].items() if k != "pending_jobs"}, "output": str(args.out)}, ensure_ascii=False))
    # The already-running campaign treats exit 0 as report completion. Prevent
    # its old in-memory controller from publishing success with a pending repair.
    queue_path = args.results.parent/"capacity_queue/status.json"
    queue = read_json(queue_path) if queue_path.exists() else {}
    return 1 if queue.get("status") == "complete" and not report["all_full_workloads_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
