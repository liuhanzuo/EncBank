"""Read-only Phase-A LoCoMo aggregation; no model, CUDA, or scoring imports.

Only complete, admitted 5090 attempts enter descriptive summaries. Confidence
intervals require the entire ten-document, four-arm paired campaign. This tool
never stitches partial stateful attempts, extrapolates a crossover, or chooses
an attempt by its speed/score. Output artifacts are created only when invoked.
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

HERE = Path(__file__).resolve().parent
ARMS = ("j0", "prefix", "fix_all", "cacheblend16")
PREFIXES = (1, 10, 50, 80, "full")
PAIRS = (("fix_all", "j0"), ("prefix", "j0"), ("cacheblend16", "j0"),
         ("fix_all", "prefix"), ("cacheblend16", "prefix"), ("fix_all", "cacheblend16"))
SEED = 20260909
GPU = "NVIDIA GeForce RTX 5090"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def number(value, name, integer=False):
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{name}: not numeric")
    require(math.isfinite(value) and value >= 0, f"{name}: nonfinite or negative")
    if integer:
        require(isinstance(value, int), f"{name}: not integer")
    return value


def close(a, b, name):
    require(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7), f"{name}: {a} != {b}")


def normalized_path(value):
    return str(value).replace("\\", "/").rstrip("/").casefold()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*p
    low, high = math.floor(x), math.ceil(x)
    return values[low]*(high-x)+values[high]*(x-low) if low != high else values[low]


def hardware_contract(hardware):
    require(hardware["timing_eligible"] is True, "Hardware not timing eligible")
    require(hardware["device_name"] == GPU and hardware["device"] == "cuda:0", "Wrong GPU/device")
    require(hardware["platform"] == "Windows", "Remote/non-Windows timing is invalid")
    for key, expected in (("torch_cpu_threads", 2), ("torch_interop_threads", 16),
                          ("omp_num_threads", "2"), ("mkl_num_threads", "2"),
                          ("tokenizers_parallelism", "false")):
        require(hardware.get(key) == expected, f"Thread policy mismatch: {key}")
    gate = hardware["gpu_admission"]
    require(gate["comparison"] == "strictly_less_than", "Admission was not strict")
    threshold = number(gate["effective_idle_slack_gib"], "effective threshold")
    require(0 < threshold <= 5, "Admission threshold exceeds 5 GiB")
    for name in ("initial_used_gib", "recheck_used_gib"):
        require(number(gate[name], name) < threshold, f"{name}: strict admission failed")
    require(gate["other_python_compute_processes"] == [], "Another model was admitted")
    require(bool(hardware.get("hostname")), "Missing actual hostname")
    return {k: hardware.get(k) for k in ("hostname", "device", "device_name", "device_uuid",
        "platform", "torch", "transformers", "cuda_runtime", "torch_cpu_threads",
        "torch_interop_threads", "omp_num_threads", "mkl_num_threads", "tokenizers_parallelism")}


def load_fixture(path):
    path = Path(path).resolve()
    manifest = read_json(path/"manifest.json")
    require(manifest["protocol"] == "locomo-real-multiquery-reuse-v1", "Only Phase A LoCoMo supported")
    require(manifest["document_count"] == 10 and manifest["categories"] == [1, 2, 3, 4], "Wrong Phase A scope")
    documents = read_json(path/manifest["documents_file"])
    queries = [json.loads(s) for s in (path/manifest["queries_file"]).read_text(encoding="utf-8").splitlines() if s.strip()]
    require(len(documents) == 10 and len(queries) == manifest["unique_queries"], "Fixture counts differ")
    dm, qm = {d["document_id"]: d for d in documents}, {q["id"]: q for q in queries}
    require(len(dm) == 10 and len(qm) == len(queries), "Duplicate fixture identity")
    visited = []
    for document in documents:
        order = document["query_ids_ordered"]
        require(order == document["prefixes"]["full"] and len(order) == document["query_count"], "Wrong fixture stream")
        require(len(set(order)) == len(order), "Duplicate fixture question in stream")
        for prefix in (1, 10, 50, 80):
            require(document["prefixes"][str(prefix)] == order[:prefix] and len(order) >= prefix, "Wrong fixture prefix")
        for i, qid in enumerate(order):
            row = qm[qid]
            require(row["document_id"] == document["document_id"] and row["stream_index"] == i, "Wrong fixture question order")
            require(row["sample"]["category"] in (1, 2, 3, 4), "Unrequested score category")
            require(row["sample"]["task"] == f"category_{row['sample']['category']}", "Fixture task/category differs")
            n, q = len(document["context_ids"]), len(row["query_ids"])
            selected = row["selected_indices"]
            require(isinstance(selected, list) and len(set(selected)) == len(selected), "Duplicate fixture chunks")
            require(all(type(j) is int and 0 <= j < (n+511)//512 for j in selected), "Fixture chunk outside document")
            require(len(selected) == min(12, (n+511)//512), "Wrong fixture retrieval budget")
            pack = row["pack"]
            require(pack["selected_indices"] == selected and pack["context_tokens"] == n and pack["query_tokens"] == q, "Fixture pack fields differ")
            require(pack["read_pack_tokens"] == 1+q+sum(min(512, n-512*j) for j in selected), "Fixture read length differs")
            visited.append(qid)
    require(len(visited) == len(queries) and set(visited) == set(qm), "Unassigned fixture rows")
    return manifest, dm, qm


def validate_attempt(path, document, expected, fixture_path):
    """Return validated raw measurements and contracts; never execute a scorer."""
    path = Path(path)
    marker, config = read_json(path/"COMPLETED.json"), read_json(path/"config.json")
    store = read_json(path/"store_cost.json")
    store_meta = read_json(path/"store/store.json")
    records = [json.loads(s) for s in (path/"measurements.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    arm, docid = config["arm"], document["document_id"]
    require(arm in ARMS, "Unknown arm")
    require(marker["status"] == "complete", "Missing complete marker")
    require(marker["smoke"] is False and config["smoke"] is False and marker["timing_eligible"] is True, "Smoke/ineligible attempt")
    require(config["protocol"] == "real-question-reuse-v1" and config["dataset"] == marker["dataset"] == "locomo", "Wrong protocol/dataset")
    require(config["document"] == marker["document"] == docid and marker["arm"] == arm, "Wrong document/arm")
    require(config["query_limit"] == 0, "Partial query-limit run is not full")
    require(config["ids"] == [q["id"] for q in expected], "Config query order differs")
    require(marker["expected_queries"] == marker["completed_queries"] == marker["new_generations"] == len(expected) == len(records), "Incomplete query denominator")
    require(marker.get("extra_reference_generations", 0) == 0, "Full timing includes smoke reference work")
    require(normalized_path(config["fixtures"]) == normalized_path(Path(fixture_path).resolve()), "Wrong fixture path")
    require(config["dtype"] == "bfloat16" and config["j"] == 12 and config["chunk_size"] == 512 and config["topk"] == 12, "Wrong model/read protocol")
    require(config["query_order_seed"] == SEED and config["answer_cache"] is False, "Wrong order seed/answer reuse")
    require(config["source_context_truncation"] == "none", "Source was truncated")
    require(not config.get("adapter"), "Unexpected model adapter")
    hw = hardware_contract(marker["hardware"])
    signature = store_meta["signature"]
    require(normalized_path(signature["model_id"]) == normalized_path(config["model"]), "Store checkpoint differs")
    require(signature["arm"] == arm and not signature.get("adapter"), "Store arm/adapter differs")
    require(signature["dtype"] == "torch.bfloat16", "Store dtype differs")
    require(signature["j"] == (0 if arm in ("j0", "prefix") else 12), "Store split differs")
    require(store_meta["n_tokens"] == len(document["context_ids"]), "Stored document size differs")
    model_config = signature["model_config"]
    require(model_config.get("model_type") == "qwen3", "Wrong model architecture")
    for key in ("hidden_size", "num_hidden_layers", "num_key_value_heads", "head_dim"):
        number(model_config[key], key, integer=True)
    write, startup = store["write"], store["startup"]
    for key in ("write_total_s", "write_compute_s", "write_device_to_cpu_s", "write_serialize_s",
                "payload_tensor_bytes", "raw_token_bytes", "serialized_bytes", "write_peak_allocated_bytes"):
        number(write[key], key)
    for key in ("startup_load_s", "startup_read_bytes", "resident_cpu_tensor_bytes"):
        number(startup[key], key)
    require(startup["tier"] == "cpu", "Wrong cache tier for Phase A")
    number(store["document_prepare_s"], "document_prepare_s")
    close(store["fixed_cost_s"], store["document_prepare_s"]+write["write_total_s"]+startup["startup_load_s"], "Fixed cost")
    budget = number(config["cache_budget_bytes"], "cache_budget_bytes", integer=True)
    require(store["budget_bytes"] == budget and write["payload_tensor_bytes"] <= budget, "Inconsistent cache budget")
    if arm == "fix_all":
        n = store_meta["n_tokens"]+1
        expected_payload = n*(model_config["hidden_size"]*2+4*signature["j"]*model_config["num_key_value_heads"]*model_config["head_dim"])
        close(write["payload_tensor_bytes"], expected_payload, "V2 representation decomposition")
    for row, reference in zip(records, expected):
        require(row["id"] == reference["id"] and row["document_id"] == docid and row["arm"] == arm, "Record identity differs")
        require(row["stream_index"] == reference["stream_index"] and row["task"] == reference["sample"]["task"], "Record order/category differs")
        require(row["timing_eligible"] is True and row["smoke_reference_equal"] is None, "Ineligible raw record")
        require(row["query_ids"] == reference["query_ids"], "Actual query token IDs differ")
        require(row["query_tokens"] == len(reference["query_ids"]), "Query token count differs")
        require(row["selected_indices"] == reference["selected_indices"] == reference["pack"]["selected_indices"], "Actual selected pack differs")
        cap = reference["sample"]["max_new_tokens"]
        require(row["max_new_tokens"] == cap, "Answer cap differs")
        ids = row["generated_ids"]
        require(isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids), "Invalid generated IDs")
        require(0 < len(ids) == row["generated_tokens"] <= cap, "Invalid output length")
        require(type(row["terminated_by_eos"]) is bool and row["terminated_by_eos"] == (len(ids) < cap), "EOS/cap accounting differs")
        require(isinstance(row["prediction"], str) and isinstance(row["scored_prediction"], str), "Missing actual output text")
        require(number(row["score"], "score") <= 1, "Score is not 0..1")
        stats, timing = row["stats"], row["timings"]
        require(stats["selected_indices"] == row["selected_indices"] and stats["read_tokens"] == reference["pack"]["read_pack_tokens"], "Measured pack differs")
        require(stats["query_tokens"] == row["query_tokens"] and stats["generated_tokens"] == len(ids), "Measured token counts differ")
        require(stats["fixed_generation_length"] is False and stats["capture_calls"] == 0, "Fixed generation or document recapture in query")
        require(stats["decode_steps"] == len(ids)-1+int(row["terminated_by_eos"]), "Decode forward count differs")
        for key in ("total_s", "ttft_s", "decode_s", "load_s", "transfer_s", "transfer_bytes",
                    "peak_allocated_bytes", "incremental_peak_bytes", "cache_resident_bytes", "cache_budget_bytes"):
            number(stats[key], key)
        require(stats["cache_budget_bytes"] == budget and stats["cache_resident_bytes"] <= budget, "Per-query cache exceeds common budget")
        for key in ("query_tokenization_s", "external_retrieval_s", "text_decode_s", "ttft_s", "query_total_s"):
            number(timing[key], key)
        for key in ("cache_fill_transfer_bytes", "cache_fill_s", "evicted_bytes"):
            number(stats.get(key, 0), key)
        close(timing["ttft_s"], timing["query_tokenization_s"]+timing["external_retrieval_s"]+stats["ttft_s"], "TTFT boundary")
        close(timing["query_total_s"], timing["query_tokenization_s"]+timing["external_retrieval_s"]+stats["total_s"]+timing["text_decode_s"], "Query total boundary")
        require(stats["total_s"] >= stats["ttft_s"] and timing["query_total_s"] >= timing["ttft_s"], "Total before first token")
    paired_contract = {"model": normalized_path(config["model"]), "model_config": model_config,
        "dtype": config["dtype"], "chunk_size": config["chunk_size"], "topk": config["topk"],
        "budget_bytes": budget, "hardware": hw,
        "input_sha256": digest({"context_ids": document["context_ids"], "queries": [
            {"id": r["id"], "query_ids": r["query_ids"], "selected_indices": r["selected_indices"],
             "cap": r["sample"]["max_new_tokens"], "task": r["sample"]["task"]} for r in expected]})}
    return {"path": str(path.resolve()), "arm": arm, "document_id": docid,
            "records": records, "store": store, "store_metadata": store_meta,
            "contract": paired_contract, "completed_at": marker.get("finished_at")}


def paired_document(attempts):
    require(set(attempts) == set(ARMS), "All four arms required for a paired document")
    base = attempts["j0"]["contract"]
    for arm in ARMS:
        require(attempts[arm]["contract"] == base, f"Cross-arm contract differs: {arm}")
    return True


def quality(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["task"]].append(row["score"])
    return {name: {"n": len(values), "sum_f1": sum(values), "mean_f1": statistics.mean(values),
                   "f1_percent": 100*statistics.mean(values)} for name, values in sorted(groups.items())}


def prefix_summary(attempt, count):
    records, store = attempt["records"][:count], attempt["store"]
    require(len(records) == count and count > 0, "Unobserved prefix requested")
    stats = [r["stats"] for r in records]
    write, startup = store["write"], store["startup"]
    online = [r["timings"]["query_total_s"] for r in records]
    lengths = [r["generated_tokens"] for r in records]
    representation = write["payload_tensor_bytes"]
    if attempt["arm"] == "prefix":
        kv_fixed, kv_max = 0, max(s["cache_resident_bytes"] for s in stats)
        kv_semantics = "recorded unique CPU prefix-KV resident bytes; filled online"
    elif attempt["arm"] == "fix_all":
        sig = attempt["store_metadata"]["signature"]
        n = attempt["store_metadata"]["n_tokens"]+1
        cfg = sig["model_config"]
        hidden = n*cfg["hidden_size"]*2
        kv_fixed = n*4*sig["j"]*cfg["num_key_value_heads"]*cfg["head_dim"]
        close(representation, hidden+kv_fixed, "V2 representation decomposition")
        kv_max, kv_semantics = kv_fixed, "BF16 lower-KV component inferred from recorded dimensions; total representation byte count verified"
    else:
        kv_fixed = representation
        kv_max = representation
        kv_semantics = "stored full-layer KV only" if attempt["arm"] == "cacheblend16" else "no persistent KV; raw-token store only"
    fill_bytes = sum(s.get("cache_fill_transfer_bytes", 0) for s in stats)
    return {"Q": count, "fixed_cost_s": store["fixed_cost_s"], "online_sum_s": sum(online),
        "cumulative_total_s": store["fixed_cost_s"]+sum(online), "mean_online_query_s": statistics.mean(online),
        "ttft_p50_s": percentile([r["timings"]["ttft_s"] for r in records], .5),
        "ttft_p95_s": percentile([r["timings"]["ttft_s"] for r in records], .95),
        "quality_by_category": quality(records), "output_tokens_sum": sum(lengths),
        "output_tokens_mean": statistics.mean(lengths), "output_tokens_range": [min(lengths), max(lengths)],
        "decode_forward_calls": sum(s["decode_steps"] for s in stats),
        "eos_terminated_queries": sum(r["terminated_by_eos"] for r in records),
        "cap_exhausted_queries": sum(not r["terminated_by_eos"] for r in records),
        "persistent_representation_tensor_bytes": representation, "persistent_kv_tensor_bytes": kv_fixed,
        "maximum_resident_kv_bytes": kv_max, "persistent_kv_count_semantics": kv_semantics,
        "persistent_raw_token_bytes": write["raw_token_bytes"], "persistent_serialized_store_bytes": write["serialized_bytes"],
        "prefix_online_kv_is_serialized": False if attempt["arm"] == "prefix" else None,
        "maximum_cross_query_representation_resident_bytes": max(s["cache_resident_bytes"] for s in stats),
        "final_cross_query_representation_resident_bytes": stats[-1]["cache_resident_bytes"],
        "peak_allocated_bytes_including_write": max(write["write_peak_allocated_bytes"], max(s["peak_allocated_bytes"] for s in stats)),
        "online_incremental_peak_allocated_bytes": max(s["incremental_peak_bytes"] for s in stats),
        "online_h2d_bytes": sum(s["transfer_bytes"] for s in stats),
        "online_h2d_s": sum(s["transfer_s"] for s in stats),
        "online_cache_fill_d2h_bytes": fill_bytes,
        "online_cache_fill_s_includes_insert": sum(s.get("cache_fill_s", 0) for s in stats),
        "initial_write_d2h_tensor_payload_bytes": representation,
        "initial_write_d2h_s": write["write_device_to_cpu_s"],
        "d2h_bytes_semantics": "initial tensor payload count plus recorded online cache-fill transfer count; no interconnect bandwidth claim",
        "initial_load_s": startup["startup_load_s"], "initial_load_bytes": startup["startup_read_bytes"],
        "cache_budget_bytes": store["budget_bytes"], "evicted_bytes": sum(s.get("evicted_bytes", 0) for s in stats)}


def observed_break_even(candidate, baseline):
    """Earliest observed cumulative prefix staying strictly ahead to trace end."""
    require(len(candidate["records"]) == len(baseline["records"]), "Unpaired break-even trace")
    c, b = candidate["store"]["fixed_cost_s"], baseline["store"]["fixed_cost_s"]
    differences = []
    for cr, br in zip(candidate["records"], baseline["records"]):
        require(cr["id"] == br["id"], "Break-even query order differs")
        c += cr["timings"]["query_total_s"]
        b += br["timings"]["query_total_s"]
        differences.append(c-b)
    first, sustained = None, True
    for i in range(len(differences)-1, -1, -1):
        sustained = sustained and differences[i] < -1e-9
        if sustained:
            first = i+1
    return {"observed_sustained_from_Q": first, "observed_through_Q": len(differences),
            "final_difference_s_candidate_minus_baseline": differences[-1],
            "beyond_observed_trace": "unknown; no linear extrapolation",
            "all_prefix_differences_s": differences}


def cluster_bootstrap(document_summaries, candidate, baseline, prefix, *, resamples=2000, seed=SEED):
    """Paired document resampling; ratios of sums for category F1, no query IID."""
    require(len(document_summaries) == 10, "Inference requires all ten paired documents")
    require(resamples >= 100, "Use at least 100 document bootstrap replicates")
    docs = sorted(document_summaries)
    pairs = [(document_summaries[d][candidate][str(prefix)], document_summaries[d][baseline][str(prefix)]) for d in docs]
    metrics = ("cumulative_total_s", "mean_online_query_s", "ttft_p50_s", "ttft_p95_s", "output_tokens_mean")
    category_names = sorted({c for a, _ in pairs for c in a["quality_by_category"]})
    def contrast(indices):
        out = {m: statistics.mean(pairs[i][0][m]-pairs[i][1][m] for i in indices) for m in metrics}
        for cat in category_names:
            n = sum(pairs[i][0]["quality_by_category"].get(cat, {}).get("n", 0) for i in indices)
            if n:
                a = sum(pairs[i][0]["quality_by_category"].get(cat, {}).get("sum_f1", 0) for i in indices)
                b = sum(pairs[i][1]["quality_by_category"].get(cat, {}).get("sum_f1", 0) for i in indices)
                out["f1_percentage_points/"+cat] = 100*(a-b)/n
        return out
    estimates = contrast(range(10))
    samples = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(resamples):
        for metric, value in contrast([rng.randrange(10) for _ in range(10)]).items():
            samples[metric].append(value)
    return {"candidate": candidate, "baseline": baseline, "prefix": prefix,
        "difference_direction": "candidate minus baseline; negative time favors candidate",
        "cluster": "document", "documents": 10, "seed": seed, "resamples": resamples,
        "aggregation": "equal-weight document means for time/length; pooled query F1 per category within resampled document clusters",
        "category_empty_resamples": "omitted for that category only, count disclosed; no re-draw",
        "multiplicity": "descriptive pointwise 95% intervals; no multiple-comparison correction",
        "metrics": {m: {"estimate": value, "ci95": [percentile(samples[m], .025), percentile(samples[m], .975)],
                        "valid_resamples": len(samples[m])} for m, value in estimates.items()}}


def aggregate(results_root, fixture_path, *, resamples=2000, seed=SEED):
    manifest, documents, queries = load_fixture(fixture_path)
    results_root = Path(results_root)
    chosen, progress, rejected, duplicates = {}, [], [], []
    for docid, document in documents.items():
        chosen[docid] = {}
        expected = [queries[qid] for qid in document["query_ids_ordered"]]
        for arm in ARMS:
            parent = results_root/docid/arm/"attempts"
            valid = []
            for path in sorted(parent.glob("*")) if parent.exists() else []:
                if not path.is_dir():
                    continue
                if not (path/"COMPLETED.json").exists():
                    if (path/"status.json").exists():
                        try:
                            state = read_json(path/"status.json")
                            progress.append({"path": str(path), "document_id": docid, "arm": arm,
                                "status": state.get("status"), "completed_queries": state.get("completed_queries"),
                                "expected_queries": state.get("expected_queries")})
                        except (ValueError, OSError) as exc:
                            rejected.append({"path": str(path), "reason": repr(exc)})
                    continue
                try:
                    attempt = validate_attempt(path, document, expected, fixture_path)
                    require(attempt["arm"] == arm, "Directory arm differs")
                    if manifest.get("tokenizer_path"):
                        require(attempt["contract"]["model"] == normalized_path(manifest["tokenizer_path"]), "Checkpoint differs from fixture tokenizer model")
                    valid.append(attempt)
                except (KeyError, ValueError, TypeError, OSError, OverflowError) as exc:
                    rejected.append({"path": str(path), "reason": repr(exc)})
            if valid:
                chosen[docid][arm] = valid[0]
                duplicates.extend({"selected": valid[0]["path"], "not_selected": a["path"],
                    "policy": "earliest lexicographic valid attempt; never choose by metric"} for a in valid[1:])
    descriptive, paired, breaks, pairing_errors = {}, {}, {}, []
    for docid, arms in chosen.items():
        descriptive[docid] = {arm: {str(p): prefix_summary(a, len(a["records"]) if p == "full" else p)
            for p in PREFIXES if p == "full" or p <= len(a["records"])} for arm, a in arms.items()}
        if len(arms) == 4:
            try:
                paired_document(arms)
                paired[docid] = descriptive[docid]
                breaks[docid] = {f"{a}_minus_{b}": observed_break_even(arms[a], arms[b]) for a, b in PAIRS}
            except ValueError as exc:
                pairing_errors.append({"document_id": docid, "reason": str(exc)})
    complete = len(paired) == 10
    intervals = []
    if complete:
        for prefix in PREFIXES:
            for candidate, baseline in PAIRS:
                intervals.append(cluster_bootstrap(paired, candidate, baseline, prefix, resamples=resamples, seed=seed))
    return {"schema_version": 1, "protocol": "phase-a-reuse-aggregate-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete" if complete else "partial",
        "dataset": "locomo", "expected_documents": 10, "paired_complete_documents": len(paired),
        "expected_arm_jobs": 40, "valid_complete_arm_jobs": sum(len(v) for v in chosen.values()),
        "expected_queries_per_arm": manifest["unique_queries"],
        "valid_completed_query_records": sum(len(a["records"]) for arms in chosen.values() for a in arms.values()),
        "source": {"results_root": str(results_root.resolve()), "fixtures": str(Path(fixture_path).resolve())},
        "attempts": {d: {a: v["path"] for a, v in arms.items()} for d, arms in chosen.items()},
        "attempt_selection_policy": "earliest lexicographic fully valid attempt; no partial-attempt stitching",
        "progress": progress, "rejected_attempts": rejected, "duplicate_valid_attempts": duplicates,
        "pairing_errors": pairing_errors, "complete_arm_descriptive_summaries": descriptive,
        "paired_document_ids": sorted(paired), "observed_break_even_by_complete_paired_document": breaks,
        "paired_document_bootstrap": intervals,
        "inference_allowed": complete, "partial_policy": "descriptive complete arms/documents only; no confidence intervals or significance conclusion before all ten paired documents",
        "fixed_cost_policy": "document tokenization + entire document write + startup load once per observed session/prefix",
        "time_boundary": "sum of declared components: query template tokenization/BM25 + cache load/H2D/rotation/prefill/online fill/natural EOS decode + text decode; external retrieval CPU tensor/chunk-index construction, source file IO/model load/warmup/validation/scoring excluded",
        "measurement_scope": "one measured stateful session per selected document/arm; prefixes are cumulative slices of that trace, not independent trials",
        "quality_policy": "actual locally timed outputs; category F1 kept separate; no remote score substitution",
        "memory_policy": "persistent representations/serialized store/online GPU peak differ; prefix online KV is CPU-only and not counted as serialized files",
        "identity_evidence": "recorded raw query IDs, ordered selected indices/cap/counts and recorded store checkpoint config matched to canonical fixture; document write token equality is asserted by runner"}


def render_report(result):
    state = "完整" if result["status"] == "complete" else "部分完成"
    lines = [f"# LoCoMo 缓存复用汇总（{state}）", "",
        f"有效完整任务 {result['valid_complete_arm_jobs']}/40；四方法配对完整文档 {result['paired_complete_documents']}/10；有效记录 {result['valid_completed_query_records']}。", "",
        "固定成本按每篇文档计算一次，包含文档 tokenization、完整写入和首次加载。在线成本包含真实问题的准备、检索、缓存读取与传输、生成及文本解码。这里累计总时间是声明组件之和：不包含外部检索 CPU tensor/chunk 索引的一次性构造、模型加载、warmup、输入验证、评分和源文件 IO；没有事后估计补时。write_store 内部自己的 chunk 准备已包含在写入时间内。", ""]
    if not result["inference_allowed"]:
        lines += ["**当前只报告已完整方法的描述统计；不输出置信区间、显著性结论或未完成流的盈亏平衡结论。**", ""]
    for docid, arms in result["complete_arm_descriptive_summaries"].items():
        if not arms:
            continue
        lines += [f"## {docid}", "", "| 方法 | Q | 累计总秒 | 平均在线秒 | TTFT p50/p95 秒 | 平均输出 token | 峰值 GiB |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, prefixes in arms.items():
            for prefix, s in prefixes.items():
                lines.append(f"| {arm} | {prefix} ({s['Q']}) | {s['cumulative_total_s']:.3f} | {s['mean_online_query_s']:.3f} | {s['ttft_p50_s']:.3f}/{s['ttft_p95_s']:.3f} | {s['output_tokens_mean']:.2f} | {s['peak_allocated_bytes_including_write']/2**30:.3f} |")
        lines += ["", "完整流各类 F1（百分制，类别分别报告）：", ""]
        for arm, prefixes in arms.items():
            cells = prefixes["full"]["quality_by_category"]
            lines.append(f"- {arm}: " + "；".join(f"{k} {v['f1_percent']:.2f} (n={v['n']})" for k, v in cells.items()))
        breaks = result["observed_break_even_by_complete_paired_document"].get(docid, {})
        if breaks:
            lines += ["", "该完整文档流相对 j0 的实测持续领先起点：" + "；".join(
                f"{a} Q={breaks[a+'_minus_j0']['observed_sustained_from_Q']}" if breaks[a+'_minus_j0']["observed_sustained_from_Q"] is not None
                else f"{a} 未观测到" for a in ("fix_all", "prefix", "cacheblend16")) + "。"]
        lines += [""]
    if result["inference_allowed"]:
        lines += ["## 配对文档 bootstrap", "", "十篇文档整簇重采样，固定 seed 20260909；以下差值均为候选方法减 j0。时间负值有利于候选方法。区间为描述性的逐项 95% 区间，未做多重比较校正。", "",
                  "| 方法 | 前缀 | 平均文档累计时间差 秒 | 95% CI |", "|---|---|---:|---:|"]
        for ci in result["paired_document_bootstrap"]:
            if ci["baseline"] == "j0":
                m = ci["metrics"]["cumulative_total_s"]
                lines.append(f"| {ci['candidate']} | {ci['prefix']} | {m['estimate']:.3f} | [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] |")
        lines += [""]
    lines += ["持久表示总字节、其 KV 部分、序列化大小、在线峰值及 H2D/D2H 分项见 JSON。V2 KV 部分由已记录形状推导并与表示总字节核对；prefix 的运行时 CPU KV 没有写入磁盘，不能把它的文件大小解释为零缓存。", "",
              "盈亏平衡仅在一条完整实测流的某个累计前缀开始、此后每个前缀都持续领先至实际末尾时记录；超出观测问题数的行为未知，不作线性外推。", ""]
    if result["rejected_attempts"] or result["pairing_errors"]:
        lines += [f"未纳入：{len(result['rejected_attempts'])} 个无效 attempt，{len(result['pairing_errors'])} 个跨方法配对错误；理由保留在 JSON。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results/locomo/full")
    parser.add_argument("--fixtures", type=Path, default=HERE/"fixtures")
    parser.add_argument("--out", type=Path, default=HERE/"results/locomo/aggregate")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    result = aggregate(args.results, args.fixtures, resamples=args.resamples)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in (("aggregate.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                       ("REPORT.md", render_report(result))):
        target = args.out/name
        temp = target.with_suffix(target.suffix+".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(target)
    print(json.dumps({"status": result["status"], "complete_jobs": result["valid_complete_arm_jobs"],
        "paired_documents": result["paired_complete_documents"], "out": str(args.out.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
