"""Local-5090 real-question cache reuse; one document/arm per guarded process."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / "exp/encbank_v2_benchmarks_20260908"
for folder in (ROOT / "Encbank", ROOT / "exp", OLD, HERE):
    sys.path.insert(0, str(folder))
ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
for key, value in ENV.items():
    os.environ[key] = value
VERSION = "real-question-reuse-v1"
ARMS = ("j0", "prefix", "fix_all", "cacheblend16")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows readers/antivirus can briefly hold a handle without delete-share.
    # Reporting is outside measured regions; retry publication, never inference.
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.01 * 2**attempt, .5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_bundle(dataset, path):
    if dataset == "locomo":
        import prepare_workload as preparation
    else:
        import prepare_scbench as preparation
    manifest, documents, queries = preparation.load_workload(path)
    if isinstance(documents, dict):
        documents = list(documents.values())
    return preparation, manifest, documents, queries


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - position) + values[high] * (position - low) if high != low else values[low]


def summarize(records, fixed_cost):
    by_category = defaultdict(list)
    for row in records:
        by_category[row["task"]].append(row["score"])
    points = sorted({q for q in (1, 10, 50, 80, len(records)) if q <= len(records)})
    return {
        "queries": len(records),
        "quality_by_task": {k: {"n": len(v), "score_0_1": statistics.mean(v)} for k, v in by_category.items()},
        "score_0_1": statistics.mean(r["score"] for r in records),
        "mean_online_query_s": statistics.mean(r["timings"]["query_total_s"] for r in records),
        "ttft_p50_s": percentile([r["timings"]["ttft_s"] for r in records], .5),
        "ttft_p95_s": percentile([r["timings"]["ttft_s"] for r in records], .95),
        "generated_tokens": sum(r["generated_tokens"] for r in records),
        "peak_allocated_bytes": max(r["stats"]["peak_allocated_bytes"] for r in records),
        "transfer_bytes": sum(r["stats"]["transfer_bytes"] for r in records),
        "cache_fill_d2h_bytes": sum(r["stats"].get("cache_fill_transfer_bytes", 0) for r in records),
        "fixed_cost_s": fixed_cost,
        "cumulative": [{"Q": q, "total_s": fixed_cost + sum(r["timings"]["query_total_s"] for r in records[:q]),
                        "query_only_s": sum(r["timings"]["query_total_s"] for r in records[:q]),
                        "score_0_1": statistics.mean(r["score"] for r in records[:q])} for q in points],
        "final_cache_resident_bytes": records[-1]["stats"].get("cache_resident_bytes"),
        "max_cache_resident_bytes": max((r["stats"].get("cache_resident_bytes", 0) for r in records), default=0),
        "evicted_bytes": sum(r["stats"].get("evicted_bytes", 0) for r in records),
        "timing_boundary": "document tokenization + write + startup + query tokenization/BM25 + cache read/transfer/prefill/decode + output text decode; excludes model load, warmup, validation and scoring",
    }


def new_reader(model, tok, arm, model_id, budget):
    from serving_reuse import ReusableReader
    if arm == "prefix":
        from prefix_cache import ReusablePrefixReader
        return ReusablePrefixReader(model, 12, tok, arm="prefix", model_id=model_id, cache_budget_bytes=budget)
    if arm == "cacheblend16":
        from cacheblend_serving_reuse import ReusableContextualCacheBlend
        return ReusableContextualCacheBlend(model, 12, tok, model_id, .16)
    return ReusableReader(model, 12, tok, arm, model_id)


def fresh_reference(model, tok, arm, document, row, cap):
    import torch
    from encbank import Encbank
    from s15_ruler_lower import EncbankLower
    from reader_adapter import ExplicitPackReader
    from cacheblend_contextual import ContextualCacheBlend
    reader = EncbankLower(model, 12, tok) if arm == "fix_all" else Encbank(model, 0 if arm in ("j0", "prefix") else 12, tokenizer=tok)
    reader.write_sink = arm == "fix_all"
    if arm == "cacheblend16":
        chunks = list(torch.tensor(document["context_ids"], device="cuda:0").split(512))
        return ContextualCacheBlend(reader, .16).generate_explicit(
            [chunks[i] for i in row["selected_indices"]], row["query_ids"], reader._sink_prefix_id(), tok.eos_token_id, cap)
    stats = {"capture_step_logits": True}
    from unittest.mock import patch
    original = reader._decode_from_pack
    def capture(*args, **kwargs):
        kwargs["stats"] = stats
        return original(*args, **kwargs)
    ids = torch.tensor([document["context_ids"] + row["query_ids"]], dtype=torch.long, device="cuda:0")
    try:
        with patch.object(reader, "_decode_from_pack", side_effect=capture):
            ExplicitPackReader(reader).generate_from_ids(ids, context_token_count=len(document["context_ids"]),
                selected_indices=row["selected_indices"], chunk_size=512, max_new_tokens=cap)
        return stats["generated_ids"]
    finally:
        if hasattr(reader, "_bottom"):
            reader._bottom = None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("locomo", "scbench"), default="locomo")
    p.add_argument("--fixtures", type=Path, default=HERE / "fixtures")
    p.add_argument("--document", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--query-limit", type=int, default=0)
    p.add_argument("--prefix-budget-bytes", type=int, default=0)
    args = p.parse_args(argv)
    preparation, manifest, documents, queries = load_bundle(args.dataset, args.fixtures)
    document = next(d for d in documents if d["document_id"] == args.document)
    rows = sorted([r for r in queries if r["document_id"] == args.document], key=lambda r: r["stream_index"])
    if args.smoke:
        rows = rows[:3]
    elif args.query_limit:
        rows = rows[:args.query_limit]
    require(bool(rows) and len({r["id"] for r in rows}) == len(rows), "Empty or duplicate query stream")
    config = dict(protocol=VERSION, dataset=args.dataset, fixtures=str(args.fixtures.resolve()), document=args.document,
                  arm=args.arm, model=args.model, smoke=args.smoke, ids=[r["id"] for r in rows],
                  prefix_budget_bytes_requested=args.prefix_budget_bytes, query_limit=args.query_limit,
                  query_order_seed=20260909, chunk_size=512, topk=12, dtype="bfloat16", j=12,
                  answer_cache=False, source_context_truncation="none", source_file_io="excluded")
    args.out.mkdir(parents=True, exist_ok=True)
    from run_driver import OutputLock
    with OutputLock(args.out):
        require(not (args.out / "measurements.jsonl").exists(), "Use a fresh attempt; do not resume a stateful cache by skipping queries")
        save(args.out / "config.json", config)
        status = {"status": "waiting_for_gpu", "pid": os.getpid(), "started_at": now(),
                  "document": args.document, "arm": args.arm, "dataset": args.dataset, "expected_queries": len(rows),
                  "completed_queries": 0, "smoke": args.smoke, "timing_eligible": not args.smoke}
        save(args.out / "status.json", status)
        try:
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(16)
            from gpu_gate import acquire_gpu, _release_lock
            admission = acquire_gpu(22, None, 5.0, max_wait=86400, tag="real_question_reuse " + str(args.out))
            from serving_reuse import device_provenance
            hardware = device_provenance("cuda:0")
            hardware["gpu_admission"] = admission
            hardware["memory_cap_bytes"] = 28_000_000_000
            torch.cuda.set_per_process_memory_fraction(min(28e9 / hardware["total_memory_bytes"], 1.0))
            import official_qa_driver as driver
            from encbank import Encbank
            from encbank import selectors
            def clock():
                torch.cuda.synchronize()
                return time.perf_counter()
            torch.manual_seed(42)
            started = clock()
            model, tok = driver.load_backbone(args.model, "bfloat16", "sdpa", "cuda:0", "")
            model.eval()
            load_s = clock() - started
            config_model = model.config
            bytes_per_token = 4 * config_model.num_hidden_layers * config_model.num_key_value_heads * config_model.head_dim
            budget = args.prefix_budget_bytes or (len(document["context_ids"]) + 1) * bytes_per_token
            config["cache_budget_bytes"] = budget
            config["cache_budget_policy"] = "same absolute document full-depth BF16 KV capacity; exact prefix uses leaf-LRU; V2/CB full document tensors fit"
            save(args.out / "config.json", config)
            status.update(status="running", hardware=hardware, model_load_s_excluded=load_s,
                          cache_budget_bytes=budget, max_new_tokens_policy="official natural EOS")
            save(args.out / "status.json", status)
            # Warm the model using an unrelated short context; no evaluated KV survives.
            started = clock()
            warm = Encbank(model, 0, tokenizer=tok)
            warm_sink = warm.write_chunk([warm._sink_prefix_id()])
            warm_context = warm.write_chunks([document["context_ids"][:64]])
            warm._decode_from_pack(warm_sink, warm_context, rows[0]["query_ids"][:16], tok.eos_token_id, 3, True)
            del warm, warm_sink, warm_context
            status["warmup_s_excluded"] = clock() - started
            started = time.perf_counter()
            context_ids = tok.encode(document["formatted_context"], add_special_tokens=False)
            document_prepare_s = time.perf_counter() - started
            require(context_ids == document["context_ids"], "Document tokenizer differs from prepared workload")
            chunks = list(torch.tensor(context_ids, dtype=torch.long).split(512))
            reader = new_reader(model, tok, args.arm, args.model, budget)
            started = clock()
            write = reader.write_store(context_ids, args.out / "store", 512)
            write_wall = clock() - started
            startup = reader.open_store(args.out / "store", "cpu")
            require(write["payload_tensor_bytes"] <= budget, "Document payload exceeds the common byte budget")
            fixed_cost = document_prepare_s + write["write_total_s"] + startup["startup_load_s"]
            save(args.out / "store_cost.json", {"write": write, "startup": startup,
                "write_wall_s_diagnostic": write_wall, "document_prepare_s": document_prepare_s,
                "fixed_cost_s": fixed_cost, "budget_bytes": budget})
            results = []
            with (args.out / "measurements.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    gc.collect()
                    clock()
                    started = time.perf_counter()
                    query_ids = tok.encode(row["formatted_query"], add_special_tokens=False)
                    question_ids = tok.encode(row["retrieval_question"], add_special_tokens=False)
                    query_prepare_s = time.perf_counter() - started
                    started = time.perf_counter()
                    selected = list(selectors.select_context_chunk_indices("bm25", chunks, question_ids, 12, None))
                    retrieval_s = time.perf_counter() - started
                    require(query_ids == row["query_ids"] and selected == row["selected_indices"], "Runtime query/retrieval differs from prepared inputs")
                    cap = min(8, row["sample"]["max_new_tokens"]) if args.smoke else row["sample"]["max_new_tokens"]
                    generated, stats = reader.query_ids(query_ids, selected_indices=selected, max_new_tokens=cap,
                        force_length=False, capture_logits=False)
                    if args.arm != "prefix":
                        stats.update(cache_resident_bytes=write["payload_tensor_bytes"], cache_budget_bytes=budget,
                            cache_hit_context_tokens=0 if args.arm == "j0" else stats["read_tokens"]-len(query_ids),
                            cache_miss_context_tokens=stats["read_tokens"]-len(query_ids) if args.arm == "j0" else 0,
                            cache_hit_semantics="j0 has no cross-query reuse; V2/CB hits are stored chunk representations, not exact full-layer prefix hits")
                    started = time.perf_counter()
                    prediction = tok.decode(generated, skip_special_tokens=True).strip()
                    text_decode_s = time.perf_counter() - started
                    score, scored_prediction = preparation.score_query(prediction, row)
                    require(0 <= score <= 1 and all(isinstance(t, int) for t in generated), "Invalid score or tokens")
                    record = {"id": row["id"], "document_id": args.document, "arm": args.arm,
                        "stream_index": row["stream_index"], "task": row["sample"]["task"],
                        "prediction": prediction, "scored_prediction": scored_prediction, "score": score,
                        "generated_ids": generated, "generated_tokens": len(generated), "max_new_tokens": cap,
                        "terminated_by_eos": len(generated) < cap, "selected_indices": selected,
                        "query_tokens": len(query_ids), "query_ids": query_ids, "stats": stats, "timestamp": now(),
                        "timings": {"document_tokenization_reused": True, "query_tokenization_s": query_prepare_s,
                            "external_retrieval_s": retrieval_s, "text_decode_s": text_decode_s,
                            "ttft_s": query_prepare_s + retrieval_s + stats["ttft_s"],
                            "query_total_s": query_prepare_s + retrieval_s + stats["total_s"] + text_decode_s},
                        "smoke_reference_equal": None, "timing_eligible": not args.smoke}
                    if args.smoke:
                        expected = fresh_reference(model, tok, args.arm, document, row, cap)
                        record["smoke_reference_equal"] = generated == list(expected)
                        require(record["smoke_reference_equal"], f"Reuse differs from fresh reference: {row['id']}; actual={generated}, expected={expected}")
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    results.append(record)
                    status.update(completed_queries=len(results), latest_query=row["id"], updated_at=now())
                    save(args.out / "status.json", status)
                    print(f"[{args.dataset}/{args.document}/{args.arm}] {len(results)}/{len(rows)} score={score:.3f} total={record['timings']['query_total_s']:.3f}s", flush=True)
            summary = summarize(results, fixed_cost)
            summary["peak_allocated_bytes_including_write"] = max(write["write_peak_allocated_bytes"], summary["peak_allocated_bytes"])
            summary["persistent_store_serialized_bytes"] = write["serialized_bytes"]
            summary["persistent_store_tensor_bytes"] = write["payload_tensor_bytes"]
            save(args.out / "summary.json", summary)
            reader.close_store()
            status.update(status="complete", finished_at=now(), reference_checks_passed=args.smoke,
                          new_generations=len(results), extra_reference_generations=len(results) if args.smoke else 0)
            save(args.out / "status.json", status)
            save(args.out / "COMPLETED.json", status)
            _release_lock()
        except BaseException as exc:
            status.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
            save(args.out / "status.json", status)
            save(args.out / "FAILED.json", status)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
