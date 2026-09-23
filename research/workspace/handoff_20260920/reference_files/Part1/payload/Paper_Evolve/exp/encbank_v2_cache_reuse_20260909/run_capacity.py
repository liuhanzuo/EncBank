"""Finite CPU cache, real multi-document request traces on the local RTX 5090."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from run_reuse import (ARMS, HERE, ROOT, VERSION, fresh_reference, load_bundle,
                       now, require, save, summarize)
from local_execution_pause import require_unpaused

CAPACITY_VERSION = "real-question-capacity-v1"


def cohorts(dataset, documents):
    if dataset == "locomo":
        return {"locomo-00": documents}
    groups = {}
    for task in ("scbench_qa_eng", "scbench_choice_eng"):
        rows = [d for d in documents if d["document_id"].startswith(task + "/")]
        for start in range(0, len(rows), 4):
            groups[f"{task}-{start//4:02d}"] = rows[start:start+4]
    require(sum(map(len, groups.values())) == len(documents), "SCBench documents not fully assigned")
    return groups


def request_trace(documents, queries, dataset, smoke=False):
    chosen = documents[:2] if smoke else documents
    grouped = {d["document_id"]: sorted((q for q in queries if q["document_id"] == d["document_id"]),
                                      key=lambda q: q["stream_index"]) for d in chosen}
    for key in grouped:
        if smoke:
            grouped[key] = grouped[key][:3]
        elif dataset == "locomo":
            require(len(grouped[key]) >= 80, "LoCoMo capacity trace requires 80 different questions per document")
            grouped[key] = grouped[key][:80]
    order = list(grouped)
    random.Random(20260909).shuffle(order)
    trace = [grouped[key][i] for i in range(max(map(len, grouped.values()))) for key in order if i < len(grouped[key])]
    require(len(trace) == len({q["id"] for q in trace}), "Duplicate query in capacity trace")
    return chosen, trace


def reference_working_set(documents, rows, bytes_per_token):
    used = {d["document_id"]: set() for d in documents}
    docmap = {d["document_id"]: d for d in documents}
    for row in rows:
        used[row["document_id"]].update(row["selected_indices"])
    tokens = 1
    chunk_count = 0
    for key, indices in used.items():
        count = len(docmap[key]["context_ids"])
        for index in indices:
            size = min(512, count-index*512)
            require(size > 0, "Invalid referenced chunk")
            tokens += size
            chunk_count += 1
    return {"scope": "predeclared trace referenced chunk union plus one shared sink; not the full source corpus",
            "full_depth_kv_bytes_per_token": bytes_per_token, "referenced_tokens_with_sink": tokens,
            "referenced_chunks": chunk_count, "full_depth_working_set_bytes": tokens*bytes_per_token,
            "whole_documents_full_depth_bytes": (1+sum(len(d["context_ids"]) for d in documents))*bytes_per_token}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("locomo", "scbench"), required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--fraction", type=float, choices=(.25, .5, 1.0), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    require_unpaused(HERE)
    helper, manifest, all_documents, all_queries = load_bundle(args.dataset, args.fixtures)
    documents, rows = request_trace(cohorts(args.dataset, all_documents)[args.cohort], all_queries, args.dataset, args.smoke)
    docmap = {d["document_id"]: d for d in documents}
    config = {"protocol": CAPACITY_VERSION, "dataset": args.dataset, "fixtures": str(args.fixtures.resolve()),
              "cohort": args.cohort, "documents": list(docmap), "arm": args.arm, "model": args.model,
              "fraction": args.fraction, "smoke": args.smoke, "ids": [r["id"] for r in rows],
              "chunk_size": 512, "topk": 12, "dtype": "bfloat16", "j": 12, "query_order_seed": 20260909,
              "answer_cache": False, "source_context_truncation": "none", "source_file_io": "excluded",
              "cache_fill": "on demand, synchronous; lookup/capture/D2H/LRU all charged to request",
              "raw_document_ids": "prepared once per cohort; excluded from KV budget, bytes reported",
              "j0_budget_independent": args.arm == "j0"}
    args.out.mkdir(parents=True, exist_ok=True)
    from run_driver import OutputLock
    with OutputLock(args.out):
        require(not (args.out / "measurements.jsonl").exists(), "Fresh attempt required for a stateful trace")
        save(args.out / "config.json", config)
        status = {"status": "waiting_for_gpu", "pid": os.getpid(), "started_at": now(),
                  "dataset": args.dataset, "cohort": args.cohort, "arm": args.arm,
                  "expected_queries": len(rows), "completed_queries": 0, "smoke": args.smoke,
                  "timing_eligible": not args.smoke}
        save(args.out / "status.json", status)
        try:
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(16)
            from gpu_gate import acquire_gpu, _release_lock
            require_unpaused(HERE)
            admission = acquire_gpu(22, None, 5.0, max_wait=86400, tag="real_question_capacity " + str(args.out))
            # A pause may be requested while the original gate waits for idle.
            # Never load a model merely because that wait later succeeds.
            require_unpaused(HERE)
            from serving_reuse import device_provenance
            hardware = device_provenance("cuda:0")
            hardware.update(gpu_admission=admission, memory_cap_bytes=28_000_000_000)
            torch.cuda.set_per_process_memory_fraction(min(28e9/hardware["total_memory_bytes"], 1.0))
            import official_qa_driver as driver
            from capacity_reader import make_capacity_reader
            from encbank import Encbank, selectors
            def clock():
                torch.cuda.synchronize()
                return time.perf_counter()
            torch.manual_seed(42)
            started = clock()
            require_unpaused(HERE)
            model, tok = driver.load_backbone(args.model, "bfloat16", "sdpa", "cuda:0", "")
            model.eval()
            status["model_load_s_excluded"] = clock()-started
            cfg = model.config
            working = reference_working_set(documents, rows, 4*cfg.num_hidden_layers*cfg.num_key_value_heads*cfg.head_dim)
            budget = int(working["full_depth_working_set_bytes"]*args.fraction)
            require(budget <= 32*1024**3, "Cohort exceeds the predeclared 32 GiB CPU capacity guard; do not silently alter budget")
            config.update(cache_budget_bytes=budget, working_set=working)
            save(args.out / "config.json", config)
            status.update(status="running", hardware=hardware, cache_budget_bytes=budget)
            save(args.out / "status.json", status)
            started = clock()
            warm = Encbank(model, 0, tokenizer=tok)
            wh = warm.write_chunk([warm._sink_prefix_id()])
            wc = warm.write_chunks([documents[0]["context_ids"][:64]])
            warm._decode_from_pack(wh, wc, rows[0]["query_ids"][:16], tok.eos_token_id, 3, True)
            del warm, wh, wc
            status["warmup_s_excluded"] = clock()-started
            prepared, doc_costs = {}, {}
            for doc in documents:
                started = time.perf_counter()
                ids = tok.encode(doc["formatted_context"], add_special_tokens=False)
                chunks = list(torch.tensor(ids, dtype=torch.long).split(512))
                doc_costs[doc["document_id"]] = time.perf_counter()-started
                require(ids == doc["context_ids"], "Document tokenizer differs from prepared inputs")
                prepared[doc["document_id"]] = (ids, chunks)
            fixed_cost = sum(doc_costs.values())
            reader = make_capacity_reader(model, 12, tok, args.arm, model_id=args.model,
                                          cache_budget_bytes=budget, chunk_size=512)
            save(args.out / "store_cost.json", {"fixed_cost_s": fixed_cost, "document_prepare_s": doc_costs,
                 "initial_persistent_kv_bytes": 0, "raw_token_bytes": sum(len(x[0])*8 for x in prepared.values()),
                 "serialized_kv_bytes": 0, "cache_policy": "CPU-only on-demand; no disk KV backing store"})
            results = []
            with (args.out / "measurements.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    gc.collect()
                    clock()
                    started = time.perf_counter()
                    reader.bind_document(row["document_id"], prepared[row["document_id"]][0])
                    bind_s = time.perf_counter()-started
                    started = time.perf_counter()
                    query_ids = tok.encode(row["formatted_query"], add_special_tokens=False)
                    bare = tok.encode(row["retrieval_question"], add_special_tokens=False)
                    query_prepare_s = time.perf_counter()-started
                    started = time.perf_counter()
                    selected = list(selectors.select_context_chunk_indices("bm25", prepared[row["document_id"]][1], bare, 12, None))
                    retrieval_s = time.perf_counter()-started
                    require(query_ids == row["query_ids"] and selected == row["selected_indices"], "Runtime query/retrieval differs from fixture")
                    cap = min(8, row["sample"]["max_new_tokens"]) if args.smoke else row["sample"]["max_new_tokens"]
                    generated, stats = reader.query_ids(query_ids, selected_indices=selected,
                                                        max_new_tokens=cap, force_length=False, capture_logits=False)
                    started = time.perf_counter()
                    prediction = tok.decode(generated, skip_special_tokens=True).strip()
                    decode_text_s = time.perf_counter()-started
                    score, scored = helper.score_query(prediction, row)
                    require(0 <= score <= 1, "Invalid official score")
                    require(stats["cache_resident_bytes"] <= budget, "Measured persistent cache exceeds budget")
                    extra = bind_s+query_prepare_s+retrieval_s
                    record = {"id": row["id"], "document_id": row["document_id"], "arm": args.arm,
                              "trace_index": len(results), "stream_index": row["stream_index"], "task": row["sample"]["task"],
                              "prediction": prediction, "scored_prediction": scored, "score": score,
                              "generated_ids": generated, "generated_tokens": len(generated), "max_new_tokens": cap,
                              "terminated_by_eos": len(generated)<cap, "query_ids": query_ids, "query_tokens": len(query_ids),
                              "selected_indices": selected, "stats": stats, "timestamp": now(),
                              "timings": {"document_bind_s": bind_s, "query_tokenization_s": query_prepare_s,
                                  "external_retrieval_s": retrieval_s, "text_decode_s": decode_text_s,
                                  "ttft_s": extra+stats["ttft_s"], "query_total_s": extra+stats["total_s"]+decode_text_s},
                              "timing_eligible": not args.smoke, "smoke_reference_equal": None}
                    if args.smoke:
                        expected = fresh_reference(model, tok, args.arm, docmap[row["document_id"]], row, cap)
                        record["smoke_reference_equal"] = generated == list(expected)
                        require(record["smoke_reference_equal"], f"Capacity/fresh mismatch {row['id']}: {generated} != {expected}")
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    results.append(record)
                    status.update(completed_queries=len(results), latest_query=row["id"], updated_at=now())
                    save(args.out / "status.json", status)
                    print(f"[{args.dataset}/{args.cohort}/{args.arm}/{args.fraction}] {len(results)}/{len(rows)} score={score:.3f} total={record['timings']['query_total_s']:.3f}s", flush=True)
            summary = summarize(results, fixed_cost)
            summary.update(cache_budget_bytes=budget, working_set=working,
                           phase="B; independent multi-document cold-cache trace", final_cache_info=reader.cache_info())
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
