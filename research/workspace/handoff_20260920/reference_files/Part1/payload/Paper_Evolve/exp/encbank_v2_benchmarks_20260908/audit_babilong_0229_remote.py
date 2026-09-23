"""Remote CPU audit: new complete pub/pub_sink/cbos cells, reuse prior V2/j0 audit."""
from __future__ import annotations
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
from pathlib import Path
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import sqlite3
import statistics
import sys
import time

ROOT = Path("/data/liuhanzuo/encbank_v2_20260908")
B = ROOT / "workspace/exp/encbank_v2_benchmarks_20260908"
OUT = ROOT / "outputs/diagnostics/heartbeat_babilong_20260909_0229.json"
PRIOR = ROOT / "outputs/diagnostics/heartbeat_babilong_20260909_0129.json"
TASKS = ("qa1", "qa2", "qa3", "qa5")
LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k")
ARMS = ("pub", "pub_sink", "cbos")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def summarize(vals):
    return {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}

started = time.perf_counter()
prior = read(PRIOR)
require(prior["completed_pairs_count"] == 28 and not prior["validation_errors"], "Previous V2/j0 audit incomplete")
prior_by_cell = {(c["task"], c["length"]): c for c in prior["complete_pairs"]}
ready, pending = {}, []
for task in TASKS:
    for length in LENGTHS:
        for arm in ARMS:
            run = ROOT / "outputs/benchmarks_v2/full/babilong" / arm / f"{task}_{length}"
            receipt = read(run / "COMPLETED.json") if (run / "COMPLETED.json").exists() else None
            n = sum(f.get("records", 0) for f in receipt.get("files", [])) if receipt else 0
            if receipt and receipt.get("status") == "completed" and n == 100:
                ready.setdefault((task, length), []).append(arm)
            else:
                pending.append({"task": task, "length": length, "arm": arm, "run": str(run),
                                "status": "no_complete_n100_receipt", "completed_receipt_records": n,
                                "score_percent": None})
print(json.dumps({"complete_baseline_cells": sum(map(len, ready.values())), "pending_baseline_cells": len(pending),
                  "ready": {f"{t}/{l}": a for (t,l),a in ready.items()}}, indent=2), flush=True)

sys.path[:0] = [str(B), str(ROOT / "workspace/Encbank")]
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer
from prepare_babilong import load_task, format_prompt, _module
from benchmark_pack import BOUNDARY, tokenize_explicit_prompt
from reader_adapter import ExplicitPackReader, _plain

require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-8B", local_files_only=True)
signature = inspect.signature(ExplicitPackReader.__new__(ExplicitPackReader).generate_from_ids)
metrics = _module("metrics")
result = {"timestamp": datetime.now(timezone.utc).isoformat(), "scope": "New complete BABILong pub/pub_sink/cbos method cells only",
          "prior_v2_j0_audit": str(PRIOR), "prior_v2_j0_recomputed": False,
          "planned_new_baseline_cells": 84, "writeback": [], "comparisons": [], "pending_cells": pending,
          "display_rows": [], "validation_errors": [],
          "protocol": {"model": "Qwen3-8B BF16, frozen base weights", "requested_j": 12,
                       "effective_j": {"pub": 12, "pub_sink": 12, "cbos": 36},
                       "reader_class": {"pub": "Encbank", "pub_sink": "Encbank", "cbos": "EncbankLower"},
                       "write_sink": {"pub": False, "pub_sink": True, "cbos": True},
                       "data": str(B / "data/babilong"), "n_per_method_cell": 100,
                       "retrieval": "BM25 top4 chunk512, shared source and ordered selected pack",
                       "prompt": "Official DEFAULT_PROMPTS/DEFAULT_TEMPLATE, no chat wrapper, independent explicit context/query boundary, no truncation/padding",
                       "generation_cap": 20, "generation_cap_scope": "Encbank benchmark setting, not a universal BABILong cap",
                       "metric": str(B / "vendor/babilong/babilong/metrics.py") + "::compare_answers / TASK_LABELS",
                       "cache_check": "Reconstruct exact input IDs plus bound generation defaults -> native per-method SQLite key/prediction/n_tokens; never read elapsed_s"},
          "limitations": ["Only complete n100 baseline cells are scored; incomplete outputs never enter displayed means.",
                          "The original 28-pair V2/j0 audit is reused; its model outputs are not regenerated or rescored.",
                          "0k means no added distractor haystack, not an empty prompt; 0k baseline results do not establish long-context baseline ranking.",
                          "QA5 is a three-argument relation task with one supporting fact, not three-support-fact reasoning.",
                          "cbos is isolated full-depth KV reuse, not CacheBlend selective recomputation.",
                          "Descriptive paired means and per-example outcomes only; no significance or universal superiority claim.",
                          "No GPU model work, remote timing consumption, paper edit, or queue change."]}

for (task, length), arms in ready.items():
    source = load_task(task, length)
    require(len(source) == 100, "Prepared source is not n100")
    old = prior_by_cell[task, length]
    reference_rows = {arm: csv_rows(old["methods"][arm]["source"]) for arm in ("fix_all", "j0")}
    require(all(len(v) == 100 for v in reference_rows.values()), "Previously audited reference CSV incomplete")
    rows_by_arm, receipts, dbs, configs, paths = {}, {}, {}, {}, {}
    try:
        for arm in arms:
            run = ROOT / "outputs/benchmarks_v2/full/babilong" / arm / f"{task}_{length}"
            receipt, config = read(run / "COMPLETED.json"), read(run / "run_config.json")
            require(receipt["status"] == "completed" and receipt["new_generations"] == 100 and receipt["reused_generations"] == 0, "Incomplete or reused-generation scope")
            require(receipt["reader"]["class"] == result["protocol"]["reader_class"][arm] and receipt["reader"]["arm"] == arm, "Wrong actual reader")
            require(receipt["reader"]["requested_j"] == 12 and receipt["reader"]["effective_j"] == result["protocol"]["effective_j"][arm], "Wrong actual split")
            require(receipt["reader"]["write_sink"] == result["protocol"]["write_sink"][arm], "Wrong actual write-sink policy")
            opts = config["driver_options"]
            require(config["benchmark"] == "babilong" and config["arm"] == arm, "Wrong run method")
            expected = {"model_path": str(ROOT / "models/Qwen3-8B"), "selector": "bm25", "topk": 4, "chunk_size": 512,
                        "tasks": [task], "lengths": [length], "max_new_tokens": 20, "limit": 100, "num_shards": 1, "shard_index": 0,
                        "dtype": "bfloat16", "lora_adapter": "", "baseline": "none", "sink_tokens": "bos",
                        "iter_rounds": 0, "iter_hop_topk": 2, "iter_score": "meanpool", "attn_impl": "sdpa"}
            require(all(opts.get(k) == v for k,v in expected.items()), "Protocol options differ")
            for reference_arm in ("fix_all", "j0"):
                reference_run = Path(old["methods"][reference_arm]["completion_receipt"]).parent
                require(opts == read(reference_run / "run_config.json")["driver_options"], "Baseline/reference options differ")
            require(len(receipt["files"]) == 1 and receipt["files"][0]["records"] == 100, "Wrong CSV count")
            path = Path(receipt["files"][0]["file"])
            require(path.is_relative_to(run / "attempts") and path.name == f"{task}_{length}_official_explicit.csv", "Wrong output attempt")
            rows = csv_rows(path)
            require(len(rows) == 100 and [int(r["index"]) for r in rows] == list(range(100)), "Wrong CSV row count/order")
            require([r["id"] for r in rows] == [f"{task}/{length}/{i}" for i in range(100)], "Wrong CSV source IDs")
            db = sqlite3.connect(f"file:{run / 'generations.sqlite3'}?mode=ro", uri=True)
            require(db.execute("select count(*) from generations").fetchone()[0] == 100, "Wrong cache count")
            rows_by_arm[arm], receipts[arm], configs[arm], dbs[arm], paths[arm] = rows, receipt, config, db, path
        values = {a: [] for a in arms}
        packs = []
        for i, sample in enumerate(source):
            marked = format_prompt(dict(sample, input=sample["input"].strip() + BOUNDARY), task)
            ids, count, selected, pack = tokenize_explicit_prompt(tokenizer, marked, sample["question"], 512, "bm25", 4, None, 0, 2, "meanpool")
            bound = signature.bind(ids, context_token_count=count, selected_indices=selected, chunk_size=512, max_new_tokens=20)
            bound.apply_defaults()
            opts = dict(bound.arguments)
            for name in ("input_ids", "dense_retriever", "tokenizer"):
                opts.pop(name, None)
            key = hashlib.sha256(json.dumps({"tokens": _plain(ids), "generation": _plain(opts)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            require(key == old["ordered_cache_keys"][i], "Input/token/cache identity differs from previously audited V2/j0 source")
            for arm in arms:
                row = rows_by_arm[arm][i]
                require(row["target"] == sample["target"] and row["question"] == sample["question"] and row["task"] == task and row["length"] == length, "Source fields differ")
                require(row["status"] == "ok" and row["output"] != "[OOM]", "Failed output")
                score = float(metrics.compare_answers(sample["target"], row["output"], sample["question"], metrics.TASK_LABELS[task]))
                require(score == float(row["score"]) and score in (0.0, 1.0), "Official score mismatch")
                require(json.loads(row["pack"]) == pack, "Pack reconstruction mismatch")
                cached = dbs[arm].execute("select prediction,n_tokens from generations where key=?", (key,)).fetchone()
                require(cached is not None and json.loads(cached[0]) == row["output"] and cached[1] == ids.numel(), "Native input/cache key/prediction mismatch")
                for ref in ("fix_all", "j0"):
                    for field in ("index", "id", "task", "length", "target", "question", "pack"):
                        require(row[field] == reference_rows[ref][i][field], "Baseline/reference source or pack differs")
                values[arm].append(score)
            packs.append(pack)
        display = {"task": task, "length": length, "n_per_complete_method": 100,
                   "scores_percent": {a: None for a in ARMS}, "prior_reference_scores_percent": old["scores_percent"],
                   "actual_tokens": {k: summarize([p[k] for p in packs]) for k in ("input_tokens", "context_tokens", "query_tokens", "read_pack_tokens")}}
        for arm in arms:
            score = sum(values[arm])
            record = {"benchmark": "babilong", "task": task, "length": length, "arm": arm, "n": 100, "expected_n": 100,
                      "score_percent": score, "display": f"{score:.2f}", "complete": True,
                      "official_rescore_matches": 100, "source_id_checks": 100, "reconstructed_pack_checks": 100,
                      "exact_generation_cache_checks": 100, "prior_reference_input_key_matches": 100,
                      "paired_source_and_pack": True, "generation_cap": 20, "new_generations": 100, "reused_generations": 0,
                      "source": str(paths[arm]), "completion_receipt": str(Path(paths[arm]).parents[3] / "COMPLETED.json"),
                      "reader": {k: receipts[arm]["reader"][k] for k in ("class", "arm", "requested_j", "effective_j", "write_sink")},
                      "finished_at": receipts[arm]["finished_at"]}
            # Keep the actual stable run marker, not an attempt-local marker.
            record["completion_receipt"] = str(ROOT / "outputs/benchmarks_v2/full/babilong" / arm / f"{task}_{length}" / "COMPLETED.json")
            result["writeback"].append(record)
            display["scores_percent"][arm] = score
            for ref in ("fix_all", "j0"):
                ref_values = [float(r["score"]) for r in reference_rows[ref]]
                require(sum(ref_values) == old["scores_percent"][ref], "Reference stored scores changed since prior audit")
                result["comparisons"].append({"task": task, "length": length, "n": 100,
                    "baseline_arm": arm, "baseline_score_percent": score, "reference_arm": ref,
                    "reference_score_percent": old["scores_percent"][ref], "reference_minus_baseline_pp": old["scores_percent"][ref]-score,
                    "paired_source_and_pack": True, "reference_score_from_prior_audit": True,
                    "baseline_only_correct": sum(x == 1 and y == 0 for x,y in zip(values[arm], ref_values)),
                    "reference_only_correct": sum(x == 0 and y == 1 for x,y in zip(values[arm], ref_values)),
                    "both_correct": sum(x == y == 1 for x,y in zip(values[arm], ref_values)),
                    "both_incorrect": sum(x == y == 0 for x,y in zip(values[arm], ref_values))})
            print(f"[audit] {task}/{length}/{arm}: {score:.0f}; score/source/token/pack/cache=100/100", flush=True)
        result["display_rows"].append(display)
    except Exception as exc:
        result["validation_errors"].append({"task": task, "length": length, "arms": arms, "error": repr(exc)})
        print(f"[audit ERROR] {task}/{length}: {exc!r}", flush=True)
    finally:
        for db in dbs.values():
            db.close()

require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
result.update(completed_new_baseline_cells=len(result["writeback"]), newly_audited_predictions=100*len(result["writeback"]),
              pending_baseline_cells=len(pending), all_new_baseline_grid_complete=len(result["writeback"]) == 84,
              audit_wall_seconds=time.perf_counter()-started,
              runtime={"ssh_alias": "longjing-1", "host": os.uname().nodename, "python": sys.executable,
                       "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                       "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "cuda_initialized": torch.cuda.is_initialized(), "model_loaded": False},
              audited_at=datetime.now(timezone.utc).isoformat())
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "complete": result["completed_new_baseline_cells"], "pending": len(pending),
                  "errors": result["validation_errors"], "rows": result["display_rows"], "cpu_s": result["audit_wall_seconds"]}, indent=2), flush=True)
if result["validation_errors"]:
    raise SystemExit(1)
