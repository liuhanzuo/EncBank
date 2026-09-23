"""Read-only remote CPU audit of complete BABILong V2/j0 task-length pairs."""
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

ROOT = Path("/data/liuhanzuo/comem_v2_20260908")
B = ROOT / "workspace/exp/comem_v2_benchmarks_20260908"
OUT = ROOT / "outputs/diagnostics/heartbeat_babilong_20260909_0129.json"
TASKS = ("qa1", "qa2", "qa3", "qa5")
LENGTHS = ("0k", "1k", "2k", "4k", "8k", "16k", "32k")
ARMS = ("fix_all", "j0")
BASELINE = "2026-09-08T16:33:56.472750+00:00"  # Previous qa3/2k audit.

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def summary(values):
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}

started = time.perf_counter()
snapshot_at = datetime.now(timezone.utc).isoformat()
pairs, pending = [], []
for task in TASKS:
    for length in LENGTHS:
        methods = {}
        for arm in ARMS:
            folder = ROOT / "outputs/benchmarks_v2/full/babilong" / arm / f"{task}_{length}"
            marker = folder / "COMPLETED.json"
            receipt = read(marker) if marker.exists() else None
            valid_complete = bool(receipt and receipt.get("status") == "completed"
                                  and sum(f.get("records", 0) for f in receipt.get("files", [])) == 100)
            methods[arm] = {"run": str(folder), "complete_receipt_n100": valid_complete,
                            "receipt_status": receipt.get("status") if receipt else None,
                            "finished_at": receipt.get("finished_at") if receipt else None}
        record = {"task": task, "length": length, "methods": methods}
        if all(m["complete_receipt_n100"] for m in methods.values()):
            pairs.append(record)
        else:
            pending.append(record)
print(json.dumps({"snapshot_at": snapshot_at, "complete_pairs": len(pairs),
                  "pending_pairs": len(pending), "cells": [f"{p['task']}/{p['length']}" for p in pairs]}, indent=2), flush=True)

sys.path[:0] = [str(B), str(ROOT / "workspace/COMem")]
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import AutoTokenizer
from prepare_babilong import load_task, format_prompt, _module
from benchmark_pack import BOUNDARY, tokenize_explicit_prompt
from reader_adapter import ExplicitPackReader, _plain

require(os.environ["CUDA_VISIBLE_DEVICES"] == "" and not torch.cuda.is_initialized(), "CPU audit cannot initialize CUDA")
tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-8B", local_files_only=True)
signature = inspect.signature(ExplicitPackReader.__new__(ExplicitPackReader).generate_from_ids)
official_metrics = _module("metrics")
result = {
    "timestamp": snapshot_at, "execution": "longjing-1 remote CPU only; CUDA hidden; tokenizer and scoring; no model",
    "baseline_audit_timestamp": BASELINE,
    "planned_tasks": list(TASKS), "planned_lengths": list(LENGTHS), "planned_pairs": 28,
    "complete_pairs": [], "pending_pairs": pending, "writeback": [], "validation_errors": [],
    "matrix_cell_format": "CoMem V2 / j0; official accuracy percent; n=100 per method; null means incomplete pair",
    "matrix": {t: {length: None for length in LENGTHS} for t in TASKS},
    "protocol": {
        "model": str(ROOT / "models/Qwen3-8B"), "model_dtype": "bfloat16", "v2_split": 12, "j0_effective_split": 0,
        "data": str(B / "data/babilong"), "expected_samples_per_task_length_method": 100,
        "prompt": "Vendored official DEFAULT_PROMPTS and DEFAULT_TEMPLATE, including instruction/examples/post-prompt",
        "chat_template": False, "context_query_tokenization": "independent, explicit boundary; no padding", "truncation": "none",
        "selector": "BM25", "topk": 4, "chunk_size": 512,
        "generation": "greedy native explicit reader; max_new_tokens=20 (CoMem evaluation setting, not a universal BABILong requirement)",
        "metric": str(B / "vendor/babilong/babilong/metrics.py") + "::compare_answers / TASK_LABELS",
        "cache_identity": "Reconstructed exact input token IDs plus all bound generation options SHA256 -> existing read-only per-method SQLite prediction/n_tokens; no elapsed_s read",
        "paired_inputs": "Same raw prepared source, ordered IDs, target, question, task, length, selected indices and entire reconstructed pack",
    },
    "limitations": [
        "Only the requested prepared qa1/qa2/qa3/qa5 x 0k/1k/2k/4k/8k/16k/32k grid is covered; this is not all twenty BABILong tasks or larger lengths.",
        "0k is the official zero-distractor length label; prompts still contain context and examples.",
        "Document length labels differ from actual model-tokenized lengths and selected read-pack lengths; actual ranges are exported per cell.",
        "BM25 top4 retrieval may omit supporting facts. These task accuracies do not by themselves prove evidence-present multi-fact reasoning.",
        "QA2 and QA3 require two and three supporting facts; QA5 is a three-argument relation task, not three-support-fact reasoning.",
        "No remote latency or memory measurements are consumed or exported; local timing results are not audited here.",
        "Only complete n100 paired cells enter the matrix; pending cells never contribute partial accuracy.",
    ],
}

for pair in pairs:
    task, length = pair["task"], pair["length"]
    source = load_task(task, length)
    require(len(source) == 100, f"{task}/{length}: source n must be 100")
    raw_rows, configs, receipts, dbs, outputs = {}, {}, {}, {}, {}
    try:
        for arm in ARMS:
            run = Path(pair["methods"][arm]["run"])
            receipt, config = read(run / "COMPLETED.json"), read(run / "run_config.json")
            receipts[arm], configs[arm] = receipt, config
            require(receipt["status"] == "completed" and receipt["new_generations"] == 100 and receipt["reused_generations"] == 0, f"{task}/{length}/{arm}: completion/generation scope")
            require(receipt["reader"]["class"] == ("CoMemLower" if arm == "fix_all" else "CoMem") and receipt["reader"]["arm"] == arm, "Wrong reader class/arm")
            require(receipt["reader"]["requested_j"] == 12 and receipt["reader"]["effective_j"] == (12 if arm == "fix_all" else 0), "Wrong effective split")
            require(config["benchmark"] == "babilong" and config["arm"] == arm, "Wrong run config method")
            opts = config["driver_options"]
            expected = {"model_path": str(ROOT / "models/Qwen3-8B"), "selector": "bm25", "topk": 4, "chunk_size": 512,
                        "tasks": [task], "lengths": [length], "max_new_tokens": 20, "limit": 100, "num_shards": 1, "shard_index": 0,
                        "dtype": "bfloat16", "lora_adapter": "", "baseline": "none", "sink_tokens": "bos",
                        "iter_rounds": 0, "iter_hop_topk": 2, "iter_score": "meanpool", "attn_impl": "sdpa"}
            require(all(opts.get(k) == v for k, v in expected.items()), "Wrong protocol options")
            require(len(receipt["files"]) == 1 and receipt["files"][0]["records"] == 100, "Wrong CSV scope")
            output = Path(receipt["files"][0]["file"])
            require(output.is_relative_to(run / "attempts") and output.name == f"{task}_{length}_official_explicit.csv", "Wrong active output location")
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == 100 and [int(row["index"]) for row in rows] == list(range(100)), "Wrong row count/order")
            require([row["id"] for row in rows] == [f"{task}/{length}/{i}" for i in range(100)], "Wrong source IDs")
            db = sqlite3.connect(f"file:{run / 'generations.sqlite3'}?mode=ro", uri=True)
            require(db.execute("select count(*) from generations").fetchone()[0] == 100, "Wrong successful cache count")
            raw_rows[arm], dbs[arm], outputs[arm] = rows, db, output
        require(configs["fix_all"]["driver_options"] == configs["j0"]["driver_options"], "Paired driver options differ")
        values = {arm: [] for arm in ARMS}
        packs, ordered_keys = [], []
        for index, sample in enumerate(source):
            marked = format_prompt(dict(sample, input=sample["input"].strip() + BOUNDARY), task)
            ids, context_count, selected, pack = tokenize_explicit_prompt(tokenizer, marked, sample["question"], 512, "bm25", 4, None, 0, 2, "meanpool")
            bound = signature.bind(ids, context_token_count=context_count, selected_indices=selected, chunk_size=512, max_new_tokens=20)
            bound.apply_defaults()
            generation = dict(bound.arguments)
            generation.pop("input_ids")
            generation.pop("dense_retriever", None)
            generation.pop("tokenizer", None)
            key = hashlib.sha256(json.dumps({"tokens": _plain(ids), "generation": _plain(generation)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for arm in ARMS:
                row = raw_rows[arm][index]
                require(row["target"] == sample["target"] and row["question"] == sample["question"] and row["task"] == task and row["length"] == length, "CSV source target/question/task/length mismatch")
                require(row["status"] == "ok" and row["output"] != "[OOM]", "Failed output cannot be scored")
                value = float(official_metrics.compare_answers(sample["target"], row["output"], sample["question"], official_metrics.TASK_LABELS[task]))
                require(value in (0.0, 1.0) and value == float(row["score"]), "Official per-row rescore mismatch")
                require(json.loads(row["pack"]) == pack, "Actual reconstructed pack mismatch")
                cached = dbs[arm].execute("select prediction,n_tokens from generations where key=?", (key,)).fetchone()
                require(cached is not None and json.loads(cached[0]) == row["output"] and cached[1] == ids.numel(), "Exact input/cache prediction mismatch")
                values[arm].append(value)
            for field in ("index", "id", "task", "length", "target", "question", "pack"):
                require(raw_rows["fix_all"][index][field] == raw_rows["j0"][index][field], "Paired source or pack differs")
            packs.append(pack)
            ordered_keys.append(key)
        scores = {arm: sum(values[arm]) for arm in ARMS}
        cell = {"task": task, "length": length, "n_per_method": 100, "complete": True,
                "new_since_prior_audit_timestamp": max(receipts[a]["finished_at"] for a in ARMS) > BASELINE,
                "scores_percent": scores, "display": f"{scores['fix_all']:.0f} / {scores['j0']:.0f}",
                "v2_minus_j0_pp": scores["fix_all"] - scores["j0"],
                "paired_source_and_pack": True, "official_rescore_matches": 200, "source_checks": 200,
                "exact_token_cache_checks": 200, "unique_reconstructed_packs": 100,
                "ordered_cache_keys": ordered_keys,
                "actual_tokens": {k: summary([p[k] for p in packs]) for k in ("input_tokens", "context_tokens", "query_tokens", "context_chunks", "read_pack_tokens")},
                "selected_chunk_count": summary([len(p["selected_indices"]) for p in packs]),
                "paired_outcomes": {"both_correct": sum(l == r == 1 for l, r in zip(values['fix_all'], values['j0'])),
                                    "v2_only_correct": sum(l == 1 and r == 0 for l, r in zip(values['fix_all'], values['j0'])),
                                    "j0_only_correct": sum(l == 0 and r == 1 for l, r in zip(values['fix_all'], values['j0'])),
                                    "both_incorrect": sum(l == r == 0 for l, r in zip(values['fix_all'], values['j0']))},
                "methods": {}}
        for arm in ARMS:
            record = {"benchmark": "babilong", "task": task, "length": length, "arm": arm, "n": 100, "expected_n": 100,
                      "score_percent": scores[arm], "display": f"{scores[arm]:.2f}", "complete": True,
                      "official_rescore_matches": 100, "source_id_checks": 100, "reconstructed_pack_checks": 100,
                      "exact_generation_cache_checks": 100, "generation_cap": 20, "new_generations": 100, "reused_generations": 0,
                      "source": str(outputs[arm]), "completion_receipt": str(Path(pair['methods'][arm]['run']) / "COMPLETED.json"),
                      "finished_at": receipts[arm]["finished_at"]}
            cell["methods"][arm] = record
            result["writeback"].append(record)
        result["complete_pairs"].append(cell)
        result["matrix"][task][length] = {"fix_all": scores["fix_all"], "j0": scores["j0"], "n_per_method": 100, "display": cell["display"]}
        print(f"[audit] {task}/{length}: {cell['display']}; input/pack/cache/official metric 200/200", flush=True)
    except Exception as exc:
        result["validation_errors"].append({"task": task, "length": length, "error": repr(exc)})
        print(f"[audit ERROR] {task}/{length}: {exc!r}", flush=True)
    finally:
        for db in dbs.values():
            db.close()

require(not torch.cuda.is_initialized(), "Audit unexpectedly initialized CUDA")
result.update(completed_pairs_count=len(result["complete_pairs"]), pending_pairs_count=len(pending),
              audited_method_cells=len(result["writeback"]), audited_predictions=100*len(result["writeback"]),
              new_pairs_since_prior_audit=[f"{c['task']}/{c['length']}" for c in result["complete_pairs"] if c["new_since_prior_audit_timestamp"]],
              all_planned_pairs_complete=len(result["complete_pairs"]) == 28 and not result["validation_errors"],
              audit_wall_seconds=time.perf_counter()-started,
              runtime={"host": os.uname().nodename, "python": sys.executable, "torch_cpu_threads": torch.get_num_threads(),
                       "torch_interop_threads": torch.get_num_interop_threads(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                       "cuda_initialized": torch.cuda.is_initialized(), "model_loaded": False},
              audited_at=datetime.now(timezone.utc).isoformat())
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUT), "completed_pairs": len(result['complete_pairs']), "pending_pairs": len(pending),
                  "validation_errors": result["validation_errors"], "audit_wall_seconds": result["audit_wall_seconds"], "matrix": result["matrix"]}, indent=2), flush=True)
if result["validation_errors"]:
    raise SystemExit(1)
