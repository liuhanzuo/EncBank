"""Verified fixed-pack support intervention using the existing official QA reader.

Accuracy only. An external remote queue owns the GPU, with a fresh idle check
after CPU validation. Natural controls are reused from the original main runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
from unittest.mock import patch

import official_qa_driver as qa
from prepare_oracle_support import ARMS, CHUNK, TOPK, SEED, digest, require, make_pack, forced_indices, source_samples

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_and_validate(args):
    from transformers import AutoTokenizer
    manifest = read_json(args.inputs.parent / "manifest.json")
    require(manifest["status"] == "ready" and manifest["cuda_initialized"] is False, "Fixture not ready")
    require(hashlib.sha256(args.inputs.read_bytes()).hexdigest() == args.expected_inputs_sha256 == manifest["files"]["inputs.jsonl"], "Input fixture changed")
    for filename, expected in manifest["files"].items():
        require(hashlib.sha256((args.inputs.parent / filename).read_bytes()).hexdigest() == expected, f"Fixture file changed: {filename}")
    indices = [int(i) for i in args.indices.split(",")]
    require(indices and len(indices) == len(set(indices)), "Empty or duplicate requested indices")
    all_rows = [json.loads(line) for line in args.inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in all_rows if r["task"] == args.task and r["index"] in indices]
    require([r["index"] for r in rows] == indices, "Requested fixture indices are missing or reordered")
    require(not args.smoke_only or len(rows) == 1 and "smoke" in args.out.parts, "Smoke/full outputs must be separate")
    contexts = read_json(args.inputs.parent / "contexts.json")
    samples = {(benchmark, s["id"]): s for benchmark, s in source_samples()}
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    checked_contexts, checked = set(), []
    for row in rows:
        require(digest({k: v for k, v in row.items() if k != "fixture_row_sha256"}) == row["fixture_row_sha256"], "Fixed row changed")
        sample = samples[(row["source_benchmark"], row["id"])]
        require({k: v for k, v in sample.items() if k != "marked_prompt"} == row["sample"], "Source question/answer/annotation/cap changed")
        formatted = tok.apply_chat_template([{"role": "user", "content": sample["marked_prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(formatted.count(qa.BOUNDARY) == 1, "Invalid context/query boundary")
        context_text, query_text = formatted.split(qa.BOUNDARY)
        context = contexts[row["context_key"]]
        require(context_text == context["context_text"] and query_text == row["query_text"], "Prompt/template changed")
        context_ids = context["context_ids"]
        if row["context_key"] not in checked_contexts:
            require(tok.encode(context_text, add_special_tokens=False) == context_ids, "Context tokenization changed")
            require(digest(context_ids) == context["context_ids_sha256"], "Context IDs changed")
            checked_contexts.add(row["context_key"])
        query_ids = tok.encode(query_text, add_special_tokens=False)
        bare = tok.encode(sample.get("retrieval_question", sample["question"]), add_special_tokens=False)
        require(query_ids == row["query_ids"] and bare == row["bare_question_ids"], "Question tokenization changed")
        require(digest(context_ids + query_ids) == row["input_ids_sha256"], "Full input token IDs changed")
        chunks = [context_ids[i:i + CHUNK] for i in range(0, len(context_ids), CHUNK)]
        scores = qa.selectors.bm25_scores(chunks, bare)
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        natural = sorted(ranked[:TOPK])
        oracle = forced_indices(row["required_chunks"], ranked, natural)
        require(oracle is not None and set(row["required_chunks"]).issubset(oracle), "Required evidence exceeds budget")
        require(make_pack(context_ids, query_ids, natural) == row["natural_pack"], "Natural retrieval pack changed")
        require(make_pack(context_ids, query_ids, oracle) == row["oracle_pack"], "Oracle pack changed")
        require(len(natural) == len(oracle), "Different chunk budget")
        checked.append((row, context_ids, query_ids))
    require(not qa.torch.cuda.is_initialized(), "CPU validation initialized CUDA")
    return checked, manifest


def wait_for_remote_gpu(metadata, metadata_path):
    require(os.environ.get("COMEM_REMOTE_QUEUE") == "1", "Accuracy runner requires remote queue admission")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible.isdigit(), "Exactly one explicit remote GPU is required")
    from remote_queue import gpu_idle
    while True:
        require(not qa.torch.cuda.is_initialized(), "CUDA initialized before fresh idle admission")
        idle, reading = gpu_idle(int(visible), 512)
        metadata["gpu_admission"] = {"physical_gpu": int(visible), "checked_at": qa.utc_now(),
            "max_memory_mib": 512, "max_utilization_percent": 5, **reading}
        metadata["status"] = "loading_model" if idle else "waiting_for_gpu"
        qa.write_json(metadata_path, metadata)
        if idle:
            return
        print(json.dumps({"status": "waiting_for_gpu", "reading": reading}), flush=True)
        time.sleep(30)


def verify_actual_chunks(current, selected):
    actual = [c.detach().cpu().tolist() if hasattr(c, "detach") else list(c) for c in selected]
    require(actual == current["expected_chunks"], "Actual reader pack differs from the fixed oracle tokens")
    current["actual_pack_checks"] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--expected-inputs-sha256", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if args.validate_only:
        rows, _ = load_and_validate(args)
        print(json.dumps({"status": "cpu_validated", "arm": args.arm, "task": args.task,
            "inputs": len(rows), "cuda_initialized": qa.torch.cuda.is_initialized()}))
        return 0
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with qa.OutputLock(out):
        if (out / "COMPLETED.json").exists():
            marker, config = read_json(out / "COMPLETED.json"), read_json(out / "run_config.json")
            require(config["options"]["inputs_sha256"] == args.expected_inputs_sha256 and config["arm"] == args.arm
                and config["options"]["task"] == args.task and config["options"]["indices"] == args.indices
                and config["options"]["model"] == args.model and config["options"]["smoke_only"] == args.smoke_only,
                "Completed output is for a different requested unit")
            predictions = [json.loads(line) for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            require(marker["status"] == "completed" and marker["n"] == len(predictions), "Invalid completion marker")
            require([{k: row[k] for k in ("index", "id", "task")} for row in predictions] == config["expected"]
                and all(row["status"] == "ok" and math.isfinite(row["score"]) and 0 <= row["score"] <= 1 for row in predictions),
                "Completed predictions differ from the requested source selection")
            print(f"Already complete: {out}")
            return 0
        attempt_number = 1
        while (out / "attempts" / f"{attempt_number:04d}").exists():
            attempt_number += 1
        attempt = out / "attempts" / f"{attempt_number:04d}"
        attempt.mkdir(parents=True)
        metadata_path = attempt / "metadata.json"
        metadata = {"status": "validating_cpu_inputs", "started_at": qa.utc_now(), "output_dir": str(out),
            "attempt_dir": str(attempt), "attempt": attempt_number, "benchmark": "oracle_support", "arm": args.arm}
        qa.write_json(metadata_path, metadata)
        cache = None
        try:
            checked, manifest = load_and_validate(args)
            expected = [{"index": row["index"], "id": row["id"], "task": row["task"]} for row, _, _ in checked]
            files = [Path(__file__), HERE / "prepare_oracle_support.py", HERE / "official_qa_driver.py",
                HERE / "reader_adapter.py", qa.ROOT / "exp/s15_ruler_lower.py", qa.ROOT / "COMem/comem/model.py",
                qa.ROOT / "COMem/comem/selectors.py", HERE / "protocol_sources/longbench/metrics.py",
                HERE / "protocol_sources/locomo/task_eval/evaluation.py", args.inputs, args.inputs.parent / "contexts.json"]
            config = {"schema_version": 1, "benchmark": "oracle_support", "arm": args.arm, "method": args.arm,
                "options": {"benchmark": "oracle_support", "arm": args.arm, "model": args.model, "j": 12,
                    "adapter": "", "selector": "bm25", "topk": TOPK, "chunk_size": CHUNK, "device": "cuda:0",
                    "dtype": "bfloat16", "attn_impl": "sdpa", "seed": SEED, "max_samples": -1,
                    "inputs_sha256": args.expected_inputs_sha256, "task": args.task, "indices": args.indices,
                    "pack_variant": "oracle", "smoke_only": args.smoke_only, "out": str(out)},
                "expected": expected, "files": [qa.file_info(path) for path in files],
                "protocol": {"template": "unchanged official_qa_driver chat template; thinking disabled",
                    "metric": "unchanged official_qa_driver.score_prediction; task/category kept separate",
                    "decode": "greedy native reader; first-token EOS suppression; original official task cap",
                    "intervention": "complete mapped support chunks forced into top-k, remaining slots highest natural BM25; both packs chronological",
                    "budget": "identical selected chunk count; actual token count may differ if the final source chunk is selected",
                    "natural": "reuse original main completed predictions after exact-source/token/config/pack checks; never substitute oracle for natural",
                    "locomo": "annotated full turn plus its session DATE header; annotations need not be exhaustive",
                    "scope": "accuracy only; runtime metadata is not a speed or GPU-memory comparison"}}
            contract = out / "run_config.json"
            if contract.exists():
                require(read_json(contract) == config, "Output contract changed; preserve old attempts and use a new output")
            else:
                qa.write_json(contract, config)
            qa.write_json(attempt / "run_config.json", config)
            metadata.update(n=len(checked), expected=expected, cpu_validated_inputs=len(checked))
            wait_for_remote_gpu(metadata, metadata_path)
            qa.torch.manual_seed(SEED)
            model, tok = qa.load_backbone(args.model, "bfloat16", "sdpa", "cuda:0", "")
            cache = qa.GenerationCache(out / "generations.sqlite3")
            def on_construct(info):
                metadata["reader"] = info
                qa.write_json(metadata_path, metadata)
            reader = qa.make_reader_factory(args.arm, cache, on_construct, explicit_pack=True)(model, 12, tokenizer=tok)
            native, current = reader.reader.reader, {}
            original = native.build_bottom if hasattr(native, "build_bottom") else native.write_chunks
            if hasattr(native, "build_bottom"):
                def checked_write(sink_id, selected):
                    require(sink_id == native._sink_prefix_id(), "Reader sink differs")
                    verify_actual_chunks(current, selected)
                    return original(sink_id, selected)
                patched_name = "build_bottom"
            else:
                def checked_write(selected, *extra, **kwargs):
                    verify_actual_chunks(current, selected)
                    return original(selected, *extra, **kwargs)
                patched_name = "write_chunks"
            metadata["status"] = "running"
            values, results = defaultdict(list), []
            with patch.object(native, patched_name, checked_write), (attempt / "predictions.jsonl").open("w", encoding="utf-8") as stream:
                for number, (fixed, context_ids, query_ids) in enumerate(checked, 1):
                    sample, pack = fixed["sample"], fixed["oracle_pack"]
                    require(pack["read_pack_tokens"] + sample["max_new_tokens"] <= model.config.max_position_embeddings, "Read pack exceeds model position budget")
                    chunks = [context_ids[i:i + CHUNK] for i in range(0, len(context_ids), CHUNK)]
                    current.update(expected_chunks=[chunks[i] for i in pack["selected_indices"]], actual_pack_checks=0)
                    before_hits = cache.hits
                    input_ids = qa.torch.tensor([context_ids + query_ids], dtype=qa.torch.long, device="cuda:0")
                    pred = reader.generate_from_ids(input_ids, context_token_count=len(context_ids),
                        selected_indices=pack["selected_indices"], chunk_size=CHUNK, max_new_tokens=sample["max_new_tokens"])
                    require(current["actual_pack_checks"] == 1 or cache.hits == before_hits + 1,
                        "Neither actual reader pack verification nor exact-input cache reuse occurred")
                    score, scored = qa.score_prediction(pred, sample)
                    require(math.isfinite(score) and 0 <= score <= 1, "Invalid official score")
                    result = dict(sample, task=fixed["task"], source_task=fixed["source_task"], source_benchmark=fixed["source_benchmark"],
                        method=args.arm, pack_variant="oracle", pred=pred, scored_prediction=scored, score=score, status="ok",
                        pack=pack, natural_pack=fixed["natural_pack"], oracle_pack=pack,
                        required_chunks=fixed["required_chunks"], input_ids_sha256=fixed["input_ids_sha256"],
                        fixture_row_sha256=fixed["fixture_row_sha256"], read_pack_token_delta=fixed["read_pack_token_delta"],
                        natural_all_required_visible=fixed["natural_all_required_visible"],
                        actual_pack_checks=current["actual_pack_checks"], exact_input_cache_reuse=cache.hits == before_hits + 1)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    results.append(result)
                    values[fixed["task"]].append(score)
                    print(f"[oracle_support/{args.arm}] {number}/{len(checked)} {fixed['id']} score={score:.4f}", flush=True)
            require([{k: row[k] for k in ("index", "id", "task")} for row in results] == expected, "Final prediction IDs/count differ")
            scores = {"benchmark": "oracle_support", "n": len(results), "tasks": {task: {"n": len(v),
                "score": 100 * sum(v) / len(v), "scale": "0..100", "metric": "f1"} for task, v in values.items()},
                "completion_scope": "exact requested support-intervention selection; no cross-task macro"}
            qa.write_json(attempt / "scores.json", scores)
            metadata.update(status="completed", n=len(results), scores=scores)
        except BaseException as exc:
            metadata.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            metadata["finished_at"] = qa.utc_now()
            if cache:
                metadata.update(reused_generations=cache.hits, new_generations=cache.misses)
                cache.close()
            qa.write_json(metadata_path, metadata)
            qa.write_json(out / "metadata.json", metadata)
        for name in ("predictions.jsonl", "scores.json"):
            temporary = out / (name + ".tmp")
            shutil.copyfile(attempt / name, temporary)
            temporary.replace(out / name)
        qa.write_json(out / "COMPLETED.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
