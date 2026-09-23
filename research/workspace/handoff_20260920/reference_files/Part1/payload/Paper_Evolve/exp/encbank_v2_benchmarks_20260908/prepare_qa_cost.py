"""Prepare exact LongBench QA inputs on CPU; never load model weights or time GPU work.

Uses official_qa_driver.longbench_samples/tokenize_pack directly and checks every
pack against both completed quality arms. The output is an input contract for a
future local timing runner, not a latency result or a GPU queue submission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TASKS = ("qasper", "hotpotqa")
ARMS = ("fix_all", "j0")
COUNTS = {task: 200 for task in TASKS}
CAPS = {"qasper": 128, "hotpotqa": 32}
PACK_KEYS = ("input_tokens", "context_tokens", "query_tokens", "context_chunks",
             "selected_indices", "read_pack_tokens", "truncation", "context_query_boundary")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_predictions(root, task, arm, expected_count=200):
    """Only the two completed full shards are eligible; smoke files cannot enter."""
    predictions, sources = {}, []
    for shard in range(2):
        folder = Path(root) / arm / f"{task}_s{shard}of2"
        marker = read_json(folder / "COMPLETED.json")
        config = read_json(folder / "run_config.json")
        options = config["options"]
        required = dict(benchmark="longbench", arm=arm, j=12, selector="bm25", topk=12,
                        chunk_size=512, dtype="bfloat16", attn_impl="sdpa", seed=42,
                        max_samples=-1, num_shards=2, shard_index=shard, tasks=[task], adapter="")
        require(all(options.get(k) == v for k, v in required.items()),
                f"Different quality protocol: {folder}")
        require(marker.get("status") == "completed", f"Incomplete quality shard: {folder}")
        path = folder / "predictions.jsonl"
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        expected = {(r["index"], r["id"], r["task"]) for r in config["expected"]}
        require(marker["n"] == len(rows) == len(expected), f"Shard count mismatch: {folder}")
        require({(r["index"], r["id"], r["task"]) for r in rows} == expected,
                f"Shard IDs differ from run contract: {folder}")
        for row in rows:
            key = (row["index"], row["id"])
            require(key not in predictions, f"Duplicate quality row: {task}/{arm}/{key}")
            require(row["task"] == task and row["index"] % 2 == shard,
                    f"Incorrect shard membership: {task}/{arm}/{key}")
            require(row["status"] == "ok" and isinstance(row["pred"], str)
                    and math.isfinite(row["score"]) and 0 <= row["score"] <= 1,
                    f"Invalid prediction: {task}/{arm}/{key}")
            require(row["max_new_tokens"] == CAPS[task], f"Wrong official cap: {task}/{arm}")
            predictions[key] = row
        sources.append({"file": str(path.resolve()), "records": len(rows),
                        "bytes": path.stat().st_size})
    require(len(predictions) == expected_count, f"Expected {expected_count}: {task}/{arm}")
    return predictions, sources


def validate_pair(sample, pack, pair):
    key = f'{sample["task"]}/{sample["id"]}'
    for arm in ARMS:
        row = pair[arm]
        for field in ("index", "id", "task", "question", "answers", "max_new_tokens"):
            require(row[field] == sample[field], f"Source/prediction {field} differs: {key}/{arm}")
        require(all(row["pack"][k] == pack[k] for k in PACK_KEYS),
                f"Recomputed official pack differs: {key}/{arm}")
    require(pair["fix_all"]["pack"] == pair["j0"]["pack"], f"V2/Replay pack differs: {key}")


def native_decode_accounting(generated_ids, cap, eos_id):
    """For a future runner using the unchanged Encbank native decode core.

The core suppresses EOS on step zero, stops on later EOS without appending it,
and otherwise fills the cap. A terminal EOS therefore costs one decode forward
beyond the returned-token list's usual G-1 calls. Never apply this helper to
retokenized prediction text; it requires the actual native return list.
"""
    require(1 <= len(generated_ids) <= cap, "Expected actual native generated IDs within cap")
    require(eos_id not in generated_ids, "Native returned IDs must exclude terminal EOS")
    ended = len(generated_ids) < cap
    return {"generated_tokens": len(generated_ids), "terminated_by_eos": ended,
            "termination_reason": "eos" if ended else "max_new_tokens",
            "decode_forward_calls": len(generated_ids) if ended else len(generated_ids) - 1,
            "sampled_tokens_including_terminal_eos": len(generated_ids) + int(ended)}


def reference_output(row, tokenizer):
    # Historical driver stores only decode(..., skip_special_tokens=True).strip().
    # Re-encoding is descriptive and cannot reconstruct the native token stream.
    return {"prediction": row["pred"], "official_score": row["score"],
            "decoded_text_retokenized_tokens": len(tokenizer.encode(row["pred"], add_special_tokens=False)),
            "generated_ids": None, "generated_tokens": None, "decode_forward_calls": None,
            "termination_reason": None,
            "length_provenance": "unavailable: historical quality output kept stripped decoded text only"}


def prepare_record(tok, source, sample, pair, driver):
    ids, n_context, selected, pack = driver.tokenize_pack(tok, sample, 512, "bm25", 12)
    validate_pair(sample, pack, pair)
    formatted = tok.apply_chat_template([{"role": "user", "content": sample["marked_prompt"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    context_text, query_text = formatted.split(driver.BOUNDARY)
    require(source["context"] in context_text, "Source context altered or truncated")
    require(source["input"] in query_text, "Source question altered or truncated")
    values = ids[0].tolist()
    require(pack["truncation"] == "none" and len(values) == pack["input_tokens"],
            "Input token count/truncation mismatch")
    spans = [{"index": index, "start": index * 512, "end": min((index + 1) * 512, n_context)}
             for index in selected]
    require(pack["read_pack_tokens"] == 1 + len(values[n_context:])
            + sum(s["end"] - s["start"] for s in spans), "Pack span count mismatch")
    sink_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    require(sink_id is not None and tok.eos_token_id is not None, "Missing native sink/EOS ID")
    return {"schema_version": 1, "task": sample["task"], "index": sample["index"], "id": sample["id"],
            "question": sample["question"], "answers": sample["answers"],
            "marked_prompt": sample["marked_prompt"], "context_ids": values[:n_context],
            "query_ids": values[n_context:], "retrieval_question_ids": tok.encode(sample["question"], add_special_tokens=False),
            "context_token_count": n_context, "selected_indices": selected,
            "selected_chunk_spans": spans, "pack": pack, "chunk_size": 512,
            "sink_id": int(sink_id), "eos_id": int(tok.eos_token_id),
            "max_new_tokens": sample["max_new_tokens"], "first_token_eos_suppressed": True,
            "later_eos_stops_generation": True, "fixed_generation_length": False,
            "reference_outputs": {arm: reference_output(pair[arm], tok) for arm in ARMS}}


def make_plan(out, records):
    jobs = []
    for phase in ("smoke", "full"):
        for task in TASKS:
            rows = records[task]
            longest = max(rows, key=lambda r: r["pack"]["read_pack_tokens"])["index"]
            smoke_indices = sorted({0, longest if longest != 0 else 1})
            for arm in ARMS:
                jobs.append({"id": f"qa_cost/{phase}/{task}/{arm}", "phase": phase,
                    "task": task, "arm": arm, "effective_j": 12 if arm == "fix_all" else 0,
                    "input_file": str((out / f"{task}.inputs.jsonl").resolve()),
                    "indices": smoke_indices if phase == "smoke" else list(range(COUNTS[task])),
                    "timed_repetitions": 1 if phase == "smoke" else 3,
                    "unmeasured_warmups": 0 if phase == "smoke" else 1,
                    "max_new_tokens": CAPS[task], "allow_later_eos": True,
                    "status": "prepared_waiting_for_timing_runner"})
    return {"schema_version": 1, "status": "inputs_ready_runner_not_implemented",
            "measurement_results": False, "gpu_started": False,
            "runner_entrypoint": None, "scheduler_modified": False,
            "required_gpu": "NVIDIA GeForce RTX 5090", "remote_timing_allowed": False,
            "gpu_admission": "shared exp/gpu_gate.py; initial and locked recheck MiB/1024 strictly <5; no other model",
            "thread_policy": {"torch_cpu_threads": 2, "torch_interop_threads": 16,
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"},
            "model": "/srv/encbank/legacy_workspace/models/Qwen3-8B", "dtype": "bfloat16", "attn_impl": "sdpa",
            "adapter": None, "seed": 42, "smoke_generation_calls": 8,
            "full_unique_method_examples": 800, "full_timed_generation_calls": 2400,
            "full_unmeasured_warmup_calls": 4, "jobs": jobs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/srv/encbank/legacy_workspace/models/Qwen3-8B"))
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/longbench")
    parser.add_argument("--prediction-root", type=Path,
                        default=HERE / "results/remote/outputs/benchmarks_v2/full/longbench")
    parser.add_argument("--out", type=Path, default=HERE / "results/protocol/qa_cost")
    args = parser.parse_args(argv)
    # Process-local isolation: this preparation can run alongside the healthy GPU queue.
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    from transformers import AutoTokenizer
    import official_qa_driver as driver
    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_config = read_json(args.model / "config.json")
    require(model_config["model_type"] == "qwen3" and model_config["num_hidden_layers"] == 36,
            "Expected the same Qwen3-8B backbone")
    _, _, _, _, caps = driver.official_protocol()
    require(all(caps[t] == CAPS[t] for t in TASKS), "Official task caps changed")
    all_records, provenance, summaries = {}, [], {}
    for task in TASKS:
        raw = [json.loads(s) for s in (args.data_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        require(len(raw) == COUNTS[task], f"Expected full 200 source rows for {task}")
        samples = list(driver.longbench_samples(SimpleNamespace(tasks=[task], data_dir=args.data_dir,
            max_samples=-1, num_shards=1, shard_index=0)))
        predictions = {}
        for arm in ARMS:
            predictions[arm], files = load_predictions(args.prediction_root, task, arm)
            provenance.extend(files)
        expected_keys = {(s["index"], s["id"]) for s in samples}
        require(all(set(predictions[a]) == expected_keys for a in ARMS), f"Full source IDs mismatch: {task}")
        rows = []
        for sample in samples:
            key = (sample["index"], sample["id"])
            row = prepare_record(tok, raw[sample["index"]], sample,
                                 {a: predictions[a][key] for a in ARMS}, driver)
            require(row["pack"]["read_pack_tokens"] + CAPS[task] <= model_config["max_position_embeddings"],
                    f"Read pack exceeds configured model positions: {task}/{key}")
            rows.append(row)
        all_records[task] = rows
        summaries[task] = {"source_rows": len(raw), "prepared_rows": len(rows),
            "matched_quality_rows_per_arm": len(rows), "all_packs_match": True,
            "official_cap": CAPS[task], "raw_generated_lengths_available": False,
            "context_token_range": [min(r["context_token_count"] for r in rows), max(r["context_token_count"] for r in rows)],
            "read_pack_token_range": [min(r["pack"]["read_pack_tokens"] for r in rows), max(r["pack"]["read_pack_tokens"] for r in rows)],
            "reference_scores": {a: statistics.mean(r["reference_outputs"][a]["official_score"] for r in rows) * 100 for a in ARMS}}
        print(json.dumps({"task": task, "verified_rows": len(rows), "pack_match": True}), flush=True)
    require(not torch.cuda.is_initialized(), "CPU preparation unexpectedly initialized CUDA")
    args.out.mkdir(parents=True, exist_ok=True)
    for task, rows in all_records.items():
        target = args.out / f"{task}.inputs.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, target)
    plan = make_plan(args.out, all_records)
    plan["model"] = str(args.model.resolve())
    (args.out / "task_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    report = {"status": "inputs_ready_runner_not_implemented", "measurement_results": False,
              "prepared_at": datetime.now(timezone.utc).isoformat(), "cuda_initialized": False,
              "model": str(args.model.resolve()), "tasks": summaries, "prediction_sources": provenance,
              "input_files": [{"file": str((args.out / f"{t}.inputs.jsonl").resolve()),
                               "bytes": (args.out / f"{t}.inputs.jsonl").stat().st_size} for t in TASKS],
              "historical_lengths": "Raw generated IDs/EOS/decode counts were not recorded. Retokenized text is not actual generation length.",
              "implementation": "Direct official_qa_driver.longbench_samples and tokenize_pack; no model loaded; no timing runner called"}
    (args.out / "PREPARED.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "prepared_samples": 400, "out": str(args.out.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
