"""CPU-only supporting-dialogue retrieval coverage for completed LoCoMo outputs.

This is an observational diagnostic of annotated dialogue availability. It does
not assume annotations are exhaustive and does not claim causal reading ability.
The original generated predictions and scores are never changed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import re

import official_qa_driver as qa


def turn_spans(context, conversation):
    """Find complete rendered turns in their chronological source order."""
    spans, cursor = {}, 0
    sessions = sorted(int(k.split("_")[1]) for k in conversation if re.fullmatch(r"session_\d+", k))
    for i in sessions:
        for turn in conversation[f"session_{i}"]:
            fragment = turn["speaker"] + ' said, "' + turn["text"] + '"\n'
            if "blip_caption" in turn:
                fragment += " and shared %s." % turn["blip_caption"]
            fragment += "\n"
            start = context.find(fragment, cursor)
            if start < 0:
                raise ValueError("Source turn was not preserved in the recorded context format")
            cursor = start + len(fragment)
            if turn.get("dia_id"):
                spans[turn["dia_id"]] = (start, cursor)
    return spans


def evidence_ids(evidence):
    ids = []
    for raw in evidence or []:
        if not isinstance(raw, str):
            raise ValueError("Unsupported evidence annotation type")
        for item in re.split(r"[;,]\s*", raw):
            item = item.strip()
            if item and item not in ids:
                ids.append(item)
    return ids


def diagnostic(row, span_map, offsets, chunk_size):
    labels = evidence_ids(row.get("evidence", []))
    selected = set(row["pack"]["selected_indices"])
    starts, ends = [s for s, _ in offsets], [e for _, e in offsets]
    records = []
    for label in labels:
        if label not in span_map:
            records.append({"id": label, "resolved": False})
            continue
        start, end = span_map[label]
        first = bisect_left(ends, start + 1)
        last = bisect_left(starts, end)
        token_indices = list(range(first, last))
        covered = sum(i // chunk_size in selected for i in token_indices)
        records.append({"id": label, "resolved": True, "tokens": len(token_indices),
            "retrieved_tokens": covered, "any_token_visible": covered > 0,
            "complete_turn_visible": bool(token_indices) and covered == len(token_indices)})
    resolved = bool(records) and all(item["resolved"] for item in records)
    return {"id": row["id"], "index": row["index"], "category": row["category"],
        "score": row["score"], "evidence": records, "all_annotations_resolved": resolved,
        "all_support_turns_fully_retrieved": resolved and all(item["complete_turn_visible"] for item in records),
        "all_support_turns_partly_retrieved": resolved and all(item["any_token_visible"] for item in records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--model", help="Local tokenizer path override; never loads model weights")
    p.add_argument("--locomo-data", type=Path, default=qa.HERE / "data/locomo/locomo10.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    config = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    if config["benchmark"] != "locomo" or not (args.run / "COMPLETED.json").exists():
        raise ValueError("Use one completed official_qa_driver LoCoMo run")
    options = argparse.Namespace(**config["options"])
    options.locomo_data = str(args.locomo_data)
    samples = {s["id"]: s for s in qa.locomo_samples(options)}
    rows = [json.loads(line) for line in (args.run / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if set(samples) != {r["id"] for r in rows} or len(rows) != len(samples):
        raise ValueError("Predicted source IDs do not exactly match the run selection")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model or options.model, local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise ValueError("Offset-aligned evidence coverage requires a fast tokenizer")
    source = json.loads(args.locomo_data.read_text(encoding="utf-8"))
    context_cache, results = {}, []
    for row in rows:
        ci = int(re.fullmatch(r"conv(\d+)_qa\d+", row["id"]).group(1))
        if ci not in context_cache:
            formatted = tok.apply_chat_template([{"role": "user", "content": samples[row["id"]]["marked_prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            context = formatted.split(qa.BOUNDARY)[0]
            offsets = tok(context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            context_cache[ci] = turn_spans(context, source[ci]["conversation"]), offsets
        span_map, offsets = context_cache[ci]
        if len(offsets) != row["pack"]["context_tokens"]:
            raise ValueError("Tokenizer context differs from the recorded generation; do not diagnose it")
        results.append(diagnostic(row, span_map, offsets, options.chunk_size))
    groups = defaultdict(list)
    for row in results:
        if row["all_annotations_resolved"] and row["category"] != 5:
            groups["all_support_retrieved" if row["all_support_turns_fully_retrieved"] else "some_support_not_fully_retrieved"].append(row["score"])
    report = {"n": len(results), "annotated_resolved_n": sum(r["all_annotations_resolved"] for r in results),
        "groups": {name: {"n": len(v), "f1_0_100": 100 * sum(v) / len(v)} for name, v in groups.items()},
        "scope": "observational annotation coverage; incomplete/unknown annotations separate; no oracle intervention",
        "records": results}
    output = args.out or args.run / "evidence_diagnostic.json"
    qa.write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
