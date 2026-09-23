"""Measure actual HotpotQA supporting-fact coverage in recorded CoMem packs.

Only strictly aligned official supporting sentences have sentence-level truth.
Unaligned sentences remain unknown; an answer-string hit is never substituted.
Document-title coverage is a separate weaker diagnostic.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path

import official_qa_driver as qa


def span_tokens(offsets, start, end):
    starts, ends = [s for s, _ in offsets], [e for _, e in offsets]
    return range(bisect_left(ends, start + 1), bisect_left(starts, end))


def coverage(mapping, offsets, source_shift, selected, chunk_size):
    selected = set(selected)
    facts = []
    for fact in mapping["supporting_facts"]:
        spans = fact["context_char_spans"]
        matches = []
        for start, end in spans:
            tokens = list(span_tokens(offsets, source_shift + start, source_shift + end))
            chunks = sorted({i // chunk_size for i in tokens})
            matches.append({"chunks": chunks, "token_n": len(tokens),
                "retrieved_token_n": sum(i // chunk_size in selected for i in tokens),
                "fully_retrieved": bool(tokens) and all(i // chunk_size in selected for i in tokens)})
        documents = []
        for start, end in fact.get("support_document_char_spans", []):
            tokens = list(span_tokens(offsets, source_shift + start, source_shift + end))
            documents.append(any(i // chunk_size in selected for i in tokens))
        facts.append({"title": fact["title"], "sent_id": fact["sent_id"],
            "alignment": fact["alignment"], "span_coverage": matches,
            "fully_retrieved": any(item["fully_retrieved"] for item in matches) if matches else None,
            "document_any_token_retrieved": any(documents) if documents else None})
    fully_aligned = bool(facts) and all(f["fully_retrieved"] is not None for f in facts)
    return {"facts": facts, "all_facts_aligned": fully_aligned,
        "all_facts_fully_retrieved": all(f["fully_retrieved"] for f in facts) if fully_aligned else None,
        "all_support_documents_touched": all(f["document_any_token_retrieved"] for f in facts) if facts and all(
            f["document_any_token_retrieved"] is not None for f in facts) else None}


def budgeted_oracle_indices(facts, ranked_indices, topk):
    """CPU preparation only; choose a same-budget forced pack or an explicit skip.

    This does not run a model or mutate any original benchmark prediction.
    A caller must label its separate oracle intervention and generation outputs.
    """
    if not facts or any(not f["span_coverage"] for f in facts):
        return {"status": "unaligned_support", "indices": None}
    required = set()
    for fact in facts:
        # Prefer the occurrence adding the fewest chunks, then its source order.
        spans = fact["span_coverage"]
        chosen = min(spans, key=lambda s: (len(set(s["chunks"]) - required), s["chunks"]))
        required.update(chosen["chunks"])
    if len(required) > topk:
        return {"status": "support_exceeds_budget", "required_chunks": sorted(required), "indices": None}
    filled = list(sorted(required))
    for index in ranked_indices:
        if len(filled) >= topk:
            break
        if index not in required:
            filled.append(index)
    return {"status": "ready", "required_chunks": sorted(required), "indices": sorted(filled), "topk": topk}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--mapping", type=Path, default=qa.HERE / "data/gold_support/gold_support_mapping.json")
    p.add_argument("--longbench", type=Path, default=qa.HERE / "data/longbench/hotpotqa.jsonl")
    p.add_argument("--model", help="Local tokenizer override; model weights are never loaded")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    config = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    if config["benchmark"] != "longbench" or not (args.run / "COMPLETED.json").exists():
        raise ValueError("Use a completed official LongBench QA run")
    rows = [json.loads(line) for line in (args.run / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["task"] == "hotpotqa"]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("No HotpotQA predictions or duplicate source IDs")
    mappings = {r["id"]: r for r in json.loads(args.mapping.read_text(encoding="utf-8"))["records"]}
    raw_rows = {str(row["_id"]): row for row in (json.loads(line) for line in args.longbench.read_text(encoding="utf-8").splitlines() if line.strip())}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model or config["options"]["model"], local_files_only=True, use_fast=True)
    if not tok.is_fast:
        raise ValueError("A fast tokenizer with offsets is required")
    _, _, _, prompts, _ = qa.official_protocol()
    records, groups = [], defaultdict(list)
    for row in rows:
        raw, mapping = raw_rows[row["id"]], mappings[row["id"]]
        if mapping["question"] != raw["input"] or mapping["context_chars"] != len(raw["context"]):
            raise ValueError("Mapping does not correspond to this LongBench source")
        marked = prompts["hotpotqa"].format(**dict(raw, context=raw["context"] + qa.BOUNDARY))
        context = tok.apply_chat_template([{"role": "user", "content": marked}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False).split(qa.BOUNDARY)[0]
        shift = context.find(raw["context"])
        if shift < 0 or context.find(raw["context"], shift + 1) >= 0:
            raise ValueError("Raw context missing or ambiguously repeated in chat input")
        offsets = tok(context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(offsets) != row["pack"]["context_tokens"]:
            raise ValueError("Tokenizer/context changed since generation")
        result = coverage(mapping, offsets, shift, row["pack"]["selected_indices"], config["options"]["chunk_size"])
        result.update(id=row["id"], index=row["index"], score=row["score"])
        result["oracle_pack_preparation"] = budgeted_oracle_indices(result["facts"], row["pack"]["selected_indices"], config["options"]["topk"])
        records.append(result)
        flag = result["all_facts_fully_retrieved"]
        name = "support_alignment_unknown" if flag is None else "all_facts_retrieved" if flag else "some_fact_not_fully_retrieved"
        groups[name].append(row["score"])
    known = [r for r in records if r["all_facts_aligned"]]
    report = {"n": len(records), "sentence_coverage_known_n": len(known),
        "all_facts_retrieved_n": sum(r["all_facts_fully_retrieved"] is True for r in records),
        "all_facts_retrieved_fraction_among_aligned": sum(r["all_facts_fully_retrieved"] for r in known) / len(known) if known else None,
        "all_support_documents_touched_n": sum(r["all_support_documents_touched"] is True for r in records),
        "groups": {name: {"n": len(v), "f1_0_100": 100 * sum(v) / len(v)} for name, v in groups.items()},
        "scope": "strict exact-support sentence coverage; unmatched revised Wikipedia sentences stay unknown; document touch is weaker; oracle packs prepared only, no oracle generation",
        "records": records}
    output = args.out or args.run / "gold_support_diagnostic.json"
    qa.write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
