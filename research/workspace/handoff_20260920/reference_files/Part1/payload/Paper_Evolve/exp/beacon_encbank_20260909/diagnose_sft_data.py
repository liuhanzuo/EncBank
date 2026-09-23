"""CPU-only evidence-coverage diagnostics for the unchanged SFT preparation.

Loads the tokenizer and canonical JSONL, never a model. Evidence/answers are
examined after question-only retrieval and cannot affect selected chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import unicodedata


ROOT = Path(__file__).resolve().parent


def normalized(text: str) -> str:
    """NFKC, casefold and whitespace only; punctuation/numbers remain intact."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def contains_snippet(haystack: str, needle: str) -> bool:
    """Continuous normalized substring with alphanumeric edge boundaries."""
    if not needle:
        return False
    start = 0
    while True:
        position = haystack.find(needle, start)
        if position < 0:
            return False
        end = position + len(needle)
        left_ok = not (needle[0].isalnum() and position > 0 and haystack[position - 1].isalnum())
        right_ok = not (needle[-1].isalnum() and end < len(haystack) and haystack[end].isalnum())
        if left_ok and right_ok:
            return True
        start = position + 1


def continuous_runs(indices) -> list[list[int]]:
    indices = list(indices)
    if indices != sorted(set(indices)) or any(not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError("Selected chunk indices must be unique, nonnegative and in source order")
    runs = []
    for index in indices:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def inspect_evidence(context: str, full_chunks, selected_indices, evidence, tokenizer) -> dict:
    """Strict snippet coverage, distinguishing source errors from retrieval misses."""
    indices = list(selected_indices)
    runs = continuous_runs(indices)
    if indices and indices[-1] >= len(full_chunks):
        raise ValueError("Selected chunk index exceeds original document")
    source = normalized(context)
    decoded_source = normalized(decode(tokenizer, [token for chunk in full_chunks for token in chunk]))
    chunk_texts = {i: normalized(decode(tokenizer, chunk)) for i, chunk in enumerate(full_chunks)}
    run_texts = [(run, normalized(decode(tokenizer, [token for i in run for token in full_chunks[i]]))) for run in runs]
    # Used only to expose a would-be false positive; it never counts as coverage.
    forbidden_join = normalized(decode(tokenizer, [token for i in indices for token in full_chunks[i]]))
    snippets = []
    seen = set()
    for text in evidence:
        value = normalized(text)
        if not value or value in seen:
            continue
        seen.add(value)
        in_source = contains_snippet(source, value)
        in_decoded_source = contains_snippet(decoded_source, value)
        source_single = [i for i, chunk in chunk_texts.items() if contains_snippet(chunk, value)]
        selected_single = [i for i in indices if contains_snippet(chunk_texts[i], value)]
        matched_runs = [run for run, run_text in run_texts if contains_snippet(run_text, value)]
        selected_match = bool(matched_runs)
        locatable = in_source and in_decoded_source
        covered = locatable and selected_match
        snippets.append({"text": text, "normalized_characters": len(value),
                         "source_text_match": in_source, "decoded_full_document_match": in_decoded_source,
                         "source_single_chunk_matches": source_single,
                         "source_requires_multiple_chunks": in_decoded_source and not source_single,
                         "selected_single_chunk_matches": selected_single,
                         "selected_contiguous_run_matches": matched_runs,
                         "retrieved_complete_snippet": covered,
                         "retrieved_across_adjacent_chunk_boundary": covered and not selected_single,
                         "nonadjacent_join_would_false_match": not selected_match and len(runs) > 1 and contains_snippet(forbidden_join, value),
                         "status": "source_text_not_locatable" if not in_source else
                                   "tokenizer_roundtrip_mismatch" if not in_decoded_source else
                                   "retrieved_complete_snippet" if covered else "source_locatable_but_not_fully_retrieved"})
    total = len(snippets)
    source_count = sum(item["source_text_match"] for item in snippets)
    locatable_count = sum(item["source_text_match"] and item["decoded_full_document_match"] for item in snippets)
    retrieved_count = sum(item["retrieved_complete_snippet"] for item in snippets)
    category = ("no_text_evidence_annotation" if not total else
                "source_annotation_not_fully_locatable" if source_count < total else
                "tokenizer_roundtrip_mismatch" if locatable_count < total else
                "all_annotated_snippets_retrieved" if retrieved_count == total else
                "no_complete_annotated_snippet_retrieved" if retrieved_count == 0 else
                "some_annotated_snippets_retrieved")
    return {"category": category, "selected_contiguous_runs": runs,
            "annotated_snippets": total, "source_locatable_snippets": source_count,
            "source_and_decoded_locatable_snippets": locatable_count,
            "retrieved_complete_snippets": retrieved_count,
            "all_annotations_locatable": bool(total) and locatable_count == total,
            "all_annotations_retrieved": bool(total) and retrieved_count == total,
            "snippets": snippets}


def describe(records: list[dict]) -> dict:
    snippets = [item for row in records for item in row["evidence"]["snippets"]]
    locatable = [item for item in snippets if item["source_text_match"] and item["decoded_full_document_match"]]
    all_locatable_rows = [row for row in records if row["evidence"]["all_annotations_locatable"]]
    return {"examples": len(records), "documents": len({row["document_id"] for row in records}),
            "question_categories": dict(Counter(row["evidence"]["category"] for row in records)),
            "primary_answer_kinds": dict(Counter(row["primary_answer_kind"] for row in records)),
            "answer_truncated_examples": sum(row["answer_truncated"] for row in records),
            "questions_with_all_annotations_locatable": len(all_locatable_rows),
            "questions_with_all_annotations_retrieved": sum(row["evidence"]["all_annotations_retrieved"] for row in records),
            "all_snippet_question_recall_on_fully_locatable_annotations":
                sum(row["evidence"]["all_annotations_retrieved"] for row in all_locatable_rows) / len(all_locatable_rows) if all_locatable_rows else None,
            "annotated_snippets": len(snippets), "locatable_snippets": len(locatable),
            "retrieved_complete_snippets": sum(item["retrieved_complete_snippet"] for item in locatable),
            "strict_snippet_recall_on_locatable_annotations":
                sum(item["retrieved_complete_snippet"] for item in locatable) / len(locatable) if locatable else None,
            "retrieved_snippets_crossing_adjacent_chunk_boundaries": sum(item["retrieved_across_adjacent_chunk_boundary"] for item in snippets),
            "locatable_snippets_requiring_multiple_source_chunks": sum(item["source_requires_multiple_chunks"] for item in locatable),
            "rejected_nonadjacent_join_false_matches": sum(item["nonadjacent_join_would_false_match"] for item in snippets),
            "original_context_tokens": {"min": min((r["original_context_tokens"] for r in records), default=0),
                                        "median": statistics.median(r["original_context_tokens"] for r in records) if records else 0,
                                        "max": max((r["original_context_tokens"] for r in records), default=0)},
            "answer_target_tokens": sum(row["answer_target_tokens"] for row in records)}


def diagnose(rows, tokenizer, *, chunk_size=512, max_chunks=7, max_question_tokens=256,
             max_answer_tokens=128, seed=42, fixed_eval_limit=32) -> dict:
    from evaluate_sft import prepare_examples
    prepared = prepare_examples(rows, tokenizer, chunk_size=chunk_size, max_chunks=max_chunks,
                                max_question_tokens=max_question_tokens, max_answer_tokens=max_answer_tokens, seed=seed)
    by_id = {row["id"]: row for row in rows}
    document_chunks, records = {}, []
    for example in prepared:
        source = by_id[example.id]
        if example.document_id not in document_chunks:
            ids = tokenizer.encode(source["context"], add_special_tokens=False)
            document_chunks[example.document_id] = tuple(tuple(ids[start:start + chunk_size]) for start in range(0, len(ids), chunk_size))
        chunks = document_chunks[example.document_id]
        if tuple(chunks[i] for i in example.selected_chunk_indices) != example.document_chunks:
            raise AssertionError("Diagnostic chunk reconstruction differs from canonical preparation")
        evidence = inspect_evidence(source["context"], chunks, example.selected_chunk_indices, source.get("evidence", []), tokenizer)
        kinds = source.get("answer_kinds") or ["unknown"]
        records.append({"id": example.id, "document_id": example.document_id, "source": example.source,
                        "split": example.split, "question": example.question,
                        "primary_answer_kind": kinds[0], "all_reference_answer_kinds": kinds,
                        "raw_primary_answer_tokens": len(tokenizer.encode(example.references[0], add_special_tokens=False)),
                        "answer_target_tokens": len(example.answer_ids), "answer_truncated": example.answer_truncated,
                        "original_context_tokens": example.original_context_tokens,
                        "selected_chunk_indices": list(example.selected_chunk_indices),
                        "selected_context_tokens": sum(len(chunk) for chunk in example.document_chunks),
                        "prompt_tokens": len(example.prompt_ids), "evidence": evidence})
    fixed_ids = [example.id for example in sorted(prepared, key=lambda ex: hashlib.sha256(f"{seed}:{ex.id}".encode()).hexdigest())[:fixed_eval_limit]]
    fixed_set = set(fixed_ids)
    return {"preparation": prepared.preparation_summary, "skipped": prepared.skipped,
            "full_dev": describe(records), "fixed_eval_subset": {**describe([row for row in records if row["id"] in fixed_set]), "ids": fixed_ids},
            "records": records}


def render_report(result: dict) -> str:
    lines = ["# QASPER SFT development-data diagnostic", "",
             "CPU/tokenizer only. The data, question-only BM25 selector and ongoing training are unchanged.", "",
             "A hit requires a complete normalized continuous evidence string. This is annotated-paragraph coverage, not a semantic claim that every answer fact is available.", "",
             "| Cohort | Questions | Documents | All annotations locatable | All annotations retrieved | Complete snippet recall | Answer truncations |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("full_dev", "fixed_eval_subset"):
        item = result[name]
        recall = item["strict_snippet_recall_on_locatable_annotations"]
        lines.append(f"| {name} | {item['examples']} | {item['documents']} | {item['questions_with_all_annotations_locatable']} | {item['questions_with_all_annotations_retrieved']} | {100 * recall:.2f}% | {item['answer_truncated_examples']} |" if recall is not None else
                     f"| {name} | {item['examples']} | {item['documents']} | 0 | 0 | unavailable | {item['answer_truncated_examples']} |")
    for name in ("full_dev", "fixed_eval_subset"):
        item = result[name]
        lines.extend(["", f"## {name}", "", "Question categories:", ""])
        lines.extend(f"- {key}: {count}" for key, count in item["question_categories"].items())
        lines.extend(["", "Primary answer kinds: " + ", ".join(f"{key}={count}" for key, count in item["primary_answer_kinds"].items()) + ".",
                      f"Retrieved complete snippets spanning adjacent selected chunks: {item['retrieved_snippets_crossing_adjacent_chunk_boundaries']}. Rejected artificial matches across unselected gaps: {item['rejected_nonadjacent_join_false_matches']}."])
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {text}" for text in result["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/qasper_sft/dev.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=Path("/srv/encbank/legacy_workspace/models/Qwen3-8B"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/qasper_sft/diagnostics")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    if torch.cuda.is_initialized():
        raise RuntimeError("This diagnostic must not initialize CUDA")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if any(row.get("split") != "dev" for row in rows):
        raise ValueError("This bounded diagnostic accepts only the development split")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    result = diagnose(rows, tokenizer)
    result.update(format="beacon-sft-data-diagnostic-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  input_file=str(args.data.resolve()), input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  tokenizer_path=str(args.tokenizer.resolve()), transformers_version=transformers.__version__,
                  cuda_initialized=torch.cuda.is_initialized(), model_loaded=False,
                  normalization="Unicode NFKC + casefold + collapse whitespace; preserve punctuation and numbers; require continuous substring with alphanumeric edge boundaries",
                  boundaries="Only source-adjacent selected chunk indices are concatenated; concatenate IDs then decode to preserve within-word/Unicode boundaries",
                  limitations=["Evidence is the union of retained human reference annotations, often whole paragraphs; all-snippet retrieval is not equivalent to recovering exactly all logically necessary answer facts.",
                               "A missing complete paragraph may still leave the answer-bearing sentence available. No bag-of-words or partial-overlap score is called complete coverage.",
                               "Source annotations not found in the full original text are separate from retrieval misses. Tokenizer roundtrip mismatches are also separate.",
                               "Only normalized exact continuous text is matched; paraphrases, citation/OCR variations and visual information may be missed.",
                               "Nonadjacent selected chunks cannot be concatenated to invent evidence. Adjacent-run containers are not minimal evidence token spans.",
                               "This diagnoses unchanged inputs, not model accuracy or whether Encbank/Beacon successfully uses retrieved evidence.",
                               "The fixed subset uses the trainer's seed-42 ID ordering; the 32-question and full 100-question cohorts are reported separately."])
    if result["cuda_initialized"]:
        raise RuntimeError("Unexpected CUDA initialization")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"full_dev": result["full_dev"], "fixed_eval_subset": {key: value for key, value in result["fixed_eval_subset"].items() if key != "ids"},
                      "report": str((args.output / "REPORT.md").resolve()), "cuda_initialized": result["cuda_initialized"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
