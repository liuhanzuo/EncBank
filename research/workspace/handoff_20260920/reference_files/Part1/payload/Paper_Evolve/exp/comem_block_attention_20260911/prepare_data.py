"""Tokenize the existing, leakage-screened Qasper train-only QA pilot on CPU.

No dataset is downloaded. This reuses the established question-only BM25 and
chat template. Whole answers that exceed the declared budget are filtered,
never shortened. The last question tokens are located using tokenizer offsets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "beacon_comem_20260909"))

FORMAT = "sparse-comem-qasper-prepared-v1"


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_sources(train, dev):
    if not train or not dev:
        raise ValueError("Both canonical train and dev sources must be nonempty")
    for expected, rows in (("train", train), ("dev", dev)):
        for row in rows:
            if row.get("source") != "allenai/qasper:train:v0.3" or row.get("split") != expected:
                raise ValueError("This pilot accepts only the screened official-train Qasper split")
    if {r["document_id"] for r in train} & {r["document_id"] for r in dev}:
        raise ValueError("Canonical training and development documents overlap")
    fingerprints = lambda rows: {hashlib.sha256(r["context"].encode()).hexdigest() for r in rows}
    if fingerprints(train) & fingerprints(dev):
        raise ValueError("Canonical training and development contexts overlap")


def question_probe_indices(tokenizer, question, prompt_ids, width=16):
    from evaluate_sft import PROMPT_INSTRUCTION
    messages = [{"role": "user", "content": f"{PROMPT_INSTRUCTION}\n\nQuestion: {question}"}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(prompt_ids):
        raise ValueError("Rendered prompt tokenization differs from prepared chat prompt")
    start = rendered.rfind(question)
    if start < 0:
        raise ValueError("Question not found in rendered chat prompt")
    end = start + len(question)
    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
               if b > start and a < end and b > a]
    if not indices:
        raise ValueError("Question has no identifiable prompt tokens")
    return indices[-width:]


def select_examples(examples, limit, seed):
    valid = [ex for ex in examples if not ex.answer_truncated]
    ordered = sorted(valid, key=lambda ex: (hashlib.sha256(f"{seed}:{ex.id}".encode()).hexdigest(), ex.id))
    return ordered[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True, help="Screened canonical Qasper train.jsonl")
    ap.add_argument("--dev", required=True, help="Document-disjoint canonical Qasper dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-limit", type=int, default=1000)
    ap.add_argument("--dev-limit", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--max-question-tokens", type=int, default=256)
    ap.add_argument("--max-answer-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if min(args.train_limit, args.dev_limit, args.chunk_size, args.max_chunks,
           args.max_question_tokens, args.max_answer_tokens) < 1:
        raise ValueError("All limits must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from transformers import AutoTokenizer
    from evaluate_sft import prepare_examples
    train, dev = read_rows(args.train), read_rows(args.dev)
    validate_sources(train, dev)
    out = Path(args.out)
    spec = {**vars(args), "train_sha256": digest(args.train), "dev_sha256": digest(args.dev),
            "format": FORMAT, "probe_rule": "last-16-question-token-offsets-exclude-assistant-markers",
            "answer_policy": "filter-over-budget-whole-targets-never-truncate"}
    spec.pop("out")
    metadata_path = out / "preparation.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old["spec"] != spec or any(digest(out / f"{s}.jsonl") != old["products"][s]["sha256"] for s in ("train", "dev")):
            raise ValueError("Existing preparation differs; choose a new output directory")
        print(json.dumps({"event": "preparation_exists", **old}, ensure_ascii=False))
        return
    if any((out / f"{s}.jsonl").exists() for s in ("train", "dev")):
        raise ValueError("Incomplete existing preparation; choose a new output directory")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    products = {}
    out.mkdir(parents=True, exist_ok=True)
    for split, rows, limit in (("train", train, args.train_limit), ("dev", dev, args.dev_limit)):
        examples = prepare_examples(rows, tokenizer, chunk_size=args.chunk_size,
            max_chunks=args.max_chunks, max_question_tokens=args.max_question_tokens,
            max_answer_tokens=args.max_answer_tokens, seed=args.seed)
        chosen = select_examples(examples, limit, args.seed)
        if not chosen:
            raise ValueError(f"No valid {split} examples after filters")
        path = out / f"{split}.jsonl"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for ex in chosen:
                row = asdict(ex)
                row["format"] = FORMAT
                row["probe_indices"] = question_probe_indices(tokenizer, ex.question, ex.prompt_ids)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
        lengths = [sum(map(len, ex.document_chunks)) + len(ex.prompt_ids) + len(ex.answer_ids) for ex in chosen]
        products[split] = {"examples": len(chosen), "requested_limit": limit,
            "sha256": digest(path), "preparation": examples.preparation_summary,
            "over_budget_answers_filtered": sum(ex.answer_truncated for ex in examples),
            "max_total_tokens": max(lengths), "min_total_tokens": min(lengths),
            "source": "official-Qasper-train-only", "metric": "normalized-max-reference-token-F1-on-pilot"}
    metadata = {"spec": spec, "products": products}
    atomic_json(metadata_path, metadata)
    print(json.dumps({"event": "preparation_complete", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
