"""Prepare a small, document-disjoint QASPER-train SFT pilot; CPU/stdlib only.

Only qasper-train-v0.3.json is read from the official train/dev archive. The
official dev/test answers never become training or development examples here.
LongBench contexts are used solely as an exclusion list, never as supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import tarfile
import unicodedata
import urllib.request


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
SOURCE_CARD = "https://huggingface.co/datasets/allenai/qasper"
TRAIN_MEMBER = "qasper-train-v0.3.json"
BENCHMARK_TASKS = ("qasper", "narrativeqa", "hotpotqa", "2wikimqa", "musique", "multifieldqa_en")
SHINGLE_WORDS = 16
MIN_MATCHING_SHINGLES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingle_hash(words: list[str], start: int, width: int = SHINGLE_WORDS) -> bytes:
    return hashlib.blake2b(" ".join(words[start:start + width]).encode("utf-8"), digest_size=16).digest()


def download_archive(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, temporary.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    temporary.replace(path)


def load_official_train(path: Path) -> dict:
    # No extractall: only this exact member is opened, never the official dev.
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(TRAIN_MEMBER)
        if not member.isfile():
            raise ValueError("Official train archive member must be a regular file")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Missing official train member")
        rows = json.load(stream)
    if not isinstance(rows, dict) or not rows:
        raise ValueError("QASPER train must be a nonempty paper-ID mapping")
    return rows


def paper_context(paper: dict) -> str:
    """All textual content in source order, without question/evidence injection."""
    parts = [paper.get("title", ""), paper.get("abstract", "")]
    for section in paper.get("full_text", []):
        parts.append(section.get("section_name", ""))
        parts.extend(section.get("paragraphs", []))
    captions = [item.get("caption", "") for item in paper.get("figures_and_tables", [])]
    if any(captions):
        parts.append("Figure and table captions")
        parts.extend(captions)
    return "\n\n".join(text.strip() for text in parts if isinstance(text, str) and text.strip())


def answer_text(annotation: dict) -> str | None:
    answer = annotation.get("answer", {})
    if answer.get("unanswerable"):
        return None
    text = str(answer.get("free_form_answer") or "").strip()
    if text:
        return text
    spans = [str(value).strip() for value in answer.get("extractive_spans", []) if str(value).strip()]
    if spans:
        return ", ".join(spans)
    if isinstance(answer.get("yes_no"), bool):
        return "Yes" if answer["yes_no"] else "No"
    return None


def question_rows(paper_id: str, paper: dict, counts: Counter) -> list[dict]:
    context = paper_context(paper)
    if not context:
        counts["empty_document"] += 1
        return []
    rows = []
    for question in paper.get("qas", []):
        counts["source_questions"] += 1
        query = str(question.get("question") or "").strip()
        if not query:
            counts["empty_question"] += 1
            continue
        answers, evidence, kinds = [], [], []
        for annotation in question.get("answers", []):
            text = answer_text(annotation)
            if text is None:
                counts["unanswerable_or_empty_annotations"] += 1
                continue
            answer = annotation["answer"]
            original_evidence = [str(x).strip() for x in answer.get("evidence", []) if str(x).strip()]
            textual_evidence = [x for x in original_evidence if not x.startswith("FLOAT SELECTED")]
            if original_evidence and not textual_evidence:
                counts["figure_only_annotations"] += 1
                continue
            if text not in answers:
                answers.append(text)
            evidence.extend(x for x in textual_evidence if x not in evidence)
            kind = "free_form" if answer.get("free_form_answer") else "extractive" if answer.get("extractive_spans") else "yes_no"
            if kind not in kinds:
                kinds.append(kind)
        if not answers:
            counts["questions_without_text_answer"] += 1
            continue
        question_id = str(question.get("question_id") or fingerprint(query))
        rows.append({"id": f"qasper-train:{paper_id}:{question_id}",
                     "document_id": paper_id, "context": context, "question": query,
                     "answer": answers[0], "answers": answers,
                     "evidence": evidence, "answer_kinds": kinds,
                     "source": "allenai/qasper:train:v0.3", "split": "unassigned"})
    return rows


class BenchmarkExclusions:
    """Detect same titles/contexts and substantial verbatim document overlap.

    Benchmark 16-word shingles are sampled at stride 16; source shingles scan at
    stride 1, so offsets need not align. At least 8 distinct benchmark shingles
    (128 benchmark words, apart from repetitions) trigger conservative exclusion.
    This is a documented text-overlap check, not a proof against all paraphrases.
    """
    def __init__(self, records: list[tuple[str, str]]):
        self.texts = []
        self.labels = []
        self.exact = defaultdict(list)
        self.index = defaultdict(list)
        seen = set()
        for label, text in records:
            canonical = normalize(text)
            key = hashlib.sha256(canonical.encode()).hexdigest()
            if not canonical or key in seen:
                continue
            seen.add(key)
            i = len(self.texts)
            self.texts.append(canonical)
            self.labels.append(label)
            self.exact[key].append(i)
            words = canonical.split()
            shingles = {shingle_hash(words, start) for start in range(0, len(words) - SHINGLE_WORDS + 1, SHINGLE_WORDS)}
            for value in shingles:
                self.index[value].append(i)

    def match(self, title: str, context: str) -> dict | None:
        canonical = normalize(context)
        exact = self.exact.get(hashlib.sha256(canonical.encode()).hexdigest())
        if exact:
            return {"kind": "exact_context", "benchmark": self.labels[exact[0]]}
        normalized_title = normalize(title)
        if len(normalized_title.split()) >= 5 and len(normalized_title) >= 35:
            for i, text in enumerate(self.texts):
                if normalized_title in text:
                    return {"kind": "paper_title_in_context", "benchmark": self.labels[i]}
        words = canonical.split()
        hits = Counter()
        seen = set()
        for start in range(len(words) - SHINGLE_WORDS + 1):
            value = shingle_hash(words, start)
            if value in seen:
                continue
            seen.add(value)
            for i in self.index.get(value, ()):
                hits[i] += 1
                if hits[i] >= MIN_MATCHING_SHINGLES:
                    return {"kind": "16_word_shingle_overlap", "benchmark": self.labels[i],
                            "matching_distinct_shingles_at_least": hits[i]}
        return None


def load_benchmark_contexts(directory: Path) -> tuple[list[tuple[str, str]], list[dict]]:
    contexts, receipts = [], []
    for task in BENCHMARK_TASKS:
        path = directory / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Required held-out exclusion file missing: {path}")
        count = 0
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                text = row.get("context")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing context in exclusion source {path}")
                contexts.append((f"longbench/{task}/{count}", text))
                count += 1
        receipts.append({"path": str(path.resolve()), "task": task, "rows": count,
                         "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return contexts, receipts


def split_documents(groups: dict[str, list[dict]], train_limit: int, dev_limit: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    if train_limit < 1 or dev_limit < 1:
        raise ValueError("Train/dev limits must be positive")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    dev_docs, remaining, dev_total = [], [], 0
    for paper_id in ordered:
        if dev_total < dev_limit:
            dev_docs.append(paper_id)
            dev_total += len(groups[paper_id])
        else:
            remaining.append(paper_id)
    dev = [dict(row, split="dev") for key in dev_docs for row in sorted(groups[key], key=lambda x: x["id"])][:dev_limit]
    train = [dict(row, split="train") for key in remaining for row in sorted(groups[key], key=lambda x: x["id"])][:train_limit]
    # Questions from reserved dev documents beyond the cap never leak into train.
    if {r["document_id"] for r in train} & {r["document_id"] for r in dev}:
        raise AssertionError("Document split leakage")
    if {fingerprint(r["context"]) for r in train} & {fingerprint(r["context"]) for r in dev}:
        raise AssertionError("Text-identical document split leakage")
    return train, dev, {"seed": seed, "method": "sha256(seed:paper_id) sorted; reserve whole dev papers before row caps",
                        "reserved_dev_document_ids": dev_docs,
                        "dev_questions_dropped_at_cap": dev_total - len(dev),
                        "train_questions_dropped_at_cap": sum(len(groups[key]) for key in remaining) - len(train)}


def describe_rows(rows: list[dict]) -> dict:
    word_lengths = [len(row["context"].split()) for row in rows]
    return {"examples": len(rows), "documents": len({row["document_id"] for row in rows}),
            "context_words": {"min": min(word_lengths, default=0), "median": statistics.median(word_lengths) if word_lengths else 0,
                              "max": max(word_lengths, default=0)},
            "answer_kind_counts": dict(Counter(row["answer_kinds"][0] for row in rows))}


def prepare(archive: Path, benchmark_dir: Path, output: Path, train_limit: int = 2000, dev_limit: int = 100, seed: int = 20260909) -> dict:
    papers = load_official_train(archive)
    benchmark_contexts, benchmark_receipts = load_benchmark_contexts(benchmark_dir)
    exclusions = BenchmarkExclusions(benchmark_contexts)
    groups, removed, duplicate_docs = {}, [], []
    counts, seen_contexts = Counter(), {}
    for paper_id, paper in papers.items():
        rows = question_rows(paper_id, paper, counts)
        if not rows:
            continue
        context = rows[0]["context"]
        key = fingerprint(context)
        if key in seen_contexts:
            duplicate_docs.append({"document_id": paper_id, "duplicates": seen_contexts[key], "questions": len(rows)})
            continue
        seen_contexts[key] = paper_id
        match = exclusions.match(paper.get("title", ""), context)
        if match:
            removed.append({"document_id": paper_id, "questions": len(rows), **match})
            continue
        groups[paper_id] = rows
    train, dev, split_receipt = split_documents(groups, train_limit, dev_limit, seed)
    if not train or not dev:
        raise ValueError("Exclusions left no usable train or development examples")
    output.mkdir(parents=True, exist_ok=True)
    products = {}
    for split, rows in (("train", train), ("dev", dev)):
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        products[split] = {**describe_rows(rows), "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    metadata = {"format": "beacon-comem-qasper-sft-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": {"dataset": "allenai/qasper", "dataset_card": SOURCE_CARD, "license": "CC BY 4.0",
                           "download_url": SOURCE_URL, "archive": str(archive.resolve()), "archive_bytes": archive.stat().st_size,
                           "archive_sha256": sha256_file(archive), "only_member_read": TRAIN_MEMBER,
                           "official_split": "train", "source_documents": len(papers)},
                "requested_limits": {"train": train_limit, "dev": dev_limit}, "filter_counts": dict(counts),
                "benchmark_exclusion": {"sources": benchmark_receipts, "unique_contexts": len(exclusions.texts),
                                        "rule": "normalized context equality OR distinctive paper title in benchmark context OR >=8 distinct 16-word shingles",
                                        "excluded_documents": removed, "excluded_questions": sum(row["questions"] for row in removed),
                                        "limitations": "Verbatim/title overlap check; not a semantic paraphrase-contamination guarantee. Covers the six current local LongBench v1 QA files only."},
                "duplicate_source_documents_removed": duplicate_docs, "split": split_receipt, "outputs": products,
                "supervision": "First valid human answer as answer; all valid references in answers. Unanswerable/empty and figure-only annotations excluded. Evidence is diagnostic metadata only.",
                "context_policy": "Full title, abstract, all sections/paragraphs and figure/table captions in source order. No question/evidence/answer injection and no truncation/retrieval here.",
                "limitations": ["Domain-specific QASPER supervised pilot, not general instruction tuning or original Activation Beacon training replication.",
                                "Scientific QA benchmark-domain train data; report as in-domain adaptation even after document exclusions.",
                                "Visual figures/table pixels are absent; only captions included. Mixed text/figure annotation may still require visual information.",
                                "Trainer must use question-only retrieval and log support coverage; do not use evidence/answer for default selection.",
                                "Internal dev is sampled from official train at paper level; official validation/test are not used here."]}
    (output / "data_receipt.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/qasper-train-dev-v0.3.tgz")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT.parent / "comem_v2_benchmarks_20260908/data/longbench")
    parser.add_argument("--output", type=Path, default=ROOT / "data/qasper_sft")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--dev-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--download", action="store_true", help="Download the official archive only if missing")
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    receipt = prepare(args.archive, args.benchmark_dir, args.output, args.train_limit, args.dev_limit, args.seed)
    print(json.dumps({"outputs": receipt["outputs"], "excluded_documents": len(receipt["benchmark_exclusion"]["excluded_documents"]),
                      "receipt": str((args.output / "data_receipt.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
