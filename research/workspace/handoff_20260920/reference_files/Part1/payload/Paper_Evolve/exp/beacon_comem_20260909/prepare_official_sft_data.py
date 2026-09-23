"""Lossless raw-data references and conservative native CoMem SFT conversion.

Only document spans identified by observed author templates are removed from
the first user message. Unknown layouts are explicitly rejected. This performs
no tokenization, retrieval, target truncation, model loading, or training.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator, Mapping
import unicodedata


DATASET_ID = "namespace-Pt/long-llm-data"
DATASET_REVISION = "48e96d937d87395c9223caf467ad30ae8e525191"
MEMORY_PLACEHOLDER = "[Document provided in memory.]"
SOURCE_FILES = {
    "gpt_book": "gpt/one_detail_book.train.16K.json",
    "gpt_paper": "gpt/one_detail_paper.train.16K.json",
    "longalpaca": "longalpaca/train.json",
    "booksum": "booksum/train.16K.json",
    "needle": "needle/train.16K.json",
}

COMMON_HEADERS = ("Read the following context.", "Context information is below:")
HEADERS = {
    "gpt_book": COMMON_HEADERS + (
        "Story:", "Below presents one or several stories.",
        "I want you read the given book then answer my questions.",
        "You're a smart book reader. You're required to read through the following book and help me with my questions.",
    ),
    "gpt_paper": COMMON_HEADERS + (
        "Paper:", "Below presents one or several scentific papers.",
        "I want you read the given paper then answer my questions.",
        "You're a smart paper reader. You're required to read through the following paper and help me with my questions.",
    ),
    "booksum": (
        "Context:", "Below presents a story.", "Read the following context.",
        "Here is a chapter in a book. I want you read this chapter and remember it.",
        "You're a story summarizer with high intelligence. You're required to read through the following story and summarize it for me.",
    ),
    "needle": (
        "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.",
    ),
}

LONGALPACA_TEMPLATES = (
    (
        "Below is a paper. Memorize the paper and answer my question after the paper.",
        "The paper begins.", "Now the paper ends.", "longalpaca-single-paper",
    ),
    (
        "There are two papers. Memorize them and answer my question after the paper.",
        "The first paper begins.", "Now the second paper ends.", "longalpaca-two-papers",
    ),
)
LONGALPACA_BOOK_HEADER = re.compile(
    r"^Below is some paragraphs in the book, [^\n]+\. Memorize the content and answer my question after the book\.\n"
)

QA_START = re.compile(
    r"^(?:What|Who|Whose|Which|Where|When|Why|How|According to|Describe|Define|State|Name|"
    r"Explain|List|Identify|Provide|Give|Compare|Summarize|Determine|Is|Are|Was|Were|Do|Does|Did|"
    r"Can|Could|Will|Would|Should|In what|To what)\b", re.IGNORECASE,
)
QUESTION_INTRO = "Answer my question using the knowledge from the context:"
BOOKSUM_TASK = re.compile(
    r"^(?:Summarize .+\.|What is .+ talking about\?|Please generate a summarization about .+\.)$",
    re.DOTALL,
)


class UnsupportedRecord(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def exact_document_id(context: str) -> str:
    return "doc:" + hashlib.sha256(context.encode("utf-8")).hexdigest()


def normalized_document_group(context: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", context).split())
    return "group:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def group_split(group_id: str, *, seed: int = 42, dev_fraction: float = 0.05) -> str:
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1)")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "dev" if fraction < dev_fraction else "train"


def _check_qa_suffix(value: str, source: str) -> None:
    task = value.strip()
    if not task or len(task) > 8192 or "\n\n" in task:
        raise UnsupportedRecord("unsupported_terminal_task_layout")
    if source == "booksum":
        if not BOOKSUM_TASK.fullmatch(task):
            raise UnsupportedRecord("unsupported_booksum_task")
        return
    if task.startswith(QUESTION_INTRO):
        task = task[len(QUESTION_INTRO):].lstrip()
    if not QA_START.match(task):
        raise UnsupportedRecord("unsupported_question_prefix")
    # Observed questions may be imperative (Define/State/Describe), so a question
    # mark alone is not the delimiter. Embedded paragraphs are not guessed apart.


def parse_first_user(text: str, source: str) -> dict:
    """Return exact source spans; reconstruction must be character-identical."""
    if not isinstance(text, str):
        raise UnsupportedRecord("non_string_first_user")
    if source == "longalpaca":
        book_header = LONGALPACA_BOOK_HEADER.match(text)
        matches = [item for item in LONGALPACA_TEMPLATES if text.startswith(item[0] + "\n")]
        if book_header is not None:
            closing = "Now the material ends."
            start = book_header.end()
            if text.count(closing) != 1:
                raise UnsupportedRecord("ambiguous_longalpaca_book_marker")
            end = text.find(closing, start)
            if end < start or not text[end + len(closing):].strip():
                raise UnsupportedRecord("missing_longalpaca_book_instruction")
            parser = "longalpaca-book-excerpt"
        elif len(matches) != 1:
            raise UnsupportedRecord("unsupported_longalpaca_header")
        else:
            header, opening, closing, parser = matches[0]
            opening_at = text.find(opening, len(header))
            if opening_at < 0 or text.count(opening) != 1 or text.count(closing) != 1:
                raise UnsupportedRecord("ambiguous_longalpaca_markers")
            if text[len(header):opening_at].strip():
                raise UnsupportedRecord("unexpected_text_before_document_marker")
            start = opening_at + len(opening)
            end = text.find(closing, start)
            if end < start or not text[end + len(closing):].strip():
                raise UnsupportedRecord("missing_longalpaca_document_or_instruction")
    elif source in HEADERS:
        candidates = [header for header in HEADERS[source] if text.startswith(header + "\n")]
        if len(candidates) != 1:
            raise UnsupportedRecord("unsupported_document_header")
        header = candidates[0]
        if header == "Context information is below:":
            prefix = header + "\n----------\n"
            closing = "\n----------\n"
            if not text.startswith(prefix):
                raise UnsupportedRecord("missing_open_context_rule")
            start = len(prefix)
            end = text.rfind(closing)
            if end < start:
                raise UnsupportedRecord("missing_close_context_rule")
            _check_qa_suffix(text[end + len(closing):], source)
            parser = f"{source}-context-rules"
        else:
            prefix = header + ("\n" if source == "needle" else "\n\n")
            if not text.startswith(prefix):
                raise UnsupportedRecord("unsupported_document_prefix_spacing")
            start = len(prefix)
            end = text.rfind("\n\n")
            if end < start:
                raise UnsupportedRecord("missing_terminal_task_separator")
            _check_qa_suffix(text[end + 2:], source)
            # Do not accidentally put a separate known question instruction into
            # document memory if a previously unseen blank-line layout is used.
            if text[start:end].rstrip().endswith(QUESTION_INTRO):
                raise UnsupportedRecord("ambiguous_question_instruction_boundary")
            parser = f"{source}-known-header-final-task"
    else:
        raise UnsupportedRecord("unsupported_source")
    context, prefix, suffix = text[start:end], text[:start], text[end:]
    if not context.strip():
        raise UnsupportedRecord("empty_document")
    assert prefix + context + suffix == text
    return {
        "context": context, "original_first_user_prefix": prefix,
        "original_first_user_suffix": suffix, "parser": parser,
        "question": prefix + MEMORY_PLACEHOLDER + suffix,
    }


def parse_record(record: Mapping, source: str, source_index: int, *, seed: int = 42,
                 dev_fraction: float = 0.05) -> tuple[dict, dict]:
    messages = record.get("conversations") if isinstance(record, Mapping) else None
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise UnsupportedRecord("incomplete_conversation")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, Mapping) or message.get("role") != expected:
            raise UnsupportedRecord("unsupported_role_order")
        if not isinstance(message.get("content"), str):
            raise UnsupportedRecord("non_string_message")
        if expected == "user" and not message["content"].strip():
            raise UnsupportedRecord("empty_user_message")
    first = parse_first_user(messages[0]["content"], source)
    # These sources currently put all document material in the first message.
    # Explicit later document wrappers are unsupported rather than read early.
    known_headers = tuple(header for group in HEADERS.values() for header in group)
    known_headers += tuple(item[0] for item in LONGALPACA_TEMPLATES)
    for message in messages[2::2]:
        if LONGALPACA_BOOK_HEADER.match(message["content"]) or any(message["content"].startswith(header + "\n") for header in known_headers):
            raise UnsupportedRecord("later_user_introduces_document_template")
    context = first.pop("context")
    doc_id, group = exact_document_id(context), normalized_document_group(context)
    conversation_id = f"official:{source}:{source_index}"
    turns = []
    for turn_index in range(len(messages) // 2):
        answer = messages[2 * turn_index + 1]["content"]
        turns.append({
            "assistant_turn_index": turn_index,
            "question": first["question"] if turn_index == 0 else messages[2 * turn_index]["content"],
            "answer": answer, "target_char_count": len(answer),
            "target_eligible": bool(answer.strip()),
            "target_ineligible_reason": None if answer.strip() else "empty_assistant_target",
            "raw_user_message_index": 2 * turn_index, "raw_assistant_message_index": 2 * turn_index + 1,
        })
    source_file = SOURCE_FILES[source]
    conversation = {
        "format": "official-beacon-comem-native-v1", "conversation_id": conversation_id,
        "document_id": doc_id, "normalized_group_id": group,
        "source": f"activation_beacon/{source}", "source_key": source,
        "source_file": source_file, "source_index": source_index,
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "split": group_split(group, seed=seed, dev_fraction=dev_fraction),
        "assistant_turn_count": len(turns), "eligible_target_count": sum(item["target_eligible"] for item in turns),
        "conversation_target_char_count": sum(item["target_char_count"] for item in turns),
        "original_first_user_prefix": first["original_first_user_prefix"],
        "original_first_user_suffix": first["original_first_user_suffix"],
        "memory_placeholder": MEMORY_PLACEHOLDER, "parser": first["parser"], "turns": turns,
        "raw_record_reference": {"source_file": source_file, "index": source_index},
    }
    document = {"document_id": doc_id, "normalized_group_id": group, "context": context,
                "context_char_count": len(context)}
    if reconstruct_original_messages(conversation, context) != messages:
        # Extra metadata on original messages stays in the original JSONL; this
        # equality checks exactly role/content, not unrelated author fields.
        original_pairs = [{"role": x["role"], "content": x["content"]} for x in messages]
        if reconstruct_original_messages(conversation, context) != original_pairs:
            raise UnsupportedRecord("original_conversation_round_trip_failed")
    return document, conversation


def reconstruct_original_messages(conversation: Mapping, context: str) -> list[dict]:
    messages = []
    for index, turn in enumerate(conversation["turns"]):
        user = turn["question"]
        if index == 0:
            user = conversation["original_first_user_prefix"] + context + conversation["original_first_user_suffix"]
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": turn["answer"]}))
    return messages


def adapted_messages(conversation: Mapping) -> list[dict]:
    return [message for turn in conversation["turns"] for message in (
        {"role": "user", "content": turn["question"]},
        {"role": "assistant", "content": turn["answer"]},
    )]


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def prepare_corpus(raw_dir: str | Path, output_dir: str | Path, *, seed: int = 42,
                   dev_fraction: float = 0.05, allow_missing: bool = False,
                   overwrite: bool = False) -> dict:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    group_split("validate", seed=seed, dev_fraction=dev_fraction)
    missing = [relative for relative in SOURCE_FILES.values() if not (raw_dir / relative).is_file()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"Waiting for complete author files (not .part): {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("documents.jsonl", "document_index.json", "conversations.jsonl", "rejected.jsonl", "processing_report.json", "PROCESSING_REPORT.md")
    if not overwrite and any((output_dir / name).exists() for name in names):
        raise FileExistsError("Output already contains a conversion; use another directory or explicit overwrite")
    document_index, group_assignments = {}, {}
    raw_conversation_fingerprints = set()
    counts = Counter(); source_stats = {}; source_groups = {}; parser_counts = Counter(); reason_counts = Counter()
    doc_tmp = output_dir / "documents.jsonl.partial"
    conv_tmp = output_dir / "conversations.jsonl.partial"
    rejected_tmp = output_dir / "rejected.jsonl.partial"
    with doc_tmp.open("wb") as documents_out, conv_tmp.open("wb") as conversations_out, rejected_tmp.open("wb") as rejected_out:
        for source, relative in SOURCE_FILES.items():
            path = raw_dir / relative
            if not path.is_file():
                continue
            stats = Counter(); source_groups[source] = set()
            with path.open("r", encoding="utf-8") as stream:
                for source_index, line in enumerate(stream):
                    stats["input_conversations"] += 1
                    try:
                        record = json.loads(line)
                        fingerprint = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
                        if fingerprint in raw_conversation_fingerprints:
                            stats["duplicate_raw_conversation_copies"] += 1
                        raw_conversation_fingerprints.add(fingerprint)
                        document, conversation = parse_record(record, source, source_index, seed=seed, dev_fraction=dev_fraction)
                    except (json.JSONDecodeError, UnsupportedRecord) as error:
                        reason = error.reason if isinstance(error, UnsupportedRecord) else "invalid_json_line"
                        rejection = {"source_key": source, "source_file": relative, "source_index": source_index,
                                     "reason": reason, "raw_record_preserved": True}
                        rejected_out.write(_json_bytes(rejection))
                        stats["rejected_conversations"] += 1
                        reason_counts[f"{source}:{reason}"] += 1
                        continue
                    doc_id, group, split = document["document_id"], document["normalized_group_id"], conversation["split"]
                    if group in group_assignments and group_assignments[group] != split:
                        raise AssertionError("A normalized document group crosses train/dev")
                    group_assignments[group] = split
                    source_groups[source].add(group)
                    if doc_id not in document_index:
                        payload = _json_bytes(document)
                        document_index[doc_id] = {"offset": documents_out.tell(), "length": len(payload), "normalized_group_id": group}
                        documents_out.write(payload)
                        counts["stored_document_characters"] += len(document["context"])
                    conversations_out.write(_json_bytes(conversation))
                    stats["accepted_conversations"] += 1
                    stats[f"{split}_conversations"] += 1
                    stats["assistant_turns"] += conversation["assistant_turn_count"]
                    stats["eligible_assistant_targets"] += conversation["eligible_target_count"]
                    stats["empty_assistant_targets"] += conversation["assistant_turn_count"] - conversation["eligible_target_count"]
                    stats["target_characters"] += conversation["conversation_target_char_count"]
                    stats[f"{split}_assistant_turns"] += conversation["assistant_turn_count"]
                    parser_counts[conversation["parser"]] += 1
            source_stats[source] = dict(stats)
            counts.update(stats)
    for temporary, name in ((doc_tmp, "documents.jsonl"), (conv_tmp, "conversations.jsonl"), (rejected_tmp, "rejected.jsonl")):
        temporary.replace(output_dir / name)
    (output_dir / "document_index.json").write_text(json.dumps(document_index, ensure_ascii=False), encoding="utf-8")
    cross_source = Counter(group for groups in source_groups.values() for group in groups)
    report = {
        "format": "official-beacon-comem-native-v1", "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "raw_dir": str(raw_dir.resolve()), "output_dir": str(output_dir.resolve()),
        "complete_five_instruction_sources": not missing, "missing_source_files": missing,
        "counts": dict(counts), "by_source": source_stats,
        "unique_exact_documents": len(document_index), "normalized_document_groups": len(group_assignments),
        "normalized_group_split_counts": dict(Counter(group_assignments.values())),
        "cross_source_normalized_groups": sum(number > 1 for number in cross_source.values()),
        "rejection_reasons": dict(reason_counts), "parser_counts": dict(parser_counts),
        "seed": seed, "dev_fraction": dev_fraction,
        "document_id_policy": "exact-context-UTF8-SHA256; all whitespace retained",
        "split_policy": "cross-source-NFC-whitespace-collapsed-context-SHA256; deterministic-seed-hash",
        "memory_placeholder": MEMORY_PLACEHOLDER,
        "raw_conversations": "unchanged author JSONL; each canonical/rejected record references file+zero-based line index",
        "duplicate_policy": "exact raw conversation copies counted and retained under distinct source/index IDs; normalized context keeps them in one split",
        "target_policy": "all assistant turns retained; empty targets flagged for downstream whole-conversation filtering; no answer or BookSum truncation",
        "empty_target_training_policy": "canonical keeps all content; downstream tokenization explicitly excludes the whole conversation if any assistant target is empty",
        "storage_policy": "one exact context per documents record; nested conversation turns; byte-offset document index",
        "pending_before_training": [
            "LongBench and other evaluation-corpus overlap filtering (root owner)",
            "Qwen3 original full-chat tokenization/length filtering and assistant label masks",
            "conversation-level valid target token weighting and resource validation",
            "5K RedPajama LM mixture is separate and not included by this five-source converter",
        ],
        "grouping_limitation": "Exact normalized text identity only; disjoint/partly overlapping excerpts from one underlying book may need additional overlap grouping.",
        "output_bytes": {name: (output_dir / name).stat().st_size for name in names[:4]},
    }
    (output_dir / "processing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Official Activation Beacon SFT conversion", "", "No model was loaded or trained. Original author JSONL files were not modified.", "",
             f"Accepted conversations: {counts['accepted_conversations']}; rejected: {counts['rejected_conversations']}; assistant turns: {counts['assistant_turns']}.",
             f"Stored exact documents: {len(document_index)}; normalized split groups: {len(group_assignments)}.", "",
             "| Source | Input conversations | Accepted | Rejected | Assistant turns | Empty targets |", "|---|---:|---:|---:|---:|---:|"]
    for source, stats in source_stats.items():
        lines.append(f"| {source} | {stats.get('input_conversations',0)} | {stats.get('accepted_conversations',0)} | {stats.get('rejected_conversations',0)} | {stats.get('assistant_turns',0)} | {stats.get('empty_assistant_targets',0)} |")
    lines += ["", "All first-user messages round-trip exactly using stored prefix + exact context + suffix. Later user/assistant content remains unchanged. Unknown layouts are logged in rejected.jsonl; references retain their raw source records.", "",
              "BookSum summaries and all other targets remain complete. Canonical records retain empty assistant outputs and flag them; the agreed downstream tokenization policy explicitly excludes the whole affected conversation with its source/id, rather than silently dropping one turn.", "",
              "Train/dev grouping uses normalized document text across all sources. It does not certify that distinct excerpts from one book cannot overlap.", "",
              "## Pending before training", ""]
    lines.extend("- " + item for item in report["pending_before_training"])
    if missing:
        lines += ["", "Missing original files: " + ", ".join(missing)]
    (output_dir / "PROCESSING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


class DocumentStore:
    """Read only requested exact document variants, using a small in-memory LRU."""

    def __init__(self, directory: str | Path, max_cached_documents: int = 8):
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "document_index.json").read_text(encoding="utf-8"))
        self.stream = (self.directory / "documents.jsonl").open("rb")
        self.max_cached_documents = max(1, max_cached_documents)
        self.cache = OrderedDict()

    def get(self, document_id: str) -> dict:
        if document_id in self.cache:
            value = self.cache.pop(document_id)
        else:
            location = self.index[document_id]
            self.stream.seek(location["offset"])
            value = json.loads(self.stream.read(location["length"]))
            if value["document_id"] != document_id:
                raise ValueError("Document byte-offset index does not match its payload")
        self.cache[document_id] = value
        while len(self.cache) > self.max_cached_documents:
            self.cache.popitem(last=False)
        return value

    def close(self):
        self.stream.close()


def iter_canonical_conversations(directory: str | Path, split: str | None = None) -> Iterator[dict]:
    store = DocumentStore(directory)
    try:
        with (Path(directory) / "conversations.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                conversation = json.loads(line)
                if split is not None and conversation["split"] != split:
                    continue
                context = store.get(conversation["document_id"])["context"]
                yield {**conversation, "context": context,
                       "original_messages": reconstruct_original_messages(conversation, context),
                       "messages": adapted_messages(conversation)}
    finally:
        store.close()


def iter_canonical_examples(directory: str | Path, split: str | None = None) -> Iterator[dict]:
    """On-demand per-turn view. Never persists context-expanded turns to disk."""
    for conversation in iter_canonical_conversations(directory, split):
        history = []
        common = {key: value for key, value in conversation.items() if key not in ("turns", "messages", "original_messages")}
        for turn in conversation["turns"]:
            history.append({"role": "user", "content": turn["question"]})
            yield {**common, **turn, "id": f"{conversation['conversation_id']}:turn:{turn['assistant_turn_index']}",
                   "messages": deepcopy(history), "answers": [turn["answer"]],
                   "original_messages": deepcopy(conversation["original_messages"][:2 * turn["assistant_turn_index"] + 1])}
            history.append({"role": "assistant", "content": turn["answer"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent / "data" / "activation_beacon_original"
    parser.add_argument("--raw-dir", type=Path, default=default_root / "raw")
    parser.add_argument("--output-dir", type=Path, default=default_root / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_corpus(args.raw_dir, args.output_dir, seed=args.seed,
                            dev_fraction=args.dev_fraction, allow_missing=args.allow_missing, overwrite=args.overwrite)
    print(json.dumps({"counts": report["counts"], "output_dir": report["output_dir"],
                      "complete_five_instruction_sources": report["complete_five_instruction_sources"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
