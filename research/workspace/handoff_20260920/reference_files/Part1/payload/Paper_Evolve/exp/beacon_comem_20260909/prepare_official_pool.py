"""CPU-only, streaming eligibility/overlap preparation for native Beacon SFT.

This is a candidate-pool audit, not a training-budget selection. Canonical and
raw inputs remain unchanged. All failures are itemized outside the training
pool. Full PreparedConversation dictionaries are emitted one at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zlib

from prepare_sft_data import BenchmarkExclusions, load_benchmark_contexts, normalize, sha256_file
from prepare_official_sft_data import DocumentStore, adapted_messages, reconstruct_original_messages
from official_sft import ConversationFiltered, prepare_conversation

ROOT = Path(__file__).resolve().parent
ANCHOR_WORDS = 256
WINNOW_WINDOW = 64
MASK64 = (1 << 64) - 1
ROLL_BASE = 1000003


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, value):
    temporary = Path(str(path) + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def bounded_map(executor, function, values, pending=16):
    """Preserve source order without eagerly queuing the entire long-text corpus."""
    iterator = iter(values)
    queue = deque()
    for _ in range(pending):
        try:
            queue.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while queue:
        yield queue.popleft().result()
        try:
            queue.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def long_anchors(words, width=ANCHOR_WORDS, window=WINNOW_WINDOW):
    """Content-defined winnowing; each selected digest verifies all `width` words.

    Rolling hashes choose locations only; BLAKE2b-128 hashes the full normalized
    word span for equality. Rightmost-min tie breaking is deterministic. Any
    identical contiguous span of width+window-1 words contains a shared anchor,
    independent of its offset within either document. Shorter matches may be
    found, but are not guaranteed. Repeated anchors count once per document.
    """
    if len(words) < width:
        return set()
    values = [zlib.crc32(word.encode("utf-8")) + 1 for word in words]
    power = pow(ROLL_BASE, width - 1, 1 << 64)
    rolling = 0
    for value in values[:width]:
        rolling = (rolling * ROLL_BASE + value) & MASK64
    minimums, chosen = deque(), set()
    ngrams = len(words) - width + 1
    effective_window = min(window, ngrams)
    last_position = -1
    for start in range(ngrams):
        if start:
            rolling = ((rolling - values[start - 1] * power) * ROLL_BASE + values[start + width - 1]) & MASK64
        while minimums and minimums[-1][1] >= rolling:
            minimums.pop()
        minimums.append((start, rolling))
        while minimums[0][0] <= start - effective_window:
            minimums.popleft()
        if start >= effective_window - 1:
            position = minimums[0][0]
            if position != last_position:
                chosen.add(hashlib.blake2b(" ".join(words[position:position + width]).encode("utf-8"), digest_size=16).digest())
                last_position = position
    return chosen


class DisjointSet:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, value):
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def component_split(group_id, seed, dev_fraction):
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") / 2**64 < dev_fraction else "train"


BENCHMARK = None


def init_overlap(benchmark_dir):
    global BENCHMARK
    BENCHMARK = BenchmarkExclusions(load_benchmark_contexts(Path(benchmark_dir))[0])


def audit_document(row):
    normalized = normalize(row["context"])
    words = normalized.split()
    return {
        "document_id": row["document_id"],
        "normalized_hash": hashlib.sha256(normalized.encode()).hexdigest(),
        "normalized_word_count": len(words),
        "benchmark_match": BENCHMARK.match("", row["context"]),
        "anchors": long_anchors(words),
    }


def build_overlap(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "overlap_report.json").exists():
        raise ValueError("Overlap report already exists; use a fresh output directory for a new audit")
    _, benchmark_receipts = load_benchmark_contexts(args.benchmark_dir)
    documents, inverted, exact = [], {}, {}
    with ProcessPoolExecutor(args.workers, initializer=init_overlap, initargs=(str(args.benchmark_dir),)) as executor:
        for result in bounded_map(executor, audit_document, rows(args.processed / "documents.jsonl"), args.workers * 4):
            index = len(documents)
            anchors = result.pop("anchors")
            result["anchor_count"] = len(anchors)
            documents.append(result)
            exact.setdefault(result["normalized_hash"], []).append(index)
            for anchor in anchors:
                previous = inverted.get(anchor)
                if previous is None:
                    inverted[anchor] = index
                elif isinstance(previous, int):
                    inverted[anchor] = [previous, index]
                else:
                    previous.append(index)
            if len(documents) % 500 == 0:
                progress = {"phase": "overlap", "documents": len(documents), "seconds": round(time.perf_counter() - start, 1)}
                write_json(args.output / "progress.json", progress)
                print(dump(progress), flush=True)
    sets = DisjointSet(len(documents))
    for members in exact.values():
        for member in members[1:]:
            sets.union(members[0], member)
    shared_anchors = 0
    high_frequency = []
    for anchor, members in inverted.items():
        if isinstance(members, int):
            continue
        shared_anchors += 1
        if len(members) >= 64:
            high_frequency.append({"anchor_digest": anchor.hex(), "documents": len(members),
                                   "example_document_ids": [documents[i]["document_id"] for i in members[:5]]})
        # Keep common spans conservatively: a fixed needle haystack must not be
        # waived as boilerplate. Giant components are reported, never split.
        for member in members[1:]:
            sets.union(members[0], member)
    unique_anchors = len(inverted)
    del inverted
    components = defaultdict(list)
    for index in range(len(documents)):
        components[sets.find(index)].append(index)
    source_by_doc = defaultdict(Counter)
    for conversation in rows(args.processed / "conversations.jsonl"):
        source_by_doc[conversation["document_id"]][conversation["source"]] += 1
    component_rows, direct_hits, propagated = [], 0, 0
    with (args.output / "document_overlap.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for members in components.values():
            ids = sorted(documents[i]["document_id"] for i in members)
            group_id = "overlap:" + hashlib.sha256("\n".join(ids).encode()).hexdigest()
            split = component_split(group_id, args.seed, args.dev_fraction)
            hits = [documents[i] for i in members if documents[i]["benchmark_match"]]
            source_counts = Counter()
            for i in members:
                source_counts.update(source_by_doc[documents[i]["document_id"]])
            component_rows.append({"overlap_group_id": group_id, "documents": len(members), "split": split,
                                   "conversation_counts_by_source": dict(source_counts),
                                   "benchmark_excluded": bool(hits), "direct_hit_documents": len(hits),
                                   "example_document_ids": ids[:5]})
            for i in members:
                row = documents[i]
                direct_hits += bool(row["benchmark_match"])
                propagated += bool(hits) and not bool(row["benchmark_match"])
                row.update(overlap_group_id=group_id, split=split, component_documents=len(members),
                           benchmark_excluded=bool(hits),
                           benchmark_exclusion_origin_document_id=hits[0]["document_id"] if hits else None)
                stream.write(dump(row) + "\n")
    with (args.output / "overlap_components.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(component_rows, key=lambda item: (-item["documents"], item["overlap_group_id"])):
            stream.write(dump(row) + "\n")
    write_json(args.output / "high_frequency_anchors.json", sorted(high_frequency, key=lambda item: -item["documents"]))
    report = {
        "complete": True, "documents": len(documents), "components": len(components),
        "normalization": "prepare_sft_data.normalize: NFKC, casefold, Unicode word extraction, single spaces",
        "anchor_words": ANCHOR_WORDS, "winnow_window": WINNOW_WINDOW,
        "guaranteed_identical_span_words": ANCHOR_WORDS + WINNOW_WINDOW - 1,
        "anchor_equality": "BLAKE2b-128 over all 256 normalized words; rolling hash selects positions only",
        "high_frequency_policy": "All shared anchors retained, including public boilerplate; conservative over-grouping possible. No giant group is split.",
        "unique_anchors": unique_anchors, "shared_anchors": shared_anchors,
        "anchors_in_at_least_64_documents": len(high_frequency),
        "benchmark_direct_hit_documents": direct_hits, "benchmark_propagated_documents": propagated,
        "benchmark_total_excluded_documents": direct_hits + propagated,
        "benchmark_protocol": "Existing BenchmarkExclusions: normalized exact context or >=8 distinct sampled 16-word benchmark shingles; no title heuristic because canonical records lack verified titles. Matches propagate to entire overlap component.",
        "benchmark_receipts": benchmark_receipts,
        "split": {"seed": args.seed, "dev_fraction": args.dev_fraction, "unit": "entire overlap connected component"},
        "limitations": ["No verified original book/paper IDs; this is textual component separation, not book-ID separation.",
                        "Does not prove absence of paraphrase, translation, or shorter/disjoint excerpts from the same work.",
                        "Common public text can join unrelated documents and conservatively exclude or group them.",
                        "Benchmark exclusions cover only these six local LongBench files."],
        "largest_components": sorted(component_rows, key=lambda item: -item["documents"])[:20],
        "cpu_elapsed_seconds": time.perf_counter() - start,
    }
    write_json(args.output / "overlap_report.json", report)
    print(dump({key: report[key] for key in ("documents", "components", "benchmark_total_excluded_documents", "cpu_elapsed_seconds")}), flush=True)


TOKENIZER, DOCSTORE = None, None


def init_tokenizer(tokenizer_path, processed):
    global TOKENIZER, DOCSTORE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    DOCSTORE = DocumentStore(Path(processed), max_cached_documents=4)


def classify_error(exc):
    message = str(exc)
    if isinstance(exc, ConversationFiltered):
        return exc.reason
    checks = (("Explicit chat/reasoning control", "embedded_chat_control"),
              ("Unsupported message role", "unsupported_role_content"),
              ("alternate", "nonalternating_roles"),
              ("Empty message", "empty_nonassistant_message"),
              ("changed assistant", "native_template_transforms_target"),
              ("transformed", "native_template_transforms_target"),
              ("Rendering modified", "native_template_transforms_target"),
              ("boundary", "ambiguous_assistant_token_boundary"),
              ("offset", "invalid_token_offsets"),
              ("target content", "native_template_transforms_target"),
              ("target text", "native_template_transforms_target"))
    for fragment, reason in checks:
        if fragment.lower() in message.lower():
            return reason
    return "validation_" + type(exc).__name__


def tokenize_row(conversation):
    identity = {key: conversation.get(key) for key in ("conversation_id", "document_id", "source", "source_file", "source_index", "parser", "overlap_group_id", "split")}
    try:
        document = DOCSTORE.get(conversation["document_id"])
        conversation["context"] = document["context"]
        conversation["original_messages"] = reconstruct_original_messages(conversation, document["context"])
        conversation["messages"] = adapted_messages(conversation)
        prepared = prepare_conversation(conversation, TOKENIZER, min_length=7200, max_length=20000)
        value = prepared.to_dict()
        value["overlap_group_id"] = conversation["overlap_group_id"]
        value["tokenizer_fingerprint"] = conversation["tokenizer_fingerprint"]
        summary = dict(identity, status="valid", benchmark_excluded=conversation.get("benchmark_excluded", False),
            original_template_token_count=prepared.original_template_token_count,
            document_token_count=prepared.document_token_count, reader_template_token_count=prepared.reader_template_token_count,
            raw_uncompressed_read_pack_tokens=prepared.raw_uncompressed_read_pack_tokens,
            assistant_turn_count=prepared.assistant_turn_count, target_token_count=prepared.target_token_count)
        return summary, dump(value) if not summary["benchmark_excluded"] else None
    except Exception as exc:
        # Full inventory records malformed rows as failures; they NEVER enter
        # the training pool. Operational setup failures still fail the executor.
        return dict(identity, status="filtered" if isinstance(exc, ConversationFiltered) else "error",
                    reason=classify_error(exc), exception_type=type(exc).__name__, message=str(exc),
                    details=getattr(exc, "details", {}), traceback=traceback.format_exc(limit=5)), None


def tokenizer_receipt(path):
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")
    files = {name: sha256_file(path / name) for name in names if (path / name).is_file()}
    if "tokenizer.json" not in files or "tokenizer_config.json" not in files:
        raise ValueError("Need local tokenizer.json and tokenizer_config.json")
    fingerprint = "sha256:" + hashlib.sha256(dump(files).encode()).hexdigest()
    return {"tokenizer_fingerprint": fingerprint, "files": files, "local_path": str(path),
            "enable_thinking": False, "use_fast": True, "local_files_only": True}


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    return {"min": values[0], "p50": values[(len(values)-1)//2], "p95": values[int((len(values)-1)*.95)], "max": values[-1]}


def tokenize_pool(args, preflight=False):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if not preflight and (args.output / "pool_report.json").exists():
        raise ValueError("Completed-pool report already exists; use a fresh output directory for a new run")
    receipt = tokenizer_receipt(args.tokenizer)
    if preflight:
        overlap = {}
    else:
        if not json.loads((args.output / "overlap_report.json").read_text(encoding="utf-8"))["complete"]:
            raise ValueError("Overlap audit is incomplete")
        overlap = {row["document_id"]: row for row in rows(args.output / "document_overlap.jsonl")}
    selected_per_source = Counter()
    def inputs():
        for conversation in rows(args.processed / "conversations.jsonl"):
            source = conversation["source"]
            if preflight and selected_per_source[source] >= args.preflight_per_source:
                continue
            selected_per_source[source] += 1
            if not preflight:
                document_id = conversation["document_id"]
                if document_id not in overlap:
                    raise ValueError("Canonical document missing from completed overlap audit: " + document_id)
                doc = overlap[document_id]
                if not all(key in doc for key in ("overlap_group_id", "split", "benchmark_excluded")):
                    raise ValueError("Incomplete overlap metadata: " + document_id)
                if not str(doc["overlap_group_id"]).startswith("overlap:") or doc["split"] not in {"train", "dev"} or type(doc["benchmark_excluded"]) is not bool:
                    raise ValueError("Invalid overlap metadata: " + document_id)
            else:
                doc = {}
            conversation.update(overlap_group_id=doc.get("overlap_group_id", "preflight:" + conversation["document_id"]),
                                split=doc.get("split", conversation["split"]),
                                benchmark_excluded=doc.get("benchmark_excluded", False),
                                tokenizer_fingerprint=receipt["tokenizer_fingerprint"])
            yield conversation
    prefix = "preflight" if preflight else "pool"
    stats, sources, reasons = Counter(), defaultdict(Counter), Counter()
    metrics = defaultdict(lambda: defaultdict(list))
    source_groups = defaultdict(lambda: defaultdict(set))
    split_groups, split_docs = defaultdict(set), defaultdict(set)
    ids = set()
    with (args.output / f"{prefix}_inventory.jsonl").open("w", encoding="utf-8", newline="\n") as inventory, \
         (args.output / f"{prefix}_errors.jsonl").open("w", encoding="utf-8", newline="\n") as errors, \
         (args.output / ("preflight_index.jsonl" if preflight else "eligible_index.jsonl.part")).open("w", encoding="utf-8", newline="\n") as eligible_index, \
         (args.output / ("preflight_prepared.jsonl" if preflight else "eligible.jsonl.part")).open("w", encoding="utf-8", newline="\n") as prepared_stream, \
         ProcessPoolExecutor(args.workers, initializer=init_tokenizer, initargs=(str(args.tokenizer), str(args.processed))) as executor:
        for summary, serialized in bounded_map(executor, tokenize_row, inputs(), args.workers * 4):
            cid, source = summary["conversation_id"], summary["source"]
            if cid in ids:
                raise ValueError("Duplicate canonical conversation ID: " + cid)
            ids.add(cid)
            stats["input_conversations"] += 1
            stats[summary["status"]] += 1
            sources[source]["input"] += 1
            sources[source][summary["status"]] += 1
            if summary["status"] != "valid":
                reasons[summary["reason"]] += 1
                sources[source][summary["reason"]] += 1
                errors.write(dump(summary) + "\n")
            else:
                for key in ("original_template_token_count", "document_token_count", "reader_template_token_count", "raw_uncompressed_read_pack_tokens", "assistant_turn_count", "target_token_count"):
                    metrics[source][key].append(summary[key])
                if summary["benchmark_excluded"]:
                    stats["valid_but_benchmark_excluded"] += 1
                    sources[source]["valid_but_benchmark_excluded"] += 1
                else:
                    assert serialized is not None
                    prepared_stream.write(serialized + "\n")
                    eligible_index.write(dump({"id": cid, **{key: summary[key] for key in ("conversation_id", "source", "split", "document_id", "overlap_group_id", "document_token_count", "original_template_token_count", "reader_template_token_count", "raw_uncompressed_read_pack_tokens", "target_token_count", "assistant_turn_count")}, "tokenizer_fingerprint": receipt["tokenizer_fingerprint"]}) + "\n")
                    split = summary["split"]
                    stats["eligible"] += 1
                    stats["eligible_" + split] += 1
                    stats["eligible_assistant_turns"] += summary["assistant_turn_count"]
                    stats["eligible_target_tokens"] += summary["target_token_count"]
                    stats["eligible_original_template_tokens"] += summary["original_template_token_count"]
                    sources[source]["eligible_" + split] += 1
                    source_groups[source][split].add(summary["overlap_group_id"])
                    split_groups[split].add(summary["overlap_group_id"])
                    split_docs[split].add(summary["document_id"])
            inventory.write(dump(summary) + "\n")
            if stats["input_conversations"] % 100 == 0:
                progress = {"phase": prefix, **dict(stats), "cpu_elapsed_seconds": round(time.perf_counter()-start, 1)}
                write_json(args.output / "progress.json", progress)
                print(dump(progress), flush=True)
    assert not (split_groups["train"] & split_groups["dev"]), "Overlap group crosses splits"
    assert not (split_docs["train"] & split_docs["dev"]), "Exact document crosses splits"
    if not preflight:
        (args.output / "eligible.jsonl.part").replace(args.output / "eligible.jsonl")
        (args.output / "eligible_index.jsonl.part").replace(args.output / "eligible_index.jsonl")
    report = {"complete": True, "kind": prefix, **dict(stats), "by_source": dict(sources),
        "filter_and_error_reasons": dict(reasons), "valid_template_distributions_by_source": {source: {key: quantiles(values) for key, values in items.items()} for source, items in metrics.items()},
        "eligible_overlap_groups_by_source_split": {source: {split: len(values) for split, values in items.items()} for source, items in source_groups.items()},
        "eligible_overlap_groups_by_split": {split: len(values) for split, values in split_groups.items()},
        "cross_split_document_ids": 0, "cross_split_overlap_groups": 0,
        "tokenizer": receipt, "cpu_elapsed_seconds": time.perf_counter()-start,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_processed": str(args.processed.resolve()),
        "training_selection": "No pilot budget or training launch selected; eligible full-conversation candidate pool only.",
        "schema": "PreparedConversation.to_dict() plus overlap_group_id and tokenizer_fingerprint; native full-template 7200–20000; full documents/all assistant targets; no truncation.",
        "empty_assistant_policy": "Whole conversation explicitly filtered; canonical/raw remain lossless.",
        "error_policy": "Every filtered/error row is inventoried with reason, identity and details; no such row enters eligible output."}
    write_json(args.output / f"{prefix}_report.json", report)
    if preflight:
        report["projected_all_conversations_seconds_from_small_sample"] = report["cpu_elapsed_seconds"] / max(1, stats["input_conversations"]) * 23319
        write_json(args.output / f"{prefix}_report.json", report)
    print(dump({key: report.get(key) for key in ("kind", "input_conversations", "eligible", "filtered", "error", "cpu_elapsed_seconds", "projected_all_conversations_seconds_from_small_sample")}), flush=True)


def validate_pool_artifacts(processed, output):
    """Independently validate complete compact metadata; never load full tokens.

    Useful also for a job already started before a fail-closed guard was added.
    Does not certify token arrays beyond the per-conversation preparation checks.
    """
    processed, output = Path(processed), Path(output)
    report = json.loads((output / "pool_report.json").read_text(encoding="utf-8"))
    overlap_report = json.loads((output / "overlap_report.json").read_text(encoding="utf-8"))
    if not report.get("complete") or not overlap_report.get("complete"):
        raise ValueError("Cannot validate incomplete preparation")
    if not (output / "eligible.jsonl").is_file() or (output / "eligible.jsonl.part").exists():
        raise ValueError("Tokenized eligible pool is absent or incomplete")
    overlap = {row["document_id"]: row for row in rows(output / "document_overlap.jsonl")}
    canonical = {}
    for row in rows(processed / "conversations.jsonl"):
        cid = row["conversation_id"]
        if cid in canonical:
            raise ValueError("Duplicate canonical identity")
        canonical[cid] = {key: row[key] for key in ("document_id", "source")}
    if {row["document_id"] for row in canonical.values()} != set(overlap):
        raise ValueError("Canonical/overlap document sets differ")
    inventoried, expected_eligible = set(), set()
    valid_statuses = {"valid", "filtered", "error"}
    for row in rows(output / "pool_inventory.jsonl"):
        cid = row["conversation_id"]
        if cid not in canonical or cid in inventoried:
            raise ValueError("Unknown/duplicate inventory identity: " + cid)
        inventoried.add(cid)
        if any(row[key] != canonical[cid][key] for key in ("document_id", "source")):
            raise ValueError("Inventory identity metadata differs: " + cid)
        doc = overlap[row["document_id"]]
        if any(row[key] != doc[key] for key in ("overlap_group_id", "split")):
            raise ValueError("Inventory bypassed overlap assignment: " + cid)
        if not row["overlap_group_id"].startswith("overlap:") or row["status"] not in valid_statuses:
            raise ValueError("Invalid inventory group/status: " + cid)
        if row["status"] == "valid":
            if row["benchmark_excluded"] != doc["benchmark_excluded"]:
                raise ValueError("Inventory benchmark exclusion differs: " + cid)
            if not doc["benchmark_excluded"]:
                expected_eligible.add(cid)
        elif not row.get("reason"):
            raise ValueError("Failed row has no explicit reason: " + cid)
    if inventoried != set(canonical):
        raise ValueError("Not every canonical conversation was inventoried")
    selected, groups, documents = set(), defaultdict(set), defaultdict(set)
    for row in rows(output / "eligible_index.jsonl"):
        cid = row["conversation_id"]
        if cid != row["id"] or cid not in expected_eligible or cid in selected:
            raise ValueError("Unknown, excluded or duplicate eligible identity: " + cid)
        selected.add(cid)
        doc = overlap[row["document_id"]]
        if any(row[key] != doc[key] for key in ("overlap_group_id", "split")) or doc["benchmark_excluded"]:
            raise ValueError("Eligible row bypassed group/exclusion: " + cid)
        if row["tokenizer_fingerprint"] != report["tokenizer"]["tokenizer_fingerprint"]:
            raise ValueError("Tokenizer fingerprint differs: " + cid)
        if not 7200 <= row["original_template_token_count"] <= 20000 or row["target_token_count"] <= 0:
            raise ValueError("Invalid eligible length/target count: " + cid)
        groups[row["split"]].add(row["overlap_group_id"])
        documents[row["split"]].add(row["document_id"])
    if selected != expected_eligible or len(selected) != report["eligible"] or len(inventoried) != report["input_conversations"]:
        raise ValueError("Eligible/inventory/report counts differ")
    if groups["train"] & groups["dev"] or documents["train"] & documents["dev"]:
        raise ValueError("Shared document/component crosses splits")
    result = {"passed": True, "canonical_conversations": len(canonical), "unique_documents": len(overlap),
              "inventoried_conversations": len(inventoried), "eligible_conversations": len(selected),
              "all_canonical_documents_in_overlap": True, "preflight_groups_in_eligible": 0,
              "benchmark_excluded_in_eligible": 0, "cross_split_groups": 0, "cross_split_documents": 0,
              "tokenizer_fingerprint": report["tokenizer"]["tokenizer_fingerprint"],
              "validation_scope": "independent compact metadata reconciliation; token arrays were checked by prepare_conversation per row"}
    write_json(output / "pool_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "overlap", "tokenize", "validate", "all"), default="all")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/activation_beacon_original/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/activation_beacon_original/prepared")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT.parent / "comem_v2_benchmarks_20260908/data/longbench")
    parser.add_argument("--tokenizer", type=Path, default=Path("/srv/encbank/legacy_workspace/models/Qwen3-8B"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=.05)
    parser.add_argument("--preflight-per-source", type=int, default=20)
    args = parser.parse_args()
    if args.phase == "preflight":
        tokenize_pool(args, preflight=True)
    if args.phase in {"overlap", "all"}:
        build_overlap(args)
    if args.phase in {"tokenize", "all"}:
        tokenize_pool(args)
    if args.phase in {"validate", "all"}:
        print(dump(validate_pool_artifacts(args.processed, args.output)), flush=True)


if __name__ == "__main__":
    main()
