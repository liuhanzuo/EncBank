"""Prepare a fixed real-question LoCoMo stream; no model inference or GPU timing.

The official adapter remains authoritative for samples, packing and scoring.
Fixtures store each formatted document once. A query's full model input is
documents[document_id].context_ids + query.query_ids, never an old prediction.
Importing this module does not change CUDA visibility or thread settings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / "exp/encbank_v2_benchmarks_20260908"
DEFAULT_SOURCE = OLD / "data/locomo/locomo10.json"
DEFAULT_MODEL_CONFIG = OLD / "results/local/qa_cost/full/qasper/fix_all/attempts/0001/config.json"
DEFAULT_OUT = HERE / "fixtures"
SEED = 20260909
PREFIXES = (1, 10, 50, 80)
PROTOCOL = "locomo-real-multiquery-reuse-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def official():
    sys.path.insert(0, str(OLD))
    import official_qa_driver
    return official_qa_driver


def normalized_question(text):
    return re.sub(r"\s+", " ", text.strip()).lower()


def sample_for_query(document, query):
    """Restore the exact input object accepted by official tokenize_pack."""
    require(document["document_id"] == query["document_id"], "Wrong document for query")
    return {**query["sample"], "marked_prompt": document["raw_context"]
            + official().BOUNDARY + query["raw_query_suffix"]}


def full_input_ids(document, query):
    require(document["document_id"] == query["document_id"], "Wrong document for query")
    return document["context_ids"] + query["query_ids"]


def validate_retokenized(document, query, ids, n_context, selected, pack):
    """Verification can run outside a runner's measured preprocessing region."""
    actual = ids[0].tolist() if hasattr(ids, "tolist") else list(ids)
    require(actual == full_input_ids(document, query), "Full context/query token mismatch")
    require(n_context == len(document["context_ids"]), "Context boundary mismatch")
    require(list(selected) == query["selected_indices"], "Ordered selected pack mismatch")
    require(pack == query["pack"], "Pack metadata mismatch")


def retokenize_query(tokenizer, document, query, *, verify=True):
    """Full official re-tokenization for validation, not per-request cost policy.

    The serving runner can tokenize formatted_context once per document, then
    formatted_query and retrieval_question once per request. This helper performs
    the complete old path to verify that optimized preprocessing stays identical.
    """
    result = official().tokenize_pack(tokenizer, sample_for_query(document, query),
                                      query["chunk_size"], query["selector"], query["topk"])
    if verify:
        validate_retokenized(document, query, *result)
    return result


def score_query(prediction, query):
    """Return (official category score in [0,1], normalized/scored prediction)."""
    return official().score_prediction(prediction, query["sample"])


def build_workload(samples, conversations, tokenizer, *, source_path, seed=SEED,
                   chunk_size=512, selector="bm25", topk=12, prefixes=PREFIXES):
    """Build once from official samples, grouping by exact formatted context IDs."""
    qa = official()
    source_path = str(Path(source_path).resolve())
    by_conversation = defaultdict(list)
    source_ids = set()
    for sample in samples:
        require(sample["id"] not in source_ids, "Duplicate original source ID")
        source_ids.add(sample["id"])
        match = re.fullmatch(r"conv(\d+)_qa(\d+)", sample["id"])
        require(match is not None, "Unexpected official source ID")
        ci, qi = map(int, match.groups())
        require(ci < len(conversations) and qi < len(conversations[ci]["qa"]), "Invalid source position")
        original = conversations[ci]["qa"][qi]
        require(sample["category"] in (1, 2, 3, 4), "Only official categories 1-4 belong in this stream")
        require(sample["retrieval_question"] == original["question"], "Source question changed")
        require(sample["category"] == original["category"], "Source category changed")
        require(sample["answers"] == [str(original["answer"])], "Source answer changed")
        by_conversation[ci].append((qi, sample))
    require(set(by_conversation) == set(range(len(conversations))), "A conversation has no eligible queries")

    documents, queries, grouping_tokens = [], [], {}
    duplicate_count = 0
    for ci in range(len(conversations)):
        conversation = conversations[ci]
        docid = str(conversation["sample_id"])
        require(docid not in {d["document_id"] for d in documents}, "Duplicate conversation sample_id")
        deduplicated = {}
        for qi, sample in by_conversation[ci]:
            key = normalized_question(sample["retrieval_question"])
            ref = {"path": source_path, "conversation_index": ci, "sample_id": docid,
                   "qa_index": qi, "id": sample["id"], "index": sample["index"],
                   "evidence": sample.get("evidence", [])}
            if key in deduplicated:
                representative, refs = deduplicated[key]
                require(representative["category"] == sample["category"], "Duplicate question changes category")
                require(representative["answers"] == sample["answers"], "Duplicate question changes answer; adjudicate before deduplication")
                require(normalized_question(representative["question"]) == normalized_question(sample["question"]),
                        "Duplicate question changes instructions")
                refs.append(ref)
                duplicate_count += 1
            else:
                deduplicated[key] = (sample, [ref])
        ordered = list(deduplicated.values())
        # A fresh RNG per document makes its order independent of other documents.
        random.Random(seed).shuffle(ordered)
        require(len(ordered) >= max(prefixes, default=0), "A document cannot supply every required prefix")
        document = None
        order = []
        for position, (sample, refs) in enumerate(ordered):
            marked = sample["marked_prompt"]
            require(marked.count(qa.BOUNDARY) == 1, "Ambiguous original boundary")
            raw_context, raw_suffix = marked.split(qa.BOUNDARY)
            formatted = tokenizer.apply_chat_template([{"role": "user", "content": marked}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            require(formatted.count(qa.BOUNDARY) == 1, "Ambiguous formatted boundary")
            formatted_context, formatted_query = formatted.split(qa.BOUNDARY)
            ids, count, selected, pack = qa.tokenize_pack(tokenizer, sample, chunk_size, selector, topk)
            tokens = ids[0].tolist()
            context_ids, query_ids = tokens[:count], tokens[count:]
            require(context_ids == tokenizer.encode(formatted_context, add_special_tokens=False), "Formatted context token mismatch")
            require(query_ids == tokenizer.encode(formatted_query, add_special_tokens=False), "Formatted query token mismatch")
            if document is None:
                key = tuple(context_ids)
                require(key not in grouping_tokens, "Distinct conversation IDs share an identical context; explicit grouping required")
                grouping_tokens[key] = docid
                document = {"document_id": docid, "conversation_index": ci, "source_path": source_path,
                            "raw_context": raw_context, "formatted_context": formatted_context,
                            "context_ids": context_ids, "context_tokens": count,
                            "raw_source_question_count": len(by_conversation[ci]),
                            "query_count": len(ordered), "shuffle_seed": seed,
                            "grouping": "identical complete formatted context token sequence"}
            else:
                require(context_ids == document["context_ids"], "Formatted context changed within a conversation")
                require(formatted_context == document["formatted_context"] and raw_context == document["raw_context"],
                        "Context text changed within a conversation")
            clean_sample = {k: v for k, v in sample.items() if k != "marked_prompt"}
            query = {"id": sample["id"], "document_id": docid, "stream_index": position,
                     "source_index": sample["index"], "source_refs": refs, "sample": clean_sample,
                     "question": sample["question"], "retrieval_question": sample["retrieval_question"],
                     "answers": sample["answers"], "category": sample["category"],
                     "category_name": sample["category_name"], "max_new_tokens": sample["max_new_tokens"],
                     "raw_query_suffix": raw_suffix, "formatted_query": formatted_query,
                     "query_ids": query_ids, "selected_indices": list(selected), "pack": pack,
                     "chunk_size": chunk_size, "selector": selector, "topk": topk}
            validate_retokenized(document, query, ids, count, selected, pack)
            queries.append(query)
            order.append(query["id"])
        document["query_ids_ordered"] = order
        document["prefixes"] = {str(q): order[:q] for q in prefixes}
        document["prefixes"]["full"] = list(order)
        documents.append(document)
        print(json.dumps({"prepared_document": docid, "queries": len(order),
                          "context_tokens": document["context_tokens"]}), flush=True)

    categories = Counter(q["category"] for q in queries)
    manifest = {"protocol": PROTOCOL, "prepared_at": datetime.now(timezone.utc).isoformat(),
                "source": source_path, "source_official_driver": str(OLD / "official_qa_driver.py"),
                "source_official_metric": str(OLD / "protocol_sources/locomo/task_eval/evaluation.py"),
                "documents_file": "documents.json", "queries_file": "queries.jsonl",
                "document_count": len(documents), "raw_eligible_queries": len(source_ids),
                "unique_queries": len(queries), "duplicates_removed": duplicate_count,
                "category_counts": dict(sorted(categories.items())), "categories": [1, 2, 3, 4],
                "deduplication": "per conversation; lowercase and collapsed whitespace; retain first source order; require identical category and answers; retain all source references",
                "shuffle_seed": seed, "shuffle": "independent random.Random(seed).shuffle per document after source-order deduplication",
                "prefixes": list(prefixes) + ["full"], "chunk_size": chunk_size,
                "selector": selector, "topk": topk,
                "max_new_tokens": sorted({q["max_new_tokens"] for q in queries}),
                "chat_template": "official Qwen user template; add_generation_prompt=True; enable_thinking=False",
                "truncation": "none", "prior_answers_in_context": False,
                "full_input_reconstruction": "document.context_ids + query.query_ids",
                "verified_query_inputs": len(queries), "verified_context_groups": len(grouping_tokens),
                "timing_eligible": False, "measurement_results": False, "model_loaded": False,
                "cost_accounting": "CPU fixture preparation is not a latency measurement. Runner measures formatted document tokenization once per document; formatted query tokenization and BM25 once per request. Full retokenize_query is an input-validation helper; offline preparation is not charged or reported as RTX 5090 runtime."}
    return manifest, documents, queries


def validate_workload(manifest, documents, queries):
    require(manifest["protocol"] == PROTOCOL, "Wrong workload protocol")
    require(len(documents) == manifest["document_count"], "Document count mismatch")
    require(len(queries) == manifest["unique_queries"], "Query count mismatch")
    dm = {d["document_id"]: d for d in documents}
    qm = {q["id"]: q for q in queries}
    require(len(dm) == len(documents) and len(qm) == len(queries), "Duplicate document/query IDs")
    visited = []
    for doc in documents:
        order = doc["query_ids_ordered"]
        require(len(order) == doc["query_count"] and len(set(order)) == len(order), "Broken query order")
        require(doc["prefixes"]["full"] == order, "Full stream differs")
        require(len(doc["context_ids"]) == doc["context_tokens"], "Context token count mismatch")
        for n in manifest["prefixes"]:
            if n != "full":
                require(len(order) >= n and doc["prefixes"][str(n)] == order[:n], "Broken common prefix")
        for index, qid in enumerate(order):
            require(qid in qm, "Query missing from stream")
            query = qm[qid]
            require(query["document_id"] == doc["document_id"] and query["stream_index"] == index, "Wrong query group/order")
            require(query["pack"]["context_tokens"] == len(doc["context_ids"]), "Wrong query context length")
            require(query["pack"]["query_tokens"] == len(query["query_ids"]), "Wrong query length")
            require(query["pack"]["input_tokens"] == len(full_input_ids(doc, query)), "Wrong full input length")
            require(query["selected_indices"] == query["pack"]["selected_indices"], "Wrong ordered selection")
            require(query["answers"] == query["sample"]["answers"] and query["category"] == query["sample"]["category"], "Wrong scoring fields")
            visited.append(qid)
    require(set(visited) == set(qm) and len(visited) == len(queries), "Unassigned query")
    return True


def load_workload(path=DEFAULT_OUT):
    """Return (manifest, document-list, query-list); no tokenizer/model load."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    manifest = read_json(path / "manifest.json")
    documents = read_json(path / manifest["documents_file"])
    queries = [json.loads(line) for line in (path / manifest["queries_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_workload(manifest, documents, queries)
    return manifest, documents, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    from transformers import AutoTokenizer
    config = read_json(args.model_config) if args.tokenizer is None else None
    tokenizer_path = args.tokenizer or Path(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    conversations = read_json(args.source)
    options = SimpleNamespace(locomo_data=str(args.source), seed=SEED, categories=[1, 2, 3, 4],
                              max_samples=-1, num_shards=1, shard_index=0)
    manifest, documents, queries = build_workload(list(official().locomo_samples(options)),
        conversations, tokenizer, source_path=args.source)
    require(len(documents) == 10 and len(queries) == 1529 and manifest["raw_eligible_queries"] == 1540,
            "Unexpected official ten-conversation release/deduplication counts")
    require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    manifest.update(tokenizer_path=str(tokenizer_path.resolve()),
                    tokenizer_origin_config=str(args.model_config.resolve()) if config else None,
                    runtime={"torch_cpu_threads": torch.get_num_threads(),
                             "torch_interop_threads": torch.get_num_interop_threads(),
                             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                             "cuda_initialized": False, "model_loaded": False})
    validate_workload(manifest, documents, queries)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "documents.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "queries.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_workload(args.out)
    print(json.dumps({"out": str(args.out.resolve()), "documents": len(documents), "queries": len(queries),
                      "duplicates_removed": manifest["duplicates_removed"], "timing_eligible": False}), flush=True)


if __name__ == "__main__":
    main()
