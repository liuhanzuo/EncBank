"""Standard-library contracts for the fixed-pack, distinct-query pilot."""
from __future__ import annotations

from pathlib import Path
import random

from infra_protocol import digest, json_digest, read_json

HERE = Path(__file__).resolve().parent
VERSION = "sparse-comem-fixed-pack-reuse-v1"
QUALITY_VERSION = "sparse-comem-trained-reuse-quality-v1"


def source_identity():
    names = ("reuse_protocol.py", "reuse_benchmark.py", "sparse_reader.py",
             "native_prefix_reader.py", "native_infra_readers.py")
    result = {name: digest(HERE / name) for name in names}
    result["comem/model.py"] = digest(HERE.parents[1] / "COMem/comem/model.py")
    result["train_8b_baseline.py"] = digest(HERE.parent / "comem_v2_benchmarks_20260908/train_8b_baseline.py")
    return result


def recipe(args):
    return {"arm": "D0" if args.arm in ("D0", "NATIVE", "FULL") else args.arm,
            "j": args.j, "m": args.m, "rho": args.rho,
            "rank": args.rank, "alpha": args.alpha}


def workload(tokenizer, config, args, *, documents=None, prompts=None):
    """Arm-independent token stream. Distinct prompts have exactly equal lengths."""
    count = int(args.reuse_requests)
    if count < 2 or args.prompt_tokens < 2 or args.document_tokens < 1 or args.chunk_size < 1:
        raise ValueError("Need at least two requests, two prompt tokens and nonempty chunks")
    rng = random.Random(args.seed)
    special = set(tokenizer.all_special_ids)
    vocab = int(config.vocab_size)
    choices = [i for i in range(3, min(vocab, 30000)) if i not in special]
    if len(choices) < count:
        raise ValueError("Vocabulary cannot supply distinct request markers")
    if documents is None:
        document = [rng.choice(choices) for _ in range(args.document_tokens)]
        documents = [document[i:i + args.chunk_size] for i in range(0, len(document), args.chunk_size)]
    if prompts is None:
        text = tokenizer.encode("Read the supplied records and answer the current question. ", add_special_tokens=False)
        if not text:
            raise ValueError("Tokenizer generated an empty prompt seed")
        stem = (text * (1 + args.prompt_tokens // len(text)))[:args.prompt_tokens]
        markers = rng.sample(choices, count)
        prompts = [stem[:-1] + [marker] for marker in markers]
    documents, prompts = [list(x) for x in documents], [list(x) for x in prompts]
    if (sum(map(len, documents)) != args.document_tokens or any(not x for x in documents)
            or any(len(x) > args.chunk_size for x in documents)):
        raise ValueError("Document input differs from declared length/chunk limit")
    if len(prompts) != count or any(len(x) != args.prompt_tokens for x in prompts):
        raise ValueError("Prompt input differs from declared request count/length")
    if len(set(map(tuple, prompts))) != count:
        raise ValueError("Requests must use different prompts; repetition is not a request stream")
    if any(type(i) is not int or i < 0 or i >= vocab for seq in documents + prompts for i in seq):
        raise ValueError("Invalid token ID")
    sink = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if sink is None or not 0 <= sink < vocab:
        raise ValueError("A valid common sink token is required")
    if args.document_tokens + 1 + args.prompt_tokens + args.generation_tokens > config.max_position_embeddings:
        raise ValueError("Document, sink, query and output exceed the unchanged model window")
    return {"chunks": documents, "prompts": prompts, "sink_id": int(sink),
            "probe_indices": list(range(max(0, args.prompt_tokens - 16), args.prompt_tokens)),
            "token_sha256": json_digest({"chunks": documents, "prompts": prompts, "sink_id": sink}),
            "source": "deterministic synthetic IDs and distinct equal-length prompt markers; no accuracy claim"}


def validate_reuse_mode(args):
    if not getattr(args, "reuse_requests", 0):
        return None
    if args.reuse_requests not in (2, 10):
        raise ValueError("The initial fixed-pack pilot supports 2 or 10 distinct queries")
    if args.reader_implementation != "reference" or args.adapter_kind != "trained":
        raise ValueError("Reuse pilot requires reference readers and the matching trained adapter")
    if args.repetitions != 1:
        raise ValueError("Use one real query stream; repetitions cannot represent cache reuse")
    if args.arm not in ("D0", "NATIVE", "A", "B", "D1", "FULL"):
        raise ValueError("Unknown reuse arm")
    expected = "block_hot" if args.arm in ("A", "B") else "cold_hj"
    if args.cache_mode != expected:
        raise ValueError(f"Reuse arm {args.arm} requires cache_mode={expected}")
    if any(getattr(args, name, False) for name in ("profile_reader_only", "backend_model_parity",
            "shared_state_diagnostic_only", "same_math_diagnostic_only", "validate_optimized_decode", "validate_backend")):
        raise ValueError("Reuse timings cannot be combined with another diagnostic mode")
    path = getattr(args, "reuse_quality_receipt", None)
    payload = read_json(path) if path else None
    if not isinstance(payload, dict):
        raise ValueError("waiting_quality: a matching trained-reader quality receipt is required")
    quality_arm = "D0" if args.arm == "NATIVE" else args.arm
    entry = payload.get("arms", {}).get(quality_arm) if "arms" in payload else payload
    if (not isinstance(entry, dict) or entry.get("protocol") != QUALITY_VERSION
            or entry.get("status") != "complete" or entry.get("passed") is not True
            or entry.get("arm") != quality_arm):
        raise ValueError("waiting_quality: trained reuse quality did not pass for this arm")
    sources = source_identity()
    if (entry.get("source_sha256") != sources or entry.get("recipe") != recipe(args)
            or entry.get("adapter_sha256") != digest(args.adapter)
            or entry.get("model_config_sha256") != digest(Path(args.model) / "config.json")):
        raise ValueError("waiting_quality: receipt sources/model/adapter/recipe differ")
    checks = entry.get("checks", {})
    required = ("greedy_equal", "selected_route_equal", "repeat_query_equal", "cache_unchanged", "finite_logits", "storage_roundtrip")
    if any(checks.get(key) is not True for key in required):
        raise ValueError("waiting_quality: incomplete trained-reuse semantic checks")
    if entry.get("unique_queries", 0) < 2 or entry.get("document_blocks", 0) < 2:
        raise ValueError("waiting_quality: receipt lacks multiple blocks and distinct questions")
    if entry.get("real_input") is not True or entry.get("unique_source_documents", 0) < 2:
        raise ValueError("waiting_quality: verify two real source documents, each with two questions")
    return {"protocol": VERSION, "requests": args.reuse_requests, "workflow": "fixed_ordered_pack",
            "source_sha256": sources, "quality_receipt_sha256": digest(path), "quality_arm": quality_arm,
            "quality_scope": "bounded trained-model implementation probe; not a benchmark accuracy estimate"}


def summarize_reuse(records, setup_s, lifecycle_s):
    if not records or setup_s < 0 or lifecycle_s < setup_s:
        raise ValueError("Invalid stream accounting")
    total_decode_s = sum(r["decode_wall_s"] for r in records)
    decode_steps = sum(r["decode_steps"] for r in records)
    return {"requests": len(records), "generated_tokens": sum(r["generated_tokens"] for r in records),
            "setup_including_storage_s": setup_s, "all_queries_with_store_build_s": lifecycle_s,
            "first_query_with_store_build_s": records[0]["cumulative_e2e_s"],
            "first_token_with_store_build_s": records[0]["cumulative_first_token_s"],
            "query_service_sum_s": sum(r["query_e2e_s"] for r in records),
            "mean_query_ttft_s": sum(r["ttft_s"] for r in records) / len(records),
            "decode_tps": decode_steps / total_decode_s if total_decode_s else None,
            "amortized_generated_tokens_per_s": sum(r["generated_tokens"] for r in records) / lifecycle_s,
            "ttft_scope": "per-query TTFT excludes one-time setup; cumulative TTFT includes setup",
            "total_scope": "continuous wall time includes cache construction, persistence, one load/transfer and all query/cleanup/reporting gaps"}
