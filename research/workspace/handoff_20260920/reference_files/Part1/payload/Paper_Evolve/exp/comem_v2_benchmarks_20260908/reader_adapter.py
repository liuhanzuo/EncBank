"""Process-local reader selection and durable, exact-input generation reuse."""
from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import time
from pathlib import Path

ARMS = ("pub", "pub_sink", "fix_all", "fix_none", "j0", "cbos", "cacheblend16")
LOWER_ARMS = {"fix_all", "fix_none", "cbos"}
LOWER_SELECTORS = {"bm25", "recency", "oracle", "iter_bm25", "iter_bm25_adaptive", "dense_bge"}


class GenerationFailure(Exception):
    """Bypass legacy drivers' OOM-to-prediction conversion; allow clean resume."""


def validate_options(args, arm):
    if getattr(args, "baseline", "none") != "none":
        raise ValueError("Use --arm with --baseline none; other baseline factories bypass the selected reader.")
    if getattr(args, "score_only", False):
        raise ValueError("Score-only is not a generation run; use the original driver's scorer on the recorded output_dir.")
    if args.top_prepay_b != 0 or args.reuse_kv_blockdiag:
        raise ValueError("These named arms require top_prepay_b=0 and block_diagonal=False; neither option is ignored.")
    if arm in LOWER_ARMS:
        if args.sink_tokens != "bos":
            raise ValueError("CoMemLower's lower/upper pack positions require --sink_tokens bos.")
        if args.selector not in LOWER_SELECTORS:
            raise ValueError(f"CoMemLower does not provide hidden states to selector={args.selector!r}.")
        if arm in {"fix_all", "fix_none"} and args.resume_j <= 0:
            raise ValueError("Lower-band arms require j>0; use --arm j0 for full replay.")
    if not args.model_path:
        raise ValueError("An explicit --model is required.")
    if args.chunk_size <= 0 or args.topk <= 0:
        raise ValueError("chunk_size and topk must be positive.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards.")


def _plain(value):
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class GenerationCache:
    """SQLite commits each successful sample; interrupted transactions roll back."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS generations (key TEXT PRIMARY KEY, prediction TEXT NOT NULL, n_tokens INTEGER NOT NULL, elapsed_s REAL NOT NULL)")
        self.db.commit()
        self.hits = self.misses = 0

    def generate(self, reader, input_ids, kwargs):
        signature = inspect.signature(reader.generate_from_ids)
        bound = signature.bind(input_ids, **kwargs)
        bound.apply_defaults()
        options = dict(bound.arguments)
        options.pop("input_ids")
        if options.get("stats") is not None:
            raise ValueError("Generation reuse is for quality evaluation, not latency/statistics measurement.")
        # These objects are fixed by the run contract; their addresses must not enter a key.
        options.pop("dense_retriever", None)
        options.pop("tokenizer", None)
        content = {"tokens": _plain(input_ids), "generation": _plain(options)}
        key = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cached = self.db.execute("SELECT prediction FROM generations WHERE key=?", (key,)).fetchone()
        if cached is not None:
            self.hits += 1
            return json.loads(cached[0])
        started = time.monotonic()
        try:
            prediction = reader.generate_from_ids(input_ids, **kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise GenerationFailure("Generation ran out of memory; completed samples are cached. " + str(exc)) from exc
            raise
        if prediction == "[OOM]":
            raise GenerationFailure("Reader returned an OOM marker; it is not a scored prediction.")
        self.db.execute("INSERT INTO generations VALUES (?, ?, ?, ?)",
                        (key, json.dumps(prediction), int(input_ids.numel()), time.monotonic() - started))
        self.db.commit()
        self.misses += 1
        return prediction

    def close(self):
        self.db.close()


class CachedReader:
    def __init__(self, reader, cache):
        self.reader, self.cache = reader, cache

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def generate_from_ids(self, input_ids, **kwargs):
        if type(self.reader).__name__ == "CoMemLower":
            if kwargs.get("sink_tokens", "bos") != "bos" or not kwargs.get("use_kv_cache", True):
                raise ValueError("CoMemLower requires a BOS pack sink and KV-cache decode.")
            if kwargs.get("selector", "bm25") not in LOWER_SELECTORS:
                raise ValueError("This selector is unsupported by CoMemLower.")
        return self.cache.generate(self.reader, input_ids, kwargs)


def make_reader_factory(arm, cache, on_construct=None, explicit_pack=False):
    # The existing research module applies the Windows repeat_kv SDPA workaround.
    # Import it for ALL arms so the in-process kernel policy is held constant.
    from comem import CoMem
    from s15_ruler_lower import CoMemLower
    if arm == "cacheblend16" and not explicit_pack:
        raise ValueError("cacheblend16 requires the official_qa_driver explicit-pack interface")

    def factory(model, resume_j, top_prepay_b=0, block_diagonal=False, tokenizer=None):
        if top_prepay_b != 0 or block_diagonal:
            raise ValueError("top_prepay_b/block_diagonal cannot be discarded when constructing a named arm.")
        if arm in LOWER_ARMS:
            if getattr(model.config, "model_type", None) != "qwen3":
                raise ValueError("The current CoMemLower capture implementation is validated for dense Qwen3 only.")
            effective_j = int(model.config.num_hidden_layers) if arm == "cbos" else int(resume_j)
            if effective_j <= 0:
                raise ValueError("Lower-band reader requires a positive split.")
            reader = CoMemLower(model, effective_j, tokenizer=tokenizer,
                                lower_layers=[] if arm == "fix_none" else None)
        else:
            effective_j = 0 if arm == "j0" else int(resume_j)
            reader = CoMem(model, resume_j=effective_j, top_prepay_b=top_prepay_b,
                           block_diagonal=block_diagonal, tokenizer=tokenizer)
            reader.write_sink = arm == "pub_sink"
        if on_construct:
            on_construct({"class": type(reader).__name__, "arm": arm,
                          "requested_j": int(resume_j), "effective_j": effective_j,
                          "write_sink": reader.write_sink,
                          "model_config": model.config.to_dict(),
                          "kernel_policy": "s15 repeat_kv: use_gqa_in_sdpa=False (all arms)"})
        if explicit_pack:
            cacheblend = None
            if arm == "cacheblend16":
                from cacheblend_contextual import ContextualCacheBlend
                cacheblend = ContextualCacheBlend(reader, .16)
            reader = ExplicitPackReader(reader, cacheblend=cacheblend)
        return CachedReader(reader, cache)

    return factory


class ExplicitPackReader:
    """Use a complete query and an externally fixed retrieval pack for every arm.

    The legacy last-chunk interface can split a question across the read/store
    boundary. This adapter keeps context/query tokenization explicit and never
    pads, drops, or silently truncates either side. Selected indices are part of
    the durable generation-cache key.
    """
    def __init__(self, reader, cacheblend=None):
        self.reader = reader
        self.cacheblend = cacheblend

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def generate_from_ids(self, input_ids, *, context_token_count, selected_indices,
                          chunk_size=512, max_new_tokens=128):
        import torch
        reader = self.reader
        tokens = input_ids[0]
        if not 0 <= context_token_count < tokens.numel():
            raise ValueError("The explicit pack must contain a nonempty complete query")
        chunks = list(tokens[:context_token_count].split(chunk_size)) if context_token_count else []
        if len(set(selected_indices)) != len(selected_indices) or any(
                i < 0 or i >= len(chunks) for i in selected_indices):
            raise ValueError("Invalid or duplicate retrieval indices")
        selected = [chunks[i] for i in selected_indices]
        query = tokens[context_token_count:].tolist()
        # Match write/read sinks. Qwen has no BOS, so use the existing write
        # helper's EOS fallback instead of accidentally using a chat role token.
        sink_id = reader._sink_prefix_id()
        _, eos_id = reader._bos_eos(reader.tokenizer)
        with torch.no_grad():
            try:
                if self.cacheblend is not None:
                    generated = self.cacheblend.generate_explicit(selected, query, sink_id,
                        eos_id, max_new_tokens)
                    return reader.tokenizer.decode(generated, skip_special_tokens=True).strip()
                if hasattr(reader, "build_bottom"):
                    sink_hj, selected_hj = reader.build_bottom(sink_id, selected)
                else:
                    saved = reader.write_sink
                    try:
                        reader.write_sink = False
                        sink_hj = reader.write_chunk([sink_id])
                    finally:
                        reader.write_sink = saved
                    selected_hj = reader.write_chunks(selected)
                generated = reader._decode_from_pack(sink_hj, selected_hj, query,
                    eos_id, max_new_tokens, True, n_context_chunks=len(chunks))
            finally:
                if hasattr(reader, "_bottom"):
                    reader._bottom = None
        return reader.tokenizer.decode(generated, skip_special_tokens=True).strip()
