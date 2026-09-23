"""Exact, context-only prefix KV reuse for the independent serving experiment.

The cache contains CPU copies of full-layer, post-RoPE KV for actual ordered
document token prefixes (including the one shared sink). Query/answer KV is
never committed. Radix edges hold at most 512 tokens and share common prefixes;
an edge can be split at an arbitrary token. A byte budget applies to unique
resident KV tensor storage, excluding the common raw-token document store and
Python indexing metadata. Prefix-safe leaf LRU evicts suffixes before parents.

This is a serial reference implementation, not vLLM/PagedAttention. Prefix
lookup, H2D, missing-context prefill, CPU cache fill/eviction and generation are
charged to query time. No GPU tensor is retained by this cache across queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "encbank_v2_benchmarks_20260908"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
# Importing the existing harness preserves its s15 repeat_kv SDPA policy.
from serving_reuse import ReusableReader, save_json, save_tensor_file, tensor_bytes


@dataclass(eq=False)
class _Node:
    tokens: tuple = ()
    kv: tuple = ()
    parent: Optional["_Node"] = None
    children: dict = field(default_factory=dict)
    touched: int = 0

    @property
    def nbytes(self):
        return tensor_bytes(self.kv)


class PrefixReader(ReusableReader):
    """Same-pack full recompute with exact CPU prefix reuse between requests."""

    def __init__(self, model, j, tokenizer, arm="prefix", model_id="",
                 cache_budget_bytes=None):
        if arm != "prefix":
            raise ValueError("PrefixReader requires arm='prefix'")
        if cache_budget_bytes is not None and (isinstance(cache_budget_bytes, bool)
                or int(cache_budget_bytes) != cache_budget_bytes or cache_budget_bytes < 0):
            raise ValueError("cache_budget_bytes must be a nonnegative integer or None")
        # This also rejects active LoRA and unsupported models/devices.
        super().__init__(model, j, tokenizer, arm="j0", model_id=model_id)
        self.arm = "prefix"
        self.requested_j = int(j)
        self.cache_budget_bytes = None if cache_budget_bytes is None else int(cache_budget_bytes)
        self.block_size = 512
        cfg = self.cm.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv_bytes_per_token = (self.cm.num_layers * 2 * cfg.num_key_value_heads
                                   * head_dim * next(model.parameters()).element_size())
        self._root = _Node()
        self._resident_bytes = 0
        self._tick = 0
        self._evictions = 0
        self._evicted_bytes = 0

    def signature(self):
        value = super().signature()
        value.update(prefix_protocol="exact-context-prefix-v1", prefix_block_tokens=self.block_size,
                     persistent_key_space="all-layer post-RoPE KV of exact ordered context tokens",
                     prefix_cache_budget_scope="unique CPU KV tensor storage bytes; raw IDs/index metadata excluded")
        return value

    def _nodes(self):
        todo = list(self._root.children.values())
        while todo:
            node = todo.pop()
            yield node
            todo.extend(node.children.values())

    def cache_info(self):
        """Direct unique backing-storage count; never returns GPU cache objects."""
        seen = {}
        nodes = list(self._nodes())
        for node in nodes:
            for pair in node.kv:
                for value in pair:
                    if value.device.type != "cpu" or value.requires_grad:
                        raise RuntimeError("Persistent prefix KV must be detached CPU tensors")
                    storage = value.untyped_storage()
                    seen[(str(value.device), storage.data_ptr())] = storage.nbytes()
        actual = sum(seen.values())
        if actual != self._resident_bytes:
            raise RuntimeError("Prefix cache accounting differs from unique tensor storage")
        return {"resident_bytes": actual, "nodes": len(nodes),
                "resident_context_tokens": sum(len(n.tokens) for n in nodes),
                "budget_bytes": self.cache_budget_bytes,
                "evicted_nodes_total": self._evictions, "evicted_bytes_total": self._evicted_bytes,
                "budget_scope": "unique CPU KV tensor storage; raw-token store and Python metadata excluded",
                "policy": "radix prefix sharing; prefix-safe leaf LRU; no persistent GPU KV"}

    def reset_cache(self):
        self._root = _Node()
        self._resident_bytes = 0
        self._tick = 0
        self._evictions = self._evicted_bytes = 0

    @torch.no_grad()
    def write_store(self, context_ids, path, chunk_size=512):
        """Persist raw document tokens only; exact KV is filled on actual requests."""
        path = Path(path)
        if chunk_size <= 0 or (path.exists() and any(path.iterdir())):
            raise ValueError("Write requires a positive chunk size and a new empty store")
        path.mkdir(parents=True, exist_ok=True)
        cm = self.cm
        baseline = 0
        if cm.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cm.device)
            baseline = torch.cuda.memory_allocated(cm.device)
        started = self.clock()
        ids = torch.as_tensor(context_ids, dtype=torch.long, device="cpu").reshape(-1).clone()
        if not ids.numel():
            raise ValueError("Document must be nonempty")
        chunks = [x.clone() for x in ids.split(chunk_size)]
        serial = self.clock()
        save_tensor_file(chunks, path / "tokens.pt")
        metadata = {"schema_version": 1, "signature": self.signature(), "n_tokens": ids.numel(),
                    "n_chunks": len(chunks), "chunk_size": chunk_size,
                    "payload_tensor_bytes": 0, "raw_token_bytes": tensor_bytes(chunks),
                    "validity": "Exact checkpoint/tokenizer and ordered prefix tokens",
                    "prefix_fill": "on request; CPU KV fill and eviction charged to query"}
        save_json(metadata, path / "store.json")
        serial_end = self.clock()
        peak = torch.cuda.max_memory_allocated(cm.device) if cm.device.type == "cuda" else 0
        return {"write_total_s": serial_end-started, "write_compute_s": 0.0,
                "write_device_to_cpu_s": 0.0, "write_serialize_s": serial_end-serial,
                "serialized_bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                "payload_tensor_bytes": 0, "raw_token_bytes": tensor_bytes(chunks),
                "n_document_tokens": ids.numel(), "n_document_chunks": len(chunks), "capture_calls": 0,
                "write_peak_allocated_bytes": peak, "write_incremental_peak_bytes": max(0, peak-baseline)}

    def open_store(self, path, tier="cpu"):
        """Load raw IDs; keep valid prefix KV when switching documents.

        Call close_store/reset_cache between independent cold-cache traces.
        Disk-resident KV is deliberately unsupported by this CPU-cache baseline.
        """
        if tier != "cpu":
            raise ValueError("The exact prefix baseline caches KV on CPU; tier must be 'cpu'")
        started = self.clock()
        path = Path(path)
        metadata = json.loads((path / "store.json").read_text(encoding="utf-8"))
        if metadata["signature"] != json.loads(json.dumps(self.signature())):
            raise ValueError("Prefix store checkpoint/configuration does not match reader")
        chunks = torch.load(path / "tokens.pt", map_location="cpu", weights_only=True)
        if sum(x.numel() for x in chunks) != metadata["n_tokens"]:
            raise ValueError("Invalid raw-token store")
        self.path, self.tier, self.metadata, self.chunks = path, tier, metadata, chunks
        self.payloads = None
        return {"startup_load_s": self.clock()-started,
                "startup_read_bytes": (path/"tokens.pt").stat().st_size+(path/"store.json").stat().st_size,
                "resident_cpu_tensor_bytes": tensor_bytes(chunks)+self._resident_bytes,
                "tier": tier, "disk_cache_policy": "raw tokens loaded once; KV is CPU-only"}

    def close_store(self):
        self.path = None
        self.payloads = None
        self.chunks = []
        self.reset_cache()

    def _match(self, tokens):
        node, offset, matched = self._root, 0, []
        self._tick += 1
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                break
            count = 0
            for a, b in zip(child.tokens, tokens[offset:]):
                if a != b:
                    break
                count += 1
            if not count:
                break
            child.touched = self._tick
            matched.append((child, count))
            offset += count
            if count < len(child.tokens):
                break
            node = child
        return offset, matched

    @staticmethod
    def _slice_cpu(kv, start, end):
        return tuple(tuple(t[..., start:end, :].detach().contiguous().clone() for t in pair)
                     for pair in kv)

    def _split(self, node, count):
        """Split an existing edge without keeping oversized backing-storage views."""
        old_bytes = node.nbytes
        tail = _Node(node.tokens[count:], self._slice_cpu(node.kv, count, len(node.tokens)),
                     node, node.children, node.touched)
        for child in tail.children.values():
            child.parent = tail
        node.tokens, node.kv = node.tokens[:count], self._slice_cpu(node.kv, 0, count)
        node.children = {tail.tokens[0]: tail}
        assert node.nbytes+tail.nbytes == old_bytes

    def _reserve(self, need, protected):
        if self.cache_budget_bytes is None:
            return True
        while self._resident_bytes+need > self.cache_budget_bytes:
            leaves = [n for n in self._nodes() if not n.children and n not in protected]
            if not leaves:
                return False
            victim = min(leaves, key=lambda n: n.touched)
            del victim.parent.children[victim.tokens[0]]
            self._resident_bytes -= victim.nbytes
            self._evictions += 1
            self._evicted_bytes += victim.nbytes
        return True

    def _commit(self, tokens, new_start, new_cpu_kv):
        node, offset, protected = self._root, 0, {self._root}
        added = 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is not None:
                count = 0
                for a, b in zip(child.tokens, tokens[offset:]):
                    if a != b:
                        break
                    count += 1
                # A complete shorter prefix already exists as a view of this edge.
                if offset+count == len(tokens):
                    break
                if count < len(child.tokens):
                    self._split(child, count)
                child.touched = self._tick
                offset += count
                node = child
                protected.add(node)
                continue
            count = min(self.block_size, len(tokens)-offset)
            need = count*self.kv_bytes_per_token
            if not self._reserve(need, protected):
                break
            if offset < new_start:
                raise RuntimeError("Attempted to commit context without its computed KV")
            payload = self._slice_cpu(new_cpu_kv, offset-new_start, offset-new_start+count)
            child = _Node(tuple(tokens[offset:offset+count]), payload, node, touched=self._tick)
            assert child.nbytes == need
            node.children[child.tokens[0]] = child
            self._resident_bytes += need
            added += need
            node = child
            protected.add(node)
            offset += count
        return added

    @torch.no_grad()
    def query_ids(self, query_ids, *, selected_indices=None, selector="bm25", topk=12,
                  bare_question_ids=None, max_new_tokens=16, force_length=False,
                  capture_logits=False):
        if self.path is None or not self.chunks:
            raise ValueError("Open a complete raw-token document store before querying")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        cm = self.cm
        baseline = 0
        if cm.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cm.device)
            baseline = torch.cuda.memory_allocated(cm.device)
        started = self.clock()
        query = torch.as_tensor(query_ids, dtype=torch.long, device="cpu").reshape(-1)
        if not query.numel():
            raise ValueError("Query must be nonempty")
        if selected_indices is None:
            from encbank import selectors
            selected_indices = selectors.select_context_chunk_indices(
                selector, self.chunks, list(bare_question_ids) if bare_question_ids is not None else query.tolist(), topk)
        indices = [int(i) for i in selected_indices]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= len(self.chunks) for i in indices):
            raise ValueError("Invalid or duplicate selected indices")
        context = (int(cm._sink_prefix_id()),)+tuple(t for i in indices for t in self.chunks[i].tolist())
        retrieval_end = self.clock()
        hit, matched = self._match(context)
        lookup_end = self.clock()
        previous_evictions, previous_evicted_bytes = self._evictions, self._evicted_bytes
        # Materialize only the legal matched prefix, clone/copy away from the
        # persistent CPU tensors, and build a fresh per-query GPU/CPU cache.
        cache = DynamicCache(config=cm.config)
        for layer in range(cm.num_layers):
            if not hit:
                break
            pair = []
            for which in (0, 1):
                pieces = [node.kv[layer][which][..., :count, :] for node, count in matched]
                value = pieces[0] if len(pieces)==1 else torch.cat(pieces, dim=-2)
                pair.append(value.to(device=cm.device, copy=True))
            cache.update(pair[0], pair[1], layer)
        suffix = torch.tensor([context[hit:]+tuple(query.tolist())], dtype=torch.long, device=cm.device)
        transfer_end = self.clock()
        hidden = cm.embed_tokens(suffix)
        positions = torch.arange(hit, len(context)+query.numel(), device=cm.device).unsqueeze(0)
        mask = create_causal_mask(config=cm.config, inputs_embeds=hidden, attention_mask=None,
                                  past_key_values=cache, position_ids=positions)
        rope = cm.rotary_emb(hidden, position_ids=positions)
        hidden = cm._run_layers(hidden, slice(0, cm.num_layers), mask, positions, rope,
                                past_key_values=cache, use_cache=True)
        logits = cm.lm_head(cm.norm(hidden[:, -1:, :]))
        prefill_end = self.clock()
        # Only document KV up to context length is eligible. Do not save even
        # one query token, including on EOS/exception/zero-capacity paths.
        limit = len(context) if self.cache_budget_bytes is None else min(
            len(context), self.cache_budget_bytes//self.kv_bytes_per_token)
        fill_transfer_bytes = added_bytes = 0
        if limit > hit:
            cpu_kv = tuple(tuple(value[..., hit:limit, :].detach().to(device="cpu", copy=True).contiguous()
                                 for value in (cache.layers[layer].keys, cache.layers[layer].values))
                           for layer in range(cm.num_layers))
            fill_transfer_bytes = tensor_bytes(cpu_kv)
            added_bytes = self._commit(context[:limit], hit, cpu_kv)
            del cpu_kv
        info = self.cache_info()
        if self.cache_budget_bytes is not None and info["resident_bytes"] > self.cache_budget_bytes:
            raise RuntimeError("Persistent prefix KV exceeded its byte budget")
        fill_end = self.clock()
        _, eos = cm._bos_eos(cm.tokenizer)
        first_logits = logits[0, -1].float().clone()
        if eos is not None:
            first_logits[eos] = -float("inf")
        step_logits = [first_logits.cpu().clone()] if capture_logits else []
        token = int(first_logits.argmax().item())
        generated = [token]
        first_end = self.clock()
        pack_pos = len(context)+query.numel()
        decode_steps = 0
        for _ in range(1, max_new_tokens):
            logits = cm.decode_step(token, None, cache, 0, pack_pos)
            pack_pos += 1
            decode_steps += 1
            next_logits = logits[0, -1].float().clone()
            if force_length and eos is not None:
                next_logits[eos] = -float("inf")
            if capture_logits:
                step_logits.append(next_logits.cpu().clone())
            token = int(next_logits.argmax().item())
            if not force_length and eos is not None and token == eos:
                break
            generated.append(token)
        ended = self.clock()
        peak = torch.cuda.max_memory_allocated(cm.device) if cm.device.type == "cuda" else 0
        stats = {"selected_indices": indices, "read_tokens": len(context)+query.numel(),
                 "query_tokens": query.numel(), "context_tokens": len(context),
                 "retrieval_s": retrieval_end-started, "prefix_lookup_s": lookup_end-retrieval_end,
                 "load_s": lookup_end-retrieval_end, "load_bytes": 0,
                 "transfer_s": transfer_end-lookup_end,
                 "transfer_bytes": hit*self.kv_bytes_per_token+suffix.numel()*suffix.element_size(),
                 "rotate_prepare_s": 0.0, "read_prefill_s": prefill_end-transfer_end,
                 "cache_fill_s": fill_end-prefill_end, "cache_fill_transfer_bytes": fill_transfer_bytes,
                 "cache_fill_added_bytes": added_bytes,
                 "ttft_s": first_end-started, "decode_s": ended-first_end, "total_s": ended-started,
                 "generated_tokens": len(generated), "decode_steps": decode_steps,
                 "fixed_generation_length": force_length, "peak_allocated_bytes": peak,
                 "incremental_peak_bytes": max(0, peak-baseline), "capture_calls": 0,
                 "cache_hit_context_tokens": hit, "cache_miss_context_tokens": len(context)-hit,
                 "cache_hit_fraction": hit/len(context), "cache_resident_bytes": info["resident_bytes"],
                 "cache_budget_bytes": self.cache_budget_bytes, "cache_nodes": info["nodes"],
                 "evicted_nodes": self._evictions-previous_evictions,
                 "evicted_bytes": self._evicted_bytes-previous_evicted_bytes,
                 "cache_policy": info["policy"], "cache_budget_scope": info["budget_scope"],
                 "prefill_context_tokens_computed": len(context)-hit,
                 "prefill_query_tokens_computed": query.numel(),
                 "cache_fill_charged_before_first_token": True,
                 "resident_raw_token_bytes": tensor_bytes(self.chunks)}
        if capture_logits:
            stats["step_logits"] = step_logits
        return generated, stats


# Explicit aliases let callers use the existing harness naming convention.
ReusablePrefixReader = PrefixReader
ExactPrefixReader = PrefixReader
