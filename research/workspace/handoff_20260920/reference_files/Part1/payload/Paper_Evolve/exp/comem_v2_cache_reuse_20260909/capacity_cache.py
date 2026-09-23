"""CPU tensor-byte LRU and fixed input-only traces for phase B.

No model or GPU work is launched here. Only floating cache tensors count toward
the persistent budget. Raw token inputs, metadata, active-query bypass payloads,
and temporary capture/copy buffers are separate accounting categories.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import random
import time
from typing import Any, Callable


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class CacheRequest:
    namespace: str
    document_id: str
    chunk_index: int
    token_ids: tuple[int, ...]
    kind: str = "chunk"
    _digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        require(bool(self.namespace), "Cache namespace must identify the exact model and payload semantics")
        require(self.kind in {"chunk", "sink"}, "Unknown cache payload kind")
        tokens = tuple(int(x) for x in self.token_ids)
        require(bool(tokens) and all(x >= 0 for x in tokens), "Nonempty nonnegative token IDs required")
        if self.kind == "sink":
            require(self.document_id == "__shared_sink__" and self.chunk_index == -1 and len(tokens) == 1,
                    "Sink must be one explicitly shared standalone token")
        else:
            require(bool(self.document_id) and self.chunk_index >= 0, "Invalid chunk identity")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "_digest", hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest())

    @property
    def key(self):
        return (self.namespace, self.kind, self.document_id, self.chunk_index, self._digest)

    @property
    def token_count(self):
        return len(self.token_ids)

    @property
    def label(self):
        return {"namespace": self.namespace, "kind": self.kind, "document_id": self.document_id,
                "chunk_index": self.chunk_index, "tokens": self.token_count, "token_digest": self._digest}


@dataclass
class BuildResult:
    payload: Any
    phases: dict[str, float] = field(default_factory=dict)


def make_requests(document_id, context_ids, selected_indices, *, chunk_size=512, sink_id, namespace):
    """Touch only selected raw chunks; do not hash/copy the whole document."""
    require(chunk_size > 0, "Positive chunk size required")
    indices = [int(i) for i in selected_indices]
    n = len(context_ids)
    require(n > 0 and len(indices) == len(set(indices)), "Nonempty document and distinct selected indices required")
    require(all(0 <= i < (n + chunk_size - 1) // chunk_size for i in indices), "Selected index outside document")
    result = [CacheRequest(namespace, "__shared_sink__", -1, (int(sink_id),), "sink")]
    for i in indices:
        piece = context_ids[i * chunk_size:min((i + 1) * chunk_size, n)]
        tokens = piece.tolist() if hasattr(piece, "tolist") else piece
        result.append(CacheRequest(namespace, str(document_id), i, tuple(tokens)))
    return result


def _map_tensor_tree(value, fn):
    import torch
    if torch.is_tensor(value):
        require(value.layout == torch.strided and value.is_floating_point(), "Cache payload must contain dense floating tensors, never raw token IDs")
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensor_tree(v, fn) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tensor_tree(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_tensor_tree(v, fn) for v in value]
    raise ValueError("Cache payload tree must contain tensors only; metadata belongs outside its budget")


def compact_cpu_payload(payload):
    require(isinstance(payload, dict) and set(payload) in ({"kv"}, {"h", "kv"}),
            "Expected V2 h+kv or CacheBlend kv payload; raw tokens/metadata are not cache tensors")
    # Clone even a contiguous view: otherwise a tiny slice can retain its parent's
    # much larger storage, or a builder's later mutation can change the cache.
    result = _map_tensor_tree(payload, lambda t: t.detach().to("cpu").contiguous().clone())
    require(payload_inventory(result)["logical_tensor_bytes"] > 0, "Empty cache tensor payload")
    return result


def payload_inventory(payload):
    storages, logical, count = {}, 0, 0
    def visit(t):
        nonlocal logical, count
        require(t.device.type == "cpu", "Persistent cache tensor must reside on CPU")
        storage = t.untyped_storage()
        storages[(str(t.device), storage.data_ptr())] = storage.nbytes()
        logical += t.numel() * t.element_size()
        count += 1
        return t
    _map_tensor_tree(payload, visit)
    return {"logical_tensor_bytes": logical, "unique_storage_bytes": sum(storages.values()),
            "tensor_count": count, "unique_storages": len(storages)}


@dataclass
class _Entry:
    request: CacheRequest
    payload: Any
    nbytes: int
    pinned: bool = False


class QueryLease:
    def __init__(self, cache, payloads, stats):
        self._cache = cache
        self.payloads = payloads
        self.stats = stats
        self.released = False

    def release(self):
        if self.released:
            return
        self._cache._release(self)
        self.payloads.clear()
        self.released = True
        self.stats["released"] = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class ByteLRU:
    """Serial, per-method CPU LRU shared by all documents and queries.

    All requested cache hits are pinned at acquire entry. Misses evict only
    unpinned LRU entries. A miss that cannot fit is available transiently for this
    query, without admission or truncation. release() must run in finally.
    """
    def __init__(self, budget_bytes, *, clock: Callable[[], float] = time.perf_counter):
        require(type(budget_bytes) is int and budget_bytes >= 0, "Budget must be a nonnegative integer byte count")
        self.budget_bytes = budget_bytes
        self.clock = clock
        self._entries = OrderedDict()
        self.resident_bytes = 0
        self._active = None
        self.totals = Counter()

    def snapshot(self):
        return {"budget_bytes": self.budget_bytes, "resident_bytes": self.resident_bytes,
                "persistent_bytes": self.resident_bytes,
                "resident_entries": len(self._entries), "active_query": self._active is not None,
                "lru_to_mru": [{**e.request.label, "tensor_bytes": e.nbytes, "pinned": e.pinned}
                               for e in self._entries.values()], "totals": dict(self.totals)}

    info = snapshot

    def clear(self, *, reset_stats=False):
        require(self._active is None, "Cannot clear a cache with an active query lease")
        removed = {"entries": len(self._entries), "tensor_bytes": self.resident_bytes}
        self._entries.clear()
        self.resident_bytes = 0
        if reset_stats:
            self.totals.clear()
        return removed

    def acquire(self, requests, builder):
        require(self._active is None, "Only one serial query lease may be active")
        requests = list(requests)
        require(all(isinstance(r, CacheRequest) for r in requests), "Expected CacheRequest objects")
        unique = OrderedDict((r.key, r) for r in requests)
        multiplicity = Counter(r.key for r in requests)
        started = self.clock()
        stats = {"requests": len(requests), "unique_requests": len(unique), "request_tokens": sum(r.token_count for r in requests),
                 "context_tokens_requested": sum(r.token_count for r in requests if r.kind == "chunk"),
                 "context_tokens_hit": 0, "context_tokens_miss": 0,
                 "hit_entries": 0, "miss_entries": 0, "hit_tensor_bytes": 0, "miss_tensor_bytes": 0,
                 "build_calls": 0, "chunk_build_calls": 0, "sink_build_calls": 0,
                 "build_s": 0.0, "cpu_copy_s": 0.0, "builder_phases_s": {},
                 "bypass_entries": 0, "bypass_context_tokens": 0, "bypass_tensor_bytes": 0,
                 "transient_tensor_bytes": 0, "active_payload_tensor_bytes": 0,
                 "persistent_bytes_before": self.resident_bytes, "evictions": [], "events": [], "released": False}
        ready = {}
        self._active = True
        try:
            # Protect a later requested hit against an earlier miss's eviction.
            for key, request in unique.items():
                if key in self._entries:
                    entry = self._entries[key]
                    require(entry.request.token_ids == request.token_ids, "Token digest collision")
                    entry.pinned = True
            for key, request in unique.items():
                event = {**request.label, "occurrences": multiplicity[key]}
                if key in self._entries:
                    entry = self._entries[key]
                    self._entries.move_to_end(key)
                    ready[key] = entry.payload
                    nbytes = entry.nbytes
                    stats["hit_entries"] += 1
                    stats["hit_tensor_bytes"] += nbytes
                    if request.kind == "chunk":
                        stats["context_tokens_hit"] += request.token_count * multiplicity[key]
                    event.update(status="hit", tensor_bytes=nbytes)
                else:
                    stats["miss_entries"] += 1
                    stats["build_calls"] += 1
                    stats[request.kind + "_build_calls"] += 1
                    if request.kind == "chunk":
                        stats["context_tokens_miss"] += request.token_count * multiplicity[key]
                    t0 = self.clock()
                    built = builder(request)
                    t1 = self.clock()
                    stats["build_s"] += t1 - t0
                    if isinstance(built, BuildResult):
                        for name, value in built.phases.items():
                            require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, "Invalid builder phase duration")
                            stats["builder_phases_s"][name] = stats["builder_phases_s"].get(name, 0.0) + value
                        raw = built.payload
                    else:
                        raw = built
                    payload = compact_cpu_payload(raw)
                    del raw, built
                    t2 = self.clock()
                    stats["cpu_copy_s"] += t2 - t1
                    inventory = payload_inventory(payload)
                    nbytes = inventory["unique_storage_bytes"]
                    require(nbytes == inventory["logical_tensor_bytes"], "Compact owned payload storage differs from tensor bytes")
                    stats["miss_tensor_bytes"] += nbytes
                    ready[key] = payload
                    # An individually oversized item cannot ever enter; do not
                    # pointlessly evict useful entries while bypassing it.
                    if nbytes <= self.budget_bytes:
                        while self.resident_bytes + nbytes > self.budget_bytes:
                            victim = next((k for k, e in self._entries.items() if not e.pinned), None)
                            if victim is None:
                                break
                            old = self._entries.pop(victim)
                            self.resident_bytes -= old.nbytes
                            stats["evictions"].append({**old.request.label, "tensor_bytes": old.nbytes})
                    if self.resident_bytes + nbytes <= self.budget_bytes:
                        self._entries[key] = _Entry(request, payload, nbytes, pinned=True)
                        self.resident_bytes += nbytes
                        event.update(status="miss_admitted", tensor_bytes=nbytes)
                    else:
                        stats["bypass_entries"] += 1
                        stats["bypass_tensor_bytes"] += nbytes
                        stats["transient_tensor_bytes"] += nbytes
                        if request.kind == "chunk":
                            stats["bypass_context_tokens"] += request.token_count
                        event.update(status="miss_bypass", tensor_bytes=nbytes,
                                     reason="entry_exceeds_budget" if nbytes > self.budget_bytes else "active_query_pins_fill_budget")
                stats["active_payload_tensor_bytes"] += nbytes
                stats["events"].append(event)
                require(self.resident_bytes <= self.budget_bytes, "Persistent capacity exceeded")
            stats.update(evicted_entries=len(stats["evictions"]),
                         evicted_tensor_bytes=sum(e["tensor_bytes"] for e in stats["evictions"]),
                         persistent_bytes_after=self.resident_bytes,
                         persistent_plus_transient_bytes_after_acquire=self.resident_bytes + stats["transient_tensor_bytes"],
                         acquire_s=self.clock() - started)
            stats["lookup_admission_s"] = max(0.0, stats["acquire_s"] - stats["build_s"] - stats["cpu_copy_s"])
            stats["capture_s"] = stats["builder_phases_s"].get("capture_s", 0.0)
            lease = QueryLease(self, [ready[r.key] for r in requests], stats)
            self._active = lease
            for key in ("requests", "unique_requests", "request_tokens", "context_tokens_requested", "context_tokens_hit",
                        "context_tokens_miss", "hit_entries", "miss_entries", "build_calls", "chunk_build_calls",
                        "sink_build_calls", "bypass_entries", "bypass_tensor_bytes", "evicted_entries", "evicted_tensor_bytes"):
                self.totals[key] += stats[key]
            self.totals["queries"] += 1
            return lease
        except BaseException:
            for entry in self._entries.values():
                entry.pinned = False
            self._active = None
            self.totals["failed_acquires"] += 1
            raise

    def _release(self, lease):
        require(self._active is lease, "Lease does not own the active query")
        for entry in self._entries.values():
            entry.pinned = False
        self._active = None


def interleaved_trace(documents, queries, *, seed=20260909, per_document_limit=80, minimum_queries=80):
    """Seeded document round-robin preserving each document's fixed query order."""
    dm = {d["document_id"]: d for d in documents}
    qm = {q["id"]: q for q in queries}
    require(len(dm) == len(documents) and len(qm) == len(queries), "Duplicate fixture IDs")
    doc_order = list(dm)
    random.Random(seed).shuffle(doc_order)
    orders = {}
    for did in doc_order:
        order = list(dm[did]["query_ids_ordered"])
        require(len(order) >= minimum_queries, "Document has fewer real queries than the required scope")
        if per_document_limit is not None:
            require(per_document_limit >= minimum_queries and len(order) >= per_document_limit, "Requested prefix unavailable")
            order = order[:per_document_limit]
        require(len(order) == len(set(order)), "Repeated query within document")
        for i, qid in enumerate(order):
            require(qid in qm and qm[qid]["document_id"] == did and qm[qid]["stream_index"] == i, "Query order/group mismatch")
        orders[did] = order
    requests = []
    for i in range(max(map(len, orders.values()), default=0)):
        for did in doc_order:
            if i >= len(orders[did]):
                continue
            qid = orders[did][i]
            requests.append({"request_index": len(requests), "document_id": did, "query_id": qid,
                             "document_query_index": i, "selected_indices": list(qm[qid]["selected_indices"])})
    return {"protocol": "fixed-multidocument-round-robin-v1", "seed": seed, "document_order": doc_order,
            "per_document_limit": per_document_limit, "minimum_queries": minimum_queries,
            "per_document_queries": {d: len(v) for d, v in orders.items()}, "requests": requests,
            "method_results_used": False, "measurements": False}


def fixed_kv_budgets(documents, trace, model_config, *, chunk_size=512, element_size=2,
                     fractions=(Fraction(1, 4), Fraction(1, 2), Fraction(1, 1))):
    """Input-only full-depth KV reference: accessed chunk union + one shared sink.

    These are nominal persistent payload budgets, not allocations or proof that
    the host has enough RAM. Actual cached payload bytes are checked by ByteLRU.
    """
    cfg = model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config)
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))
    require(min(layers, heads, head_dim, chunk_size, element_size) > 0, "Positive model dimensions required")
    per_token = 2 * layers * heads * head_dim * element_size
    dm = {d["document_id"]: d for d in documents}
    unique, participating = set(), set()
    for request in trace["requests"]:
        did = request["document_id"]
        require(did in dm, "Trace document missing")
        participating.add(did)
        n = len(dm[did]["context_ids"])
        indices = request["selected_indices"]
        require(len(indices) == len(set(indices)), "Duplicate selected chunk")
        for index in indices:
            require(type(index) is int and 0 <= index < (n + chunk_size - 1) // chunk_size, "Trace chunk out of bounds")
            unique.add((did, index))
    chunk_tokens = sum(min(chunk_size, len(dm[did]["context_ids"]) - index * chunk_size) for did, index in unique)
    sink_tokens = 1 if trace["requests"] else 0
    reference = (chunk_tokens + sink_tokens) * per_token
    full_tokens = sum(len(dm[did]["context_ids"]) for did in participating)
    budgets = []
    for fraction in fractions:
        f = Fraction(fraction)
        require(0 < f <= 1, "Budget fractions must be in (0,1]")
        budgets.append({"numerator": f.numerator, "denominator": f.denominator,
                        "fraction": float(f), "bytes": reference * f.numerator // f.denominator,
                        "GiB": (reference * f.numerator // f.denominator) / 2**30})
    return {"reference_scope": "union of chunks selected by the fixed trace, plus one globally shared standalone sink",
            "full_kv_bytes_per_token": per_token, "dtype_element_size": element_size,
            "layers": layers, "kv_heads": heads, "head_dim": head_dim,
            "participating_documents": len(participating), "referenced_unique_chunks": len(unique),
            "referenced_chunk_tokens": chunk_tokens, "shared_sink_tokens": sink_tokens,
            "reference_full_kv_bytes": reference,
            "all_participating_document_tokens": full_tokens,
            "all_document_full_kv_bytes_including_shared_sink": (full_tokens + sink_tokens) * per_token,
            "budgets": budgets, "same_absolute_bytes_for_all_methods": True,
            "j0_persistent_kv_bytes": 0, "raw_tokens_and_metadata_in_kv_budget": False,
            "active_query_transients_in_persistent_budget": False,
            "prefix_cache_scope_note": "Apply the same absolute byte limits to actual prefix-node KV. The isolated-chunk-union reference does not guarantee that every ordered prefix fits at 100%.",
            "measurements": False, "host_ram_feasibility_verified": False}
