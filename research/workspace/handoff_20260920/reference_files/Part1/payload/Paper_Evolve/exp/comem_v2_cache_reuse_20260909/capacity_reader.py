"""On-demand, byte-budgeted multi-document readers; no model/GPU entry point.

bind_document prepares only raw CPU tokens. V2/CacheBlend populate the global
CPU cache solely when an actual selected chunk is missing. The old reader's
load region includes lookup, capture, D2H and eviction. Query leases protect
active payloads, including budget-bypassed temporaries, and are always released.
The exact-prefix arm uses the independently tested prefix trie; j0 has no
persistent model cache. These are serial experiment adapters, not a server.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "comem_v2_benchmarks_20260908"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
from serving_reuse import ReusableReader, tensor_bytes
from cacheblend_serving_reuse import ReusableContextualCacheBlend
from prefix_cache import PrefixReader
from capacity_cache import ByteLRU, CacheRequest, BuildResult


class _RawDocumentBinding:
    def _init_binding(self, chunk_size):
        if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = int(chunk_size)
        self.document_id = None
        self.chunks = []

    @torch.no_grad()
    def bind_document(self, document_id, context_ids):
        """Switch raw document tokens without clearing valid model caches.

        The caller must charge this returned preparation cost. Tokenization is
        separate. No full-document model computation or offline KV occurs here.
        """
        if getattr(self, "_query_active", False) or getattr(self, "_active_lease", None) is not None:
            raise RuntimeError("Cannot change documents during an active query")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a nonempty string")
        started = self.clock()
        ids = torch.as_tensor(context_ids, dtype=torch.long, device="cpu").reshape(-1)
        if not ids.numel():
            raise ValueError("Document must be nonempty")
        self.chunks = [chunk.detach().contiguous().clone() for chunk in ids.split(self.chunk_size)]
        self.document_id = document_id
        self.path = Path("__bound_raw_document__")  # Existing read guard only; never opened.
        self.tier, self.payloads = "cpu", None
        self.metadata = {"n_tokens": ids.numel(), "n_chunks": len(self.chunks),
                         "chunk_size": self.chunk_size, "raw_token_bytes": tensor_bytes(self.chunks)}
        return {"bind_document_s": self.clock()-started,
                "raw_token_bytes": tensor_bytes(self.chunks), "n_document_tokens": ids.numel(),
                "n_document_chunks": len(self.chunks), "capture_calls": 0,
                "offline_kv_tensor_bytes": 0, "document_id": document_id}


class _CapacityMixin(_RawDocumentBinding):
    def _init_capacity(self, cache_budget_bytes, chunk_size):
        self._init_binding(chunk_size)
        if (cache_budget_bytes is None or isinstance(cache_budget_bytes, bool)
                or int(cache_budget_bytes) != cache_budget_bytes or cache_budget_bytes < 0):
            raise ValueError("An explicit nonnegative integer byte budget is required")
        self.cache_budget_bytes = int(cache_budget_bytes)
        self._capacity_cache = None if self.arm == "j0" else ByteLRU(
            self.cache_budget_bytes, clock=self.clock)
        signature = json.dumps(self.signature(), sort_keys=True, separators=(",", ":"), default=str)
        self.cache_namespace = "capacity-reader-v1:" + hashlib.sha256(signature.encode()).hexdigest()
        self._active_lease = None
        self._query_active = False
        self._last_capacity_stats = {}
        self._captures = self._sink_captures = self._chunk_captures = 0

    def write_store(self, *args, **kwargs):
        raise ValueError("Capacity readers use bind_document; offline KV writes are not supported")

    def open_store(self, *args, **kwargs):
        raise ValueError("Capacity readers use bind_document; disk stores are not supported")

    def _requests(self, indices):
        requests = [CacheRequest(namespace=self.cache_namespace, document_id="__shared_sink__",
                    chunk_index=-1, token_ids=(int(self.cm._sink_prefix_id()),), kind="sink")]
        requests.extend(CacheRequest(namespace=self.cache_namespace, document_id=self.document_id,
                         chunk_index=i, token_ids=tuple(self.chunks[i].tolist()), kind="chunk")
                        for i in indices)
        return requests

    @torch.no_grad()
    def _build_request(self, request):
        started = self.clock()
        if self.arm == "cacheblend16":
            kv, _ = self.cb.prefill_chunk_full(request.token_ids)
            payload = {"kv": kv}
        else:
            payload = self._capture_payload(request.token_ids, sink=request.kind == "sink")
        self._captures += 1
        self._sink_captures += int(request.kind == "sink")
        self._chunk_captures += int(request.kind == "chunk")
        # ByteLRU owns compact detached CPU copies and actual unique-storage
        # billing. It also measures D2H; do not clone here or charge twice.
        return BuildResult(payload, phases={"capture_s": self.clock()-started})

    def _load_selected(self, indices):
        if self.arm == "j0":
            return super()._load_selected(indices)
        if self._active_lease is not None:
            raise RuntimeError("A query may acquire its selected payloads only once")
        lease = self._capacity_cache.acquire(self._requests(indices), self._build_request)
        self._active_lease = lease
        self._last_capacity_stats = lease.stats
        return lease.payloads, 0

    def cache_info(self):
        if self._capacity_cache is None:
            return {"persistent_bytes": 0, "budget_bytes": self.cache_budget_bytes,
                    "entries": 0, "policy": "no persistent model cache"}
        return self._capacity_cache.info()

    def reset_cache(self):
        if self._query_active or self._active_lease is not None:
            raise RuntimeError("Cannot reset a cache during an active query")
        if self._capacity_cache is not None:
            self._capacity_cache.clear()

    def close_store(self):
        self.reset_cache()
        self.path, self.document_id, self.payloads = None, None, None
        self.chunks = []
        if hasattr(self.cm, "_bottom"):
            self.cm._bottom = None

    @torch.no_grad()
    def query_ids(self, query_ids, **kwargs):
        if self._query_active:
            raise RuntimeError("Capacity readers are serial; concurrent queries are unsupported")
        started = self.clock()
        self._query_active = True
        self._last_capacity_stats = {}
        self._captures = self._sink_captures = self._chunk_captures = 0
        try:
            generated, stats = super().query_ids(query_ids, **kwargs)
        finally:
            release_started = self.clock()
            try:
                if self._active_lease is not None:
                    self._active_lease.release()
            finally:
                self._active_lease = None
                self._query_active = False
                release_ended = self.clock()
        # The cache's lease stats preserve request-time hits/misses. Snapshot
        # after release is the live persistent inventory, excluding bypasses.
        cap = dict(self._last_capacity_stats)
        # A complete key/LRU inventory remains available through cache_info()
        # for the final diagnostic. Do not rebuild/serialize that growing list
        # on every query. Release cannot change the persistent byte count.
        info = {"persistent_bytes": self._capacity_cache.resident_bytes if self._capacity_cache else 0,
                "budget_bytes": self.cache_budget_bytes, "active_query": False}
        context_tokens = stats["read_tokens"]-stats["query_tokens"]
        sink_hit = sum(event["tokens"] for event in cap.get("events", [])
                       if event["kind"] == "sink" and event["status"] == "hit")
        document_hit = cap.get("context_tokens_hit", 0)
        hit_tokens = document_hit+sink_hit
        stats.update(document_id=self.document_id, cache_budget_bytes=self.cache_budget_bytes,
                     capacity_cache=cap, cache_inventory=info,
                     capture_calls=self._captures, document_capture_calls=self._chunk_captures,
                     sink_capture_calls=self._sink_captures, query_capture_calls=0,
                     resident_raw_token_bytes=tensor_bytes(self.chunks),
                     context_tokens=context_tokens, cache_lookup_applicable=self.arm != "j0",
                     cache_hit_context_tokens=hit_tokens,
                     cache_miss_context_tokens=context_tokens-hit_tokens,
                     cache_hit_fraction=hit_tokens/context_tokens,
                     cache_hit_document_tokens=document_hit, cache_hit_sink_tokens=sink_hit,
                     cache_token_scope="flat context includes shared sink; nested ByteLRU context counts document tokens only",
                     cache_resident_bytes=info.get("persistent_bytes", 0),
                     evicted_nodes=cap.get("evicted_entries", 0),
                     evicted_bytes=cap.get("evicted_tensor_bytes", 0),
                     cache_bypass_tokens=sum(event["tokens"] for event in cap.get("events", [])
                                             if event["status"] == "miss_bypass"),
                     cache_bypass_bytes=cap.get("bypass_tensor_bytes", 0),
                     cache_active_payload_bytes=cap.get("active_payload_tensor_bytes", 0),
                     cache_fill_transfer_bytes=cap.get("miss_tensor_bytes", 0),
                     cache_fill_s=cap.get("build_s", 0.0)+cap.get("cpu_copy_s", 0.0),
                     cache_fill_direction_scope="GPU payload D2H" if self.cm.device.type == "cuda"
                                                else "CPU correctness test: CPU compact copy, no D2H",
                     cache_capture_s=cap.get("capture_s", 0.0),
                     cache_build_s=cap.get("build_s", 0.0), cache_cpu_copy_s=cap.get("cpu_copy_s", 0.0),
                     cache_release_s=release_ended-release_started,
                     cache_budget_scope="unique persistent CPU payload tensor storage; raw IDs/metadata and active bypasses excluded",
                     model_cache_policy="no cache" if self.arm == "j0" else "on-demand chunk ByteLRU; shared sink; pinned query leases")
        ended = self.clock()
        # Old load_s already contains lookup/build/copy/eviction. Include the
        # outer bookkeeping/release as a separate, auditable total component;
        # it is not mislabeled as decode compute or silently left uncharged.
        stats["reader_wrapper_s"] = max(0.0, ended-started-stats["total_s"])
        stats["base_query_total_s"] = stats["total_s"]
        stats["total_s"] += stats["reader_wrapper_s"]
        return generated, stats


class CapacityReader(_CapacityMixin, ReusableReader):
    def __init__(self, model, j, tokenizer, arm="fix_all", model_id="", cache_budget_bytes=None,
                 chunk_size=512):
        if arm not in {"j0", "fix_all"}:
            raise ValueError("CapacityReader supports only j0 and fix_all")
        ReusableReader.__init__(self, model, j, tokenizer, arm=arm, model_id=model_id)
        self._init_capacity(cache_budget_bytes, chunk_size)


class CapacityCacheBlendReader(_CapacityMixin, ReusableContextualCacheBlend):
    def __init__(self, model, j, tokenizer, arm="cacheblend16", model_id="", cache_budget_bytes=None,
                 chunk_size=512):
        if arm != "cacheblend16":
            raise ValueError("CapacityCacheBlendReader requires cacheblend16")
        ReusableContextualCacheBlend.__init__(self, model, j, tokenizer, model_id=model_id)
        self._init_capacity(cache_budget_bytes, chunk_size)


class CapacityPrefixReader(_RawDocumentBinding, PrefixReader):
    def __init__(self, model, j, tokenizer, arm="prefix", model_id="", cache_budget_bytes=None,
                 chunk_size=512):
        PrefixReader.__init__(self, model, j, tokenizer, arm=arm, model_id=model_id,
                              cache_budget_bytes=cache_budget_bytes)
        self._init_binding(chunk_size)

    @torch.no_grad()
    def query_ids(self, query_ids, **kwargs):
        generated, stats = super().query_ids(query_ids, **kwargs)
        stats.update(document_id=self.document_id, cache_lookup_applicable=True,
                     document_capture_calls=0, sink_capture_calls=0, query_capture_calls=0,
                     model_cache_policy="exact context-prefix radix trie; on-demand CPU fill")
        return generated, stats


def make_capacity_reader(model, j, tokenizer, arm="fix_all", model_id="", cache_budget_bytes=None,
                         chunk_size=512):
    classes = {"j0": CapacityReader, "fix_all": CapacityReader,
               "cacheblend16": CapacityCacheBlendReader, "prefix": CapacityPrefixReader}
    if arm not in classes:
        raise ValueError(f"Unsupported Phase B arm: {arm}")
    return classes[arm](model, j, tokenizer, arm=arm, model_id=model_id,
                        cache_budget_bytes=cache_budget_bytes, chunk_size=chunk_size)
