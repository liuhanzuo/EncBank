"""Small real-tensor CPU capacity tests; no model loads or GPU measurements."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import copy
import unittest
import torch

from capacity_cache import (BuildResult, ByteLRU, CacheRequest, compact_cpu_payload,
                            fixed_kv_budgets, interleaved_trace, make_requests,
                            payload_inventory)

torch.set_num_threads(2)
if torch.get_num_interop_threads() != 16:
    torch.set_num_interop_threads(16)


def request(name, tokens=(1, 2), namespace="model/v2", document="doc"):
    return CacheRequest(namespace, document, name, tuple(tokens))


def payload(req):
    # Two FP32 scalars per token: actual payload is 8 bytes/token.
    values = torch.tensor(req.token_ids, dtype=torch.float32).reshape(1, 1, -1, 1)
    return {"kv": [(values, values + 1)]}


class CacheTests(unittest.TestCase):
    def test_real_hit_miss_eviction_and_rebuild(self):
        sink = CacheRequest("model/v2", "__shared_sink__", -1, (0,), "sink")
        a, b, c = (request(i) for i in range(3))
        calls = []
        def build(r):
            calls.append(r.key)
            return payload(r)
        cache = ByteLRU(40)  # sink 8 + two chunks of 16.
        with cache.acquire([sink, a, b], build) as first:
            self.assertEqual(first.stats["context_tokens_miss"], 4)
            self.assertEqual(first.stats["persistent_bytes_after"], 40)
        with cache.acquire([sink, b, c], build) as second:
            self.assertEqual(second.stats["context_tokens_hit"], 2)
            self.assertEqual(second.stats["context_tokens_miss"], 2)
            self.assertEqual(second.stats["evictions"][0]["chunk_index"], 0)
        with cache.acquire([sink, a], build) as third:
            self.assertEqual(third.stats["context_tokens_hit"], 0)
            self.assertEqual(third.stats["build_calls"], 1)
        self.assertEqual(calls.count(a.key), 2)  # Historical retrieval overlap is not a cache hit.
        self.assertEqual(cache.snapshot()["totals"]["queries"], 3)

    def test_later_hit_is_pinned_before_earlier_miss(self):
        a, b, c = (request(i) for i in range(3))
        cache = ByteLRU(32)
        with cache.acquire([a, b], payload):
            pass
        with cache.acquire([c, a], payload) as lease:
            self.assertEqual(lease.stats["hit_entries"], 1)
            self.assertEqual(lease.stats["evictions"][0]["chunk_index"], 1)
            self.assertEqual(lease.stats["bypass_entries"], 0)

    def test_oversized_query_preserves_all_payloads_with_explicit_bypass(self):
        sink = CacheRequest("model/v2", "__shared_sink__", -1, (0,), "sink")
        chunks = [request(i) for i in range(3)]
        cache = ByteLRU(24)
        lease = cache.acquire([sink] + chunks, payload)
        self.assertEqual(len(lease.payloads), 4)
        self.assertEqual(lease.stats["persistent_bytes_after"], 24)
        self.assertEqual(lease.stats["transient_tensor_bytes"], 32)
        self.assertEqual(lease.stats["active_payload_tensor_bytes"], 56)
        self.assertEqual(lease.stats["bypass_context_tokens"], 4)
        self.assertEqual(lease.stats["persistent_plus_transient_bytes_after_acquire"], 56)
        lease.release()
        lease.release()  # finally may safely repeat release.
        self.assertEqual(lease.payloads, [])
        self.assertFalse(cache.info()["active_query"])
        self.assertEqual(cache.resident_bytes, 24)

    def test_individually_oversized_item_does_not_evict_other_cache_entries(self):
        cache = ByteLRU(16)
        a = request(0)
        huge = request(1, range(10))
        with cache.acquire([a], payload):
            pass
        with cache.acquire([huge], payload) as lease:
            self.assertEqual(lease.stats["events"][0]["reason"], "entry_exceeds_budget")
            self.assertEqual(lease.stats["evicted_entries"], 0)
        with cache.acquire([a], payload) as lease:
            self.assertEqual(lease.stats["hit_entries"], 1)

    def test_zero_budget_and_duplicate_request_capture_once(self):
        a = request(0)
        calls = []
        def build(r):
            calls.append(r)
            return payload(r)
        with ByteLRU(0).acquire([a, a], build) as lease:
            self.assertEqual(len(calls), 1)
            self.assertIs(lease.payloads[0], lease.payloads[1])
            self.assertEqual(lease.stats["context_tokens_miss"], 4)
            self.assertEqual(lease.stats["bypass_tensor_bytes"], 16)

    def test_actual_storage_compacts_views_and_owns_external_values(self):
        parent = torch.arange(4096, dtype=torch.float32)
        view = parent[10:12]
        original = {"kv": [(view, view)]}
        self.assertEqual(payload_inventory(original)["unique_storage_bytes"], 16384)
        compact = compact_cpu_payload(original)
        self.assertEqual(payload_inventory(compact)["unique_storage_bytes"], 16)
        parent[10] = -100
        self.assertEqual(float(compact["kv"][0][0][0]), 10)

    def test_raw_ids_metadata_and_nonfloating_tensors_are_not_kv_capacity(self):
        invalid = ({"tokens": torch.tensor([1])},
                   {"kv": [(torch.tensor([1]), torch.tensor([2]))]},
                   {"kv": [], "metadata": "not cache"},
                   {"kv": [3]})
        for bad in invalid:
            with self.subTest(bad=type(bad)):
                with self.assertRaises(ValueError):
                    compact_cpu_payload(bad)

    def test_shared_sink_cross_document_and_exact_input_namespace_keys(self):
        cache = ByteLRU(128)
        a = make_requests("a", [1, 2, 3, 4], [0], chunk_size=2, sink_id=9, namespace="model-v2")
        b = make_requests("b", [1, 2, 3, 4], [0], chunk_size=2, sink_id=9, namespace="model-v2")
        with cache.acquire(a, payload):
            pass
        with cache.acquire(b, payload) as lease:
            self.assertEqual(lease.stats["hit_entries"], 1)
            self.assertEqual(lease.stats["sink_build_calls"], 0)
            self.assertEqual(lease.stats["chunk_build_calls"], 1)
        changed = make_requests("a", [8, 2, 3, 4], [0], chunk_size=2, sink_id=9, namespace="model-v2")
        self.assertNotEqual(a[1].key, changed[1].key)
        other_model = make_requests("a", [1, 2, 3, 4], [0], chunk_size=2, sink_id=9, namespace="model-cb")
        self.assertNotEqual(a[0].key, other_model[0].key)

    def test_failure_unpins_and_releases_the_serial_lock(self):
        cache = ByteLRU(32)
        a, b = request(0), request(1)
        with cache.acquire([a], payload):
            pass
        def fail(_):
            raise RuntimeError("capture failed")
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            cache.acquire([a, b], fail)
        self.assertFalse(cache.info()["active_query"])
        self.assertFalse(any(e["pinned"] for e in cache.info()["lru_to_mru"]))
        with cache.acquire([a], payload):
            with self.assertRaisesRegex(ValueError, "serial"):
                cache.acquire([b], payload)
            with self.assertRaisesRegex(ValueError, "active"):
                cache.clear()
        self.assertEqual(cache.clear()["tensor_bytes"], 16)

    def test_timed_builder_and_cpu_copy_are_separate_and_capture_is_a_subphase(self):
        ticks = iter(range(20))
        cache = ByteLRU(64, clock=lambda: float(next(ticks)))
        with cache.acquire([request(0)], lambda r: BuildResult(payload(r), {"capture_s": 0.25})) as lease:
            self.assertEqual(lease.stats["build_s"], 1)
            self.assertEqual(lease.stats["cpu_copy_s"], 1)
            self.assertEqual(lease.stats["capture_s"], 0.25)
            self.assertEqual(lease.stats["acquire_s"], 4)


class TraceBudgetTests(unittest.TestCase):
    def fixture(self):
        docs, queries = [], []
        for did, length in (("a", 5), ("b", 8)):
            order = [f"{did}-{i}" for i in range(3)]
            docs.append({"document_id": did, "context_ids": list(range(length)), "query_ids_ordered": order})
            for i, qid in enumerate(order):
                queries.append({"id": qid, "document_id": did, "stream_index": i,
                                "selected_indices": [0] if i < 2 else [1]})
        return docs, queries

    def test_trace_is_deterministic_interleaved_and_preserves_real_query_order(self):
        docs, queries = self.fixture()
        plan = interleaved_trace(docs, queries, per_document_limit=3, minimum_queries=3)
        self.assertEqual(plan, interleaved_trace(docs, queries, per_document_limit=3, minimum_queries=3))
        self.assertEqual(len(plan["requests"]), 6)
        self.assertEqual([r["document_query_index"] for r in plan["requests"]], [0, 0, 1, 1, 2, 2])
        self.assertNotEqual(plan["requests"][0]["document_id"], plan["requests"][1]["document_id"])
        with self.assertRaisesRegex(ValueError, "fewer"):
            interleaved_trace(docs, queries, minimum_queries=80)

    def test_union_budget_counts_tail_chunks_and_single_shared_sink(self):
        docs, queries = self.fixture()
        trace = interleaved_trace(docs, queries, per_document_limit=3, minimum_queries=3)
        cfg = {"num_hidden_layers": 2, "num_key_value_heads": 1, "head_dim": 2}
        b = fixed_kv_budgets(docs, trace, cfg, chunk_size=4)
        # Four chunks: a=4+1, b=4+4, and exactly one global sink token.
        self.assertEqual(b["referenced_chunk_tokens"], 13)
        self.assertEqual(b["shared_sink_tokens"], 1)
        self.assertEqual(b["full_kv_bytes_per_token"], 16)
        self.assertEqual(b["reference_full_kv_bytes"], 224)
        self.assertEqual([x["bytes"] for x in b["budgets"]], [56, 112, 224])
        self.assertEqual(b["j0_persistent_kv_bytes"], 0)
        # Q2 uses each document's first chunk only; repeated visits count once.
        short = interleaved_trace(docs, queries, per_document_limit=2, minimum_queries=2)
        small = fixed_kv_budgets(docs, short, cfg, chunk_size=4)
        self.assertEqual(small["referenced_chunk_tokens"], 8)
        self.assertEqual(small["all_participating_document_tokens"], 13)

    def test_qwen_gqa_reference_and_real_tensor_footprints(self):
        cfg = {"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128}
        docs, queries = self.fixture()
        trace = interleaved_trace(docs, queries, per_document_limit=2, minimum_queries=2)
        b = fixed_kv_budgets(docs, trace, cfg, chunk_size=4)
        self.assertEqual(b["full_kv_bytes_per_token"], 144 * 1024)
        pairs = lambda n: [(torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16),
                            torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16)) for _ in range(n)]
        full = compact_cpu_payload({"kv": pairs(36)})
        v2 = compact_cpu_payload({"h": torch.zeros(1, 1, 4096, dtype=torch.bfloat16), "kv": dict(enumerate(pairs(12)))})
        self.assertEqual(payload_inventory(full)["unique_storage_bytes"], 144 * 1024)
        self.assertEqual(payload_inventory(v2)["unique_storage_bytes"], 56 * 1024)


if __name__ == "__main__":
    unittest.main()
