"""CPU-only integration tests for on-demand readers and actual byte budgets."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  TOKENIZERS_PARALLELISM="false")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from transformers import Qwen3Config, Qwen3ForCausalLM
from capacity_reader import make_capacity_reader, ReusableReader, ReusableContextualCacheBlend


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2


class CapacityReaderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(771)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=2048,
                            bos_token_id=1, eos_token_id=2, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        self.model = Qwen3ForCausalLM(config).eval()
        self.tokenizer = Tokenizer()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.counter = 0
        self.doc_a = [10,11,12,13,20,21,22,23,30,31,32,33]
        self.doc_b = [10,11,12,13,40,41,42,43,50,51,52,53]

    def tearDown(self):
        self.tmp.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    def reader(self, arm, budget=10**8):
        reader = make_capacity_reader(self.model, 1, self.tokenizer, arm=arm,
                                      cache_budget_bytes=budget, chunk_size=4)
        prepared = reader.bind_document("A", self.doc_a)
        self.assertEqual(prepared["capture_calls"], 0)
        self.assertEqual(prepared["offline_kv_tensor_bytes"], 0)
        self.assertEqual(prepared["raw_token_bytes"], len(self.doc_a)*8)
        return reader

    def fresh(self, arm, document):
        self.counter += 1
        if arm == "cacheblend16":
            reader = ReusableContextualCacheBlend(self.model, 1, self.tokenizer)
        else:
            reader = ReusableReader(self.model, 1, self.tokenizer,
                                    arm="j0" if arm == "prefix" else arm)
        reader.write_store(document, self.path/str(self.counter), chunk_size=4)
        reader.open_store(self.path/str(self.counter))
        return reader

    def compare(self, reader, reference, indices=(0,1), query=(60,61), force=True):
        kwargs = dict(selected_indices=indices, max_new_tokens=5,
                      force_length=force, capture_logits=True)
        with torch.enable_grad():
            generated, stats = reader.query_ids(query, **kwargs)
            expected, old = reference.query_ids(query, **kwargs)
        self.assertEqual(generated, expected)
        self.assertEqual(stats["decode_steps"], old["decode_steps"])
        self.assertEqual(len(stats["step_logits"]), len(old["step_logits"]))
        for actual, target in zip(stats["step_logits"], old["step_logits"]):
            torch.testing.assert_close(actual, target, atol=2e-6, rtol=2e-5)
            self.assertFalse(actual.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertLessEqual(stats["cache_resident_bytes"], stats["cache_budget_bytes"])
        self.assertAlmostEqual(stats["ttft_s"]+stats["decode_s"]+stats.get("reader_wrapper_s", 0),
                               stats["total_s"], places=8)
        self.assertEqual(stats["query_capture_calls"], 0)
        return stats

    def test_all_arms_cold_and_warm_equal_old_reader(self):
        for arm in ("j0", "fix_all", "cacheblend16", "prefix"):
            with self.subTest(arm=arm):
                reader, reference = self.reader(arm), self.fresh(arm, self.doc_a)
                cold = self.compare(reader, reference)
                warm = self.compare(reader, reference, query=(70,71,72), force=False)
                if arm in {"fix_all", "cacheblend16"}:
                    self.assertEqual(cold["capture_calls"], 3)
                    self.assertEqual(cold["document_capture_calls"], 2)
                    self.assertEqual(cold["sink_capture_calls"], 1)
                    self.assertEqual(warm["capture_calls"], 0)
                    self.assertEqual(warm["cache_hit_context_tokens"], 9)
                    self.assertGreater(cold["cache_cpu_copy_s"], 0)
                    self.assertGreaterEqual(cold["load_s"], cold["cache_build_s"])
                    self.assertEqual(cold["cache_fill_transfer_bytes"], cold["cache_resident_bytes"])
                    self.assertEqual(warm["cache_fill_transfer_bytes"], 0)
                    self.assertAlmostEqual(cold["cache_fill_s"],
                        cold["cache_build_s"]+cold["cache_cpu_copy_s"], places=8)
                    self.assertIn("no D2H", cold["cache_fill_direction_scope"])
                if arm == "j0":
                    self.assertEqual(warm["cache_resident_bytes"], 0)
                    self.assertFalse(warm["cache_lookup_applicable"])

    def test_cross_document_cache_survives_and_sink_is_shared_once(self):
        for arm in ("fix_all", "cacheblend16", "prefix"):
            with self.subTest(arm=arm):
                reader = self.reader(arm)
                ref_a = self.fresh(arm, self.doc_a)
                self.compare(reader, ref_a)
                reader.bind_document("B", self.doc_b)
                second = self.compare(reader, self.fresh(arm, self.doc_b))
                if arm != "prefix":
                    self.assertEqual(second["cache_hit_context_tokens"], 1)
                    self.assertEqual(second["capture_calls"], 2)
                    self.assertEqual(second["sink_capture_calls"], 0)
                else:
                    self.assertEqual(second["cache_hit_context_tokens"], 5)
                reader.bind_document("A", self.doc_a)
                reused = self.compare(reader, ref_a)
                self.assertEqual(reused["cache_hit_context_tokens"], 9)
                self.assertEqual(reused["capture_calls"], 0)

    def test_same_document_id_changed_tokens_cannot_reuse_stale_chunk(self):
        for arm in ("fix_all", "cacheblend16", "prefix"):
            with self.subTest(arm=arm):
                reader = self.reader(arm)
                self.compare(reader, self.fresh(arm, self.doc_a))
                reader.bind_document("A", self.doc_b)
                changed = self.compare(reader, self.fresh(arm, self.doc_b))
                self.assertEqual(changed["cache_hit_context_tokens"], 5)
                if arm != "prefix":
                    self.assertEqual(changed["capture_calls"], 1)

    def test_zero_budget_bypasses_without_skipping_model_work(self):
        for arm in ("fix_all", "cacheblend16"):
            with self.subTest(arm=arm):
                reader, reference = self.reader(arm, 0), self.fresh(arm, self.doc_a)
                for _ in range(2):
                    stats = self.compare(reader, reference)
                    self.assertEqual(stats["capture_calls"], 3)
                    self.assertEqual(stats["cache_hit_context_tokens"], 0)
                    self.assertEqual(stats["cache_resident_bytes"], 0)
                    self.assertGreater(stats["capacity_cache"]["transient_tensor_bytes"], 0)

    def test_pinned_pack_bypass_and_cross_document_eviction_remain_correct(self):
        # FP32 tiny model: V2 per token = hidden(128)+one-layer KV(128).
        for arm, bytes_per_token in (("fix_all", 256), ("cacheblend16", 384)):
            with self.subTest(arm=arm):
                reader = self.reader(arm, 5*bytes_per_token)
                ref_a = self.fresh(arm, self.doc_a)
                self.compare(reader, ref_a)
                again = self.compare(reader, ref_a)
                self.assertEqual(again["cache_hit_context_tokens"], 5)
                self.assertEqual(again["capture_calls"], 1)
                reader.bind_document("B", self.doc_b)
                self.compare(reader, self.fresh(arm, self.doc_b))
                reader.bind_document("A", self.doc_a)
                back = self.compare(reader, ref_a)
                self.assertEqual(back["capture_calls"], 2)
                self.assertLessEqual(back["cache_resident_bytes"], 5*bytes_per_token)

    def test_failure_releases_query_lease_and_recovers(self):
        for arm in ("fix_all", "cacheblend16"):
            with self.subTest(arm=arm):
                reader, reference = self.reader(arm), self.fresh(arm, self.doc_a)
                method = "_prepare_pack" if arm == "cacheblend16" else "_prepare_read"
                with patch.object(reader, method, side_effect=RuntimeError("injected read failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        reader.query_ids([60], selected_indices=[0,1], max_new_tokens=2)
                self.assertIsNone(reader._active_lease)
                self.assertFalse(reader._query_active)
                recovered = self.compare(reader, reference)
                self.assertEqual(recovered["capture_calls"], 0)
                reader.close_store()
                self.assertEqual(reader.cache_info()["persistent_bytes"], 0)
                with self.assertRaises(ValueError):
                    reader.query_ids([60], selected_indices=[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
