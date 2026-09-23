"""Native Encbank and full-recompute controls for the sparse infra harness.

Pure reader definitions: no model loading, device selection, admission or timers.
Both controls keep the EXACT model object (including active upper-layer LoRA)
passed by the harness. The full-recompute label must therefore say "same upper
LoRA", unless the caller explicitly supplies an unadapted base model.

Full recompute persists raw CPU token IDs and computes all layers per request.
Both controls use normal KV-cached, one-token decode. There is no cross-request
prefix reuse. Document-KV bytes below count the document portion of the combined
native cache; they are not an additional allocation or GPU peak measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1] / "Encbank") not in sys.path:
    sys.path.insert(0, str(HERE.parents[1] / "Encbank"))
from encbank.model import Encbank


@dataclass
class NativeQueryState:
    bottom_cache: object
    top_cache: object
    query_position: int
    pack_position: int
    route_stats: dict


class NativeEncbankReader:
    """Native dense Encbank execution of the passed model and its active adapter."""

    store_kind = "cold_hj"
    implementation = "native-encbank-layer-calls-cached-decode"
    model_contract = "same passed model object and active adapter; no merge or replacement"

    def __init__(self, encbank):
        if encbank.top_prepay_b != 0 or encbank.block_diagonal:
            raise ValueError("Native control requires ordinary exact-resume Encbank")
        self.encbank = encbank
        self.model = encbank.model
        self.j = int(encbank.resume_j)
        self.L = int(encbank.num_layers)

    def eval(self):
        self.model.eval()
        return self

    def _check_eval(self):
        if self.model.training:
            raise ValueError("Use model.eval() for native infra controls")

    @torch.no_grad()
    def write_chunk(self, token_ids):
        self._check_eval()
        return self.encbank.write_chunk(token_ids)

    def _prepare_documents(self, sink, docs):
        cm = self.encbank
        converted = [value.to(device=cm.device, dtype=cm.dtype) for value in docs]
        sink = None if sink is None else sink.to(device=cm.device, dtype=cm.dtype)
        return sink, converted

    def _route_stats(self, sink, docs, cache, prompt_length, probes):
        lengths = [int(value.shape[1]) for value in docs]
        candidate = sum(lengths)
        sink_tokens = 0 if sink is None else int(sink.shape[1])
        context = candidate + sink_tokens
        by_layer, document_bytes = {}, 0
        for layer in range(self.j, self.L):
            entry = cache.layers[layer]
            if entry.keys.shape[-2] != context + prompt_length or entry.values.shape[-2] != context + prompt_length:
                raise RuntimeError("Native prefill KV does not contain the exact document+prompt pack")
            by_layer[str(layer)] = context
            document_bytes += sum(value[..., :context, :].numel() * value.element_size()
                                  for value in (entry.keys, entry.values))
        return {"selection_source": "all-blocks", "selected_indices": list(range(len(docs))),
                "candidate_blocks": len(docs), "selected_blocks": len(docs),
                "candidate_tokens": candidate, "selected_tokens": candidate,
                "target_retain_ratio": 1.0, "actual_retain_ratio": 1.0,
                "token_budget": candidate, "budget_overflow_tokens": 0,
                "sink_tokens": sink_tokens, "resume_j": self.j, "fusion_layer": self.j,
                "probe_mode": "native-dense", "probe_indices": list(probes or []),
                "probe_rows_fallback": False, "routing_computed": False,
                "hot_hits": 0, "hot_misses": len(docs),
                "original_query_start": context, "query_tokens": prompt_length,
                "document_kv_tokens_by_layer": by_layer,
                "document_kv_bytes": int(document_bytes), "unselected_late_kv_tokens": 0,
                "document_kv_bytes_scope": "document slice of combined document+query native KV; not separate allocation",
                "implementation": self.implementation, "model_contract": self.model_contract,
                "cross_request_kv_reuse": False, "cached_decode": True}

    @torch.no_grad()
    def prefill(self, sink, docs, prompt, probe_indices=None):
        self._check_eval()
        cm = self.encbank
        sink, documents = self._prepare_documents(sink, docs)
        prompt_ids = cm._as_ids(prompt)
        if prompt_ids.shape[1] < 1:
            raise ValueError("Prompt must be nonempty")
        q_h, bottom, q_pos = cm.write_prefill(prompt_ids)
        logits, top, pack_pos = cm.read_prefill(sink, documents, q_h)
        stats = self._route_stats(sink, documents, top, int(q_pos), probe_indices)
        return logits, NativeQueryState(bottom, top, int(q_pos), int(pack_pos), stats)

    @torch.no_grad()
    def decode_step(self, token_id, state):
        self._check_eval()
        logits = self.encbank.decode_step(token_id, state.bottom_cache, state.top_cache,
                                       state.query_position, state.pack_position)
        state.query_position += 1
        state.pack_position += 1
        return logits


class FullRecomputeReader(NativeEncbankReader):
    """Same model/upper LoRA, j=0; token store, full prefill, normal cached decode."""

    store_kind = "raw_token_ids"
    implementation = "full-recompute-same-upper-lora-native-layer-cached-decode"

    def __init__(self, encbank):
        # Constructing this wrapper references the existing model. It neither
        # copies parameters nor removes the caller's upper-layer adapter.
        full = Encbank(encbank.model, resume_j=0, tokenizer=encbank.tokenizer)
        super().__init__(full)

    @torch.no_grad()
    def write_chunk(self, token_ids):
        # Raw token storage, deliberately no embedding or model/device work.
        ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").reshape(1, -1).clone()
        if not ids.shape[1]:
            raise ValueError("Token store chunk must be nonempty")
        return ids

    def _prepare_documents(self, sink, docs):
        cm = self.encbank
        def embed(value):
            if value.dtype != torch.long:
                raise ValueError("Full-recompute store must contain int64 token IDs, not h_j")
            return cm.embed_tokens(cm._as_ids(value))
        return None if sink is None else embed(sink), [embed(value) for value in docs]


def self_test():
    """Tiny CPU-only real-Qwen tests; no checkpoint download or CUDA calls."""
    import unittest
    from transformers import Qwen3Config, Qwen3ForCausalLM
    sys.path.insert(0, str(HERE.parent / "encbank_v2_benchmarks_20260908"))
    from train_8b_baseline import attach_lora
    from sparse_reader import SparseEncbankReader

    class Tests(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(928)
            config = Qwen3Config(vocab_size=101, hidden_size=64, intermediate_size=128,
                num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                max_position_embeddings=256, attention_dropout=0., bos_token_id=1, eos_token_id=2)
            config._attn_implementation = "sdpa"
            self.model = Qwen3ForCausalLM(config).cpu().eval()
            modules = attach_lora(self.model, 2, 4, 4., torch.float32)
            with torch.no_grad():
                for module in modules.values():
                    module.B.normal_(std=.01)
            self.base = Encbank(self.model, resume_j=2)
            self.chunks = [[3, 4, 5, 6], [7, 8, 9, 10]]
            self.prompt = [21, 22, 23]

        def test_native_matches_sparse_D0_prefill_and_decode(self):
            reader = NativeEncbankReader(self.base)
            sparse = SparseEncbankReader(self.base, fusion_layer=4, retain_ratio=1., probe_mode="dense").eval()
            docs = [reader.write_chunk(chunk) for chunk in self.chunks]
            sink = reader.write_chunk([1])
            with torch.no_grad():
                actual, state = reader.prefill(sink, docs, self.prompt)
                expected, reference = sparse.prefill(sink, docs, self.prompt)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                for token in (31, 32, 33):
                    actual = reader.decode_step(token, state)
                    expected = sparse.decode_step(token, reference)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            self.assertFalse(state.route_stats["routing_computed"])
            self.assertIs(reader.model, self.model)

        def test_full_store_is_tokens_and_prefill_matches_same_adapted_HF_model(self):
            reader = FullRecomputeReader(self.base)
            calls = []
            hooks = [layer.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
                     for layer in self.base.layers]
            try:
                docs = [reader.write_chunk(chunk) for chunk in self.chunks]
                sink = reader.write_chunk([1])
                self.assertEqual(calls, [])  # No offline document model compute.
                self.assertEqual(docs[0].dtype, torch.long)
                self.assertEqual(docs[0].device.type, "cpu")
                logits, state = reader.prefill(sink, docs, self.prompt)
                packed = [1] + sum(self.chunks, []) + self.prompt
                with torch.no_grad():
                    expected = self.model(input_ids=torch.tensor([packed]), use_cache=False).logits[:, -1:]
                torch.testing.assert_close(logits, expected, atol=3e-6, rtol=3e-5)
                for token in (31, 32, 33):
                    calls.clear()
                    logits = reader.decode_step(token, state)
                    self.assertEqual(calls, [1] * 6)  # Every layer receives just one token.
                    packed.append(token)
                    with torch.no_grad():
                        expected = self.model(input_ids=torch.tensor([packed]), use_cache=False).logits[:, -1:]
                    torch.testing.assert_close(logits, expected, atol=3e-6, rtol=3e-5)
                self.assertIs(reader.model, self.model)
                self.assertEqual(set(state.route_stats["document_kv_tokens_by_layer"]), set(map(str, range(6))))
            finally:
                for hook in hooks:
                    hook.remove()

        def tearDown(self):
            self.assertFalse(torch.cuda.is_initialized())

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    raise SystemExit(0 if self_test() else 1)
