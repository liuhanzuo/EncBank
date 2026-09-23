"""Check contextual selection and full-recompute identity on tiny Qwen3."""
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Encbank"))
sys.path.insert(0, str(ROOT / "exp"))
from transformers import Qwen3Config, Qwen3ForCausalLM
from encbank import Encbank
from cacheblend_contextual import ContextualCacheBlend


class ContextualSelectionTest(unittest.TestCase):
    def test_context_changes_selection_signal_and_full_recompute_is_exact(self):
        torch.manual_seed(73)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
        cm = Encbank(model, resume_j=2)
        cb = ContextualCacheBlend(cm, .16)
        pieces = [torch.tensor([[1]]), torch.tensor([[11, 12, 13, 14]]),
                  torch.tensor([[31, 32, 33, 34, 35]]), torch.tensor([[51, 52]])]
        caches = [cb.prefill_chunk_full(ids)[0] for ids in pieces]
        merged = cb.concat_kv_reindex(caches, [0, 1, 5, 10])
        packed = torch.cat(pieces, 1)
        stats = {}
        full, indices, _ = cb.read(packed, merged, 1, 2, 1.0, stats)
        reference = model(packed, use_cache=False).logits
        torch.testing.assert_close(full, reference, atol=2e-6, rtol=2e-5)
        self.assertGreater(stats["context_v_error_max"], 1e-8)
        self.assertEqual(indices.tolist(), list(range(12)))
        stats = {}
        sparse, indices, _ = cb.read(packed, merged, 1, 2, .16, stats)
        self.assertTrue(torch.isfinite(sparse).all())
        self.assertEqual(stats["n_recompute_ctx"], 1)
        self.assertEqual(len(indices), 4)
        self.assertTrue({0, 10, 11}.issubset(indices.tolist()))
        # An all-selected port must preserve stock autoregressive token decisions.
        cb.recompute_ratio = 1.0
        actual = cb.generate_explicit(pieces[1:-1], pieces[-1], 1, None, 5)
        expected = []
        running = packed
        for _ in range(5):
            token = int(model(running, use_cache=False).logits[0, -1].argmax())
            expected.append(token)
            running = torch.cat([running, torch.tensor([[token]])], 1)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
