"""Standard-library-only request and accounting contract tests."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from reuse_protocol import (QUALITY_VERSION, recipe, summarize_reuse,
                            validate_reuse_mode, workload)


class Tokenizer:
    all_special_ids = [0, 1, 2]
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        return [20, 21, 22, 23]


def arguments(**kwargs):
    args = dict(reuse_requests=10, prompt_tokens=8, document_tokens=64, chunk_size=16,
        seed=7, generation_tokens=8, reader_implementation="reference", adapter_kind="trained",
        repetitions=1, arm="D0", cache_mode="cold_hj", j=2, m=4, rho=.5, rank=4, alpha=4.)
    args.update(kwargs)
    return SimpleNamespace(**args)


class Tests(unittest.TestCase):
    def test_distinct_prompts_and_identical_arm_inputs(self):
        cfg = SimpleNamespace(vocab_size=101, max_position_embeddings=256)
        outputs = [workload(Tokenizer(), cfg, arguments(arm=arm)) for arm in ("D0", "A", "B", "D1", "FULL")]
        self.assertTrue(all(x == outputs[0] for x in outputs))
        self.assertEqual(len({tuple(p) for p in outputs[0]["prompts"]}), 10)
        self.assertEqual({len(p) for p in outputs[0]["prompts"]}, {8})
        self.assertEqual(sum(map(len, outputs[0]["chunks"])), 64)

    def test_same_prompt_repetition_and_window_overflow_rejected(self):
        cfg = SimpleNamespace(vocab_size=101, max_position_embeddings=256)
        with self.assertRaises(ValueError):
            workload(Tokenizer(), cfg, arguments(), prompts=[[3] * 8] * 10)
        cfg.max_position_embeddings = 80
        with self.assertRaises(ValueError):
            workload(Tokenizer(), cfg, arguments())

    def test_zero_preserves_old_protocol(self):
        self.assertIsNone(validate_reuse_mode(SimpleNamespace(reuse_requests=0)))

    def test_setup_counted_once_and_decode_uses_n_minus_one(self):
        records = [dict(generated_tokens=4, decode_steps=3, decode_wall_s=2,
                       query_e2e_s=3, ttft_s=1, cumulative_e2e_s=8, cumulative_first_token_s=6),
                   dict(generated_tokens=4, decode_steps=3, decode_wall_s=2,
                       query_e2e_s=3, ttft_s=1, cumulative_e2e_s=11, cumulative_first_token_s=9)]
        out = summarize_reuse(records, 5, 11)
        self.assertEqual(out["all_queries_with_store_build_s"], 11)
        self.assertEqual(out["first_query_with_store_build_s"], 8)
        self.assertEqual(out["decode_tps"], 1.5)
        self.assertEqual(out["amortized_generated_tokens_per_s"], 8 / 11)

    def test_gate_requires_matching_complete_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.json"
            args = arguments(reuse_quality_receipt=path, model=tmp, adapter=Path(tmp) / "adapter.pt")
            good = {"protocol": QUALITY_VERSION, "status": "complete", "passed": True, "arm": "D0",
                    "recipe": recipe(args), "source_sha256": {"a": "123"},
                    "adapter_sha256": "hash", "model_config_sha256": "hash", "unique_queries": 2,
                    "document_blocks": 2, "real_input": True, "unique_source_documents": 2,
                    "checks": dict.fromkeys(("greedy_equal", "selected_route_equal",
                        "repeat_query_equal", "cache_unchanged", "finite_logits", "storage_roundtrip"), True)}
            with patch("reuse_protocol.source_identity", return_value={"a": "123"}), patch("reuse_protocol.digest", return_value="hash"):
                path.write_text(json.dumps(good), encoding="utf-8")
                self.assertEqual(validate_reuse_mode(args)["requests"], 10)
                for mutate in (lambda d: d.update(passed=False), lambda d: d.update(arm="B"),
                               lambda d: d.update(adapter_sha256="other"),
                               lambda d: d["checks"].update(cache_unchanged=False)):
                    bad = copy.deepcopy(good)
                    mutate(bad)
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "waiting_quality"):
                        validate_reuse_mode(args)

    def test_repetition_and_wrong_backend_are_not_reuse(self):
        for change in ({"repetitions": 10}, {"reader_implementation": "backend_v3"},
                       {"arm": "B", "cache_mode": "cold_hj"}):
            with self.assertRaises(ValueError):
                validate_reuse_mode(arguments(**change))


if __name__ == "__main__":
    unittest.main()
