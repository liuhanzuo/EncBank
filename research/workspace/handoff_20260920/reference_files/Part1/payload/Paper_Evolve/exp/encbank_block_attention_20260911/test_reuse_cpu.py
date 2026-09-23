"""Tiny Qwen3 tests. Run only on the remote CPU with CUDA_VISIBLE_DEVICES=''."""
from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1] / "Encbank", HERE.parent / "encbank_v2_benchmarks_20260908"):
    sys.path.insert(0, str(path))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from encbank.model import Encbank
from train_8b_baseline import attach_lora
from reuse_benchmark import run_reuse_quality_probe
from test_reuse_protocol import Tokenizer, arguments


class TinyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="reuse_cpu_", dir=HERE)
        root = Path(self.tmp.name)
        torch.manual_seed(231)
        cfg = Qwen3Config(vocab_size=101, hidden_size=64, intermediate_size=96,
            num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=256, attention_dropout=0., bos_token_id=1, eos_token_id=2)
        cfg._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(cfg).cpu().eval()
        modules = attach_lora(model, 2, 4, 4., torch.float32)
        with torch.no_grad():
            for module in modules.values():
                module.B.normal_(std=.01)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.encbank = Encbank(model, resume_j=2)
        cfg.to_json_file(root / "config.json")
        (root / "adapter.pt").write_bytes(b"tiny-test-identity-only")
        self.args = arguments(reuse_requests=2, document_tokens=16, chunk_size=8,
            generation_tokens=4, model=str(root), adapter=str(root / "adapter.pt"))
        self.chunks = [[3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18]]
        self.prompts = [[31, 32, 33], [41, 42, 43, 44]]

    def test_all_arm_reference_reuse_and_fresh_query_state(self):
        for arm in ("D0", "A", "B", "D1", "FULL"):
            with self.subTest(arm=arm):
                self.args.arm = arm
                out = run_reuse_quality_probe(self.args, torch, self.encbank, Tokenizer(),
                    chunks=self.chunks, prompts=self.prompts, source_ids=["tiny-q0", "tiny-q1"],
                    probe_indices=[[0, 1, 2], [1, 2, 3]])
                self.assertTrue(out["passed"], json.dumps(out["checks"]))
                self.assertEqual(len(out["observations"]), 3)
                self.assertTrue(all(out["checks"].values()))
                self.assertTrue(out["storage_roundtrip"]["passed"])
                self.assertGreater(out["storage_roundtrip"]["cold_file_bytes"], 0)
                self.assertEqual(out["storage_roundtrip"]["hot_file_bytes"] > 0, arm in ("A", "B"))
                if arm == "D0":
                    self.assertEqual(set(out["observations"][0]["paths"]), {"reuse", "reference_cold", "native_cold"})
                if arm == "FULL":
                    self.assertIn("direct_hf", out["observations"][0]["paths"])

    def test_generated_synthetic_inputs_do_not_qualify_real_model_timing(self):
        self.args.arm = "A"
        out = run_reuse_quality_probe(self.args, torch, self.encbank, Tokenizer())
        self.assertFalse(out["passed"])
        self.assertFalse(out["real_input"])

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.tmp.cleanup()


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES='' before this remote CPU test")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    unittest.main(verbosity=2)
