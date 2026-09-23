"""Native tiny-Qwen3 numerical/stop semantics and queue contracts; CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
import copy
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(2)
from transformers import Qwen3Config, Qwen3ForCausalLM
from qa_cost_runner import measure_native, native_reference, load_job, HERE
from qa_cost_bootstrap import serving_dependency_ready
from encbank import Encbank
from s15_ruler_lower import EncbankLower


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def decode(self, ids, **kwargs):
        return " ".join(str(int(i)) for i in ids)


def fixture(arm):
    torch.manual_seed(27)
    cfg = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048)
    cfg._attn_implementation = "sdpa"
    model, tok = Qwen3ForCausalLM(cfg).eval(), Tokenizer()
    reader = EncbankLower(model, 2, tok) if arm == "fix_all" else Encbank(model, 0, tokenizer=tok)
    reader.write_sink = arm == "fix_all"
    row = {"context_token_count": 26, "selected_indices": [3, 1], "chunk_size": 8,
           "max_new_tokens": 5, "eos_id": 2}
    ids = torch.tensor([list(range(3, 29)) + [31, 32, 33]])
    return reader, row, ids


class QACostRunnerTests(unittest.TestCase):
    def test_real_native_tokens_equal_reference_and_reset(self):
        for arm in ("fix_all", "j0"):
            reader, row, ids = fixture(arm)
            original_instance_keys = set(reader.__dict__)
            for selected in ([3, 1], [0, 2], [3, 1]):
                row["selected_indices"] = selected
                actual = measure_native(reader, ids, row)
                ref_text, ref_ids = native_reference(reader, ids, row)
                self.assertEqual(actual["generated_ids"], ref_ids)
                self.assertEqual(actual["prediction"], ref_text)
                self.assertEqual(actual["actual_decode_forward_calls"], actual["decode_forward_calls"])
                self.assertIsNone(getattr(reader, "_bottom", None))
                self.assertEqual(set(reader.__dict__), original_instance_keys)
                self.assertGreater(actual["selected_read_state_logical_tensor_bytes"], 0)
                self.assertGreaterEqual(actual["timings"]["native_adapter_generation_s"], actual["timings"]["native_adapter_ttft_s"])

    def test_real_native_eos_suppressed_first_then_terminal_forward_counted(self):
        for arm in ("fix_all", "j0"):
            reader, row, ids = fixture(arm)
            # Use the real native loop with deterministic logits at each head call.
            def forced(hidden):
                out = torch.full((*hidden.shape[:-1], 96), -10.0)
                out[..., 2] = 20  # EOS wins, except native step zero suppresses it.
                out[..., 7] = 10
                return out
            with patch.object(reader.lm_head, "forward", side_effect=forced):
                result = measure_native(reader, ids, row)
            self.assertEqual(result["generated_ids"], [7])
            self.assertTrue(result["terminated_by_eos"])
            self.assertEqual(result["actual_decode_forward_calls"], 1)
            self.assertEqual(result["sampled_tokens_including_terminal_eos"], 2)

    def test_cap_exhaustion_uses_n_minus_one_actual_forwards(self):
        reader, row, ids = fixture("fix_all")
        def forced(hidden):
            out = torch.zeros((*hidden.shape[:-1], 96))
            out[..., 7] = 10
            return out
        with patch.object(reader.lm_head, "forward", side_effect=forced):
            result = measure_native(reader, ids, row)
        self.assertEqual(result["generated_ids"], [7]*5)
        self.assertFalse(result["terminated_by_eos"])
        self.assertEqual(result["actual_decode_forward_calls"], 4)

    def test_hooks_restore_after_reader_failure(self):
        reader, row, ids = fixture("fix_all")
        original_keys = set(reader.__dict__)
        with patch.object(reader, "read_prefill", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                measure_native(reader, ids, row)
        self.assertIsNone(reader._bottom)
        self.assertEqual(set(reader.__dict__), original_keys)

    def test_actual_prepared_400_inputs_and_finite_job_scope(self):
        from qa_cost_runner import read_json
        path = HERE / "results/protocol/qa_cost/task_plan.json"
        total, smoke, warmups = 0, 0, 0
        for planned in read_json(path)["jobs"]:
            job, inputs, config = load_job(path, planned["id"])
            self.assertEqual(len(inputs), 200)
            if config["phase"] == "full":
                total += len(job["indices"])*config["repetitions"]
                warmups += config["warmups"]
            else:
                smoke += len(job["indices"])
            self.assertFalse(config["generation_cache"])
            self.assertFalse(config["whole_document_write_once"])
        self.assertEqual((total, smoke, warmups), (2400, 8, 4))
        self.assertFalse(torch.cuda.is_initialized())

    def test_waits_for_complete_scope_and_actual_child_exit(self):
        state = {"status": "completed", "requested_campaign": "all",
                 "jobs": {f"base/full/job{i}": {"status": "complete", "cells": 12} for i in range(10)}}
        self.assertTrue(serving_dependency_ready(state, []))
        processes = [{"Name": "python.exe", "CommandLine": "python F:/Paper_Evolve/exp/encbank_v2_benchmarks_20260908/serving_reuse.py --out existing"}]
        self.assertFalse(serving_dependency_ready(state, processes))
        state["jobs"].pop("base/full/job9")
        self.assertFalse(serving_dependency_ready(state, []))


if __name__ == "__main__":
    unittest.main()
