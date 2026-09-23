"""Real tiny-Qwen3 equivalence, durable reuse and accounting checks; no GPU."""
import tempfile
import unittest
import json
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from serving_reuse import ARMS, ReusableReader, make_questions, tensor_bytes
import serving_reuse
from encbank import Encbank
from s15_ruler_lower import EncbankLower
from reader_adapter import ExplicitPackReader

torch.set_num_threads(1)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def decode(self, ids, **kwargs):
        return " ".join(str(int(i)) for i in ids)
    def encode(self, text, **kwargs):
        return [3+ord(c)%90 for c in text]


def model_and_tokenizer():
    torch.manual_seed(27)
    cfg = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048)
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval(), Tokenizer()


def reference(model, tokenizer, arm, context, query, indices, chunk_size):
    reader = EncbankLower(model, 2, tokenizer) if arm == "fix_all" else Encbank(
        model, 0 if arm == "j0" else 2, tokenizer=tokenizer)
    reader.write_sink = arm in {"fix_all", "pub_sink"}
    stats = {"capture_step_logits": True}
    original_decode = reader._decode_from_pack
    def instrument(*args, **kwargs):
        kwargs["stats"] = stats
        return original_decode(*args, **kwargs)
    ids = torch.cat([context, torch.tensor(query)]).unsqueeze(0)
    with patch.object(reader, "_decode_from_pack", side_effect=instrument):
        ExplicitPackReader(reader).generate_from_ids(ids, context_token_count=len(context),
            selected_indices=indices, chunk_size=chunk_size, max_new_tokens=5)
    return stats["generated_ids"], stats["step_logits"]


class ServingReuseTests(unittest.TestCase):
    def test_all_arms_match_fresh_explicit_packs_after_disk_reopen(self):
        model, tok = model_and_tokenizer()
        context = torch.tensor(list(range(3, 29)))  # Last document chunk is short.
        cases = [([31, 32, 33], [0, 2]), ([41, 42, 43, 44], [3, 1]), ([31, 32, 33], [2, 0])]
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ARMS:
                reader = ReusableReader(model, 2, tok, arm, "tiny")
                path = Path(tmp)/arm
                write = reader.write_store(context, path, 8)
                self.assertEqual(write["n_document_tokens"], 26)
                self.assertEqual(write["serialized_bytes"], sum(p.stat().st_size for p in path.iterdir()))
                if arm == "fix_all":
                    self.assertEqual(write["capture_calls"], 5)
                if arm == "j0":
                    self.assertEqual(write["payload_tensor_bytes"], 0)
                for tier in ("disk", "cpu"):
                    with self.subTest(arm=arm, tier=tier):
                        # A different reader process-equivalent object loads saved tensors.
                        reopened = ReusableReader(model, 2, tok, arm, "tiny")
                        startup = reopened.open_store(path, tier)
                        self.assertGreater(startup["resident_cpu_tensor_bytes"], 0)
                        fail_name = "_capture_lower" if arm == "fix_all" else "write_chunk"
                        with patch.object(reopened.cm, fail_name, side_effect=AssertionError("query recaptured document")):
                            for query, indices in cases:
                                actual, stats = reopened.query_ids(query, selected_indices=indices,
                                    max_new_tokens=5, capture_logits=True)
                                expected, logits = reference(model, tok, arm, context, query, indices, 8)
                                self.assertEqual(actual, expected)
                                for a, b in zip(stats["step_logits"], logits):
                                    self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=2e-5))
                                self.assertEqual(stats["capture_calls"], 0)
                                self.assertEqual(stats["decode_steps"], len(stats["step_logits"])-1)
                                self.assertGreaterEqual(stats["total_s"], stats["ttft_s"])
                                if tier == "cpu":
                                    self.assertEqual(stats["load_bytes"], 0)
                        reopened.close_store()

    def test_payloads_remain_on_cpu_and_unchanged_and_fixed_g(self):
        model, tok = model_and_tokenizer()
        context = torch.tensor(list(range(3, 19)))
        with tempfile.TemporaryDirectory() as tmp:
            reader = ReusableReader(model, 2, tok)
            reader.write_store(context, tmp, 8)
            reader.open_store(tmp, "cpu")
            original = reader.payloads["chunk_000000.pt"]["kv"][0][0].clone()
            for question in ([21, 22], [23, 24]):
                ids, stats = reader.query_ids(question, selected_indices=[0, 1], max_new_tokens=7, force_length=True)
                self.assertEqual(len(ids), 7)
                self.assertEqual(stats["decode_steps"], 6)
                self.assertTrue(torch.equal(original, reader.payloads["chunk_000000.pt"]["kv"][0][0]))
            self.assertEqual(reader.payloads["chunk_000000.pt"]["h"].device.type, "cpu")

    def test_retrieval_and_input_errors(self):
        model, tok = model_and_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            reader = ReusableReader(model, 2, tok)
            reader.write_store(list(range(3, 27)), tmp, 8)
            reader.open_store(tmp, "disk")
            _, recency = reader.query_ids([28, 29], selector="recency", topk=1, max_new_tokens=2)
            _, bm25 = reader.query_ids([4, 5], selector="bm25", topk=1, max_new_tokens=2)
            self.assertEqual(recency["selected_indices"], [2])
            self.assertEqual(bm25["selected_indices"], [0])
            for bad in ([0, 0], [4], [-1]):
                with self.assertRaises(ValueError):
                    reader.query_ids([22], selected_indices=bad)
            with self.assertRaises(ValueError):
                reader.query_ids([], selected_indices=[0])
            with self.assertRaises(ValueError):
                reader.write_store([1, 2], tmp)
            with self.assertRaises(ValueError):
                ReusableReader(model, 2, tok, model_id="different").open_store(tmp)
            self.assertEqual(len({q["text"] for q in make_questions(list(range(3, 27)), tok, 100, 8)}), 100)

    def test_cli_q_prefixes_fixed_g_and_full_document_write_once(self):
        model, tok = model_and_tokenizer()
        capture_count = 0
        original = EncbankLower._capture_lower
        def count_capture(reader, ids):
            nonlocal capture_count
            capture_count += 1
            return original(reader, ids)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            context = Path(tmp)/"document.txt"
            context.write_text("A document about apples, oranges and pears. "*10, encoding="utf-8")
            output = Path(tmp)/"output"
            argv = ["--model", "tiny", "--context-file", str(context), "--out", str(output),
                    "--device", "cpu", "--cpu-test-only", "--dtype", "float32", "--j", "2", "--chunk-size", "8",
                    "--topk", "2", "--lengths", "32", "--query-counts", "1", "3",
                    "--generation-lengths", "2", "4"]
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
                 patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                 patch.object(EncbankLower, "_capture_lower", count_capture):
                self.assertEqual(serving_reuse.main(argv), 0)
            self.assertEqual(capture_count, 5)  # 4 document chunks + 1 sink, all workloads.
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(summary), 32)
            for cell in summary:
                self.assertEqual(cell["query_totals"]["generated_tokens"], cell["Q"]*cell["G"])
                self.assertEqual(cell["query_totals"]["decode_steps"], cell["Q"]*(cell["G"]-1))
                result = output/f'queries_32_{cell["arm"]}_{cell["tier"]}_g{cell["G"]}.jsonl'
                rows = [json.loads(line) for line in result.read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                self.assertAlmostEqual(cell["query_totals"]["total_s"], sum(r["total_s"] for r in rows[:cell["Q"]]))
                self.assertAlmostEqual(cell["end_to_end_total_s"], cell["write"]["write_total_s"]+
                    cell["startup"]["startup_load_s"]+cell["query_totals"]["total_s"])
                self.assertFalse(cell["source_preprocessing_charged"])
            self.assertEqual(json.loads((output/"COMPLETED.json").read_text())["status"], "complete")
            self.assertFalse(json.loads((output/"COMPLETED.json").read_text())["hardware"]["timing_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
