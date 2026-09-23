"""CPU-only contracts: real tiny Qwen3 readers plus interruption/resume of native drivers."""
import argparse
import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_driver
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from reader_adapter import GenerationCache, GenerationFailure, CachedReader, make_reader_factory, validate_options

torch.set_num_threads(1)


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text, add_special_tokens=True, return_tensors=None):
        ids = ([1] if add_special_tokens else []) + [3 + ord(c) % 80 for c in text]
        return torch.tensor([ids]) if return_tensors else ids
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if int(i) not in (1, 2))


def tiny_model():
    torch.manual_seed(19)
    config = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=8192)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval(), TinyTokenizer()


class AdapterTests(unittest.TestCase):
    def test_all_arms_match_direct_reader_and_cache_survives_reopen(self):
        from comem import CoMem
        from s15_ruler_lower import CoMemLower
        model, tok = tiny_model()
        ids = torch.tensor([[1] + list(range(3, 25))])
        kwargs = dict(chunk_size=8, max_new_tokens=3, selector="recency", topk=1)
        with tempfile.TemporaryDirectory() as temp:
            for arm in (a for a in run_driver.ARMS if a != "cacheblend16"):
                with self.subTest(arm=arm):
                    db = Path(temp) / (arm + ".sqlite3")
                    cache = GenerationCache(db)
                    constructed = []
                    wrapped = make_reader_factory(arm, cache, constructed.append)(model, 2, tokenizer=tok)
                    j = 0 if arm == "j0" else 3 if arm == "cbos" else 2
                    direct = CoMemLower(model, j, tokenizer=tok,
                        lower_layers=[] if arm == "fix_none" else None) if arm in {"fix_all", "fix_none", "cbos"} else CoMem(model, j, tokenizer=tok)
                    if arm not in {"fix_all", "fix_none", "cbos"}:
                        direct.write_sink = arm == "pub_sink"
                    expected = direct.generate_from_ids(ids, **kwargs)
                    self.assertEqual(wrapped.generate_from_ids(ids, **kwargs), expected)
                    self.assertEqual(constructed[0]["effective_j"], j)
                    cache.close()
                    cache = GenerationCache(db)
                    wrapped = make_reader_factory(arm, cache)(model, 2, tokenizer=tok)
                    self.assertEqual(wrapped.generate_from_ids(ids, **kwargs), expected)
                    self.assertEqual(cache.hits, 1)
                    self.assertEqual(cache.misses, 0)
                    cache.close()

    def test_constructor_and_generation_variants_are_rejected(self):
        model, tok = tiny_model()
        with tempfile.TemporaryDirectory() as temp:
            cache = GenerationCache(Path(temp) / "cache.db")
            factory = make_reader_factory("fix_all", cache)
            with self.assertRaisesRegex(ValueError, "cannot be discarded"):
                factory(model, 1, top_prepay_b=1, tokenizer=tok)
            with self.assertRaises(ValueError):
                factory(model, 1, block_diagonal=True, tokenizer=tok)
            reader = factory(model, 1, tokenizer=tok)
            with self.assertRaises(ValueError):
                reader.generate_from_ids(torch.tensor([[1, 3]]), sink_tokens="none")
            with self.assertRaises(ValueError):
                reader.generate_from_ids(torch.tensor([[1, 3]]), selector="reader_attn")
            cache.close()

    def test_oom_is_not_cached_or_converted_to_a_scored_prediction(self):
        class BrokenReader:
            def generate_from_ids(self, input_ids, *, max_new_tokens=2):
                raise RuntimeError("CUDA out of memory")
        with tempfile.TemporaryDirectory() as temp:
            cache = GenerationCache(Path(temp) / "cache.db")
            with self.assertRaises(GenerationFailure):
                cache.generate(BrokenReader(), torch.tensor([[1, 3]]), {})
            self.assertEqual(cache.db.execute("SELECT count(*) FROM generations").fetchone()[0], 0)
            cache.close()

    def test_native_longeval_interruption_resume_and_completed_skip(self):
        module = importlib.import_module("eval.longeval")
        original_class = module.CoMem
        model, tok = tiny_model()
        def prompt(_target, _tok, rng):
            label = str(rng.randrange(10000))
            return "a b c d " + label + "?", "123456", label, 1
        original_generate = CachedReader.generate_from_ids
        calls = 0
        def interrupt(reader, *a, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise GenerationFailure("intentional CPU test interruption")
            return original_generate(reader, *a, **kw)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "run"
            argv = ["--benchmark", "longeval", "--arm", "fix_all", "--out", str(output),
                    "--model", "cpu-test-model", "--j", "1", "--device", "cpu", "--dtype", "float32",
                    "--lengths", "1k", "--n", "2", "--selector", "recency", "--chunk_size", "8", "--max_new_tokens", "2"]
            with patch.object(module, "load_backbone", return_value=(model, tok)), patch.object(module, "build_lines_prompt", prompt):
                with patch.object(CachedReader, "generate_from_ids", interrupt), self.assertRaises(GenerationFailure):
                    run_driver.main(argv)
                self.assertFalse((output / "COMPLETED.json").exists())
                self.assertEqual(run_driver.main(argv), 0)
            result = json.loads((output / "COMPLETED.json").read_text())
            self.assertEqual(result["reused_generations"], 1)
            self.assertEqual(result["new_generations"], 1)
            self.assertEqual(result["files"][0]["records"], 2)
            completed_mtime = (output / "COMPLETED.json").stat().st_mtime_ns
            with patch.object(module, "load_backbone", side_effect=AssertionError("completed run loaded model")):
                self.assertEqual(run_driver.main(argv), 0)
            self.assertEqual((output / "COMPLETED.json").stat().st_mtime_ns, completed_mtime)
            with self.assertRaisesRegex(ValueError, "different options"):
                run_driver.main(argv + ["--n", "3"])
        self.assertIs(module.CoMem, original_class)

    def test_missing_longbench_cell_fails_without_completed_receipt(self):
        module = importlib.import_module("eval.longbench")
        model, tok = tiny_model()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            argv = ["--benchmark", "longbench", "--out", temp, "--model", "cpu-test-model", "--j", "1",
                    "--device", "cpu", "--dtype", "float32", "--tasks", "qasper", "--n", "1"]
            with patch.object(module, "load_backbone", return_value=(model, tok)), patch.object(module, "load_longbench_dataset", return_value={}):
                with self.assertRaisesRegex(ValueError, "no samples"):
                    run_driver.main(argv)
            self.assertFalse((Path(temp) / "COMPLETED.json").exists())

    def test_native_babilong_output_and_official_scoring(self):
        module = importlib.import_module("babilong_driver")
        model, tok = tiny_model()
        class FakeReader:
            def generate_from_ids(self, _ids, **kwargs):
                self.kwargs = kwargs
                return "kitchen"
        reader = FakeReader()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            argv = ["--benchmark", "babilong", "--out", temp, "--model", "cpu-test-model", "--j", "1",
                    "--device", "cpu", "--tasks", "qa1", "--lengths", "0k", "--n", "1"]
            with patch.object(module, "load_backbone", return_value=(model, tok)), patch.object(run_driver, "make_reader_factory", return_value=lambda *a, **kw: reader):
                self.assertEqual(run_driver.main(argv), 0)
            done = json.loads((Path(temp) / "COMPLETED.json").read_text())
            self.assertEqual(done["files"][0]["records"], 1)
            self.assertLessEqual(len(reader.kwargs["selected_indices"]), 4)
            self.assertIn("context_token_count", reader.kwargs)
            self.assertTrue((Path(done["output_dir"]) / "scores_official.json").exists())

    def test_infinitebench_explicit_query_and_official_score(self):
        module = importlib.import_module("infinitebench_driver")
        model, tok = tiny_model()
        class FakeReader:
            def generate_from_ids(self, ids, **kwargs):
                self.query_tokens = ids.numel() - kwargs["context_token_count"]
                return "a bird"
        reader = FakeReader()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            data = Path(temp) / "data"
            data.mkdir()
            (data / "longbook_qa_eng.jsonl").write_text(json.dumps({"id": "source-1",
                "input": "What animal was it?", "context": "A bird was there.", "answer": ["a bird"]}) + "\n")
            out = Path(temp) / "out"
            argv = ["--benchmark", "infinitebench", "--out", str(out), "--model", "cpu-test-model",
                "--j", "1", "--device", "cpu", "--tasks", "longbook_qa_eng", "--n", "1",
                "--data_dir", str(data), "--chunk_size", "8"]
            with patch.object(module, "load_backbone", return_value=(model, tok)), patch.object(run_driver,
                    "make_reader_factory", return_value=lambda *a, **kw: reader):
                self.assertEqual(run_driver.main(argv), 0)
            done = json.loads((out / "COMPLETED.json").read_text())
            records = [json.loads(s) for s in Path(done["files"][0]["file"]).read_text().splitlines()]
            self.assertEqual(records[0]["score"], 1)
            self.assertEqual(records[0]["id"], "source-1")
            self.assertGreater(reader.query_tokens, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
