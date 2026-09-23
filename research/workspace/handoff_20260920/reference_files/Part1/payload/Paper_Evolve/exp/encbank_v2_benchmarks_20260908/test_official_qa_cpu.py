"""CPU tests for official scoring, complete-query packs and resumable QA outputs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import official_qa_driver as qa
from reader_adapter import ARMS, GenerationCache, make_reader_factory
from test_adapter_cpu import tiny_model, TinyTokenizer

torch.set_num_threads(1)


class ChatTokenizer(TinyTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if tokenize or not add_generation_prompt or enable_thinking:
            raise AssertionError("The controlled nonthinking generation template changed")
        return "USER\n" + messages[0]["content"] + "\nASSISTANT\n"


class OfficialQATests(unittest.TestCase):
    def test_official_metrics_and_invalid_adversarial_output(self):
        self.assertEqual(qa.score_prediction("", {"answers": [""]})[0], 0)
        self.assertEqual(qa.score_prediction("the cat", {"answers": ["cat"]})[0], 1)
        self.assertEqual(qa.score_prediction("running", {"category": 4, "answers": ["runs"]})[0], 1)
        self.assertEqual(qa.score_prediction("red, blue", {"category": 1, "answers": ["blue, red"]})[0], 1)
        self.assertEqual(qa.score_prediction("red", {"category": 3, "answers": ["red; alternative"]})[0], 1)
        row = {"category": 5, "answers": ["No information available"],
               "option_mapping": {"a": "distractor", "b": "No information available"}}
        for pred in ("", "(a)", "(a) or (b)", "nonsense"):
            self.assertEqual(qa.score_prediction(pred, row)[0], 0, pred)
        for pred in ("(b)", "b", "Answer: b", "No information available"):
            self.assertEqual(qa.score_prediction(pred, row)[0], 1, pred)

    def test_full_locomo_schema_captions_dates_and_stable_options(self):
        parser = qa.build_parser()
        args = parser.parse_args(["--benchmark", "locomo", "--model", "test", "--out", "unused"])
        args.categories = [1, 2, 3, 4, 5]
        all_rows = list(qa.locomo_samples(args))
        self.assertEqual(len(all_rows), 1986)
        self.assertEqual({c: sum(r["category"] == c for r in all_rows) for c in args.categories},
                         {1: 282, 2: 321, 3: 96, 4: 841, 5: 446})
        self.assertTrue(any(" and shared " in r["marked_prompt"] for r in all_rows))
        temporal = next(r for r in all_rows if r["category"] == 2)
        self.assertIn("Use DATE of CONVERSATION", temporal["question"])
        args.categories = [5]
        args.num_shards, args.shard_index = 3, 1
        subset = list(qa.locomo_samples(args))
        mappings = {r["id"]: r["option_mapping"] for r in all_rows}
        self.assertTrue(all(r["option_mapping"] == mappings[r["id"]] for r in subset))

    def test_explicit_pack_preserves_query_longer_than_chunk_for_every_arm(self):
        model, _ = tiny_model()
        tok = ChatTokenizer()
        sample = {"marked_prompt": "z" * 97 + qa.BOUNDARY + "complete question " * 6,
                  "question": "complete question"}
        ids, n_context, selected, pack = qa.tokenize_pack(tok, sample, 16, "recency", 2)
        self.assertGreater(pack["query_tokens"], 16)
        self.assertEqual(pack["context_tokens"], 102)
        self.assertEqual(selected, [5, 6])
        with tempfile.TemporaryDirectory() as temp:
            for arm in ARMS:
                with self.subTest(arm=arm):
                    cache = GenerationCache(Path(temp) / (arm + ".db"))
                    reader = make_reader_factory(arm, cache, explicit_pack=True)(model, 2, tokenizer=tok)
                    pred = reader.generate_from_ids(ids, context_token_count=n_context,
                        selected_indices=selected, chunk_size=16, max_new_tokens=3)
                    self.assertIsInstance(pred, str)
                    self.assertEqual(reader.generate_from_ids(ids, context_token_count=n_context,
                        selected_indices=selected, chunk_size=16, max_new_tokens=3), pred)
                    self.assertEqual(cache.hits, 1)
                    cache.close()

    def test_qasper_main_outputs_complete_and_resume_without_model_load(self):
        model, _ = tiny_model()
        tok = ChatTokenizer()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            data = Path(temp) / "data"
            data.mkdir()
            (data / "qasper.jsonl").write_text(json.dumps({"_id": "official-id-1", "context": "A study.",
                "input": "What was studied?", "answers": ["birds"]}) + "\n", encoding="utf-8")
            out = Path(temp) / "out"
            argv = ["--benchmark", "longbench", "--tasks", "qasper", "--model", "cpu-test-model",
                "--out", str(out), "--data-dir", str(data), "--device", "cpu", "--dtype", "float32",
                "--j", "1", "--chunk-size", "32", "--topk", "2"]
            with patch.object(qa, "load_backbone", return_value=(model, tok)):
                self.assertEqual(qa.main(argv), 0)
            done = json.loads((out / "COMPLETED.json").read_text())
            rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(done["n"], 1)
            self.assertEqual(rows[0]["id"], "official-id-1")
            self.assertGreater(rows[0]["pack"]["query_tokens"], 32)
            self.assertEqual(rows[0]["status"], "ok")
            with patch.object(qa, "load_backbone", side_effect=AssertionError("loaded complete model")):
                self.assertEqual(qa.main(argv), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
