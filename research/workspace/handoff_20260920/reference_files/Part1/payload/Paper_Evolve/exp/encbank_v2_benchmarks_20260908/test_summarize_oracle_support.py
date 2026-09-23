"""Protect scientific pairing and incomplete-result behavior without loading a model."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from summarize_oracle_support import paired_summary, natural_generation_key, verify_natural_cache


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = {"selected_indices": [0, 2], "read_pack_tokens": 100}
        self.oracle_pack = {"selected_indices": [1, 2], "read_pack_tokens": 100}
        self.sample = {"id": "a", "index": 0, "task": "hotpotqa", "question": "Question?",
                       "answers": ["Answer"], "max_new_tokens": 32}
        self.fixture = {"id": "a", "index": 0, "source_benchmark": "longbench",
            "source_task": "hotpotqa", "task": "hotpotqa/oracle", "sample": self.sample,
            "natural_pack": self.pack, "oracle_pack": self.oracle_pack,
            "input_ids_sha256": "input", "fixture_row_sha256": "fixture",
            "natural_all_required_visible": False, "read_pack_token_delta": 0}
        self.oracle_plan = {"arms": ["fix_all"], "jobs": [self.job("oracle_support", "hotpotqa/oracle")]}
        self.main_plan = {"jobs": [self.job("longbench", "hotpotqa")]}
        self.options = {"j": 12, "selector": "bm25", "topk": 12, "chunk_size": 512,
            "seed": 42, "dtype": "bfloat16", "attn_impl": "sdpa", "adapter": "", "model": "/model"}
        self.natural = dict(self.sample, pack=self.pack, pred="Answer", score=0.5, status="ok")
        self.oracle = dict(self.sample, task="hotpotqa/oracle", source_task="hotpotqa",
            source_benchmark="longbench", pack=self.oracle_pack, natural_pack=self.pack,
            input_ids_sha256="input", fixture_row_sha256="fixture", pred="Answer", score=1.0, status="ok")

    def job(self, benchmark, task, arm="fix_all"):
        return {"id": benchmark + "_" + arm, "benchmark": benchmark, "arm": arm,
            "layout": "official_qa", "relative_output": benchmark + "/" + arm,
            "cells": [{"key": task, "indices": [0], "expected_n": 1}]}

    def write(self, kind, row, *, complete=True, options=None):
        plan = self.oracle_plan if kind == "oracle" else self.main_plan
        job = plan["jobs"][0]
        target = self.root / kind / job["relative_output"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "run_config.json").write_text(json.dumps({"benchmark": job["benchmark"],
            "arm": job["arm"], "options": options or self.options}), encoding="utf-8")
        (target / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        if complete:
            (target / "COMPLETED.json").write_text("{}", encoding="utf-8")

    def summarize(self):
        return paired_summary([self.fixture], self.oracle_plan, self.main_plan,
            self.root / "oracle", self.root / "main")

    def test_complete_matching_control(self):
        self.write("oracle", self.oracle)
        self.write("main", self.natural)
        result = self.summarize()
        self.assertTrue(result["all_planned_pairs_complete"])
        self.assertEqual(result["cells"][0]["oracle_minus_natural_pp"], 50)

    def test_missing_natural_is_pending_not_zero(self):
        self.write("oracle", self.oracle)
        result = self.summarize()
        self.assertFalse(result["all_planned_pairs_complete"])
        self.assertIsNone(result["cells"][0]["natural_f1_percent"])
        self.assertIsNone(result["cells"][0]["oracle_minus_natural_pp"])

    def test_changed_pack_blocks_pair(self):
        self.write("oracle", self.oracle)
        self.write("main", dict(self.natural, pack=self.oracle_pack))
        result = self.summarize()
        self.assertTrue(result["diagnostics"])
        self.assertFalse(result["all_planned_pairs_complete"])
        self.assertEqual(result["cells"][0]["paired_observed_n"], 0)

    def test_incomplete_shard_is_never_scored(self):
        self.write("oracle", self.oracle, complete=False)
        self.write("main", self.natural)
        result = self.summarize()
        self.assertFalse(result["cells"][0]["oracle_complete"])
        self.assertIsNone(result["cells"][0]["oracle_f1_percent"])

    def test_unplanned_natural_control_is_explicit(self):
        self.write("oracle", self.oracle)
        self.main_plan["jobs"] = []
        result = self.summarize()
        self.assertTrue(result["all_planned_pairs_complete"])
        self.assertEqual(result["cells"][0]["status"], "oracle_only_no_natural_control_planned")
        self.assertFalse(result["cells"][0]["paired_complete"])

    def test_changed_model_blocks_pair(self):
        self.write("oracle", self.oracle)
        self.write("main", self.natural, options=dict(self.options, model="/different-model"))
        result = self.summarize()
        self.assertTrue(result["diagnostics"])
        self.assertFalse(result["all_planned_pairs_complete"])

    def test_changed_source_answer_blocks_pair(self):
        self.write("oracle", self.oracle)
        self.write("main", dict(self.natural, answers=["Another answer"]))
        result = self.summarize()
        self.assertTrue(result["diagnostics"])
        self.assertFalse(result["all_planned_pairs_complete"])

    def test_identical_pack_uses_complete_natural_without_gpu_result(self):
        self.fixture["oracle_pack"] = copy.deepcopy(self.pack)
        self.oracle_plan["jobs"] = []
        self.write("main", self.natural)
        result = self.summarize()
        self.assertTrue(result["all_planned_pairs_complete"])
        self.assertEqual(result["cells"][0]["oracle_derived_from_identical_natural_n"], 1)
        self.assertEqual(result["cells"][0]["oracle_minus_natural_pp"], 0)

    def test_changed_pack_cannot_be_derived(self):
        self.oracle_plan["jobs"] = []
        self.write("main", self.natural)
        result = self.summarize()
        self.assertFalse(result["all_planned_pairs_complete"])
        self.assertTrue(result["diagnostics"])

    def test_exact_generation_key_detects_changed_tokens(self):
        fixture = dict(self.fixture, context_key="context", query_ids=[9])
        contexts = {"context": {"context_ids": [1, 2, 3]}}
        key, count = natural_generation_key(fixture, contexts)
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE generations (key TEXT, prediction TEXT, n_tokens INTEGER)")
        db.execute("INSERT INTO generations VALUES (?,?,?)", (key, json.dumps(self.natural["pred"]), count))
        self.assertEqual(verify_natural_cache(db, fixture, self.natural, contexts), key)
        with self.assertRaises(ValueError):
            verify_natural_cache(db, dict(fixture, query_ids=[10]), self.natural, contexts)


if __name__ == "__main__":
    unittest.main()
