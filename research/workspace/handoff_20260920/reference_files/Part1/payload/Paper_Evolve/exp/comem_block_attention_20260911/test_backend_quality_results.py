"""CPU-only fixtures; no models, GPUs, SSH, or real result directories."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import backend_quality_results as quality


def sources():
    return {"backend_quality_results.py": hashlib.sha256(Path(quality.__file__).read_bytes()).hexdigest(),
            "evaluate_backend_quality.py": "b" * 64}


def record(identity, ordinal, branch, prediction="red", references=None, **updates):
    refs = ["red"] if references is None else references
    row = {"schema": quality.RECORD_SCHEMA, "run_id": "test-run", "recipe_sha256": "a" * 64,
        "source_sha256": sources(), "ordinal": ordinal, "id": identity, "document_id": "doc:" + identity,
        "source": "allenai/qasper:train:v0.3", "branch": branch, "status": "complete",
        "prediction": prediction, "references": refs, "generated_ids": [8, 99], "finish_reason": "eos",
        "eos_token_id": 99, "stop_token_ids": [99, 100], "max_new_tokens": 128,
        "selected_chunk_indices": [5, 12], "prompt_ids": [1, 2, 3], "probe_indices": [1, 2],
        "candidate_tokens": 16, "question": "Which color?", "shared_writer_hj_unchanged": True,
        "formal_inference_timing": False, "formal_inference_memory": False,
        "route_stats": {"selected_indices": [0, 1], "candidate_blocks": 2, "selected_blocks": 2,
                        "candidate_tokens": 16, "selected_tokens": 16, "block_scores": [0.2, 0.8]}}
    row.update(quality.score_prediction(prediction, refs))
    row.update(updates)
    return row


def aggregate(rows, ids=("q0",)):
    return quality.aggregate_paired(rows, ids, run_id="test-run", recipe_sha256="a" * 64,
                                    source_sha256=sources())


def complete_fixture():
    ids = [f"q{i}" for i in range(8)]
    recipe = {"mode": "smoke", "ordered_ids": ids, "seed": 42, "max_new_tokens": 128,
        "j": 12, "m": 16, "probe_mode": "dense", "retain_ratio": 1., "rank": 32, "alpha": 32.,
        "branches": list(quality.BRANCHES), "decoding": "independent-free-greedy-natural-eos",
        "teacher_forced_ce": False, "model": "fixture-only", "init_adapter_sha256": "c" * 64,
        "train_sha256": "d" * 64, "dev_sha256": "e" * 64}
    recipe_sha = quality.canonical_hash(recipe)
    rows = [record(identity, ordinal, branch, recipe_sha256=recipe_sha)
            for ordinal, identity in enumerate(ids) for branch in quality.BRANCHES]
    summary = quality.aggregate_paired(rows, ids, run_id="test-run", recipe_sha256=recipe_sha,
                                       source_sha256=sources())
    profiles = {}
    for branch in quality.BRANCHES:
        profiles[branch] = {"status": "complete", "prefix_equal": True,
            "normal_generation_prefix": [8, 99], "profile_generated_ids": [8, 99],
            "formal_inference_timing": False, "formal_inference_memory": False,
            "profile": {"status": "complete", "timing_eligible": False,
                "backend_operator_evidence": [{"name": "aten::_scaled_dot_product_attention_math", "count": 36}],
                "metadata": {"run_id": "test-run", "id": ids[0], "branch": branch,
                             "recipe_sha256": recipe_sha}}}
    summary["profiles"] = profiles
    return {"metadata": {"recipe": recipe, "recipe_sha256": recipe_sha, "run_id": "test-run",
            "source_sha256": sources(), "formal_inference_timing": False, "formal_inference_memory": False},
        "status": {"status": "complete", "run_id": "test-run", "records_completed": 16},
        "summary": summary, "records": rows}


class AggregateTests(unittest.TestCase):
    def test_scoring_empty_punctuation_articles_multiple_refs(self):
        self.assertEqual(quality.score_prediction("The RED!", ["blue", "red"]),
                         {"token_f1": 1., "exact_match": 1.})
        self.assertEqual(quality.score_prediction("", [""]), {"token_f1": 1., "exact_match": 1.})
        self.assertAlmostEqual(quality.score_prediction("red red blue", ["red blue"])["token_f1"], .8)

    def test_valid_matched_macro_scale_and_wins(self):
        rows = [record("q0", 0, "decode_v2", "wrong"), record("q0", 0, "backend_v3"),
                record("q1", 1, "decode_v2"), record("q1", 1, "backend_v3", "red blue")]
        result = aggregate(rows, ["q0", "q1"])
        self.assertEqual(result["status"], "complete")
        metrics = result["metrics"]
        self.assertEqual(metrics["branches"]["decode_v2"]["token_f1"], .5)
        self.assertAlmostEqual(metrics["branches"]["backend_v3"]["token_f1_percent"], 100 * 5 / 6)
        self.assertAlmostEqual(metrics["paired"]["token_f1"]["mean_delta"], 1 / 3)
        self.assertEqual([metrics["paired"]["token_f1"][key] for key in ("wins", "ties", "losses")], [1, 0, 1])

    def test_missing_and_failure_do_not_produce_full_metrics(self):
        self.assertEqual(aggregate([])["status"], "pending")
        partial = aggregate([record("q0", 0, "decode_v2")])
        self.assertEqual(partial["status"], "partial")
        self.assertIsNone(partial["metrics"])
        failed = record("q0", 0, "backend_v3", status="failed", error={"type": "OutOfMemoryError"})
        for field in ("prediction", "generated_ids", "token_f1", "exact_match", "stop_token_ids", "route_stats"):
            failed.pop(field)
        result = aggregate([record("q0", 0, "decode_v2"), failed])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["complete"])
        self.assertIsNone(result["metrics"])

    def test_identity_schema_source_recipe_rejected(self):
        for field, wrong in (("schema", "old"), ("run_id", "old-run"), ("recipe_sha256", "c" * 64),
                             ("source_sha256", {"old.py": "c" * 64}), ("ordinal", 1), ("id", "old")):
            with self.subTest(field=field):
                row = record("q0", 0, "backend_v3", **{field: wrong}) if field not in ("id", "ordinal") else record("q0", 0, "backend_v3")
                row[field] = wrong
                self.assertEqual(aggregate([record("q0", 0, "decode_v2"), row])["status"], "invalid")

    def test_nonfinite_wrong_score_and_percent_scores_rejected(self):
        for field, value in (("token_f1", float("nan")), ("token_f1", float("inf")),
                             ("token_f1", 100.), ("token_f1", .5), ("exact_match", .5)):
            with self.subTest(value=value):
                rows = [record("q0", 0, branch) for branch in quality.BRANCHES]
                rows[1][field] = value
                self.assertEqual(aggregate(rows)["status"], "invalid")

    def test_duplicate_and_out_of_order_rejected(self):
        row = record("q0", 0, "decode_v2")
        self.assertEqual(aggregate([row, row])["status"], "invalid")
        self.assertEqual(aggregate([record("q1", 1, "decode_v2"), row], ["q0", "q1"])["status"], "invalid")

    def test_input_mismatch_rejected_route_changes_observed(self):
        rows = [record("q0", 0, branch) for branch in quality.BRANCHES]
        rows[1]["route_stats"].update(selected_indices=[1], selected_blocks=1, selected_tokens=8,
                                        block_scores=[.1, .9])
        result = aggregate(rows)
        self.assertTrue(result["complete"])
        self.assertEqual(result["metrics"]["paired"]["route_selected_indices_equal"]["different"], 1)
        self.assertAlmostEqual(result["metrics"]["paired"]["block_scores_max_abs_difference"], .1)
        rows[1]["prompt_ids"] = [1, 2, 4]
        self.assertEqual(aggregate(rows)["status"], "invalid")

    def test_generated_prediction_and_finish_are_separate(self):
        rows = [record("q0", 0, branch) for branch in quality.BRANCHES]
        rows[1].update(generated_ids=[7] * 128, finish_reason="max_new_tokens", eos_token_id=None)
        result = aggregate(rows)
        self.assertTrue(result["complete"])
        paired = result["metrics"]["paired"]
        self.assertEqual(paired["prediction_equal"]["equal"], 1)
        self.assertEqual(paired["generated_ids_equal"]["different"], 1)
        self.assertEqual(result["metrics"]["branches"]["backend_v3"]["finish_reasons"]["max_new_tokens"], 1)

    def test_malformed_stopping_rejected(self):
        for change in ({"generated_ids": [99, 8, 99]}, {"eos_token_id": 100},
                       {"generated_ids": [8], "finish_reason": "max_new_tokens", "eos_token_id": None},
                       {"generated_ids": []}):
            rows = [record("q0", 0, branch) for branch in quality.BRANCHES]
            rows[1].update(change)
            self.assertEqual(aggregate(rows)["status"], "invalid")


class CompletionTests(unittest.TestCase):
    def test_valid_object_and_directory_recompute(self):
        value = complete_fixture()
        expected = value["metadata"]["recipe"]
        self.assertTrue(quality.validate_complete_result(value, expected, expected_source_sha256=sources()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for name in ("metadata", "status", "summary"):
                (path / (name + ".json")).write_text(json.dumps(value[name]), encoding="utf-8")
            (path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in value["records"]), encoding="utf-8")
            self.assertTrue(quality.validate_complete_result(path, expected))
            (path / "records.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                quality.validate_complete_result(path, expected)

    def test_status_failed_rejects_even_complete_records(self):
        value = complete_fixture()
        value["status"]["status"] = "failed"
        with self.assertRaises(ValueError):
            quality.validate_complete_result(value, value["metadata"]["recipe"])

    def test_modified_summary_or_recipe_or_sources_rejected(self):
        base = complete_fixture()
        for target in ("summary", "recipe", "source"):
            value = copy.deepcopy(base)
            if target == "summary":
                value["summary"]["metrics"]["branches"]["decode_v2"]["token_f1"] = .5
            elif target == "recipe":
                value["metadata"]["recipe"]["model"] = "changed"
            else:
                value["metadata"]["source_sha256"]["backend_quality_results.py"] = "f" * 64
            with self.assertRaises(ValueError):
                quality.validate_complete_result(value, {"mode": "smoke", "ordered_ids": base["metadata"]["recipe"]["ordered_ids"]})

    def test_missing_or_wrong_profile_rejected(self):
        for target in ("missing", "prefix", "source_run", "no_backend", "formal_timing"):
            value = complete_fixture()
            profiles = value["summary"]["profiles"]
            if target == "missing":
                profiles.pop("backend_v3")
            elif target == "prefix":
                profiles["backend_v3"]["profile_generated_ids"] = [0]
            elif target == "source_run":
                profiles["backend_v3"]["profile"]["metadata"]["run_id"] = "old"
            elif target == "no_backend":
                profiles["backend_v3"]["profile"]["backend_operator_evidence"] = []
            else:
                profiles["backend_v3"]["profile"]["timing_eligible"] = True
            with self.assertRaises(ValueError):
                quality.validate_complete_result(value, value["metadata"]["recipe"])

    def test_missing_files_invalid_expected_recipe_and_pinned_source(self):
        value = complete_fixture()
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            quality.validate_complete_result(temp, value["metadata"]["recipe"])
        with self.assertRaises(ValueError):
            quality.validate_complete_result(value, {})
        with self.assertRaises(ValueError):
            quality.validate_complete_result(value, value["metadata"]["recipe"], expected_source_sha256={"old.py": "d" * 64})


if __name__ == "__main__":
    unittest.main()
