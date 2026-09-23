"""CPU-only tests for matched-set gains and four-arm SFT completeness."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from aggregate_sft import ARMS, aggregate, load_evaluation, read_training_log, summarize


def record(i, score=0.0, ce=2.0, tokens=3):
    return {"id": f"q{i}", "document_id": f"paper{i // 2}", "source": "qasper:train", "split": "dev",
            "question": f"question {i}", "references": [f"answer {i}"], "selected_chunk_indices": [0, 2],
            "selected_context_tokens": 1024, "original_context_tokens": 2048, "prompt_tokens": 20,
            "answer_truncated": False, "exact_match": score, "token_f1": score,
            "answer_ce": ce, "answer_ce_sum": ce * tokens, "answer_ce_tokens": tokens}


def evaluation(records):
    result = summarize(records)
    result.update(score_scale="0-to-1", protocol={"decoding": "greedy", "max_new_tokens": 128,
        "eos_token_ids": [2], "enable_thinking": False, "answer_ce": True, "ce_reference_index": 0,
        "ce_targets": "prepared-answer-ids-including-eos-only-when-not-truncated", "hardware_timing": "not-collected"})
    return {"summary": result, "records": records}


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    for arm, (mode, ratio) in ARMS.items():
        directory = root / arm
        directory.mkdir()
        recipe = {"mode": mode, "ratio": ratio, "model": "same-base", "init_adapter_sha256": "init",
            "train_sha256": "train", "dev_sha256": "dev", "objective": "answer-only CE", "selection": "question-only BM25",
            "seed": 42, "steps": 2, "grad_accum": 4, "chunk_size": 512, "max_chunks": 7,
            "max_question_tokens": 256, "max_answer_tokens": 128, "max_new_tokens": 128,
            "eval_every": 1, "eval_limit": 2, "final_eval_limit": 4, "train_examples": 8, "dev_examples": 4}
        dump(directory / "metadata.json", {"recipe": recipe, "trainable_parameters": 10, "gpu": "3090"})
        rows = [{"step": step, "cursor": step * 4, "raw_tokens": step * 5000, "target_tokens": step * 12,
                 "loss": 1.0, "training_seconds": step * 4} for step in (1, 2)]
        (directory / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        dump(directory / "status.json", {"phase": "complete", "complete": True, "step": 2, "target_steps": 2,
            "training_examples_processed": 8, "raw_tokens": 10000, "target_tokens": 24})
        dump(directory / "eval_step0.json", evaluation([record(0), record(1)]))
        dump(directory / "eval_step1.json", evaluation([record(0, 1), record(1, 0)]))
        dump(directory / "eval_step2.json", evaluation([record(0, 1), record(1, 1), record(2, 0), record(3, 0)]))


class SFTAggregationTests(unittest.TestCase):
    def test_full_final_mean_is_not_used_as_unpaired_gain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            result = aggregate(root)
            self.assertTrue(result["complete"])
            self.assertEqual(result["arms"]["beacon4"]["evaluations"]["2"]["metrics"]["token_f1"], 0.5)
            paired = result["paired_initial_to_final_fixed_ids"]["beacon4"]
            self.assertEqual(paired["delta_token_f1_pp"], 100.0)
            self.assertEqual((paired["examples"], paired["documents"], paired["full_final_examples"]), (2, 1, 4))

    def test_ce_uses_token_weighting_and_validates_persisted_summary(self):
        rows = [record(0, ce=1, tokens=1), record(1, ce=3, tokens=3)]
        self.assertEqual(summarize(rows)["answer_ce"], 2.5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eval.json"
            value = evaluation(rows)
            dump(path, value)
            self.assertEqual(load_evaluation(path, 2)["metrics"]["answer_ce"], 2.5)
            value["summary"]["answer_ce"] = 2.0
            dump(path, value)
            with self.assertRaisesRegex(ValueError, "Summary does not match"):
                load_evaluation(path, 2)

    def test_same_count_different_ids_prevents_four_arm_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            dump(root / "beacon8/eval_step2.json", evaluation([record(0), record(1), record(2), record(9)]))
            result = aggregate(root)
            self.assertFalse(result["complete"])
            self.assertFalse(result["checkpoints"][-1]["comparable"])
            self.assertEqual(result["checkpoints"][-1]["deltas_vs_comem"], {})

    def test_changed_retrieval_blocks_paired_gain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "beacon4/eval_step2.json"
            value = json.loads(path.read_text())
            value["records"][0]["selected_chunk_indices"] = [1, 2]
            dump(path, value)
            result = aggregate(root)
            self.assertFalse(result["complete"])
            self.assertFalse(result["paired_initial_to_final_fixed_ids"]["beacon4"]["comparable"])

    def test_raw_token_budget_difference_cannot_be_called_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "pool4/train.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["raw_tokens"] += 1
            path.write_text("\n".join(json.dumps(row) for row in rows))
            status_path = root / "pool4/status.json"
            status = json.loads(status_path.read_text())
            status["raw_tokens"] += 1
            dump(status_path, status)
            result = aggregate(root)
            self.assertFalse(result["complete"])
            self.assertIn("Training raw/target token or example budgets differ", result["checkpoints"][-1]["issues"])

    def test_missing_final_eval_overrides_complete_training_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            (root / "pool4/eval_step2.json").unlink()
            result = aggregate(root)
            self.assertFalse(result["complete"])
            self.assertEqual(result["arms"]["pool4"]["state"], "running")
            self.assertEqual(result["arms"]["pool4"]["missing_evaluation_steps"], [2])

    def test_resume_invalidates_superseded_tail_and_partial_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            def row(step):
                return {"step": step, "cursor": step * 4, "raw_tokens": step * 100, "target_tokens": step * 8, "loss": 1.0}
            path.write_text("\n".join(json.dumps(row(step)) for step in (1, 2, 3, 4, 3)) + '\n{"step":', encoding="utf-8")
            steps, warnings = read_training_log(path)
            self.assertEqual(set(steps), {1, 2, 3})
            self.assertEqual(len(warnings), 2)

    def test_recipe_data_change_and_duplicate_eval_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / "beacon4/metadata.json"
            metadata = json.loads(path.read_text())
            metadata["recipe"]["dev_sha256"] = "different"
            dump(path, metadata)
            self.assertFalse(aggregate(root)["complete"])
            dump(root / "beacon8/eval_step2.json", evaluation([record(0), record(0), record(1), record(2)]))
            self.assertEqual(aggregate(root)["arms"]["beacon8"]["state"], "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
