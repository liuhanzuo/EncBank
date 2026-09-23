"""Small stdlib-only fixtures: validation failures and statistical semantics."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import aggregate_sparse as agg


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_row(split, number, answer_length=1):
    return dict(id=f"{split}-{number}", document_id=f"doc-{split}-{number}", split=split,
                source="allenai/qasper:train:v0.3", question=f"What color {number}?",
                references=["blue" if number == 0 else "red green"], prompt_ids=[10, 11, 12],
                answer_ids=[20] * answer_length, probe_indices=[1, 2], answer_truncated=False,
                document_chunks=[[3] * (2 + number) for _ in range(4)])


def make_route(row, recipe):
    lengths = list(map(len, row["document_chunks"]))
    candidate = sum(lengths)
    selected = list(range(4)) if recipe["retain_ratio"] == 1 else [0, 2]
    kept = sum(lengths[i] for i in selected)
    by_layer = {str(layer): (candidate if layer < 16 else kept) + 1 for layer in range(12, 36)}
    return dict(selected_indices=selected, candidate_blocks=4, selected_blocks=len(selected),
                candidate_tokens=candidate, selected_tokens=kept, sink_tokens=1,
                selection_source="all-blocks" if recipe["retain_ratio"] == 1 else "prompt-attention-mass",
                probe_indices=row["probe_indices"], probe_rows_fallback=False,
                probe_mode=recipe["probe_mode"], resume_j=12, fusion_layer=16,
                target_retain_ratio=recipe["retain_ratio"], actual_retain_ratio=kept / candidate,
                token_budget=int(candidate * recipe["retain_ratio"]), budget_overflow_tokens=0,
                document_kv_tokens_by_layer=by_layer, document_kv_bytes=32 * sum(by_layer.values()),
                unselected_late_kv_tokens=0)


def make_evaluation(rows, recipe, initial=False):
    ordered = sorted(rows, key=lambda row: hashlib.sha256(f"42:{row['id']}".encode()).hexdigest())
    if initial:
        ordered = ordered[:1]
    records = []
    for row in ordered:
        prediction = "wrong" if initial else ("blue" if row["id"].endswith("0") else "red")
        record = dict(id=row["id"], document_id=row["document_id"], references=row["references"],
                      prediction=prediction, source=row["source"], probe_indices=row["probe_indices"],
                      token_f1=max(agg.token_f1(prediction, ref) for ref in row["references"]),
                      exact_match=max(float(agg.normalize(prediction) == agg.normalize(ref)) for ref in row["references"]),
                      answer_ce=1. if row["id"].endswith("0") else 3., answer_ce_tokens=len(row["answer_ids"]),
                      generated_ids=[2], finish_reason="eos", candidate_tokens=sum(map(len, row["document_chunks"])),
                      route_stats=make_route(row, recipe), ce_route_stats=make_route(row, recipe))
        records.append(record)
    summary = dict(examples=len(records), token_f1=agg.mean(row["token_f1"] for row in records),
                   exact_match=agg.mean(row["exact_match"] for row in records),
                   answer_ce=sum(row["answer_ce"] * row["answer_ce_tokens"] for row in records) / sum(row["answer_ce_tokens"] for row in records),
                   score_scale="0-to-1", decoding="greedy-natural-eos", max_new_tokens=2, formal_inference_timing=False)
    return dict(summary=summary, records=records)


def fixture(root):
    data = root / "data"
    data.mkdir()
    train, dev = [make_row("train", 0), make_row("train", 1, 3)], [make_row("dev", 0), make_row("dev", 1, 3)]
    for split, rows in (("train", train), ("dev", dev)):
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    order = list(range(len(train)))
    random.Random(42).shuffle(order)
    for arm in agg.ARMS:
        directory = root / "train" / arm
        recipe = dict(arm=arm, j=12, m=16, rho=.5, retain_ratio=1. if arm in ("A", "D0") else .5,
                      probe_mode="block" if arm in ("A", "B") else "dense", smoke=False,
                      steps=2, grad_accum=1, seed=42, lr=2e-5, max_new_tokens=2, eval_limit=1, final_eval_limit=100,
                      train=str(data / "train.jsonl"), dev=str(data / "dev.jsonl"), model="/remote/models/Qwen3-8B",
                      train_sha256=agg.sha256(data / "train.jsonl"), dev_sha256=agg.sha256(data / "dev.jsonl"),
                      init_adapter_sha256="same-adapter", reader_sha256="same-reader",
                      train_ids=[row["id"] for row in train], dev_ids=[row["id"] for row in dev])
        metadata = dict(recipe=recipe, model=dict(num_hidden_layers=36, num_key_value_heads=2,
                        hidden_size=16, num_attention_heads=4, vocab_size=100), gpu="NVIDIA GeForce RTX 3090",
                        formal_inference_timing=False, formal_inference_memory=False)
        logs, raw, targets = [], 0, 0
        for step, index in enumerate(order, 1):
            row = train[index]
            raw += sum(map(len, row["document_chunks"])) + len(row["prompt_ids"]) + len(row["answer_ids"])
            targets += len(row["answer_ids"])
            logs.append(dict(step=step, cursor=step, loss=2., grad_norm=.3, raw_tokens=raw, target_tokens=targets,
                             routes=[dict(id=row["id"])], seconds=1.))
        state = dict(arm=arm, complete=True, phase="complete", step=2, target_steps=2, cursor=2,
                     raw_tokens=raw, target_tokens=targets, checkpoint="/remote/output/last.pt",
                     gradient_check=dict(reader_norm=.2, frozen_base_has_grad=False,
                                         trainable_tensors_with_grad=336, total_trainable_tensors=336))
        publish(directory / "metadata.json", metadata)
        publish(directory / "status.json", state)
        publish(directory / "eval_step2.json", make_evaluation(dev, recipe))
        publish(directory / "eval_step0.json", make_evaluation(dev, recipe, initial=True))
        (directory / "last.pt").write_bytes(b"fixture-not-a-checkpoint-never-loaded")
        (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs), encoding="utf-8")
    return data


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sparse-aggregate-test-")
        self.root = Path(self.temporary.name)
        self.data = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def change(self, name, mutate, arm="B"):
        path = self.root / "train" / arm / name
        value = json.loads(path.read_text())
        mutate(value)
        publish(path, value)

    def assert_invalid(self):
        result = agg.aggregate(self.root)
        self.assertEqual(result["status"], "invalid", result)
        self.assertFalse(result["comparison_valid"])
        self.assertNotIn("final_vs_D0", result)

    def test_valid_matched_means_scales_and_tensor_accounting(self):
        result = agg.aggregate(self.root, include_pairs=True)
        self.assertEqual(result["status"], "complete", result)
        self.assertTrue(result["comparison_valid"])
        final = result["arms"]["B"]["final"]
        self.assertAlmostEqual(final["token_f1"], 5 / 6)
        self.assertEqual(final["exact_match"], .5)
        self.assertEqual(final["answer_ce"], 2.5)  # Token-weighted, not the example mean 2.0.
        self.assertEqual(final["mean_candidate_tokens"], 10)
        self.assertEqual(final["mean_selected_tokens"], 5)
        self.assertEqual(final["mean_tensor_document_kv_bytes"], 5248)
        self.assertEqual(final["mean_analytic_document_kv_bytes"], 5248)
        self.assertEqual(result["arms"]["D0"]["final"]["mean_tensor_document_kv_bytes"], 8448)
        self.assertAlmostEqual(final["pooled_document_kv_reduction_fraction"], 1 - 5248 / 8448)
        self.assertIn("83.33", agg.render_report(result))
        self.assertEqual(result["arms"]["B"]["source_gpu"], "NVIDIA GeForce RTX 3090")
        self.assertFalse(result["formal_inference_memory"])
        self.assertEqual(len(result["final_vs_D0"]["B"]["records"]), 2)

    def test_initial_delta_uses_only_common_subset(self):
        result = agg.aggregate(self.root)
        change = result["arms"]["B"]["initial_to_final_matched"]
        self.assertEqual(change["examples"], 1)
        self.assertEqual(change["token_f1_delta"], change["after_on_identical_ids"]["token_f1"])
        self.assertNotEqual(change["token_f1_delta"], result["arms"]["B"]["final"]["token_f1"])

    def test_empty_queue_and_missing_train_directory_remain_pending(self):
        empty = self.root / "empty-queue"
        result = agg.aggregate(empty)
        self.assertEqual(result["status"], "pending")
        self.assertTrue(all(value["status"] == "pending" for value in result["arms"].values()))
        agg.write_outputs(result, empty / "aggregate")
        self.assertTrue((empty / "aggregate" / "REPORT.md").is_file())
        self.assertNotIn("0.00", agg.render_report(result))

    def test_step_count_without_final_evaluation_is_not_completion(self):
        (self.root / "train" / "B" / "eval_step2.json").unlink()
        result = agg.aggregate(self.root)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["arms"]["B"]["status"], "pending")
        self.assertFalse(result["comparison_valid"])

    def test_failure_and_nonfinite_block_comparison(self):
        for mutate in (lambda value: value.update(phase="failed", error="OOM"),
                       lambda value: value["gradient_check"].update(reader_norm=float("nan")),
                       lambda value: value["gradient_check"].update(frozen_base_has_grad=True)):
            original = json.loads((self.root / "train" / "B" / "status.json").read_text())
            self.change("status.json", mutate)
            self.assert_invalid()
            publish(self.root / "train" / "B" / "status.json", original)

    def test_id_reference_score_and_tensor_corruption_rejected(self):
        mutations = (lambda value: value["records"][0].update(id="wrong-id"),
                     lambda value: value["records"][0].update(references=["wrong reference"]),
                     lambda value: value["records"][0].update(question="different question"),
                     lambda value: value["records"][0].update(answer_ce=float("inf")),
                     lambda value: value["records"][0].update(token_f1=50.),
                     lambda value: value["summary"].update(score_scale="percent"),
                     lambda value: value["summary"].update(answer_ce=2.),
                     lambda value: value["records"][0]["route_stats"].update(document_kv_bytes=1))
        path = self.root / "train" / "B" / "eval_step2.json"
        original = json.loads(path.read_text())
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            publish(path, changed)
            self.assert_invalid()
        publish(path, original)

    def test_mismatched_training_recipe_and_consumed_tokens(self):
        self.change("metadata.json", lambda value: value["recipe"].update(lr=3e-5))
        self.assert_invalid()
        self.change("metadata.json", lambda value: value["recipe"].update(lr=2e-5))
        self.change("status.json", lambda value: value.update(target_tokens=999))
        self.assert_invalid()

    def test_metadata_only_does_not_require_or_fabricate_weights(self):
        (self.root / "train" / "B" / "last.pt").unlink()
        self.assertEqual(agg.aggregate(self.root)["status"], "pending")
        result = agg.aggregate(self.root, metadata_only=True)
        self.assertEqual(result["status"], "metadata_only")
        self.assertFalse(result["comparison_valid"])
        self.assertFalse(result["arms"]["B"]["checkpoint_present"])
        self.assertEqual(result["arms"]["B"]["final"]["examples"], 2)

    def test_remote_data_prefix_remapping_and_explicit_data_directory(self):
        for arm in agg.ARMS:
            self.change("metadata.json", lambda value: value["recipe"].update(
                train="/remote/data/train.jsonl", dev="/remote/data/dev.jsonl"), arm=arm)
        self.assertEqual(agg.aggregate(self.root)["status"], "pending")
        self.assertTrue(agg.aggregate(self.root, maps=[("/remote/data", self.data)])["comparison_valid"])
        self.assertTrue(agg.aggregate(self.root, data_dir=self.data)["comparison_valid"])
        self.assertEqual(agg.resolve_path("/remote/database/dev.jsonl", [("/remote/data", self.data)]), Path("/remote/database/dev.jsonl"))

    def test_changed_prepared_question_detected_by_consumed_file_hash(self):
        path = self.data / "dev.jsonl"
        path.write_text(path.read_text().replace("What color", "What animal"), encoding="utf-8")
        self.assert_invalid()


if __name__ == "__main__":
    unittest.main()
