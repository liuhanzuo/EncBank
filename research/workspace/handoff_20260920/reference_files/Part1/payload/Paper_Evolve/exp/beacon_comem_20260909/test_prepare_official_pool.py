"""Small CPU invariants for native-pool overlap and explicit failure handling."""
import unittest
import json
from pathlib import Path
import tempfile
from official_sft import ConversationFiltered
from prepare_official_pool import long_anchors, DisjointSet, component_split, classify_error, validate_pool_artifacts
from prepare_sft_data import BenchmarkExclusions


class PoolTests(unittest.TestCase):
    def test_winnowing_shared_span_is_offset_independent(self):
        shared = [f"shared{i}" for i in range(319)]
        first = [f"left{i}" for i in range(47)] + shared + [f"tail{i}" for i in range(22)]
        second = [f"right{i}" for i in range(113)] + shared + [f"other{i}" for i in range(31)]
        self.assertTrue(long_anchors(first) & long_anchors(second))
        self.assertEqual(long_anchors(first), long_anchors(first))

    def test_full_anchor_length_and_unrelated_text(self):
        self.assertFalse(long_anchors([f"x{i}" for i in range(255)]))
        self.assertFalse(long_anchors([f"x{i}" for i in range(500)]) & long_anchors([f"y{i}" for i in range(500)]))

    def test_transitive_groups_stay_together(self):
        groups = DisjointSet(5)
        groups.union(0, 1); groups.union(2, 3); groups.union(1, 2)
        self.assertEqual(len({groups.find(i) for i in range(4)}), 1)
        self.assertNotEqual(groups.find(0), groups.find(4))
        self.assertEqual(component_split("same", 42, .05), component_split("same", 42, .05))

    def test_reused_benchmark_match_handles_different_offsets(self):
        text = " ".join(f"word{i}" for i in range(180))
        index = BenchmarkExclusions([("bench", text)])
        self.assertEqual(index.match("", "prefix prefix " + text + " suffix")["kind"], "16_word_shingle_overlap")
        self.assertIsNone(index.match("", "unrelated words"))

    def test_failure_is_explicit_and_finely_classified(self):
        self.assertEqual(classify_error(ConversationFiltered("empty_assistant_target")), "empty_assistant_target")
        self.assertEqual(classify_error(ValueError("Explicit chat/reasoning control text needs a separate protocol")), "embedded_chat_control")
        self.assertEqual(classify_error(ValueError("Ambiguous boundary")), "ambiguous_assistant_token_boundary")
        self.assertEqual(classify_error(ValueError("unknown validation problem")), "validation_ValueError")

    def test_completed_metadata_reconciliation_and_bypass_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            processed, output = base / "processed", base / "prepared"
            processed.mkdir(); output.mkdir()
            def save(path, value):
                path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            save(processed / "conversations.jsonl", {"conversation_id": "c", "document_id": "d", "source": "s"})
            save(output / "document_overlap.jsonl", {"document_id": "d", "overlap_group_id": "overlap:g", "split": "train", "benchmark_excluded": False})
            save(output / "overlap_report.json", {"complete": True})
            save(output / "pool_report.json", {"complete": True, "eligible": 1, "input_conversations": 1, "tokenizer": {"tokenizer_fingerprint": "sha256:test"}})
            inventory = {"conversation_id": "c", "document_id": "d", "source": "s", "overlap_group_id": "overlap:g", "split": "train", "status": "valid", "benchmark_excluded": False}
            save(output / "pool_inventory.jsonl", inventory)
            save(output / "eligible_index.jsonl", {**inventory, "id": "c", "tokenizer_fingerprint": "sha256:test", "original_template_token_count": 8000, "target_token_count": 20})
            save(output / "eligible.jsonl", {"token_arrays_validated_by": "separate_prepare_conversation"})
            self.assertTrue(validate_pool_artifacts(processed, output)["passed"])
            inventory["overlap_group_id"] = "preflight:d"
            save(output / "pool_inventory.jsonl", inventory)
            with self.assertRaisesRegex(ValueError, "bypassed overlap"):
                validate_pool_artifacts(processed, output)


if __name__ == "__main__":
    unittest.main()
