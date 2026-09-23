"""CPU checks for evidence budgeting, exact pack integrity and GPU admission."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import prepare_oracle_support as prep
import run_oracle_support as run


class SupportProtocolTests(unittest.TestCase):
    def test_forced_pack_preserves_count_and_chronology(self):
        result = prep.forced_indices([1, 8], [7, 2, 4, 1, 8, 0, 3, 5, 6], [2, 4, 7])
        self.assertEqual(result, [1, 7, 8])

    def test_budget_failure_is_explicit(self):
        self.assertIsNone(prep.forced_indices([0, 1, 2], [0, 1, 2, 3], [0, 3]))

    def test_already_visible_support_keeps_exact_natural_pack(self):
        self.assertEqual(prep.forced_indices([2], [7, 2, 4, 1], [2, 4, 7]), [2, 4, 7])

    def test_minimum_union_checks_alternative_occurrences(self):
        facts = [{"span_coverage": [{"chunks": [1, 2], "token_n": 3}, {"chunks": [4], "token_n": 2}]},
                 {"span_coverage": [{"chunks": [1, 2], "token_n": 4}]}]
        self.assertEqual(prep.minimum_union(facts), [1, 2])
        self.assertIsNone(prep.minimum_union([{"span_coverage": []}]))

    def test_equal_chunk_count_can_have_different_tokens(self):
        context = list(range(600))
        full = prep.make_pack(context, [1, 2], [0])
        tail = prep.make_pack(context, [1, 2], [1])
        self.assertEqual(len(full["selected_indices"]), len(tail["selected_indices"]))
        self.assertEqual(full["read_pack_tokens"] - tail["read_pack_tokens"], 424)

    def test_actual_reader_tokens_must_match(self):
        current = {"expected_chunks": [[1, 2], [3]], "actual_pack_checks": 0}
        run.verify_actual_chunks(current, [[1, 2], [3]])
        self.assertEqual(current["actual_pack_checks"], 1)
        with self.assertRaisesRegex(ValueError, "Actual reader pack"):
            run.verify_actual_chunks(current, [[1, 2], [4]])

    def test_busy_remote_gpu_waits_without_cuda(self):
        probe = Mock(side_effect=[(False, {"memory_mib": 1000, "utilization": 40, "compute_pids": [99]}),
                                  (True, {"memory_mib": 10, "utilization": 0, "compute_pids": []})])
        metadata = {}
        with patch.dict(os.environ, {"COMEM_REMOTE_QUEUE": "1", "CUDA_VISIBLE_DEVICES": "1"}), \
             patch.dict(sys.modules, {"remote_queue": SimpleNamespace(gpu_idle=probe)}), \
             patch.object(run.qa, "write_json"), patch.object(run.time, "sleep") as sleep, \
             patch.object(run.qa.torch.cuda, "is_initialized", return_value=False):
            run.wait_for_remote_gpu(metadata, Path("unused.json"))
        self.assertEqual(probe.call_count, 2)
        probe.assert_called_with(1, 512)
        sleep.assert_called_once_with(30)
        self.assertEqual(metadata["status"], "loading_model")

    def test_full_roster_and_fixture_counts(self):
        folder = prep.HERE / "data/oracle_support"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (folder / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        roster = [json.loads(line) for line in (folder / "selection_roster.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), manifest["eligible_inputs"])
        self.assertEqual(len(roster), manifest["source_rows"])
        self.assertEqual(sum(r["status"] == "ready" for r in roster), len(rows))
        self.assertEqual(sum(r["task"] == "hotpotqa/oracle" for r in rows), 8)
        self.assertEqual(sum(r["source_benchmark"] == "longbench" and r["status"] == "support_alignment_unknown" for r in roster), 192)
        for row in rows:
            self.assertEqual(prep.digest({k: v for k, v in row.items() if k != "fixture_row_sha256"}), row["fixture_row_sha256"])
            self.assertEqual(len(row["natural_pack"]["selected_indices"]), len(row["oracle_pack"]["selected_indices"]))
            self.assertTrue(set(row["required_chunks"]).issubset(row["oracle_pack"]["selected_indices"]))
            self.assertEqual(row["natural_pack"] == row["oracle_pack"], row["natural_all_required_visible"])
        self.assertFalse(run.qa.torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
