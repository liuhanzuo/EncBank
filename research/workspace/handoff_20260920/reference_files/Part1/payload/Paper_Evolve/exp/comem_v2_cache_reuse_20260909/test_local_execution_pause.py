"""CPU-only pause checks; no tokenization, subprocess, model or CUDA load."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import aggregate_capacity
import continue_campaign
import launch_capacity
import remeasure_capacity
import run_capacity
from local_execution_pause import MARKER, LocalExecutionPaused, require_unpaused


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local_pause_test_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def marker(self):
        path = self.root/MARKER
        path.write_text("{partial-user-pause", encoding="utf-8")
        return path

    def test_unpaused_guard_is_noop(self):
        self.assertIsNone(require_unpaused(self.root))

    def test_marker_existence_blocks_and_never_changes_it(self):
        path = self.marker()
        original = path.read_bytes()
        with self.assertRaisesRegex(LocalExecutionPaused, "explicit user instruction"):
            require_unpaused(self.root)
        self.assertEqual(path.read_bytes(), original)

    def test_primary_queue_does_not_prepare_or_spawn(self):
        self.marker()
        with patch.object(launch_capacity, "HERE", self.root), patch.object(launch_capacity, "load_bundle") as load, \
             patch.object(launch_capacity.subprocess, "Popen") as spawn:
            with self.assertRaises(LocalExecutionPaused):
                launch_capacity.main(["--run"])
            load.assert_not_called()
            spawn.assert_not_called()
        self.assertFalse((self.root/"results").exists())

    def test_campaign_does_not_spawn_or_rewrite_state(self):
        self.marker()
        with patch.object(continue_campaign, "HERE", self.root), patch.object(continue_campaign.subprocess, "Popen") as spawn:
            with self.assertRaises(LocalExecutionPaused):
                continue_campaign.main()
            spawn.assert_not_called()
        self.assertFalse((self.root/"results").exists())

    def test_remeasurement_does_not_read_plan_or_spawn(self):
        self.marker()
        with patch.object(remeasure_capacity, "HERE", self.root), patch.object(remeasure_capacity, "read_json") as read, \
             patch.object(remeasure_capacity.subprocess, "Popen") as spawn:
            with self.assertRaises(LocalExecutionPaused):
                remeasure_capacity.main(["--plan", str(self.root/"missing.json")])
            read.assert_not_called()
            spawn.assert_not_called()

    def test_direct_runner_blocks_before_fixtures_or_model_import(self):
        self.marker()
        with patch.object(run_capacity, "HERE", self.root), patch.object(run_capacity, "load_bundle") as load:
            with self.assertRaises(LocalExecutionPaused):
                run_capacity.main(["--dataset", "locomo", "--fixtures", str(self.root/"missing"),
                    "--cohort", "locomo-00", "--fraction", ".25", "--arm", "prefix", "--out", str(self.root/"out")])
            load.assert_not_called()
        self.assertFalse((self.root/"out").exists())

    def test_cpu_aggregation_still_allowed_with_marker(self):
        from test_aggregate_capacity import fixture, write_attempt
        self.marker()
        docs, rows = fixture()
        jobs = aggregate_capacity.expected_jobs("scbench", docs, rows, self.root/"fixtures")
        write_attempt(self.root, jobs[0])
        result = aggregate_capacity.aggregate(jobs, self.root)
        self.assertEqual(result["progress"]["unique_measured_outputs"], 2)
        self.assertTrue((self.root/MARKER).exists())


if __name__ == "__main__":
    unittest.main()
