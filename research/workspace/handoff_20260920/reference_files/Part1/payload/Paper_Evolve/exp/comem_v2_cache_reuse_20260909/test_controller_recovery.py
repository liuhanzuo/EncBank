"""CPU-only regression for transient Windows status publication failures."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continue_campaign
import launch_capacity
import run_reuse


class ControllerStatusTests(unittest.TestCase):
    def test_both_controllers_use_bounded_reporting_writer(self):
        self.assertIs(launch_capacity.save_json, run_reuse.save)
        self.assertIs(continue_campaign.save_json, run_reuse.save)

    def test_transient_denial_retries_atomic_publish_without_losing_old_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"status.json"
            old = {"status": "waiting_for_phase_A", "completed": 36}
            path.write_text(json.dumps(old), encoding="utf-8")
            calls = []
            real_replace = os.replace
            def transient(source, destination):
                calls.append((source, destination))
                if len(calls) < 3:
                    self.assertEqual(json.loads(path.read_text()), old)
                    raise PermissionError("simulated Windows sharing violation")
                real_replace(source, destination)
            with patch.object(run_reuse.os, "replace", side_effect=transient), patch.object(run_reuse.time, "sleep") as sleep:
                launch_capacity.save_json(path, {"status": "running", "completed": 0})
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), {"status": "running", "completed": 0})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_persistent_denial_fails_bounded_and_retains_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"status.json"
            path.write_text('{"status":"old"}', encoding="utf-8")
            with patch.object(run_reuse.os, "replace", side_effect=PermissionError("persistent denial")) as replace, patch.object(run_reuse.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    continue_campaign.save_json(path, {"status": "new"})
            self.assertEqual(replace.call_count, 10)
            self.assertEqual(sleep.call_count, 9)
            self.assertEqual(json.loads(path.read_text()), {"status": "old"})
            self.assertEqual(json.loads(path.with_suffix(".json.tmp").read_text()), {"status": "new"})

    def test_other_io_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_reuse.os, "replace", side_effect=OSError("disk failure")) as replace, patch.object(run_reuse.time, "sleep") as sleep:
                with self.assertRaises(OSError):
                    continue_campaign.save_json(Path(directory)/"status.json", {})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
