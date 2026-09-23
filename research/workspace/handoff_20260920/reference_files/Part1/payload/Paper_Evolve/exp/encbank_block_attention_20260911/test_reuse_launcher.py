"""Check that an unfinished remote quality probe cannot launch a local GPU worker."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import launch_sparse_infra as launch


class QualityAdmissionTests(unittest.TestCase):
    def fixture(self, root):
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3",
            "num_hidden_layers": 36, "max_position_embeddings": 40960}))
        adapter = root / "adapter.pt"
        adapter.write_bytes(b"not loaded by a dry supervisor")
        return ["--python", sys.executable, "--model", str(model), "--adapter", str(adapter),
                "--adapter-kind", "trained", "--out", str(root / "out"), "--lengths", "32768",
                "--arms", "D0", "A", "B", "D1", "FULL", "--reuse-requests", "10",
                "--reuse-quality-receipt", str(root / "not_ready.json")]

    def test_dry_stream_retains_missing_quality_without_gpu_or_legacy_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.fixture(root)
            with patch.object(launch.subprocess, "Popen", side_effect=AssertionError("GPU worker forbidden")), \
                 patch.object(launch, "snapshot", side_effect=AssertionError("GPU query forbidden")):
                self.assertEqual(launch.main(argv), 0)
            state = json.loads((root / "out/status.json").read_text())
            jobs = json.loads((root / "out/plan.json").read_text())
            self.assertEqual(state["status"], "waiting_quality")
            self.assertEqual(len(jobs), 5)
            self.assertEqual({row["arm"]: row["cache_mode"] for row in jobs},
                {"D0": "cold_hj", "A": "block_hot", "B": "block_hot", "D1": "cold_hj", "FULL": "cold_hj"})
            self.assertTrue(all(cell["status"] == "waiting_quality" for cell in state["jobs"].values()))
            self.assertFalse((root / "out/INFRA_TABLE.tex").exists())

    def test_run_stops_before_worker_when_remote_quality_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.fixture(root) + ["--run"]
            with patch.object(launch.platform, "system", return_value="Windows"), \
                 patch.object(launch.subprocess, "Popen", side_effect=AssertionError("GPU worker forbidden")), \
                 patch.object(launch, "snapshot", side_effect=AssertionError("GPU query forbidden")):
                self.assertEqual(launch.main(argv), 0)
            state = json.loads((root / "out/status.json").read_text())
            self.assertEqual(state["status"], "waiting_quality")
            self.assertIsNone(state.get("child_pid"))
            self.assertFalse(list((root / "out").glob("*/attempts/*")))


if __name__ == "__main__":
    unittest.main()
