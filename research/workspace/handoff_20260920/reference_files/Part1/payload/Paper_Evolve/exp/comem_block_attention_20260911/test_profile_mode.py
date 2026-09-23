"""Standard-library checks keeping diagnostic profiles out of timing results.

Uses temporary stub files and dry planning only. No Torch, NVIDIA query, model,
GPU worker, or remote connection is permitted by these tests.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import infra_sparse as worker
import launch_sparse_infra as launcher
import run_reader_profiles as profiles
from compare_decode_infra import collect


class ProfileModeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stub_profile_mode_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.base / "model"
        self.model.mkdir()
        self.put(self.model / "config.json", dict(model_type="qwen3", num_hidden_layers=36,
                                                  max_position_embeddings=8192))
        self.adapter = self.base / "adapter.stub"
        self.adapter.write_bytes(b"NOT A CHECKPOINT: TEST STUB")

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def launcher_args(self, out=None):
        return ["--python", sys.executable, "--model", str(self.model), "--adapter", str(self.adapter),
                "--out", str(out or self.base / "dry"), "--lengths", "4096", "--arms", "D0", "B",
                "--repetitions", "1", "--generation-tokens", "4"]

    def test_profile_identity_is_separate_from_unprofiled_run(self):
        base = self.launcher_args()
        formal = launcher.parser().parse_args(base)
        diagnostic = launcher.parser().parse_args(base + ["--profile-reader-only"])
        job = dict(arm="D0", cache_mode="cold_hj", document_tokens=4096)
        # Hash I/O is substituted so this check does not depend on whether the
        # separately developed Torch profiler helper has landed on disk yet.
        with patch.object(launcher, "digest", side_effect=lambda p: "stub:" + Path(p).name):
            a = launcher.expected_identity(formal, job, self.adapter)
            b = launcher.expected_identity(diagnostic, job, self.adapter)
        self.assertFalse(a["profile_reader_only"])
        self.assertTrue(b["profile_reader_only"])
        self.assertIsNone(a["profiler_sha256"])
        self.assertEqual(b["profiler_sha256"], "stub:reader_profiler.py")
        self.assertNotEqual(a, b)

    def make_prior(self, profile=True, *, include_receipt=True, exit_code=0):
        ident = dict(profile_reader_only=profile, marker="test-stub")
        folder = self.base / "case"
        attempt = folder / "attempts" / "0001"
        result = dict(status="complete", identity=ident, timing_eligible=False, requests=[], summary={})
        if include_receipt:
            result["profiler_receipt"] = dict(scope="diagnostic-only-test-stub")
        self.put(attempt / "result.json", result)
        self.put(attempt / "monitor.json", dict(status="complete", exit_code=exit_code))
        return folder, attempt, ident

    def test_profile_prior_requires_matching_mode_and_successful_monitor(self):
        folder, _, ident = self.make_prior()
        self.assertEqual(launcher.find_prior(folder, ident)["status"], "complete")
        self.assertIsNone(launcher.find_prior(folder, ident | dict(profile_reader_only=False)))
        for include_receipt, exit_code in ((False, 0), (True, 1)):
            self.make_prior(include_receipt=include_receipt, exit_code=exit_code)
            self.assertEqual(launcher.find_prior(folder, ident)["status"], "invalid_incomplete_receipt")

    def test_profiler_receipt_does_not_make_formal_prior_timing_eligible(self):
        folder, _, ident = self.make_prior(profile=False)
        self.assertEqual(launcher.find_prior(folder, ident)["status"], "invalid_incomplete_receipt")

    def test_formal_comparison_refuses_completed_diagnostic(self):
        folder, attempt, _ = self.make_prior()
        self.put(folder / "status.json", dict(status="complete", jobs={
            "profile-stub": dict(status="complete", attempt=str(attempt))}))
        with self.assertRaisesRegex(ValueError, "invalid measurement"):
            collect(folder)

    def test_dry_profile_plan_avoids_gpu_processes_and_formal_report(self):
        out = self.base / "dry"
        with patch.object(launcher, "snapshot", side_effect=AssertionError("NVIDIA query")), \
             patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("GPU worker")), \
             patch.object(launcher, "render_report", side_effect=AssertionError("formal timing table")), \
             patch.object(launcher, "digest", side_effect=lambda p: "stub:" + Path(p).name):
            self.assertEqual(launcher.main(self.launcher_args(out) + ["--profile-reader-only"]), 0)
        state = json.loads((out / "status.json").read_text())
        self.assertTrue(state["profile_reader_only"])
        self.assertEqual(len(state["jobs"]), 3)
        self.assertTrue(all(j["status"] == "pending" for j in state["jobs"].values()))
        self.assertFalse((out / "INFRA_REPORT.md").exists())
        self.assertFalse((out / "INFRA_SUMMARY.json").exists())

    def test_launcher_rejects_unbounded_profile_before_gpu_or_model_io(self):
        for repetitions, generation in ((2, 4), (1, 32)):
            argv = ["--profile-reader-only", "--repetitions", str(repetitions),
                    "--generation-tokens", str(generation)]
            with self.subTest(repetitions=repetitions, generation=generation), \
                 patch.object(launcher, "read_json", side_effect=AssertionError("model read")), \
                 patch.object(launcher, "snapshot", side_effect=AssertionError("GPU query")):
                with self.assertRaisesRegex(ValueError, "Bounded profiling"):
                    launcher.main(argv)

    def test_worker_rejects_unbounded_profile_before_gate_or_torch(self):
        for repetitions, generation in ((2, 4), (1, 32)):
            args = worker.parser().parse_args(["--model", str(self.model), "--adapter", str(self.adapter),
                "--arm", "D0", "--cache-mode", "cold_hj", "--document-tokens", "4096",
                "--profile-reader-only", "--repetitions", str(repetitions),
                "--generation-tokens", str(generation), "--supervisor-pid", "100",
                "--supervisor-heartbeat", str(self.base / "heartbeat.stub"), "--out", str(self.base / "worker")])
            with self.subTest(repetitions=repetitions, generation=generation), \
                 patch.object(worker, "require_supervisor", return_value={"stub": True}), \
                 patch.object(worker.platform, "system", return_value="Windows"), \
                 patch.object(worker, "snapshot", side_effect=AssertionError("GPU query")):
                with self.assertRaisesRegex(ValueError, "bounded profiler"):
                    worker.run(args)

    def test_outer_dry_plan_requests_four_bounded_guarded_cells(self):
        out = self.base / "outer"
        argv = ["run_reader_profiles.py", "--python", sys.executable, "--out", str(out)]
        with patch.object(sys, "argv", argv), \
             patch.object(profiles.subprocess, "Popen", side_effect=AssertionError("worker launch")):
            self.assertEqual(profiles.main(), 0)
        plan = json.loads((out / "profile_plan.json").read_text())
        self.assertEqual(plan["cells"], 4)
        self.assertEqual(set(plan["commands"]), {"reference", "decode_v2"})
        for variant, command in plan["commands"].items():
            self.assertIn("--profile-reader-only", command)
            self.assertNotIn("--run", command)
            self.assertEqual(Path(command[4]).name, "launch_sparse_infra.py")
            self.assertEqual(command[command.index("--generation-tokens")+1], "4")
            self.assertEqual(command[command.index("--repetitions")+1], "1")
            self.assertEqual(command[command.index("--reader-implementation")+1], variant)


if __name__ == "__main__":
    unittest.main(verbosity=2)
