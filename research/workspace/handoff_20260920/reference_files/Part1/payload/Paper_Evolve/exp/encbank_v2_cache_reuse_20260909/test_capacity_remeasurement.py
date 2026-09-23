"""Bounded CPU tests for exclusion, exact replacement, and non-interruption."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import aggregate_capacity as agg
import capacity_timing_policy as policy
import launch_capacity
import remeasure_capacity as repair
from test_aggregate_capacity import fixture, write_attempt, save


def register(old, job, replacement="0002", status="queued"):
    cfg = agg.read_json(old/"config.json")
    value = {"protocol": "capacity-complete-trace-remeasurement-v1", "job_id": job["id"],
             "expected_queries": len(cfg["ids"]), "excluded_attempts": ["0001"],
             "replacement_attempt": replacement, "status": status,
             "expected_config_sha256": agg.digest(cfg)}
    save(old/policy.EXCLUSION, {"reason": "known external CPU overlap", "original_unchanged": True})
    save(old.parent.parent/policy.REGISTRY, value)
    return value


class RemeasurementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="remeasure_cpu_")
        self.root = Path(self.tmp.name)
        docs, rows = fixture()
        self.jobs = agg.expected_jobs("scbench", docs, rows, self.root/"fixtures")
        self.job = next(j for j in self.jobs if j["arm"] == "fix_all" and j["fraction"] == .25)
        self.old = write_attempt(self.root, self.job)

    def tearDown(self):
        self.tmp.cleanup()

    def fresh(self):
        new = self.old.parent/"0002"
        shutil.copytree(self.old, new, ignore=shutil.ignore_patterns(policy.EXCLUSION, "CPU_CONTAMINATION_NOTE.json"))
        return new

    def test_original_note_alone_is_fail_closed(self):
        save(self.old/"CPU_CONTAMINATION_NOTE.json", {"external_cpu_overlap": True})
        with self.assertRaisesRegex(ValueError, "CPU overlap"):
            agg.validate_attempt(self.old, self.job)

    def test_contaminated_complete_not_accepted_by_queue_or_aggregate(self):
        original = (self.old/"measurements.jsonl").read_bytes()
        register(self.old, self.job)
        qjob = dict(self.job, smoke=False, queries=2, ids=[r["id"] for r in self.job["rows"]], budget_bytes_expected=self.job["budget"])
        self.assertIsNone(launch_capacity.valid_complete(self.old.parent.parent, qjob, agg.MODEL))
        result = agg.aggregate(self.jobs, self.root)
        self.assertEqual(result["progress"]["unique_measured_outputs"], 0)
        self.assertEqual((self.old/"measurements.jsonl").read_bytes(), original)

    def test_partial_replacement_cannot_fallback(self):
        register(self.old, self.job)
        new = self.fresh()
        (new/"COMPLETED.json").unlink()
        result = agg.aggregate(self.jobs, self.root)
        self.assertEqual(result["progress"]["unique_measured_outputs"], 0)
        self.assertEqual(len(result["pending_timing_remeasurements"]), 1)

    def test_only_registered_complete_trace_counts_once(self):
        for job in self.jobs:
            if job["id"] != self.job["id"]:
                write_attempt(self.root, job)
        register(self.old, self.job, status="complete")
        new = self.fresh()
        save(new/"REMEASUREMENT_VALIDATED.json", {"status": "complete", "completed_queries": 2, "replacement_attempt": "0002"})
        shutil.copytree(new, self.old.parent/"0003")  # Unregistered newer attempt never wins.
        result = agg.aggregate(self.jobs, self.root)
        self.assertTrue(result["all_full_workloads_complete"])
        self.assertEqual(result["progress"]["unique_measured_outputs"], 20)
        chosen = next(a for a in result["valid_attempts"] if a["job_id"] == self.job["id"])
        self.assertEqual(Path(chosen["path"]).name, "0002")
        self.assertTrue((self.old/"measurements.jsonl").exists())

    def test_config_and_raw_identity_mismatch_rejected(self):
        register(self.old, self.job)
        new = self.fresh()
        cfg = agg.read_json(new/"config.json")
        cfg["extra_unplanned_setting"] = True
        save(new/"config.json", cfg)
        with self.assertRaisesRegex(ValueError, "complete original trace"):
            agg.validate_attempt(new, self.job)
        save(new/"config.json", agg.read_json(self.old/"config.json"))
        rows = [json.loads(r) for r in (new/"measurements.jsonl").read_text().splitlines()]
        rows[0]["query_ids"] = [99, 99]
        (new/"measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "query token IDs"):
            agg.validate_attempt(new, self.job)

    def test_dependency_identity_and_finite_command(self):
        original = {"ProcessId": 7, "CreationDate": "old", "CommandLine": "run_capacity.py --out x"}
        self.assertTrue(repair.same_process(original, [original]))
        self.assertFalse(repair.same_process(original, [dict(original, CreationDate="reused")]))
        cfg = dict(dataset="locomo", cohort="locomo-00", arm="fix_all", fraction=.25,
                   smoke=False, ids=[str(i) for i in range(800)], fixtures="fixture", model=agg.MODEL)
        command = repair.command_for(cfg, self.root/"new", "python")
        self.assertIn("run_capacity.py", " ".join(command))
        self.assertNotIn("--smoke", command)
        with self.assertRaises(ValueError):
            repair.command_for(dict(cfg, fraction=.5), self.root/"new", "python")

    def test_live_dependency_waits_without_spawning_or_terminating(self):
        planroot = self.root/"results/capacity_remeasure"
        jobid = "locomo/full/locomo-00/025/fix_all"
        target = self.root/"results/capacity"/jobid
        cfg = dict(dataset="locomo", cohort="locomo-00", arm="fix_all", fraction=.25,
                   smoke=False, ids=[str(i) for i in range(800)], fixtures="fixture", model=agg.MODEL)
        save(target/"attempts/0001/config.json", cfg)
        register(target/"attempts/0001", {"id": jobid})
        dependency = {"ProcessId": 7, "CreationDate": "old", "CommandLine": "run_capacity.py"}
        save(planroot/"plan.json", {"job_id": jobid, "replacement_attempt": "0002", "python": "python",
                                   "wait_for_processes": [dependency], "wait_for_attempt": "unused"})
        with patch.object(repair, "HERE", self.root), patch.object(repair, "bootstrap_lock", return_value=nullcontext()), \
             patch.object(repair, "processes", return_value=[dependency]), patch.object(repair.time, "sleep", side_effect=InterruptedError("bounded test")), \
             patch.object(repair.subprocess, "Popen") as launch:
            with self.assertRaises(InterruptedError):
                repair.main(["--plan", str(planroot/"plan.json")])
            launch.assert_not_called()
        state = agg.read_json(planroot/"status.json")
        self.assertEqual(state["waiting_pids"], [7])
        self.assertFalse((target/"attempts/0002").exists())

    def test_completed_replacement_recovery_never_launches(self):
        planroot = self.root/"results/capacity_remeasure"
        jobid = "locomo/full/locomo-00/025/fix_all"
        target = self.root/"results/capacity"/jobid
        cfg = dict(dataset="locomo", cohort="locomo-00", arm="fix_all", fraction=.25,
                   smoke=False, ids=[str(i) for i in range(800)], fixtures="fixture", model=agg.MODEL)
        save(target/"attempts/0001/config.json", cfg)
        register(target/"attempts/0001", {"id": jobid})
        save(target/"attempts/0002/COMPLETED.json", {"status": "complete"})
        save(self.root/"dependency/COMPLETED.json", {"status": "complete", "completed_queries": 800})
        save(planroot/"plan.json", {"job_id": jobid, "replacement_attempt": "0002", "python": "python",
                                   "wait_for_processes": [], "wait_for_attempt": str(self.root/"dependency")})
        receipt = {"status": "complete", "completed_queries": 800, "replacement_attempt": "0002"}
        with patch.object(repair, "HERE", self.root), patch.object(repair, "bootstrap_lock", return_value=nullcontext()), \
             patch.object(repair, "processes", return_value=[]), patch.object(repair, "validate_finished", return_value=receipt), \
             patch.object(repair.subprocess, "Popen") as launch:
            self.assertEqual(repair.main(["--plan", str(planroot/"plan.json")]), 0)
            launch.assert_not_called()
        self.assertEqual(agg.read_json(planroot/"status.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
