"""CPU-only heartbeat fault injection; fake owned child, no NVIDIA or process launch."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_sparse_infra as launcher
import infra_processes as processes
from infra_protocol import GIB, GPU_NAME


class StubChild:
    pid = 600000001
    _infra_root_create_time = 1234.

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self):
        if self.returncode is None:
            raise AssertionError("Test child must be stopped before a wait can finish")
        return self.returncode


class MonitorFailureChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stub_monitor_failure_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.attempt = self.base / "attempt"
        self.attempt.mkdir()
        self.beat = self.base / "heartbeat.json"
        # An isolated helper directory prevents a user's real pause marker from
        # changing the injected failure path. These files are never executed.
        self.here = self.base / "stub_code"
        self.here.mkdir()
        (self.here / "launch_sparse_infra.py").write_text("# test stub only\n")
        self.child = StubChild()

    def inject(self, successful_heartbeats, *, stopping_fails=False):
        lease = dict(worker=dict(pid=self.child.pid, create_time=1234.),
                     supervisor=dict(pid=123, create_time=1000.),
                     owned_root=dict(pid=self.child.pid, create_time=1234.), redirector=None)
        baseline = dict(name=GPU_NAME, uuid="TEST-STUB-GPU", used_bytes=2*GIB,
                        processes=[], timestamp="test-baseline")
        sample = baseline | dict(used_bytes=6*GIB, timestamp="test-sample", processes=[
            dict(pid=self.child.pid, name="test-worker-python.exe", type="C", used_bytes=4*GIB)])
        (self.attempt / "progress.json").write_text(json.dumps(
            dict(supervisor_lease=lease, baseline_snapshot=baseline)), encoding="utf-8")
        beat_calls = 0
        def heartbeat(path):
            nonlocal beat_calls
            self.assertEqual(path, self.beat)
            beat_calls += 1
            if beat_calls > successful_heartbeats:
                raise PermissionError("injected heartbeat replacement sharing conflict")
        def stop(child):
            self.assertIs(child, self.child)
            if stopping_fails:
                raise RuntimeError("injected owned-process exit not confirmed")
            child.returncode = 75
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "HERE", self.here))
            stack.enter_context(patch.object(launcher, "heartbeat", side_effect=heartbeat))
            snapshot = stack.enter_context(patch.object(launcher, "snapshot", return_value=sample))
            stopped = stack.enter_context(patch.object(launcher, "stop_owned_child", side_effect=stop))
            stack.enter_context(patch.object(launcher, "validate_worker", return_value={self.child.pid}))
            stack.enter_context(patch.object(launcher.time, "sleep", return_value=None))
            stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("process launch")))
            if stopping_fails:
                with self.assertRaisesRegex(RuntimeError, "queue stopped"):
                    launcher.monitor_worker(self.child, self.attempt, self.beat, .1)
                result = json.loads((self.attempt / "monitor.json").read_text())
            else:
                result = launcher.monitor_worker(self.child, self.attempt, self.beat, .1)
            stopped.assert_called_once_with(self.child)
            self.assertEqual(snapshot.call_count, successful_heartbeats)
        saved = json.loads((self.attempt / "monitor.json").read_text())
        self.assertEqual(result, saved)
        self.assertEqual(saved["status"], "monitor_failed")
        self.assertIn("PermissionError", saved["error"])
        self.assertIn("heartbeat", saved["error"])
        self.assertEqual(saved["child_pid"], self.child.pid)
        self.assertIn("finished_at", saved)
        return saved

    def test_first_heartbeat_failure_stops_only_owned_child_and_finalizes(self):
        result = self.inject(0)
        self.assertEqual(result["exit_code"], 75)
        self.assertIsNone(result["ownership_stop_error"])
        self.assertEqual((self.attempt / "telemetry.jsonl").read_text(), "")

    def test_failure_after_sample_preserves_telemetry_and_finishes_failed(self):
        result = self.inject(1)
        self.assertEqual(result["exit_code"], 75)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["baseline_gpu_used_bytes"], 2*GIB)
        self.assertEqual(result["peak_incremental_gpu_bytes"], 4*GIB)
        self.assertEqual(result["cap_bytes"], 28*GIB)
        self.assertEqual(result["actual_worker_pid"], self.child.pid)
        self.assertEqual(result["interference"], [])
        rows = [json.loads(s) for s in (self.attempt / "telemetry.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["signed_increment_bytes"], 4*GIB)
        self.assertEqual(rows[0]["own_process_bytes"], 4*GIB)

    def test_unconfirmed_owned_child_exit_raises_and_cannot_advance_queue(self):
        result = self.inject(0, stopping_fails=True)
        self.assertIsNone(result["exit_code"])
        self.assertIn("exit not confirmed", result["ownership_stop_error"])
        self.assertIsNone(self.child.poll())


class HeartbeatRefreshChecks(unittest.TestCase):
    """A second snapshot can be fresh; it must never relax identity or age."""
    supervisor_pid = 600000101

    def check(self, snapshots, expect_error=None):
        script = Path(__file__).resolve().parent / "infra_sparse.py"
        output = Path(tempfile.gettempdir()) / "UNCREATED-profile-heartbeat-test"
        worker_pid = os.getpid()
        class Process:
            def __init__(self, pid):
                self.pid = pid
            def create_time(inner):
                return 50. if inner.pid == self.supervisor_pid else 60.
            def parent(inner):
                return Process(self.supervisor_pid)
            def cmdline(inner):
                return ["python.exe", str(script), "--out", str(output)]
        with patch.object(processes, "read_json", side_effect=snapshots) as read, \
             patch.object(processes.time, "time", return_value=100.), \
             patch.object(processes.psutil, "Process", side_effect=Process) as lookup:
            if expect_error:
                with self.assertRaisesRegex(RuntimeError, expect_error):
                    processes.validate_supervisor(self.supervisor_pid, output / "beat.json", script, output)
                result = None
            else:
                result = processes.validate_supervisor(self.supervisor_pid, output / "beat.json", script, output)
                self.assertEqual(result["worker"]["pid"], worker_pid)
                self.assertEqual(result["supervisor"], dict(pid=self.supervisor_pid, create_time=50.))
            self.assertEqual(read.call_count, len(snapshots))
            if expect_error == "heartbeat is required":
                lookup.assert_not_called()
        return result

    def beat(self, timestamp, **changes):
        return dict(pid=self.supervisor_pid, create_time=50., unix_s=timestamp) | changes

    def test_stale_snapshot_then_fresh_is_accepted_after_one_refresh(self):
        self.check([self.beat(79.), self.beat(99.)])

    def test_missing_snapshot_then_fresh_is_accepted_after_one_refresh(self):
        self.check([None, self.beat(99.)])

    def test_two_stale_snapshots_are_rejected_without_process_lookup(self):
        self.check([self.beat(79.), self.beat(79.)], "heartbeat is required")

    def test_wrong_pid_after_refresh_is_rejected(self):
        self.check([self.beat(79.), self.beat(99., pid=123)], "heartbeat is required")

    def test_fresh_timestamp_does_not_accept_reused_supervisor_identity(self):
        self.check([self.beat(79.), self.beat(99., create_time=49.)], "process identity changed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
