"""CPU-only real Windows Python/venv ancestry and owned termination checks."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

from infra_processes import (identity, identity_alive, validate_supervisor,
                             validate_worker, terminate_owned_tree)

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
VENV_PYTHON = HERE.parents[1] / ".venv/Scripts/python.exe"


def write_json(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def cpu_worker():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-worker", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--supervisor", type=int, required=True)
    args = parser.parse_args()
    try:
        lease = validate_supervisor(args.supervisor, args.heartbeat, SCRIPT, args.out)
        write_json(args.out / "worker_identity.json", {
            "lease": lease, "worker_executable": sys.executable,
            "worker_parent": psutil.Process().ppid()})
        sys.stdin.readline()  # Parent owns process lifetime; no background GPU work.
    except Exception as error:
        write_json(args.out / "worker_error.json", {"type": type(error).__name__, "error": str(error)})
        raise


class ProcessIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sparse-infra-process-test-")
        self.root = Path(self.temporary.name)
        self.children = []

    def tearDown(self):
        for process in reversed(self.children):
            try:
                terminate_owned_tree(process)
            except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    stream.close()
        self.temporary.cleanup()

    def launch(self, executable, name, wrong_heartbeat=False):
        out = self.root / name
        out.mkdir()
        beat = self.root / f"{name}-heartbeat.json"
        supervisor = identity(os.getpid())
        write_json(beat, {**supervisor, "create_time": supervisor["create_time"] + (1 if wrong_heartbeat else 0),
                          "unix_s": time.time()})
        process = subprocess.Popen([str(executable), "-u", str(SCRIPT), "--cpu-worker", "--out", str(out),
                                    "--heartbeat", str(beat), "--supervisor", str(os.getpid())],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        process._infra_root_create_time = identity(process.pid)["create_time"]
        self.children.append(process)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            error = out / "worker_error.json"
            output = out / "worker_identity.json"
            if error.is_file():
                return process, out, None, json.loads(error.read_text())
            if output.is_file():
                return process, out, json.loads(output.read_text()), None
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=3)
                self.fail(f"CPU identity worker exited unexpectedly ({process.returncode}): {stderr}")
            time.sleep(.05)
        self.fail("CPU worker did not publish an identity within 15 seconds")

    def validated(self, process, out, record):
        lease = record["lease"]
        owned = validate_worker(process.pid, process._infra_root_create_time, os.getpid(), lease, SCRIPT, out)
        process._infra_worker_lease = lease
        self.assertIn(process.pid, owned)
        self.assertIn(lease["worker"]["pid"], owned)
        self.assertTrue(identity_alive(lease["worker"]))
        return lease, owned

    def test_real_direct_base_python_handshake(self):
        executable = getattr(sys, "_base_executable", None) or sys.executable
        process, out, record, error = self.launch(executable, "direct")
        self.assertIsNone(error, error)
        lease, owned = self.validated(process, out, record)
        self.assertEqual(lease["worker"]["pid"], process.pid)
        self.assertIsNone(lease["redirector"])
        self.assertEqual(owned, {process.pid})

    @unittest.skipUnless(os.name == "nt" and VENV_PYTHON.is_file(), "Requires the local Windows venv redirector")
    def test_real_venv_redirector_handshake_and_only_owned_tree_termination(self):
        # Independent CPU process must survive termination of the venv tree.
        foreign = subprocess.Popen([str(getattr(sys, "_base_executable", sys.executable)), "-c",
                                    "import sys; sys.stdin.readline()"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        foreign._infra_root_create_time = identity(foreign.pid)["create_time"]
        self.children.append(foreign)
        process, out, record, error = self.launch(VENV_PYTHON, "venv")
        self.assertIsNone(error, error)
        lease, owned = self.validated(process, out, record)
        self.assertNotEqual(lease["worker"]["pid"], process.pid)
        self.assertEqual(lease["redirector"]["pid"], process.pid)
        self.assertEqual(record["worker_parent"], process.pid)
        self.assertEqual(len(owned), 2)
        terminate_owned_tree(process)
        self.assertIsNotNone(process.poll())
        self.assertFalse(identity_alive(lease["worker"]))
        self.assertFalse(identity_alive(lease["owned_root"]))
        self.assertIsNone(foreign.poll(), "Terminating the owned venv subtree affected another process")

    def test_wrong_create_time_script_and_output_are_rejected(self):
        process, out, record, error = self.launch(getattr(sys, "_base_executable", sys.executable), "invalid")
        self.assertIsNone(error, error)
        lease, _ = self.validated(process, out, record)
        for field in ("worker", "owned_root", "supervisor"):
            bad = copy.deepcopy(lease)
            bad[field]["create_time"] += 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_worker(process.pid, process._infra_root_create_time, os.getpid(), bad, SCRIPT, out)
        for script, destination in ((SCRIPT.with_name("wrong_script.py"), out), (SCRIPT, out / "wrong")):
            with self.subTest(script=script, destination=destination), self.assertRaises(RuntimeError):
                validate_worker(process.pid, process._infra_root_create_time, os.getpid(), lease, script, destination)

    def test_unrelated_live_worker_rejected_even_if_command_check_mocked(self):
        first, first_out, first_record, error = self.launch(getattr(sys, "_base_executable", sys.executable), "first")
        self.assertIsNone(error, error)
        second, second_out, second_record, error = self.launch(getattr(sys, "_base_executable", sys.executable), "second")
        self.assertIsNone(error, error)
        first_lease, _ = self.validated(first, first_out, first_record)
        self.validated(second, second_out, second_record)
        bad = copy.deepcopy(first_lease)
        bad["worker"] = second_record["lease"]["worker"]
        with patch("infra_processes._own_command", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "outside"):
                validate_worker(first.pid, first._infra_root_create_time, os.getpid(), bad, SCRIPT, first_out)
        self.assertIsNone(second.poll())

    def test_worker_rejects_stale_supervisor_identity(self):
        process, out, record, error = self.launch(getattr(sys, "_base_executable", sys.executable), "badbeat", wrong_heartbeat=True)
        self.assertIsNone(record)
        self.assertEqual(error["type"], "RuntimeError")
        self.assertIn("identity changed", error["error"])
        process.wait(timeout=5)
        self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    if "--cpu-worker" in sys.argv:
        cpu_worker()
    else:
        unittest.main(verbosity=2)
