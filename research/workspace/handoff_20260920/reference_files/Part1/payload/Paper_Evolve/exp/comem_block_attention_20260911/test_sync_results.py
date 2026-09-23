"""Bounds and path checks for a remote artifact stream; no network or GPU."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import sys

from sync_results import allowed_name, extract_snapshot, remote_source


def payload(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content, kind in entries:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.size = len(content) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                item.linkname = "../../outside"
            archive.addfile(item, io.BytesIO(content) if item.size else None)
    return stream.getvalue()


class SyncTests(unittest.TestCase):
    def test_allowlist_excludes_weights_and_path_escapes(self):
        for name in ("queue.json", "train/D0/status.json", "smoke/B/eval_step1.records.jsonl"):
            self.assertTrue(allowed_name(name), name)
        for name in ("../queue.json", "/queue.json", "C:/queue.json", "train/D0/last.pt",
                     "train/D0/../status.json", "train\\D0\\status.json", "queue.json/."):
            self.assertFalse(allowed_name(name), name)

    def test_valid_pending_snapshot(self):
        entries = [("queue.json", b'{"reason":"waiting"}', tarfile.REGTYPE),
                   ("SYNC_INFO.json", b'{"checkpoint_weights_copied":false}', tarfile.REGTYPE)]
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "snapshot"
            self.assertEqual(extract_snapshot(payload(entries), target), ["SYNC_INFO.json", "queue.json"])
            self.assertEqual(json.loads((target / "queue.json").read_text())["reason"], "waiting")

    def test_original_failure_and_authorized_retry_both_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "remote"
            names = ["smoke/D0/worker.log", "smoke/D0_retry1/status.json",
                     "smoke/D0_retry1/gpu_admission.json", "smoke/D0_retry1/gpu_lease.json",
                     "smoke/D0_retry1/eval_step1.json", "controller_guard_retry1.log",
                     "launch_guard_retry1.json"]
            excluded = ["smoke/D0_retry1/last.pt", "smoke/D0_retry2/status.json",
                        "train/D0_retry1/status.json"]
            for name in names + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            with patch("sync_results.REMOTE_OUT", str(root)):
                script = remote_source()
            copied = subprocess.run([sys.executable, "-c", script], capture_output=True, check=True)
            target = Path(temporary) / "snapshot"
            self.assertEqual(extract_snapshot(copied.stdout, target), sorted(names + ["SYNC_INFO.json"]))
            info = json.loads((target / "SYNC_INFO.json").read_text())
            self.assertEqual(set(info["files"]), set(names))
            self.assertFalse(info["checkpoint_weights_copied"])
            for name in excluded:
                self.assertFalse(allowed_name(name))
                self.assertFalse((target / name).exists())

    def test_reject_symlink_duplicate_escape_and_missing_info(self):
        cases = [[("queue.json", b"", tarfile.SYMTYPE)],
                 [("queue.json", b"{}", tarfile.REGTYPE)] * 2,
                 [("../queue.json", b"{}", tarfile.REGTYPE)],
                 [("queue.json", b"{}", tarfile.REGTYPE)]]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "snapshot"
                with self.assertRaises(ValueError):
                    extract_snapshot(payload(entries), target)
                self.assertFalse((Path(temporary) / "queue.json").exists())


if __name__ == "__main__":
    unittest.main()
