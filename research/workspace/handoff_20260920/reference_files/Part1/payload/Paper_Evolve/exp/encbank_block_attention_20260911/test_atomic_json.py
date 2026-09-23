"""Real Windows atomic JSON/read-handle checks; stdlib only, no GPU/process jobs."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import infra_protocol
from infra_protocol import read_json, save_json, shared_open_text


def process_handle_count():
    import ctypes
    from ctypes import wintypes
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.GetCurrentProcess.argtypes = ()
    library.GetCurrentProcess.restype = wintypes.HANDLE
    library.GetProcessHandleCount.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    library.GetProcessHandleCount.restype = wintypes.BOOL
    count = wintypes.DWORD()
    if not library.GetProcessHandleCount(library.GetCurrentProcess(), ctypes.byref(count)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(count.value)


class AtomicJsonCommonTests(unittest.TestCase):
    def test_parse_missing_and_decode_errors_keep_default_semantics(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "heartbeat.json"
            self.assertIs(read_json(path, sentinel), sentinel)
            path.write_text("{truncated", encoding="utf-8")
            self.assertIs(read_json(path, sentinel), sentinel)
            path.write_bytes(b"\xff\xfeinvalid-utf8")
            self.assertIs(read_json(path, sentinel), sentinel)
            save_json(path, {"unix_s": 123.5, "message": "完整 JSON"})
            self.assertEqual(read_json(path), {"unix_s": 123.5, "message": "完整 JSON"})

    def test_read_json_uses_shared_stream_and_closes_on_parse_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "heartbeat.json"
            path.write_text("{invalid", encoding="utf-8")
            seen = []
            @contextmanager
            def observed(*args, **kwargs):
                with shared_open_text(*args, **kwargs) as stream:
                    seen.append(stream)
                    yield stream
            with patch("infra_protocol.shared_open_text", side_effect=observed):
                self.assertEqual(read_json(path, "default"), "default")
            self.assertTrue(seen[0].closed)

    def test_replace_permission_retry_is_bounded_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "heartbeat.json"
            save_json(path, {"version": "old"})
            with patch("infra_protocol._replace_json_file", side_effect=PermissionError("test sharing violation")) as replace, \
                 patch("infra_protocol.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    save_json(path, {"version": "new"})
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(sleep.call_count, 7)
            self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1.0)
            self.assertEqual(read_json(path), {"version": "old"})


@unittest.skipUnless(os.name == "nt", "Windows file sharing semantics required")
class WindowsSharedJsonTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "心跳 heartbeat.json"
        save_json(self.path, {"generation": 0, "payload": "原始完整内容" * 8192})
        # Initialize imports and CRT stream machinery before counting handles.
        with shared_open_text(self.path) as stream:
            stream.read(1)
        process_handle_count()

    def test_old_shared_handle_stays_open_during_twelve_atomic_replacements(self):
        expected_old = read_json(self.path)
        with shared_open_text(self.path) as old:
            for generation in range(1, 13):
                value = {"generation": generation, "payload": ("new-%d" % generation) * 10000}
                save_json(self.path, value)
                self.assertEqual(read_json(self.path), value)
                self.assertFalse(old.closed)
            # Read only after the replacements, so this is an open HANDLE test,
            # not just data that Python buffered before the writer ran.
            self.assertEqual(json.load(old), expected_old)
            old.seek(0)
            self.assertEqual(json.load(old), expected_old)
        self.assertTrue(old.closed)

    def test_multiple_open_generations_keep_their_complete_documents(self):
        streams, expected = [], []
        with ExitStack() as stack:
            for generation in range(8):
                value = {"generation": generation, "payload": "z" * (32768 + generation)}
                save_json(self.path, value)
                streams.append(stack.enter_context(shared_open_text(self.path)))
                expected.append(value)
            for stream, value in zip(streams, expected):
                self.assertEqual(json.load(stream), value)

    def test_context_exception_closes_handle_without_leak(self):
        before = process_handle_count()
        for _ in range(32):
            with self.assertRaisesRegex(RuntimeError, "body failure"):
                with shared_open_text(self.path) as stream:
                    stream.read(1)
                    raise RuntimeError("body failure")
            self.assertTrue(stream.closed)
        self.assertEqual(process_handle_count(), before)

    def test_crt_descriptor_transfer_failure_closes_original_win32_handle(self):
        import msvcrt
        before = process_handle_count()
        with patch.object(msvcrt, "open_osfhandle", side_effect=OSError("descriptor failure")):
            for _ in range(32):
                with self.assertRaisesRegex(OSError, "descriptor failure"):
                    with shared_open_text(self.path):
                        pass
        self.assertEqual(process_handle_count(), before)

    def test_text_stream_construction_failure_closes_owned_descriptor(self):
        before = process_handle_count()
        with patch("infra_protocol.os.fdopen", side_effect=ValueError("stream failure")):
            for _ in range(32):
                with self.assertRaisesRegex(ValueError, "stream failure"):
                    with shared_open_text(self.path):
                        pass
        self.assertEqual(process_handle_count(), before)

    def test_invalid_encoding_real_constructor_failure_does_not_leak(self):
        before = process_handle_count()
        for _ in range(32):
            with self.assertRaises(LookupError):
                with shared_open_text(self.path, encoding="not-a-real-codec-atomic-json-test"):
                    pass
        self.assertEqual(process_handle_count(), before)

    def test_first_creation_and_removed_destination_use_atomic_creation(self):
        path = self.path.with_name("created.json")
        with patch("infra_protocol.os.replace", wraps=os.replace) as replace:
            save_json(path, {"generation": 1})
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(read_json(path), {"generation": 1})
            path.unlink()
            save_json(path, {"generation": 2})
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(read_json(path), {"generation": 2})

    def test_missing_file_win32_open_failure_does_not_leak(self):
        missing = self.path.with_name("missing.json")
        before = process_handle_count()
        for _ in range(32):
            self.assertIsNone(read_json(missing))
        self.assertEqual(process_handle_count(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
