"""CPU checks for Windows process liveness; never invoke GPU admission."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_gate


class PidEncodingTests(unittest.TestCase):
    def result(self, stdout, code=0):
        return subprocess.CompletedProcess([], code, stdout=stdout, stderr=b"")

    def test_localized_dead_pid_message_is_not_decoded(self):
        message = "信息: 没有运行的任务匹配指定标准。\r\n".encode("gbk")
        with patch.object(gpu_gate.subprocess, "run", return_value=self.result(message)) as call:
            self.assertFalse(gpu_gate._pid_alive(103556))
        self.assertIs(call.call_args.kwargs["text"], False)

    def test_live_pid_with_localized_image_name_is_retained(self):
        row = b'"'+"测试.exe".encode("gbk")+b'","103556","Console","1","11,000 K"\r\n'
        with patch.object(gpu_gate.subprocess, "run", return_value=self.result(row)):
            self.assertTrue(gpu_gate._pid_alive(103556))

    def test_pid_must_be_exact_csv_pid_not_name_or_substring(self):
        rows = b'"103556.exe","2103556","Console","1","100 K"\r\n'
        with patch.object(gpu_gate.subprocess, "run", return_value=self.result(rows)):
            self.assertFalse(gpu_gate._pid_alive(103556))

    def test_failed_command_is_conservatively_alive(self):
        with patch.object(gpu_gate.subprocess, "run", return_value=self.result(b"", code=1)):
            self.assertTrue(gpu_gate._pid_alive(103556))

    def test_exception_is_conservatively_alive(self):
        with patch.object(gpu_gate.subprocess, "run", side_effect=OSError("unavailable")):
            self.assertTrue(gpu_gate._pid_alive(103556))


if __name__ == "__main__":
    unittest.main()
