"""Stdlib-only safety/accounting tests; no NVIDIA query, torch import or GPU work."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from infra_protocol import (GIB, MIB, GPU_NAME, make_jobs, parse_memory, parse_nvidia_xml,
    model_processes, validate_idle, validate_shape, summarize_requests, save_json, render_report)
from launch_sparse_infra import find_prior, monitor_worker


def telemetry(used=3 * GIB, processes=None):
    return {"timestamp": "fixture", "name": GPU_NAME, "uuid": "GPU-fixture", "used_bytes": used,
            "total_bytes": 32 * GIB, "processes": processes or []}


class FakeChild:
    pid = 1234
    def __init__(self, finish_after=1):
        self.calls, self.finish_after, self.returncode, self.terminated = 0, finish_after, None, False
    def poll(self):
        if self.returncode is not None:
            return self.returncode
        self.calls += 1
        if self.calls > self.finish_after:
            self.returncode = 0
        return self.returncode
    def terminate(self):
        self.terminated, self.returncode = True, -15
    def kill(self):
        self.terminated, self.returncode = True, -9
    def wait(self, timeout=None):
        return self.returncode or 0


class InfraProtocolTests(unittest.TestCase):
    def test_gib_and_unknown_process_memory(self):
        self.assertEqual(parse_memory("28672 MiB"), 28 * GIB)
        self.assertIsNone(parse_memory("N/A"))
        xml = f"""<nvidia_smi_log><driver_version>x</driver_version><gpu><product_name>{GPU_NAME}</product_name>
          <uuid>GPU-fixture</uuid><fb_memory_usage><used>4096 MiB</used><total>32768 MiB</total></fb_memory_usage>
          <processes><process_info><pid>7</pid><type>G</type><process_name>python.exe</process_name>
          <used_memory>N/A</used_memory></process_info></processes></gpu></nvidia_smi_log>"""
        snap = parse_nvidia_xml(xml)
        self.assertIsNone(snap["processes"][0]["used_bytes"])
        self.assertEqual(len(model_processes(snap)), 1)

    def test_wddm_cg_does_not_reclassify_desktop_as_model(self):
        desktop = {"pid": 4, "type": "C+G", "name": "System", "used_bytes": None}
        unclassified = {"pid": 8, "type": "C+G", "name": "A1.exe", "used_bytes": None}
        self.assertTrue(validate_idle(telemetry(processes=[desktop, unclassified])))
        for row in ({"pid": 5, "type": "G", "name": "python.exe", "used_bytes": None},
                    {"pid": 6, "type": "C", "name": "unknown.exe", "used_bytes": None},
                    {"pid": 7, "type": "M", "name": "service", "used_bytes": None}):
            self.assertFalse(validate_idle(telemetry(processes=[desktop, row])))
        self.assertFalse(validate_idle(telemetry(used=5 * GIB)))
        self.assertTrue(validate_idle(telemetry(used=5 * GIB - 1)))

    def test_missing_full_process_inventory_is_not_idle(self):
        xml = f"<nvidia_smi_log><gpu><product_name>{GPU_NAME}</product_name><fb_memory_usage><used>0 MiB</used><total>32768 MiB</total></fb_memory_usage></gpu></nvidia_smi_log>"
        with self.assertRaises(ValueError):
            parse_nvidia_xml(xml)

    def test_default_matrix_and_native_explicit_additions(self):
        jobs = make_jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({j["id"] for j in jobs}), 12)
        self.assertTrue(all(j["arm"] in ("A", "B") for j in jobs if j["cache_mode"] == "block_hot"))
        self.assertEqual(len(make_jobs([4096], arms=["NATIVE", "FULL"])), 2)

    def test_actual_total_window_no_silent_128k_extension(self):
        cfg = {"model_type": "qwen3", "num_hidden_layers": 36, "max_position_embeddings": 40960}
        self.assertEqual(validate_shape(cfg, 16384, 64, 32, 12, 16, 512)["required_total_positions"], 16481)
        with self.assertRaises(ValueError):
            validate_shape(cfg, 131072, 64, 32, 12, 16, 512)

    def test_decode_denominator_excludes_first_prefill_token(self):
        row = dict(file_load_s=1., h2d_wall_s=2., prefill_wall_s=3., ttft_s=6., decode_wall_s=2.,
                   query_e2e_s=8., decode_steps=31, peak_allocated_bytes=20 * GIB, peak_reserved_bytes=21 * GIB)
        summary = summarize_requests([row, row])
        self.assertEqual(summary["decode_tokens_per_s"], 15.5)
        self.assertEqual(summary["request_peak_allocated_bytes"], 20 * GIB)

    def test_nonzero_exit_invalidates_preexisting_complete_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            attempt = folder / "attempts/0001"
            save_json(attempt / "result.json", {"identity": {"a": 1}, "status": "complete", "timing_eligible": True})
            save_json(attempt / "monitor.json", {"status": "complete", "exit_code": 1})
            self.assertEqual(find_prior(folder, {"a": 1})["status"], "invalid_incomplete_receipt")

    def test_monitor_uses_increment_from_baseline_not_total_28(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            baseline = telemetry(3 * GIB)
            save_json(folder / "progress.json", {"baseline_snapshot": baseline})
            sample = telemetry(30 * GIB, [{"pid": 1234, "type": "C+G", "name": "python.exe", "used_bytes": None}])
            with patch("launch_sparse_infra.snapshot", return_value=sample), patch("launch_sparse_infra.time.sleep"):
                mon = monitor_worker(FakeChild(), folder, folder / "heartbeat.json", .1)
            self.assertEqual(mon["status"], "complete")
            self.assertEqual(mon["peak_incremental_gpu_bytes"], 27 * GIB)
            self.assertIsNone(mon["peak_process_gpu_bytes"])

    def test_monitor_stops_only_owned_worker_on_cap_or_foreign_model(self):
        for cause in ("cap", "foreign"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                save_json(folder / "progress.json", {"baseline_snapshot": telemetry(3 * GIB)})
                sample = telemetry(31 * GIB + MIB if cause == "cap" else 20 * GIB)
                if cause == "foreign":
                    sample["processes"] = [{"pid": 5555, "name": "ollama.exe", "type": "C+G", "used_bytes": None}]
                child = FakeChild(finish_after=2)
                with patch("launch_sparse_infra.snapshot", return_value=sample), patch("launch_sparse_infra.time.sleep"):
                    mon = monitor_worker(child, folder, folder / "heartbeat.json", .1)
                self.assertTrue(child.terminated)
                self.assertEqual(mon["status"], "incremental_cap_exceeded" if cause == "cap" else "invalid_interference")

    def test_partial_na_owned_process_rows_remain_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_json(folder / "progress.json", {"baseline_snapshot": telemetry(3 * GIB)})
            sample = telemetry(20 * GIB, [
                {"pid": 1234, "type": "C+G", "name": "python.exe", "used_bytes": 0},
                {"pid": 1234, "type": "G", "name": "python.exe", "used_bytes": None}])
            with patch("launch_sparse_infra.snapshot", return_value=sample), patch("launch_sparse_infra.time.sleep"):
                mon = monitor_worker(FakeChild(), folder, folder / "heartbeat.json", .1)
            self.assertIsNone(mon["peak_process_gpu_bytes"])
            self.assertEqual(mon["process_memory_available_samples"], 0)

    def test_pending_and_oom_are_distinct_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_jobs([4096], arms=["D0", "A"])
            states = {plan[0]["id"]: {"status": "oom", "error": "original shape"}}
            render_report(tmp, plan, states)
            text = (Path(tmp) / "INFRA_REPORT.md").read_text()
            self.assertIn("oom", text)
            self.assertIn("pending", text)
            tex = (Path(tmp) / "INFRA_TABLE.tex").read_text()
            header = next(line for line in tex.splitlines() if "Method / cache" in line)
            self.assertTrue(header.endswith(r"\\"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
