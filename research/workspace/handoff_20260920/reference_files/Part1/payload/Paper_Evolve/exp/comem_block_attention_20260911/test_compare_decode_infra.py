"""CPU-only comparison rejection checks; all measurements below are test stubs.

No GPU, Torch, SSH, or real results directories are used. The fixtures exercise
report acceptance/rejection, not the performance of either reader.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_decode_infra import compare
from infra_protocol import GIB, GPU_NAME, summarize_requests


HERE = Path(__file__).resolve().parent


class DecodeComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stub_decode_compare_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.reference, self.optimized = self.base / "reference", self.base / "decode_v2"
        self.destination = self.base / "report"
        self.write_pair()

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def write_pair(self, generation=32, repetitions=3):
        self.paths = {}
        for implementation, folder in (("reference", self.reference), ("decode_v2", self.optimized)):
            attempt = folder / "D0_cold_hj_4096" / "attempts" / "0001"
            ident = dict(protocol="TEST-STUB", reader_implementation=implementation,
                reader_sha256="reference-stub-sha", optimized_reader_sha256=(
                    "optimized-stub-sha" if implementation == "decode_v2" else None),
                infra_worker_sha256="worker-stub-sha", infra_launcher_sha256="launcher-stub-sha",
                infra_protocol_sha256="protocol-stub-sha", infra_processes_sha256="process-stub-sha",
                monitor_policy_version="stub-monitor-v1", adapter_kind="strong", adapter_sha256="adapter-stub-sha",
                arm="D0", cache_mode="cold_hj", document_tokens=4096, prompt_tokens=64,
                generation_tokens=generation, repetitions=repetitions, seed=42,
                optimized_gpu_validation=True,
                optimized_validation_sha256={"test_optimized_sparse_reader.py": "tests-stub-sha",
                                             "validate_optimized_cpu.py": "validation-stub-sha"})
            requests = []
            for i in range(repetitions):
                decode_wall, ttft = float(i + 1), .2 + .1 * i
                steps = generation - 1
                requests.append(dict(repetition=i, generated_ids=list(range(generation)),
                    generated_tokens=generation, decode_steps=steps,
                    file_load_s=.01, h2d_wall_s=.02, prefill_wall_s=ttft-.03,
                    ttft_s=ttft, decode_wall_s=decode_wall, query_e2e_s=ttft+decode_wall,
                    decode_step_wall_s=[decode_wall/steps]*steps,
                    decode_step_cuda_event_s=[decode_wall/steps/2]*steps,
                    peak_allocated_bytes=20*GIB, peak_reserved_bytes=21*GIB,
                    route_stats={"selected_indices": [0, 1], "document_kv_bytes": 123456}))
            worker_pid = 101 if implementation == "reference" else 102
            lease = dict(worker=dict(pid=worker_pid, create_time=1000.),
                         supervisor=dict(pid=100, create_time=900.),
                         owned_root=dict(pid=worker_pid, create_time=1000.), redirector=None)
            baseline = dict(name=GPU_NAME, uuid="TEST-STUB-GPU-UUID", driver_version="TEST-STUB-DRIVER",
                            used_bytes=2*GIB, total_bytes=32*GIB)
            hardware = dict(gpu=GPU_NAME, total_memory_bytes=32*GIB, torch="TEST-STUB-TORCH",
                cuda_runtime="TEST-STUB-CUDA", transformers="TEST-STUB-TRANSFORMERS", torch_cpu_threads=2,
                torch_interop_threads=16, gpu_incremental_budget_bytes=28*GIB,
                torch_allocator_cap_bytes=27.5*GIB, non_allocator_reserve_bytes=.5*GIB,
                autocast="cuda bfloat16", baseline_gpu_used_bytes=2*GIB)
            result = dict(status="complete", timing_eligible=True, identity=ident,
                pid=worker_pid, hostname="TEST-STUB-HOST", platform="Windows", supervisor_lease=lease,
                hardware=hardware, input=dict(token_sha256="test-input-sha", probe_indices=[0, 1],
                                               document_tokens=4096, prompt_tokens=64),
                optimized_gpu_validation=dict(passed=True, tests_run=8, failures=0, errors=0,
                    skipped=0, device="cuda", host="TEST-STUB-HOST",
                    source_sha256={"optimized_sparse_reader.py": "optimized-stub-sha",
                                   "sparse_reader.py": "reference-stub-sha",
                                   "test_optimized_sparse_reader.py": "tests-stub-sha",
                                   "validate_optimized_cpu.py": "validation-stub-sha"}),
                requests=requests, summary=summarize_requests(requests))
            monitor = dict(status="complete", exit_code=0, samples=10, interference=[],
                baseline_gpu_used_bytes=2*GIB, baseline_snapshot=baseline,
                peak_incremental_gpu_bytes=22*GIB, peak_process_gpu_bytes=None,
                cap_bytes=28*GIB, baseline_is_fixed_before_cuda=True,
                child_pid=worker_pid, actual_worker_pid=worker_pid, supervisor_pid=100,
                worker_lease=copy.deepcopy(lease), launcher_sha256="launcher-stub-sha",
                monitor_policy_version="stub-monitor-v1")
            self.put(attempt / "result.json", result)
            self.put(attempt / "monitor.json", monitor)
            self.put(folder / "status.json", dict(status="complete", jobs={
                "D0_cold_hj_4096": dict(status="complete", attempt=str(attempt))}))
            self.paths[implementation] = attempt

    def mutate(self, filename, callback, implementation="decode_v2"):
        path = self.paths[implementation] / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        callback(value)
        self.put(path, value)

    def run_compare(self):
        return compare(self.reference, self.optimized, self.destination)

    def assert_rejected(self):
        with self.assertRaises(ValueError):
            self.run_compare()
        self.assertFalse((self.destination / "COMPARISON.json").exists())

    def test_valid_matched_pair_preserves_weighted_decode_and_repetition_scope(self):
        result = self.run_compare()
        self.assertTrue(result["parity_passed"])
        self.assertEqual(len(result["checks"]), 3)
        for row in result["rows"]:
            self.assertEqual(row["repetitions"], 3)
            self.assertAlmostEqual(row["decode_tokens_per_s"], 93/6)
            self.assertAlmostEqual(row["ttft_s"], .3)
            self.assertAlmostEqual(row["query_e2e_s"], 2.3)

    def test_dynamic_report_does_not_claim_32_outputs_or_3_repeats(self):
        self.write_pair(generation=7, repetitions=2)
        result = self.run_compare()
        self.assertEqual(len(result["checks"]), 2)
        report = (self.destination / "COMPARISON.md").read_text(encoding="utf-8")
        self.assertIn("固定输出token数[7]", report)
        self.assertIn("重复次数[2]", report)
        self.assertNotIn("固定32-token", report)

    def test_missing_or_failed_gpu_validation_is_rejected(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.write_pair()
                self.mutate("result.json", lambda r: r.pop("optimized_gpu_validation") if missing
                            else r["optimized_gpu_validation"].update(passed=False))
                self.assert_rejected()

    def test_hardware_runtime_and_physical_gpu_changes_are_rejected(self):
        cases = [("result.json", lambda r: r["hardware"].update(gpu="NVIDIA GeForce RTX 3090")),
                 ("result.json", lambda r: r["hardware"].update(torch="other-runtime")),
                 ("result.json", lambda r: r["hardware"].update(autocast="float32")),
                 ("monitor.json", lambda m: m["baseline_snapshot"].update(uuid="another-5090")),
                 ("monitor.json", lambda m: m["baseline_snapshot"].update(driver_version="other-driver"))]
        for filename, change in cases:
            with self.subTest(file=filename, mutation=cases.index((filename, change))):
                self.write_pair()
                self.mutate(filename, change)
                self.assert_rejected()

    def test_adapter_input_and_source_changes_are_rejected(self):
        cases = [lambda r: r["identity"].update(adapter_sha256="other-adapter"),
                 lambda r: r["identity"].update(infra_worker_sha256="different-harness"),
                 lambda r: r["input"].update(token_sha256="different-input")]
        for index, change in enumerate(cases):
            with self.subTest(case=index):
                self.write_pair()
                self.mutate("result.json", change)
                self.assert_rejected()

    def test_missing_decode_token_or_step_is_rejected(self):
        for change in (lambda r: r["requests"][0]["generated_ids"].pop(),
                       lambda r: r["requests"][0].update(decode_steps=32),
                       lambda r: r["requests"].pop()):
            self.write_pair()
            self.mutate("result.json", change)
            self.assert_rejected()

    def test_monitor_receipt_cannot_be_borrowed_from_another_worker_or_source(self):
        cases = [lambda m: m.update(actual_worker_pid=999),
                 lambda m: m.update(launcher_sha256="other-launcher"),
                 lambda m: m.update(monitor_policy_version="other-policy"),
                 lambda m: m["worker_lease"]["worker"].update(create_time=2000.)]
        for index, change in enumerate(cases):
            with self.subTest(case=index):
                self.write_pair()
                self.mutate("monitor.json", change)
                self.assert_rejected()

    def test_different_host_or_gpu_validated_sources_are_rejected(self):
        cases = [lambda r: r.update(hostname="other-host"),
                 lambda r: r["optimized_gpu_validation"]["source_sha256"].update(
                     optimized_sparse_reader_py="additional-unmatched-source")]
        for index, change in enumerate(cases):
            with self.subTest(case=index):
                self.write_pair()
                self.mutate("result.json", change)
                self.assert_rejected()

    def test_failed_unobserved_interfered_or_over_budget_monitor_is_rejected(self):
        cases = [dict(exit_code=1), dict(samples=0), dict(status="monitor_failed"),
                 dict(interference=[{"pid": 999}]), dict(peak_incremental_gpu_bytes=28*GIB+1),
                 dict(baseline_gpu_used_bytes=5*GIB)]
        for changes in cases:
            with self.subTest(changes=changes):
                self.write_pair()
                self.mutate("monitor.json", lambda m: m.update(changes))
                self.assert_rejected()

    def test_token_and_route_difference_are_not_reported_as_equivalent(self):
        for key in ("generated_ids", "route_stats"):
            with self.subTest(key=key):
                self.write_pair()
                def change(result):
                    if key == "generated_ids":
                        result["requests"][1][key][3] = 999
                    else:
                        result["requests"][1][key]["selected_indices"] = [1, 2]
                self.mutate("result.json", change)
                result = self.run_compare()
                self.assertFalse(result["parity_passed"])
                self.assertEqual(sum(not c["tokens_equal"] or not c["route_equal"]
                                     for c in result["checks"]), 1)

    def test_cli_propagates_token_parity_failure(self):
        self.mutate("result.json", lambda r: r["requests"][0]["generated_ids"].__setitem__(0, 999))
        result = subprocess.run([sys.executable, str(HERE / "compare_decode_infra.py"),
            "--reference", str(self.reference), "--optimized", str(self.optimized),
            "--out", str(self.destination)], text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.destination / "COMPARISON.json").read_text(encoding="utf-8"))
        self.assertFalse(report["parity_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
