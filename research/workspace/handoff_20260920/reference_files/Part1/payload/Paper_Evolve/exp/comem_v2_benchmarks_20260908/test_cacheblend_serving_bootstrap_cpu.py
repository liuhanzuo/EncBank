"""CPU-only launch dependencies, strict admission and finite campaign checks."""
import copy
import socket
import unittest
from pathlib import Path
from cacheblend_serving_bootstrap import dependencies_ready, load_plans, child_command, hardware_valid


class BootstrapTests(unittest.TestCase):
    def states(self):
        serving = {"status": "completed", "jobs": {f"base/full/{i}": {"status": "complete", "cells": 12} for i in range(10)}}
        qa = {"status": "completed", "full_timed_generations": 2400,
              "jobs": {f"qa_cost/full/task/{i}": {"status": "complete", "timed_generations": 600} for i in range(4)}}
        return serving, qa

    def test_incomplete_campaign_or_live_model_blocks_dispatch(self):
        serving, qa = self.states()
        self.assertTrue(dependencies_ready(serving, qa, []))
        for entry in ("qa_cost_runner.py", "serving_reuse.py", "cacheblend_serving_driver.py"):
            self.assertFalse(dependencies_ready(serving, qa, [{"CommandLine": "python " + entry}]))
        qa["status"] = "running"
        self.assertFalse(dependencies_ready(serving, qa, []))

    def test_counts_cannot_be_replaced_by_status_only(self):
        serving, qa = self.states()
        serving["jobs"].pop("base/full/0")
        self.assertFalse(dependencies_ready(serving, qa, []))
        serving, qa = self.states()
        qa["jobs"]["qa_cost/full/task/0"]["timed_generations"] = 599
        self.assertFalse(dependencies_ready(serving, qa, []))

    def test_finite_scope_and_smoke_reference_only(self):
        jobs = load_plans()
        self.assertEqual([j["cells"] for j in jobs], [8, 12, 12])
        self.assertEqual([j["query_records"] for j in jobs], [8, 400, 400])
        for job in jobs:
            command = child_command(job, Path("F:/Paper_Evolve/tmp/cb_test_only"))
            self.assertEqual("--verify-fresh" in command, job["phase"] == "smoke")
            self.assertEqual(command[command.index("--gpu-idle-slack-gb")+1], "5.0")
            self.assertNotIn("--allow-eos", command)
            self.assertNotIn("--cpu-test-only", command)

    def test_both_gate_samples_must_be_strictly_below_five(self):
        hw = {"timing_eligible": True, "device_name": "NVIDIA GeForce RTX 5090", "platform": "Windows",
              "hostname": socket.gethostname(), "torch_cpu_threads": 2, "torch_interop_threads": 16,
              "omp_num_threads": "2", "mkl_num_threads": "2", "tokenizers_parallelism": "false",
              "gpu_admission": {"initial_used_gib": 4.999, "recheck_used_gib": 4.999,
                  "other_python_compute_processes": [], "comparison": "strictly_less_than"}}
        hardware_valid(hw)
        for key in ("initial_used_gib", "recheck_used_gib"):
            bad = copy.deepcopy(hw)
            bad["gpu_admission"][key] = 5.0
            with self.assertRaises(ValueError):
                hardware_valid(bad)
        hw["gpu_admission"]["other_python_compute_processes"] = ["123:python.exe"]
        with self.assertRaises(ValueError):
            hardware_valid(hw)


if __name__ == "__main__":
    unittest.main()
