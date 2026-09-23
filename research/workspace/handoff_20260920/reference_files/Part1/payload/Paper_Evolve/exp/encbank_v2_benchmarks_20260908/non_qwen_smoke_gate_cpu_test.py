"""Remote CPU boundary checks for the finite real-weight smoke admission."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
import non_qwen_smoke_driver as driver
from non_qwen_smoke_bootstrap import idle_enough, completion_ok

class GateTests(unittest.TestCase):
    def test_idle_exact_boundary_and_exclusion(self):
        self.assertTrue(idle_enough({"memory_mib": 512, "utilization": 5, "compute_pids": []}))
        for value in ({"memory_mib": 513, "utilization": 0, "compute_pids": []},
                      {"memory_mib": 0, "utilization": 6, "compute_pids": []},
                      {"memory_mib": 0, "utilization": 0, "compute_pids": [123]}):
            self.assertFalse(idle_enough(value))

    def test_native_gate_first_and_no_duplicate_benchmark_rows(self):
        selected = [{"task": "qasper", "index": i} for i in range(14)]
        refs = [selected[i] for i in (0, 3, 9, 13)]
        actions = driver.smoke_actions(("pub", "pub_sink", "fix_all", "j0", "fix_none"), selected, refs)
        self.assertEqual([x[0] for x in actions[:4]], ["native_j0_gate"] * 4)
        self.assertEqual([x[1] for x in actions[:4]], ["j0"] * 4)
        self.assertEqual(len(actions), 70)
        self.assertEqual(len({(arm, row["task"], row["index"]) for _, arm, row in actions}), 70)
        self.assertTrue(all(x[0] == "generation" for x in actions[4:]))

    def test_invalid_reference_fixture_rejected(self):
        selected = [{"task": "x", "index": i} for i in range(14)]
        with self.assertRaises(AssertionError):
            driver.smoke_actions(("pub", "pub_sink", "fix_all", "j0", "fix_none"), selected,
                                 selected[:3] + [{"task": "missing", "index": 0}])

    def test_logits_same_argmax_but_deviating_decode_rejected(self):
        first = torch.tensor([[[1., 0., -1.]]])
        reference = [first, first.clone()]
        perturbed = [first, torch.tensor([[[1., .1, -1.]]])]
        self.assertEqual(int(reference[1].argmax()), int(perturbed[1].argmax()))
        result = driver.compare_reference_logits(perturbed, reference)
        self.assertFalse(result["all_generated_step_logits_pass"])
        self.assertTrue(result["steps"][0]["allclose"])
        self.assertFalse(result["steps"][1]["allclose"])

    def test_logits_count_nonfinite_and_shape_rejected(self):
        x = torch.zeros(1, 1, 3)
        self.assertTrue(driver.compare_reference_logits([x, x], [x, x])["all_generated_step_logits_pass"])
        for actual, expected in (([x], [x, x]), ([x + float("nan")], [x]),
                                 ([torch.zeros(1, 1, 4)], [x]), ([], [])):
            self.assertFalse(driver.compare_reference_logits(actual, expected)["all_generated_step_logits_pass"])

    def predecessor_environment(self, directory):
        root = Path(directory)
        statuspath = root / "outputs/bootstrap_babi16_priority/status.json"
        state = {"status": "completed", "completed_jobs": 6, "completed_predictions": 600,
                 "pid": 91, "queue_pid": 92, "jobs": {}}
        for i in range(6):
            queuepath = root / f"queue{i}.json"
            driver.write(queuepath, {"queue_pid": 200 + i,
                                     "jobs": {f"job{i}": {"pid": 300 + i}, "olddep": {"external_dependency": True}}})
            state["jobs"][f"job{i}"] = {"status": "complete", "exit_code": 0, "model_pid": 100 + i,
                                          "queue_state": str(queuepath)}
        driver.write(statuspath, state)
        driver.write(root / "babi16_priority_full_plan.json", {"jobs": [{"id": f"job{i}"} for i in range(6)]})
        return root, statuspath, state

    def test_predecessor_all_native_receipts_and_dead_pids_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root, statuspath, state = self.predecessor_environment(temp)
            validator = SimpleNamespace(completed=lambda job: True)
            with patch.object(driver, "REMOTE", root), patch.object(driver, "HERE", root), \
                 patch.dict("sys.modules", {"native_priority_receipt": validator}), patch.object(driver, "alive", return_value=False):
                self.assertEqual(driver.predecessor_complete()["native_completion_receipts_verified"], 6)
                for key, value in (("status", "running"), ("completed_jobs", 5), ("completed_predictions", 599)):
                    changed = copy.deepcopy(state)
                    changed[key] = value
                    driver.write(statuspath, changed)
                    with self.assertRaises(AssertionError):
                        driver.predecessor_complete()
                driver.write(statuspath, state)
                validator.completed = lambda job: job["id"] != "job4"
                with self.assertRaises(AssertionError):
                    driver.predecessor_complete()

    def test_each_predecessor_pid_class_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, _ = self.predecessor_environment(temp)
            with patch.object(driver, "REMOTE", root), patch.object(driver, "HERE", root), \
                 patch.dict("sys.modules", {"native_priority_receipt": SimpleNamespace(completed=lambda job: True)}):
                for live_pid in (91, 92, 103, 203, 303):
                    with patch.object(driver, "alive", side_effect=lambda p: p == live_pid):
                        with self.assertRaises(AssertionError):
                            driver.predecessor_complete()

    def test_complete_marker_requires_70_rows_four_references_and_all_logits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(completion_ok(root))
            marker = {"status": "complete", "actual_rows": 70, "actual_extra_references": 4,
                      "all_raw_reference_ids_equal": True, "all_generated_step_logits_passed": True}
            driver.write(root / "non_qwen_SMOKE_COMPLETE.json", marker)
            self.assertFalse(completion_ok(root))
            for i in range(70):
                driver.write(root / "records" / f"{i}.json", {})
            for i in range(4):
                driver.write(root / "references" / f"{i}.json", {})
            self.assertTrue(completion_ok(root))
            marker["all_generated_step_logits_passed"] = False
            driver.write(root / "non_qwen_SMOKE_COMPLETE.json", marker)
            self.assertFalse(completion_ok(root))

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateTests))
    assert not torch.cuda.is_initialized()
    print(json.dumps({"complete": result.wasSuccessful(), "tests_run": result.testsRun, "device": "cpu",
                      "cuda_initialized": False, "threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
