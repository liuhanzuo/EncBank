"""CPU-only temporary receipts; never write mock measurements into results/."""
import json
from pathlib import Path
import tempfile
import unittest

from summarize_shared_state_diagnostic import summarize


def save(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


class SummaryChecks(unittest.TestCase):
    def fixture(self, folder):
        attempt = folder/"attempt"
        attempt.mkdir()
        source = {"shared_state_diagnostic.py": "stub-helper", "sdpa_numeric_oracle.py": "stub-oracle"}
        ident = {"shared_state_diagnostic_only": True, "profile_reader_only": False,
            "backend_model_parity": False, "arm": "D0", "cache_mode": "cold_hj", "document_tokens": 4096,
            "prompt_tokens": 64, "generation_tokens": 4, "repetitions": 1,
            "shared_state_sources_sha256": source, "infra_launcher_sha256": "stub-launcher", "monitor_policy_version": "stub-policy"}
        metric = {"max_abs": .125, "rms": .1, "signed_mean_difference": .08, "centered_rms": .06}
        oracle = {"status": "complete", "oracle_can_explain_captured_call": True,
            "captured_vs_repeat_auto_match": True, "receipt_path": str(attempt/"oracle.json"),
            "branches": {name: {"status": "complete", "versus_fp64": {"max_abs": .01, "rms": .001},
                "backend_evidence": {"operator_events": [], "cuda_kernel_timeline_observed": False}}
                for name in ("gqa_math", "repeat_math", "repeat_automatic")}}
        shared = {"status": "complete", "protocol_checks_passed": True, "formal_timing_eligible": False,
            "prefill_count": 1, "decode_steps": 3, "source_sha256": "stub-helper", "receipt_path": str(attempt/"shared.json"),
            "steps": [{"step": n, "reference_greedy_id": n, "candidate_greedy_id": n, "numerical": metric.copy()}
                for n in (1,2,3)], "same_input_oracle": {"step": 1, "layer": 0, "oracle": oracle}}
        lease = {"worker": {"pid": 123, "create_time": 9.}}
        result = {"identity": ident, "status": "complete", "timing_eligible": False, "requests": [], "summary": {},
            "shared_state_receipt": shared, "shared_state_gpu_validation": {"passed": True, "device": "cuda", "source_sha256": source},
            "supervisor_lease": lease, "pid": 123, "input": {"token_sha256": "stub-input"},
            "platform": "Windows", "hardware": {"gpu": "NVIDIA GeForce RTX 5090"}}
        monitor = {"status": "complete", "exit_code": 0, "worker_lease": lease, "actual_worker_pid": 123,
            "launcher_sha256": "stub-launcher", "monitor_policy_version": "stub-policy", "baseline_gpu_used_bytes": 2*2**30,
            "peak_incremental_gpu_bytes": 20*2**30, "samples": 2, "interference": [],
            "baseline_snapshot": {"name": "NVIDIA GeForce RTX 5090"}}
        previous = folder/"previous.json"
        save(previous, {"identity": ident.copy(), "status": "failed", "input": result["input"],
            "backend_model_parity": {"passed": False, "stages": [{"phase": "prefill", "passed": False,
                "numerical": {"max_abs": .125, "rms": .027, "greedy_equal": True}}]}})
        def write():
            save(attempt/"result.json", result); save(attempt/"monitor.json", monitor)
            save(attempt/"shared.json", shared); save(attempt/"oracle.json", oracle)
        write()
        return attempt, previous, result, shared, oracle, monitor, write

    def test_valid_preserves_separate_old_failure_and_unavailable_centered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a,p,*_=self.fixture(root)
            report=summarize(a,root/"report",p)
            self.assertTrue(report["report_checks_passed"])
            self.assertFalse(report["numerical_equivalence_asserted"])
            self.assertFalse(report["previous_separate_prefills"]["passed"])
            self.assertIsNone(report["previous_separate_prefills"]["steps"][0]["centered_rms"])
            self.assertFalse(report["same_qkv_oracle"]["branches"]["repeat_automatic"]["actual_backend"]["cuda_kernel_timeline_observed"])

    def test_oracle_capture_mismatch_is_not_explanation_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a,p,r,s,o,m,write=self.fixture(root)
            o["captured_vs_repeat_auto_match"]=False; write()
            report=summarize(a,root/"report",p)
            self.assertFalse(report["report_checks_passed"])
            self.assertFalse(report["checks"]["oracle_actual_capture_matches"])
            self.assertTrue((root/"report/SHARED_STATE_DIAGNOSTIC.md").is_file())

    def test_missing_centered_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a,p,r,s,o,m,write=self.fixture(root)
            del s["steps"][0]["numerical"]["centered_rms"]; write()
            report=summarize(a,root/"report",p)
            self.assertFalse(report["checks"]["centered_metrics_present"])
            self.assertNotIn("centered_rms",report["current"]["steps"][0]["numerical"])

    def test_monitor_wrong_worker_and_formal_contamination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a,p,r,s,o,m,write=self.fixture(root)
            m["actual_worker_pid"]=321; r["timing_eligible"]=True; r["summary"]={"ttft_s": .2}; write()
            report=summarize(a,root/"report",p)
            self.assertFalse(report["checks"]["monitor_worker_bound"])
            self.assertFalse(report["checks"]["diagnostic_only"])
            self.assertFalse(report["report_checks_passed"])

    def test_only_exact_torchversion_repr_difference_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a,p,r,s,o,m,write=self.fixture(root)
            o["torch_version"]="2.7.1+cu128"; write()
            saved=o.copy(); saved["torch_version"]=repr(o["torch_version"])
            save(a/"oracle.json",saved)
            report=summarize(a,root/"report",p)
            self.assertTrue(report["report_checks_passed"])
            self.assertFalse(report["oracle_saved_receipt_comparison"]["exact_equal"])
            self.assertTrue(report["oracle_saved_receipt_comparison"]["torch_version_repr_only"])
            saved["captured_vs_repeat_auto_match"]=False; save(a/"oracle.json",saved)
            report=summarize(a,root/"report",p)
            self.assertFalse(report["checks"]["saved_oracle_receipt_semantics_equal"])


if __name__ == "__main__":
    unittest.main()
