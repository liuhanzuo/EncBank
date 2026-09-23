"""CPU-only acceptance/rejection tests, using seven temporary measurement stubs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from compare_backend_infra import compare, CASES
from infra_protocol import case_id, summarize_requests
import test_compare_decode_infra as decode_fixtures


class BackendComparisonChecks(unittest.TestCase):
    def setUp(self):
        fixture = decode_fixtures.DecodeComparisonChecks()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.base = fixture.base
        self.seed_result = json.loads((fixture.paths["reference"] / "result.json").read_text())
        self.seed_monitor = json.loads((fixture.paths["reference"] / "monitor.json").read_text())
        self.formal = self.base / "formal"
        self.out = self.base / "backend_report"
        self.sequence = 0
        self.write_fixture()

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_fixture(self):
        self.paths = {}
        sources = {"sparse_reader.py":"reference-stub-sha", "optimized_sparse_reader.py":"optimized-stub-sha",
                   "backend_sparse_reader.py":"backend-stub-sha", "validate_backend_cpu.py":"backend-validator-stub-sha",
                   "test_backend_sparse_reader.py":"backend-tests-stub-sha"}
        for variant in ("decode_v2", "backend_v3", "native"):
            jobs = {}
            for arm, mode in ((("NATIVE", "cold_hj"),) if variant == "native" else CASES):
                key = (variant, arm, mode)
                name = case_id(arm, mode, 4096, 64, 32)
                attempt = self.formal / variant / name / "attempts" / "0001"
                r,m = copy.deepcopy(self.seed_result),copy.deepcopy(self.seed_monitor)
                ident=r["identity"]
                ident.update(reader_implementation="reference" if variant=="native" else variant,
                    arm=arm,cache_mode=mode,profile_reader_only=False,profiler_sha256=None,
                    backend_model_parity=False,backend_parity_sha256=None,optimized_gpu_validation=False,
                    optimized_validation_sha256=None,
                    optimized_reader_sha256=None if variant=="native" else "optimized-stub-sha",
                    native_reader_sha256="native-stub-sha" if variant=="native" else None,
                    backend_reader_sha256="backend-stub-sha" if variant=="backend_v3" else None,
                    backend_gpu_validation=True,backend_validation_sha256={k:v for k,v in sources.items()
                        if k in {"backend_sparse_reader.py","validate_backend_cpu.py","test_backend_sparse_reader.py"}})
                r.pop("optimized_gpu_validation",None)
                r.update(fixed_generation_ignores_eos=True,backend_gpu_validation=dict(passed=True,device="cuda",
                    tests_run=8,failures=0,errors=0,skipped=0,source_sha256=copy.deepcopy(sources),
                    same_math_backend_bitwise_passed=True,
                    same_backend_contract=dict(passed=True,tests_run=8,failures=0,errors=0,skipped=0)))
                selected=list(range(4,8)) if arm=="B" else list(range(8))
                selected_tokens=2048 if arm=="B" else 4096
                by_layer={str(l):(4097 if l<16 or arm!="B" else 2049) for l in range(12,36)}
                route=dict(selection_source="prompt-attention-mass" if arm=="B" else "all-blocks",
                    selected_indices=selected,candidate_blocks=8,selected_blocks=len(selected),candidate_tokens=4096,
                    selected_tokens=selected_tokens,target_retain_ratio=.5 if arm=="B" else 1.,
                    actual_retain_ratio=.5 if arm=="B" else 1.,token_budget=selected_tokens,budget_overflow_tokens=0,
                    sink_tokens=1,resume_j=12,fusion_layer=12 if arm=="NATIVE" else 16,
                    probe_indices=list(range(48,64)),probe_mode="native-dense" if arm=="NATIVE" else "wrapper",
                    original_query_start=4097,document_kv_tokens_by_layer=by_layer,
                    document_kv_bytes=sum(by_layer.values())*4096,unselected_late_kv_tokens=0,
                    hot_hits=8 if mode=="block_hot" else 0,hot_misses=0 if mode=="block_hot" else 8)
                if arm!="NATIVE":
                    route["block_scores"]=[.01*(i+1) for i in range(8)]
                hot=mode=="block_hot"
                store=attempt/"store"
                store.mkdir(parents=True,exist_ok=True)
                (store/"cold.pt").write_bytes(b"C"*512)
                if hot:(store/"hot.pt").write_bytes(b"H"*256)
                r["store"]=dict(cold_file_bytes=512,hot_file_bytes=256 if hot else 0,
                    cold_tensor_bytes=384,hot_tensor_bytes=128 if hot else 0,
                    request_h2d_bytes=128 if hot else 384,persistent_location=str(store))
                for q in r["requests"]:
                    q["route_stats"]=copy.deepcopy(route)
                    q["state_positions"]=dict(query_position=95,pack_position=4192)
                    q["h2d_bytes"]=r["store"]["request_h2d_bytes"]
                    if variant=="backend_v3":
                        q["decode_wall_s"] /= 2
                        q["decode_step_wall_s"]=[x/2 for x in q["decode_step_wall_s"]]
                        q["decode_step_cuda_event_s"]=[x/2 for x in q["decode_step_cuda_event_s"]]
                        q["query_e2e_s"]=q["ttft_s"]+q["decode_wall_s"]
                r["summary"]=summarize_requests(r["requests"])
                self.paths[key]=attempt
                self.save_result(key,r)
                self.put(attempt/"monitor.json",m)
                jobs[name]=dict(status="complete",attempt=str(attempt))
            self.put(self.formal/variant/"status.json",dict(status="complete",jobs=jobs))

    def save_result(self,key,r,*,refresh_summary=False):
        if refresh_summary:r["summary"]=summarize_requests(r["requests"])
        p=self.paths[key]
        self.put(p/"result.json",r)
        self.put(p/"config.json",r["identity"])
        for i,q in enumerate(r["requests"]):self.put(p/f"request_{i:03d}.json",q)

    def mutate(self,change,key=("backend_v3","B","cold_hj"),*,refresh_summary=False):
        r=json.loads((self.paths[key]/"result.json").read_text())
        change(r)
        self.save_result(key,r,refresh_summary=refresh_summary)

    def run_compare(self):
        self.sequence+=1
        return compare(self.formal,self.out/str(self.sequence))

    def test_valid_seven_cells_recomputed_ratios_and_extra_hot_storage(self):
        result=self.run_compare()
        self.assertTrue(result["parity_passed"])
        self.assertEqual(len(result["rows"]),7)
        self.assertEqual(len(result["checks"]),9)
        self.assertEqual(len(result["native_control_checks"]),6)
        for ratio in result["backend_ratios"]:
            self.assertEqual(ratio["decode_throughput_ratio"],2.)
            self.assertEqual(ratio["ttft_speedup"],1.)
        for row in result["rows"]:
            self.assertIsNone(row["sampled_process_gpu_bytes"])
            self.assertEqual(row["total_cache_file_bytes"],768 if row["cache_mode"]=="block_hot" else 512)

    def test_probe_score_roundoff_is_reported_without_false_route_failure(self):
        self.mutate(lambda r:r["requests"][1]["route_stats"]["block_scores"].__setitem__(2,.030001))
        result=self.run_compare()
        self.assertTrue(result["parity_passed"])
        self.assertAlmostEqual(max(c["block_score_max_abs"] or 0 for c in result["checks"]),.000001)

    def test_tokens_discrete_route_and_final_positions_fail_parity_but_keep_rows(self):
        cases=[lambda r:r["requests"][0]["generated_ids"].__setitem__(0,999),
               lambda r:r["requests"][0]["route_stats"]["selected_indices"].__setitem__(0,0),
               lambda r:r["requests"][0]["state_positions"].update(pack_position=2144)]
        for index,change in enumerate(cases):
            with self.subTest(case=index):
                self.write_fixture();self.mutate(change)
                result=self.run_compare()
                self.assertFalse(result["parity_passed"])
                self.assertFalse(result["backend_parity_passed"])
                self.assertEqual(len(result["rows"]),7)

    def test_both_sides_wrong_position_does_not_pass_merely_by_matching(self):
        for variant in ("decode_v2","backend_v3"):
            self.mutate(lambda r:r["requests"][0]["state_positions"].update(query_position=96),
                        (variant,"B","cold_hj"))
        self.assertFalse(self.run_compare()["backend_parity_passed"])

    def test_native_control_token_difference_is_separate_and_fails_global_parity(self):
        self.mutate(lambda r:r["requests"][0]["generated_ids"].__setitem__(0,999),("native","NATIVE","cold_hj"))
        result=self.run_compare()
        self.assertTrue(result["backend_parity_passed"])
        self.assertFalse(result["native_control_parity_passed"])
        self.assertFalse(result["parity_passed"])

    def test_recipe_adapter_input_runtime_and_native_mismatches_are_rejected(self):
        cases=[lambda r:r["identity"].update(adapter_sha256="other"),
               lambda r:r["identity"].update(infra_worker_sha256="old-harness"),
               lambda r:r["input"].update(token_sha256="different-input"),
               lambda r:r["hardware"].update(autocast="float32"),
               lambda r:r.update(hostname="remote-3090")]
        for index,change in enumerate(cases):
            with self.subTest(case=index):
                self.write_fixture();self.mutate(change,("native","NATIVE","cold_hj"))
                with self.assertRaises(ValueError):self.run_compare()

    def test_missing_failed_cpu_or_wrong_source_gpu_receipt_is_rejected(self):
        cases=[lambda r:r.pop("backend_gpu_validation"),
               lambda r:r["backend_gpu_validation"]["same_backend_contract"].update(passed=False),
               lambda r:r["backend_gpu_validation"].update(same_math_backend_bitwise_passed=False),
               lambda r:r["backend_gpu_validation"].update(device="cpu"),
               lambda r:r["backend_gpu_validation"]["source_sha256"].update({"backend_sparse_reader.py":"wrong"})]
        for index,change in enumerate(cases):
            with self.subTest(case=index):
                self.write_fixture();self.mutate(change)
                with self.assertRaises(ValueError):self.run_compare()

    def test_tiny_auto_failure_is_visible_without_invalidating_same_backend_contract(self):
        self.mutate(lambda r:r["backend_gpu_validation"].update(passed=False,failures=2))
        result=self.run_compare()
        self.assertTrue(result["parity_passed"])
        row=next(x for x in result["rows"] if x["implementation"]=="backend_v3" and x["arm"]=="B" and x["cache_mode"]=="cold_hj")
        self.assertFalse(row["tiny_auto_passed"])
        self.assertEqual(row["tiny_auto_failures"],2)
        self.assertTrue(row["tiny_same_backend_contract_passed"])
        self.assertIn("未通过(f=2,e=0)",(self.out/str(self.sequence)/"COMPARISON.md").read_text(encoding="utf-8"))

    def test_profile_partial_tokens_nonfinite_and_inconsistent_summary_are_rejected(self):
        cases=[lambda r:r["identity"].update(profile_reader_only=True),
               lambda r:r.update(backend_model_parity={"passed":True}),
               lambda r:r["requests"][0]["generated_ids"].pop(),
               lambda r:r["requests"][0].update(decode_wall_s=float("nan")),
               lambda r:r["requests"][0]["route_stats"]["block_scores"].__setitem__(0,float("nan")),
               lambda r:r["summary"].update(decode_tokens_per_s=10000.)]
        for index,change in enumerate(cases):
            with self.subTest(case=index):
                self.write_fixture();self.mutate(change)
                with self.assertRaises(ValueError):self.run_compare()

    def test_monitor_failure_process_cap_and_missing_native_are_rejected(self):
        key=("backend_v3","B","cold_hj")
        for changes in (dict(exit_code=1),dict(peak_process_gpu_bytes=29*2**30),dict(actual_worker_pid=999)):
            with self.subTest(changes=changes):
                self.write_fixture()
                p=self.paths[key]/"monitor.json"
                m=json.loads(p.read_text());m.update(changes);self.put(p,m)
                with self.assertRaises(ValueError):self.run_compare()
        self.write_fixture()
        self.put(self.formal/"native/status.json",dict(status="complete",jobs={}))
        with self.assertRaises(ValueError):self.run_compare()

    def test_actual_cache_file_size_is_required(self):
        (self.paths[("backend_v3","B","block_hot")]/"store/hot.pt").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError,"cache file byte receipt"):
            self.run_compare()

    def test_cli_returns_nonzero_on_parity_failure_with_report_preserved(self):
        self.mutate(lambda r:r["requests"][0]["generated_ids"].__setitem__(0,999))
        target=Path(__file__).with_name("compare_backend_infra.py")
        result=subprocess.run([sys.executable,str(target),"--input",str(self.formal),"--out",str(self.out)],
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertFalse(json.loads((self.out/"COMPARISON.json").read_text())["parity_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
