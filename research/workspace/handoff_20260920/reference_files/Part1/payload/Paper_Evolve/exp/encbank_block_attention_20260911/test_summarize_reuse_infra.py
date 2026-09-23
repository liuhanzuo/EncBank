"""Small real-schema fixtures for complete, failed and waiting reuse rows."""
from pathlib import Path
import copy
import json
import tempfile
import unittest

from summarize_reuse_infra import collect, render, METRICS, GIB, REUSE_VERSION, GPU


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, arm="D0", n=10):
    job = dict(id=f"n32768_q64_g32_{arm}_reuse{n}", arm=arm,
               cache_mode="block_hot" if arm in ("A", "B") else "cold_hj",
               document_tokens=32768, prompt_tokens=64, generation_tokens=32)
    reuse = dict(protocol=REUSE_VERSION, workflow="fixed_ordered_pack", requests=n,
                 quality_receipt_sha256="quality123", source_sha256={"reader": "source123"})
    identity = {k: v for k, v in job.items() if k != "id"}
    identity.update(reuse=reuse, adapter_kind="trained", reader_implementation="reference", repetitions=1,
                    monitor_policy_version="monitor-v", infra_launcher_sha256="launcher123")
    lease = dict(worker={"pid": 123, "create_time": 1.}, supervisor={"pid": 321, "create_time": 1.})
    attempt = root / job["id"] / "attempts" / "0001"
    requests = [dict(query_index=i, prompt_sha256=f"prompt{i}", generated_tokens=32, decode_steps=31,
        query_state_reused=False, shared_document_cache_unchanged=True, ttft_s=1., decode_wall_s=2.,
        query_e2e_s=3., cumulative_e2e_s=5. + 3. * (i + 1), cumulative_first_token_s=5. + 3. * i + 1.,
        cache_file_load_s=0., document_h2d_bytes=0, route_stats={"prefix_cache_hit": True}) for i in range(n)]
    result = dict(status="complete", timing_eligible=True, identity=identity, reuse=reuse,
        pid=123, supervisor_lease=lease, hardware={"gpu": GPU}, requests=requests,
        store=dict(load_count=1, transfer_count=1, document_prefix_build_count=1, setup_wall_s=5.,
            cold_file_bytes=GIB, hot_file_bytes=2 * GIB if arm in ("A", "B") else 0,
            resident_gpu_cache={"tensor_bytes": 3 * GIB, "unique_storage_bytes": 3 * GIB}),
        summary={"all_queries_with_store_build_s": 5. + 3. * n + .5},
        lifecycle_peak_allocated_bytes=20 * GIB, input={"token_sha256": "same-input"})
    monitor = dict(status="complete", exit_code=0, samples=100, interference=[], worker_lease=lease,
        actual_worker_pid=123, monitor_policy_version="monitor-v", launcher_sha256="launcher123",
        baseline_gpu_used_bytes=GIB, peak_incremental_gpu_bytes=21 * GIB, baseline_is_fixed_before_cuda=True)
    save(attempt / "config.json", identity)
    save(attempt / "result.json", result)
    save(attempt / "monitor.json", monitor)
    return job, dict(status="complete", attempt=str(attempt)), result, monitor, attempt


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def queue(self, fixtures, n=10):
        save(self.root / "plan.json", [f[0] for f in fixtures])
        save(self.root / "status.json", dict(reuse_requests=n, jobs={f[0]["id"]: f[1] for f in fixtures}))

    def test_complete_accounting_uses_setup_once_and_decode_excludes_first_token(self):
        data = fixture(self.root)
        self.queue([data])
        rows = render(self.root)
        d0 = rows[0]
        self.assertTrue(d0["eligible"])
        self.assertEqual(d0["first_query_setup_inclusive_ttft_s"], 6.)
        self.assertEqual(d0["cumulative_1_query_s"], 8.)
        self.assertEqual(d0["cumulative_10_queries_s"], 35.5)
        self.assertEqual(d0["later_queries_mean_ttft_s"], 1.)
        self.assertEqual(d0["decode_tps"], 15.5)
        self.assertEqual(d0["additional_hot_storage_bytes"], 0)
        self.assertEqual(len(rows), 5)
        self.assertTrue((self.root / "REUSE_INFRA.csv").is_file())
        self.assertTrue(all(row["status"] == "not_planned" for row in rows[1:]))

    def test_waiting_oom_and_failed_monitor_have_no_numeric_results(self):
        first, second, third = fixture(self.root, "D0"), fixture(self.root, "A"), fixture(self.root, "B")
        first[1].update(status="waiting_quality", attempt=None, error="waiting_quality: test")
        second[2].update(status="oom", error="CUDA out of memory")
        save(second[4] / "result.json", second[2])
        third[3].update(exit_code=1)
        save(third[4] / "monitor.json", third[3])
        self.queue([first, second, third])
        rows = collect(self.root)
        self.assertEqual([r["status"] for r in rows[:3]], ["waiting_quality", "oom", "invalid_monitor"])
        self.assertTrue(all(r[k] is None for r in rows for k in METRICS))

    def test_identity_and_legacy_mismatch_rejected(self):
        data = fixture(self.root)
        self.queue([data])
        data[2]["identity"]["adapter_kind"] = "strong"
        save(data[4] / "result.json", data[2])
        self.assertEqual(collect(self.root)[0]["status"], "identity_mismatch")
        save(data[4] / "config.json", data[2]["identity"])
        self.assertEqual(collect(self.root)[0]["status"], "not_matching_reuse")

    def test_two_query_smoke_does_not_extrapolate_ten(self):
        data = fixture(self.root, n=2)
        self.queue([data], n=2)
        row = collect(self.root)[0]
        self.assertTrue(row["eligible"])
        self.assertIsNone(row["cumulative_10_queries_s"])

    def test_different_input_streams_blank_both_methods(self):
        first, second = fixture(self.root, "D0"), fixture(self.root, "A")
        second[2]["input"]["token_sha256"] = "different-input"
        save(second[4] / "result.json", second[2])
        self.queue([first, second])
        rows = collect(self.root)
        self.assertEqual([r["status"] for r in rows[:2]], ["cross_method_input_mismatch"] * 2)
        self.assertTrue(all(r[k] is None for r in rows[:2] for k in METRICS))

    def test_never_scan_other_attempts_to_pick_a_faster_complete_run(self):
        data = fixture(self.root)
        self.queue([data])
        data[2]["status"] = "failed"
        save(data[4] / "result.json", data[2])
        alternate = data[4].parent / "0002"
        other = copy.deepcopy(data[2]); other["status"] = "complete"
        save(alternate / "result.json", other)
        save(alternate / "monitor.json", data[3])
        save(alternate / "config.json", other["identity"])
        row = collect(self.root)[0]
        self.assertFalse(row["eligible"])
        self.assertEqual(row["attempt"], str(data[4].resolve()))

    def test_repetition_reloading_is_not_resident_cache_reuse(self):
        data = fixture(self.root)
        data[2]["requests"][1]["cache_file_load_s"] = .5
        save(data[4] / "result.json", data[2])
        self.queue([data])
        self.assertEqual(collect(self.root)[0]["status"], "not_matching_reuse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
