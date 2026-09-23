"""CPU-only checks of cost accumulation and declared trace coverage."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from run_reuse import summarize, save
from run_capacity import cohorts, request_trace, reference_working_set
from launch_capacity import make_jobs


class ProtocolTests(unittest.TestCase):
    def test_atomic_status_retries_transient_windows_sharing(self):
        import os
        original = os.replace
        calls = []
        def transient(source, target):
            calls.append(1)
            if len(calls) <= 2:
                raise PermissionError("temporary shared file handle")
            return original(source, target)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"state.json"
            with patch("run_reuse.os.replace", side_effect=transient), patch("run_reuse.time.sleep"):
                save(target, {"completed": 12})
            self.assertEqual(json.loads(target.read_text()), {"completed": 12})
            self.assertEqual(len(calls), 3)

    def test_fixed_cost_charged_once_at_every_prefix(self):
        records = [{"task": "category_1", "score": .5, "generated_tokens": 2,
                    "timings": {"query_total_s": 2, "ttft_s": 1},
                    "stats": {"peak_allocated_bytes": 100, "transfer_bytes": 5,
                              "cache_fill_transfer_bytes": 7, "cache_resident_bytes": 10}}
                   for _ in range(10)]
        result = summarize(records, 3)
        self.assertEqual([(r["Q"], r["total_s"]) for r in result["cumulative"]], [(1, 5), (10, 23)])
        self.assertEqual(result["cache_fill_d2h_bytes"], 70)

    def test_union_deduplicates_chunks_but_not_document_identity(self):
        docs = [{"document_id": "a", "context_ids": [1]*600},
                {"document_id": "b", "context_ids": [1]*600}]
        rows = [{"document_id": "a", "selected_indices": [0, 1]},
                {"document_id": "a", "selected_indices": [1]},
                {"document_id": "b", "selected_indices": [1]}]
        result = reference_working_set(docs, rows, 4)
        self.assertEqual(result["referenced_tokens_with_sink"], 1+600+88)
        self.assertEqual(result["full_depth_working_set_bytes"], 2756)

    def test_cohorts_cover_every_scbench_question_once_per_method_budget(self):
        docs, queries = [], []
        for task, count in (("scbench_qa_eng", 9), ("scbench_choice_eng", 5)):
            for i in range(count):
                did = f"{task}/{i}"
                docs.append({"document_id": did, "context_ids": [1]*512})
                for j in range(2):
                    queries.append({"id": f"{did}_q{j}", "document_id": did, "stream_index": j, "selected_indices": [0]})
        groups = cohorts("scbench", docs)
        self.assertTrue(all(len(v) <= 4 for v in groups.values()))
        jobs = [j for j in make_jobs("scbench", docs, queries) if not j["smoke"]]
        self.assertEqual(sum(j["queries"] for j in jobs), len(queries)*10)
        for arm in ("prefix", "fix_all", "cacheblend16"):
            for fraction in (.25, .5, 1):
                ids = [qid for job in jobs if job["arm"] == arm and job["fraction"] == fraction for qid in job["ids"]]
                self.assertEqual(sorted(ids), sorted(q["id"] for q in queries))
        self.assertEqual(sum(j["queries"] for j in jobs if j["arm"] == "j0"), len(queries))

    def test_round_robin_preserves_question_order_without_repetition(self):
        docs = [{"document_id": key} for key in ("a", "b")]
        rows = [{"id": f"{key}{i}", "document_id": key, "stream_index": i}
                for key in ("a", "b") for i in range(81)]
        _, trace = request_trace(docs, rows, "locomo")
        self.assertEqual(len(trace), 160)
        self.assertNotEqual(trace[0]["document_id"], trace[1]["document_id"])
        for key in ("a", "b"):
            self.assertEqual([r["stream_index"] for r in trace if r["document_id"] == key], list(range(80)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
