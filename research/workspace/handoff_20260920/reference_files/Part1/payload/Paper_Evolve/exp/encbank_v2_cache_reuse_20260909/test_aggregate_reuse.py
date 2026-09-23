"""Synthetic JSON-only checks: no model imports, GPU calls, or real score reads."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import aggregate_reuse as agg


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def save_lines(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def fixture(root, query_count=80):
    path = root/"fixtures"
    documents, queries = [], []
    for d in range(10):
        docid = f"doc-{d:02d}"
        order = [f"{docid}/q{i}" for i in range(query_count)]
        documents.append({"document_id": docid, "context_ids": list(range(10, 18)),
            "raw_context": "Synthetic context", "formatted_context": "Synthetic context",
            "query_count": query_count, "query_ids_ordered": order,
            "prefixes": {str(p): order if p == "full" else order[:p] for p in agg.PREFIXES}})
        for i, qid in enumerate(order):
            queries.append({"id": qid, "document_id": docid, "stream_index": i,
                "query_ids": [20+i, 30], "selected_indices": [0],
                "pack": {"selected_indices": [0], "context_tokens": 8,
                         "query_tokens": 2, "read_pack_tokens": 11},
                "sample": {"task": f"category_{1+i%4}", "category": 1+i%4,
                           "max_new_tokens": 3, "answers": ["synthetic"]}})
    save(path/"manifest.json", {"protocol": "locomo-real-multiquery-reuse-v1",
        "document_count": 10, "categories": [1, 2, 3, 4], "unique_queries": len(queries),
        "documents_file": "documents.json", "queries_file": "queries.jsonl",
        "tokenizer_path": "synthetic/Qwen3"})
    save(path/"documents.json", documents)
    save_lines(path/"queries.jsonl", queries)
    return path, {d["document_id"]: d for d in documents}, {q["id"]: q for q in queries}


def hardware():
    return {"timing_eligible": True, "device_name": agg.GPU, "device": "cuda:0",
        "platform": "Windows", "hostname": "synthetic-local", "device_uuid": "synthetic-uuid",
        "torch": "synthetic", "transformers": "synthetic", "cuda_runtime": "synthetic",
        "torch_cpu_threads": 2, "torch_interop_threads": 16, "omp_num_threads": "2",
        "mkl_num_threads": "2", "tokenizers_parallelism": "false",
        "gpu_admission": {"comparison": "strictly_less_than", "effective_idle_slack_gib": 5,
            "initial_used_gib": 3, "recheck_used_gib": 3.1, "other_python_compute_processes": []}}


def create_attempt(root, fixture_path, document, queries, arm, name="0001", online_total=1.35):
    docid = document["document_id"]
    path = root/"results/locomo/full"/docid/arm/"attempts"/name
    expected = [queries[q] for q in document["query_ids_ordered"]]
    config = {"protocol": "real-question-reuse-v1", "dataset": "locomo", "smoke": False,
        "document": docid, "arm": arm, "query_limit": 0, "ids": [q["id"] for q in expected],
        "fixtures": str(fixture_path.resolve()), "model": "synthetic/Qwen3",
        "dtype": "bfloat16", "j": 12, "chunk_size": 512, "topk": 12,
        "query_order_seed": agg.SEED, "answer_cache": False, "source_context_truncation": "none",
        "cache_budget_bytes": 2592}
    marker = {"status": "complete", "smoke": False, "timing_eligible": True,
        "dataset": "locomo", "document": docid, "arm": arm,
        "expected_queries": len(expected), "completed_queries": len(expected),
        "new_generations": len(expected), "extra_reference_generations": 0, "hardware": hardware()}
    payload = {"j0": 0, "prefix": 0, "fix_all": 936, "cacheblend16": 2592}[arm]
    meta = {"signature": {"model_id": "synthetic/Qwen3", "arm": arm, "adapter": None,
        "dtype": "torch.bfloat16", "j": 0 if arm in ("j0", "prefix") else 12,
        "model_config": {"model_type": "qwen3", "hidden_size": 4, "num_hidden_layers": 36,
            "num_key_value_heads": 1, "head_dim": 2}}, "n_tokens": 8}
    store = {"document_prepare_s": 1., "fixed_cost_s": 6., "budget_bytes": 2592,
        "write": {"write_total_s": 2., "write_compute_s": .8, "write_device_to_cpu_s": .5,
            "write_serialize_s": .6, "payload_tensor_bytes": payload, "raw_token_bytes": 64,
            "serialized_bytes": payload+320, "write_peak_allocated_bytes": 10000+payload},
        "startup": {"startup_load_s": 3., "startup_read_bytes": payload+320,
            "resident_cpu_tensor_bytes": payload+64, "tier": "cpu"}}
    records = []
    for row in expected:
        records.append({"id": row["id"], "document_id": docid, "arm": arm,
            "stream_index": row["stream_index"], "task": row["sample"]["task"],
            "query_ids": row["query_ids"], "query_tokens": 2, "selected_indices": [0],
            "max_new_tokens": 3, "generated_ids": [101, 102], "generated_tokens": 2,
            "terminated_by_eos": True, "prediction": "synthetic", "scored_prediction": "synthetic",
            "score": .5, "timing_eligible": True, "smoke_reference_equal": None,
            "stats": {"selected_indices": [0], "read_tokens": 11, "query_tokens": 2,
                "generated_tokens": 2, "fixed_generation_length": False, "capture_calls": 0,
                "decode_steps": 2, "total_s": online_total-.35, "ttft_s": .3,
                "decode_s": online_total-.65, "load_s": .05, "transfer_s": .05,
                "transfer_bytes": 100, "peak_allocated_bytes": 11000+payload,
                "incremental_peak_bytes": 1000, "cache_resident_bytes": 200 if arm == "prefix" else payload,
                "cache_budget_bytes": 2592, "cache_fill_transfer_bytes": 200 if arm == "prefix" else 0,
                "cache_fill_s": .05 if arm == "prefix" else 0},
            "timings": {"query_tokenization_s": .1, "external_retrieval_s": .2,
                "text_decode_s": .05, "ttft_s": .6, "query_total_s": online_total}})
    save(path/"config.json", config)
    save(path/"COMPLETED.json", marker)
    save(path/"store_cost.json", store)
    save(path/"store/store.json", meta)
    save_lines(path/"measurements.jsonl", records)
    return path


class AggregateReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reuse-aggregate-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fp, self.docs, self.queries = fixture(self.root)
        self.document = self.docs["doc-00"]
        self.expected = [self.queries[q] for q in self.document["query_ids_ordered"]]

    def attempt(self, arm="fix_all", **kwargs):
        path = create_attempt(self.root, self.fp, self.document, self.queries, arm, **kwargs)
        return path, agg.validate_attempt(path, self.document, self.expected, self.fp)

    def test_cumulative_fixed_once_category_output_memory_and_transfer(self):
        _, attempt = self.attempt()
        one, ten = agg.prefix_summary(attempt, 1), agg.prefix_summary(attempt, 10)
        self.assertAlmostEqual(one["cumulative_total_s"], 7.35)
        self.assertAlmostEqual(ten["cumulative_total_s"], 19.5)
        self.assertAlmostEqual(ten["mean_online_query_s"], 1.35)
        self.assertAlmostEqual(ten["ttft_p50_s"], .6)
        self.assertAlmostEqual(ten["ttft_p95_s"], .6)
        self.assertEqual(ten["quality_by_category"]["category_1"]["n"], 3)
        self.assertEqual(ten["quality_by_category"]["category_1"]["f1_percent"], 50)
        self.assertEqual(ten["output_tokens_range"], [2, 2])
        self.assertEqual(ten["output_tokens_sum"], 20)
        self.assertEqual(ten["decode_forward_calls"], 20)
        self.assertEqual(ten["persistent_kv_tensor_bytes"], 864)
        self.assertEqual(ten["persistent_representation_tensor_bytes"], 936)
        self.assertEqual(ten["online_h2d_bytes"], 1000)
        _, prefix = self.attempt("prefix")
        ps = agg.prefix_summary(prefix, 10)
        self.assertEqual(ps["persistent_kv_tensor_bytes"], 0)
        self.assertEqual(ps["maximum_resident_kv_bytes"], 200)
        self.assertFalse(ps["prefix_online_kv_is_serialized"])
        self.assertEqual(ps["online_cache_fill_d2h_bytes"], 2000)

    def test_actual_query_order_pack_cap_and_natural_eos_rejected(self):
        path, good = self.attempt()
        changes = [lambda r: r[0].update(query_ids=[999]),
                   lambda r: r[0].update(selected_indices=[]),
                   lambda r: r[0].update(max_new_tokens=4),
                   lambda r: r[0].update(stream_index=1),
                   lambda r: r[0]["stats"].update(decode_steps=1),
                   lambda r: r[0]["stats"].update(capture_calls=1),
                   lambda r: r[0]["stats"].update(fixed_generation_length=True),
                   lambda r: r[0].update(score=float("nan"))]
        for change in changes:
            rows = copy.deepcopy(good["records"])
            change(rows)
            save_lines(path/"measurements.jsonl", rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                agg.validate_attempt(path, self.document, self.expected, self.fp)

    def test_strict_gate_hardware_smoke_and_incomplete_rejected(self):
        path, _ = self.attempt()
        good = agg.read_json(path/"COMPLETED.json")
        changes = [lambda r: r["hardware"]["gpu_admission"].update(initial_used_gib=5),
                   lambda r: r["hardware"]["gpu_admission"].update(recheck_used_gib=5),
                   lambda r: r["hardware"]["gpu_admission"].update(effective_idle_slack_gib=6),
                   lambda r: r["hardware"]["gpu_admission"].update(other_python_compute_processes=[123]),
                   lambda r: r["hardware"].update(device_name="NVIDIA GeForce RTX 3090"),
                   lambda r: r["hardware"].update(torch_cpu_threads=4),
                   lambda r: r.update(smoke=True), lambda r: r.update(timing_eligible=False),
                   lambda r: r.update(completed_queries=79)]
        for change in changes:
            marker = copy.deepcopy(good)
            change(marker)
            save(path/"COMPLETED.json", marker)
            with self.subTest(change=change), self.assertRaises(ValueError):
                agg.validate_attempt(path, self.document, self.expected, self.fp)

    def test_pairing_requires_input_model_budget_hardware_not_outputs_or_payload(self):
        attempts = {arm: self.attempt(arm)[1] for arm in agg.ARMS}
        self.assertTrue(agg.paired_document(attempts))
        self.assertNotEqual(attempts["j0"]["store"]["write"]["payload_tensor_bytes"],
                            attempts["fix_all"]["store"]["write"]["payload_tensor_bytes"])
        changed = copy.deepcopy(attempts)
        changed_row = changed["prefix"]["records"][0]
        changed_row.update(generated_ids=[103, 104, 105], generated_tokens=3, terminated_by_eos=False)
        changed_row["stats"].update(generated_tokens=3, decode_steps=2)
        path = Path(changed["prefix"]["path"])
        save_lines(path/"measurements.jsonl", changed["prefix"]["records"])
        changed["prefix"] = agg.validate_attempt(path, self.document, self.expected, self.fp)
        self.assertTrue(agg.paired_document(changed))
        for field in ("input_sha256", "model", "budget_bytes", "hardware"):
            changed = copy.deepcopy(attempts)
            changed["prefix"]["contract"][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                agg.paired_document(changed)

    def test_observed_break_even_sustained_only_without_extrapolation(self):
        def trace(fixed, times):
            return {"store": {"fixed_cost_s": fixed}, "records": [
                {"id": str(i), "timings": {"query_total_s": t}} for i, t in enumerate(times)]}
        baseline = trace(0, [2, 2, 2, 2])
        no = agg.observed_break_even(trace(0, [1, 1, 1, 6]), baseline)
        self.assertIsNone(no["observed_sustained_from_Q"])
        late = agg.observed_break_even(trace(2, [1, 1, 1, 1]), baseline)
        self.assertEqual(late["observed_sustained_from_Q"], 3)
        regained = agg.observed_break_even(trace(0, [1, 4, 0, 1]), baseline)
        self.assertEqual(regained["observed_sustained_from_Q"], 3)
        self.assertEqual(regained["observed_through_Q"], 4)
        self.assertIn("no linear extrapolation", regained["beyond_observed_trace"])

    def test_cluster_bootstrap_document_unit_deterministic_and_partial_prohibited(self):
        summaries = {}
        for i in range(10):
            base = {"cumulative_total_s": 20+i, "mean_online_query_s": 2+i,
                "ttft_p50_s": 1+i, "ttft_p95_s": 2+i, "output_tokens_mean": 3+i,
                "quality_by_category": {"category_1": {"n": i+1, "sum_f1": (i+1)*.5}}}
            candidate = copy.deepcopy(base)
            for key in base:
                if key != "quality_by_category":
                    candidate[key] -= 1
            candidate["quality_by_category"]["category_1"]["sum_f1"] = (i+1)*.75
            summaries[str(i)] = {"j0": {"full": base}, "fix_all": {"full": candidate}}
        a = agg.cluster_bootstrap(summaries, "fix_all", "j0", "full", resamples=100)
        b = agg.cluster_bootstrap(summaries, "fix_all", "j0", "full", resamples=100)
        self.assertEqual(a, b)
        self.assertEqual(a["cluster"], "document")
        self.assertEqual(a["metrics"]["cumulative_total_s"]["ci95"], [-1, -1])
        self.assertEqual(a["metrics"]["f1_percentage_points/category_1"]["ci95"], [25, 25])
        with self.assertRaises(ValueError):
            agg.cluster_bootstrap(dict(list(summaries.items())[:9]), "fix_all", "j0", "full")

    def test_complete_campaign_uses_all_denominators_and_five_prefixes(self):
        for document in self.docs.values():
            for arm in agg.ARMS:
                create_attempt(self.root, self.fp, document, self.queries, arm)
        result = agg.aggregate(self.root/"results/locomo/full", self.fp, resamples=100)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["valid_complete_arm_jobs"], 40)
        self.assertEqual(result["paired_complete_documents"], 10)
        self.assertEqual(result["valid_completed_query_records"], 3200)
        self.assertEqual(len(result["paired_document_bootstrap"]), 30)
        self.assertTrue(result["inference_allowed"])
        self.assertEqual(set(result["complete_arm_descriptive_summaries"]["doc-00"]["j0"]),
                         {"1", "10", "50", "80", "full"})
        self.assertIn("CPU tensor/chunk", agg.render_report(result))

    def test_partial_attempt_selection_never_stitches_or_chooses_fastest(self):
        path, _ = self.attempt("j0", name="0001")
        marker = agg.read_json(path/"COMPLETED.json")
        marker["hardware"]["gpu_admission"]["recheck_used_gib"] = 5
        save(path/"COMPLETED.json", marker)
        self.attempt("j0", name="0002", online_total=2)
        self.attempt("j0", name="0003", online_total=1)
        pending = self.root/"results/locomo/full/doc-00/prefix/attempts/0001"
        save(pending/"status.json", {"status": "running", "completed_queries": 30, "expected_queries": 80})
        result = agg.aggregate(self.root/"results/locomo/full", self.fp, resamples=100)
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["inference_allowed"])
        self.assertEqual(result["paired_document_bootstrap"], [])
        self.assertEqual(result["valid_completed_query_records"], 80)
        self.assertEqual(result["valid_complete_arm_jobs"], 1)
        self.assertEqual(len(result["rejected_attempts"]), 1)
        self.assertEqual(len(result["duplicate_valid_attempts"]), 1)
        self.assertEqual(len(result["progress"]), 1)
        self.assertTrue(result["attempts"]["doc-00"]["j0"].endswith("0002"))

    def test_default_cli_empty_campaign_generates_only_partial_artifacts(self):
        with mock.patch.object(agg, "HERE", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agg.main([]), 0)
        out = self.root/"results/locomo/aggregate"
        result = agg.read_json(out/"aggregate.json")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["valid_completed_query_records"], 0)
        self.assertEqual(result["paired_document_bootstrap"], [])
        self.assertIn("部分完成", (out/"REPORT.md").read_text(encoding="utf-8"))

    def test_corrupted_fixture_pack_is_rejected_before_reading_results(self):
        rows = list(self.queries.values())
        rows[0]["pack"]["read_pack_tokens"] += 1
        save_lines(self.fp/"queries.jsonl", rows)
        with self.assertRaises(ValueError):
            agg.aggregate(self.root/"results/locomo/full", self.fp)


if __name__ == "__main__":
    unittest.main()
