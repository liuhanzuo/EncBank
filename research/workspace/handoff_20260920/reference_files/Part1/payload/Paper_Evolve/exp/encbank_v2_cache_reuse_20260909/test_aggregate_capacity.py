"""Synthetic CPU tests: no Torch, GPU, scorer, or real result modifications."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import aggregate_capacity as agg


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture():
    docs = [{"document_id": "scbench_qa_eng/doc0", "context_tokens": 4}]
    rows = [{"id": f"q{i}", "document_id": docs[0]["document_id"], "stream_index": i,
             "query_ids": [11, 12+i], "selected_indices": [0],
             "pack": {"selected_indices": [0], "read_pack_tokens": 7},
             "sample": {"task": "scbench_qa_eng", "max_new_tokens": 4, "ground_truth": "answer"}}
            for i in range(2)]
    return docs, rows


def hardware():
    return {"timing_eligible": True, "device_name": "NVIDIA GeForce RTX 5090", "device": "cuda:0",
            "platform": "Windows", "hostname": "synthetic-only", "torch_cpu_threads": 2,
            "torch_interop_threads": 16, "omp_num_threads": "2", "mkl_num_threads": "2",
            "tokenizers_parallelism": "false", "gpu_admission": {"comparison": "strictly_less_than",
                "effective_idle_slack_gib": 5, "initial_used_gib": 3, "recheck_used_gib": 3,
                "other_python_compute_processes": []}}


def write_attempt(root, job):
    path = root/job["id"]/"attempts/0001"
    arm = job["arm"]
    cfg = {"protocol": agg.VERSION, "dataset": job["dataset"], "cohort": job["cohort"], "arm": arm,
           "fraction": job["fraction"], "cache_budget_bytes": job["budget"], "working_set": job["working_set"],
           "documents": [d["document_id"] for d in job["documents"]], "ids": [r["id"] for r in job["rows"]],
           "fixtures": job["fixtures"], "model": agg.MODEL, "dtype": "bfloat16", "chunk_size": 512,
           "topk": 12, "j": 12, "query_order_seed": agg.SEED, "answer_cache": False,
           "source_context_truncation": "none", "j0_budget_independent": arm == "j0", "smoke": False}
    marker = {k: cfg[k] for k in ("dataset", "cohort", "arm", "smoke", "cache_budget_bytes")}
    marker.update(status="complete", timing_eligible=True, expected_queries=2, completed_queries=2,
                  new_generations=2, extra_reference_generations=0, hardware=hardware())
    store = {"fixed_cost_s": .25, "document_prepare_s": {job["documents"][0]["document_id"]: .25},
             "initial_persistent_kv_bytes": 0, "serialized_kv_bytes": 0, "raw_token_bytes": 32}
    records = []
    for i, ref in enumerate(job["rows"]):
        hit = 5 if i and arm != "j0" else 0
        built = arm in ("fix_all", "cacheblend16") and not i
        fill_bytes = 1000 if arm != "j0" and not i else 0
        cap = {"released": True, "persistent_bytes_after": 1000, "miss_tensor_bytes": fill_bytes,
               "evicted_tensor_bytes": 0, "bypass_tensor_bytes": 0, "build_s": .1 if built else 0,
               "cpu_copy_s": .05 if built else 0, "build_calls": 2 if built else 0}
        cap["events"] = [{"kind": kind, "document_id": doc, "chunk_index": idx, "tokens": tokens,
                          "occurrences": 1, "status": "hit" if i else "miss_admitted", "tensor_bytes": size}
                         for kind, doc, idx, tokens, size in
                         (("sink", "__shared_sink__", -1, 1, 200), ("chunk", ref["document_id"], 0, 4, 800))]
        stats = {"document_id": ref["document_id"], "selected_indices": [0], "read_tokens": 7,
                 "query_tokens": 2, "generated_tokens": 2, "fixed_generation_length": False, "decode_steps": 2,
                 "total_s": 1.5, "ttft_s": .5, "decode_s": 1., "load_s": .2, "transfer_s": .1,
                 "transfer_bytes": 1060, "cache_fill_s": .15 if fill_bytes else 0,
                 "cache_fill_transfer_bytes": fill_bytes, "peak_allocated_bytes": 20000,
                 "incremental_peak_bytes": 5000, "cache_resident_bytes": 0 if arm == "j0" else 1000,
                 "evicted_nodes": 0, "evicted_bytes": 0, "capture_calls": 2 if built else 0,
                 "cache_hit_context_tokens": hit, "cache_miss_context_tokens": 5-hit,
                 "cache_hit_sink_tokens": int(hit > 0), "cache_budget_bytes": job["budget"],
                 "cache_lookup_applicable": arm != "j0", "document_capture_calls": int(built),
                 "sink_capture_calls": int(built), "query_capture_calls": 0,
                 "cache_bypass_tokens": 0, "cache_bypass_bytes": 0, "cache_active_payload_bytes": 1000,
                 "capacity_cache": cap}
        if arm == "prefix":
            for key in ("cache_hit_sink_tokens", "cache_bypass_tokens", "cache_bypass_bytes", "cache_active_payload_bytes"):
                del stats[key]
        records.append({"id": ref["id"], "document_id": ref["document_id"], "arm": arm,
            "trace_index": i, "stream_index": i, "task": ref["sample"]["task"], "timing_eligible": True,
            "smoke_reference_equal": None, "query_ids": ref["query_ids"], "query_tokens": 2,
            "selected_indices": [0], "max_new_tokens": 4, "generated_ids": [20, 21], "generated_tokens": 2,
            "terminated_by_eos": True, "prediction": "answer", "scored_prediction": "answer", "score": float(i),
            "stats": stats, "timings": {"document_bind_s": .1, "query_tokenization_s": .2,
                "external_retrieval_s": .3, "text_decode_s": .05, "ttft_s": 1.1, "query_total_s": 2.15}})
    save(path/"config.json", cfg)
    save(path/"COMPLETED.json", marker)
    save(path/"store_cost.json", store)
    (path/"measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    return path


class CapacityAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="capacity_aggregate_test_")
        self.root = Path(self.temp.name)
        self.docs, self.rows = fixture()
        self.jobs = agg.expected_jobs("scbench", self.docs, self.rows, self.root/"fixtures")

    def tearDown(self):
        self.temp.cleanup()

    def complete_all(self):
        return {job["id"]: write_attempt(self.root, job) for job in self.jobs}

    def test_j0_shared_not_triple_counted(self):
        self.complete_all()
        result = agg.aggregate(self.jobs, self.root)
        self.assertTrue(result["all_full_workloads_complete"])
        self.assertEqual(result["scope"]["expected_jobs"], 10)
        self.assertEqual(result["progress"]["unique_measured_outputs"], 20)
        self.assertEqual(result["progress"]["unique_j0_outputs"], 2)
        self.assertEqual(len(result["capacity_points"]), 3)
        self.assertEqual(len({p["measurement_job_ids"]["j0"] for p in result["capacity_points"]}), 1)

    def test_actual_bytes_not_raw_tokens_or_query_sums_as_peak(self):
        paths = self.complete_all()
        job = next(j for j in self.jobs if j["arm"] == "fix_all")
        a = agg.validate_attempt(paths[job["id"]], job)
        s = agg.summarize([a])
        self.assertEqual(s["maximum_persistent_cpu_representation_bytes"], 1000)
        self.assertEqual(s["cache_fill_d2h_tensor_bytes"], 1000)
        self.assertEqual(s["h2d_bytes_including_raw_ids"], 2120)
        self.assertEqual(s["maximum_total_gpu_allocated_bytes"], 20000)
        self.assertEqual(s["raw_cpu_token_bytes_by_cohort"][job["cohort"]], 32)
        self.assertEqual(s["sink_tokens_hit"], 1)
        self.assertEqual(s["sink_tokens_miss"], 1)
        self.assertEqual(s["document_tokens_hit"], 4)
        self.assertEqual(s["document_tokens_miss"], 4)
        self.assertAlmostEqual(s["cumulative_prepare_plus_query_s"], 4.55)
        self.assertAlmostEqual(s["ttft_mean_s"], 1.1)
        self.assertEqual(s["quality_by_task"]["scbench_qa_eng"]["metric"], "official_case_insensitive_substring_accuracy")

    def test_missing_or_partial_does_not_make_campaign_complete(self):
        paths = self.complete_all()
        job = next(j for j in self.jobs if j["arm"] == "cacheblend16" and j["fraction"] == .5)
        (paths[job["id"]]/"COMPLETED.json").unlink()
        result = agg.aggregate(self.jobs, self.root)
        self.assertFalse(result["all_full_workloads_complete"])
        self.assertEqual(result["progress"]["unique_measured_outputs"], 18)
        self.assertEqual(result["progress"]["valid_complete_cohorts"], 0)
        self.assertEqual(result["progress"]["paired_capacity_points"], 2)
        self.assertIn("部分进度", agg.render_report(result))

    def test_smoke_only_never_counts(self):
        job = self.jobs[0]
        path = write_attempt(self.root, job)
        marker = agg.read_json(path/"COMPLETED.json")
        marker["smoke"] = True
        save(path/"COMPLETED.json", marker)
        result = agg.aggregate(self.jobs, self.root)
        self.assertEqual(result["progress"]["unique_measured_outputs"], 0)
        self.assertEqual(len(result["rejected_complete_attempts"]), 1)

    def test_config_model_dtype_budget_order_rejected(self):
        job = self.jobs[0]
        path = write_attempt(self.root, job)
        original = agg.read_json(path/"config.json")
        for key, value in (("model", "other-checkpoint"), ("dtype", "float16"),
                           ("cache_budget_bytes", job["budget"]+1), ("ids", ["q1", "q0"]),
                           ("cohort", "wrong-cohort")):
            with self.subTest(key=key):
                cfg = copy.deepcopy(original)
                cfg[key] = value
                save(path/"config.json", cfg)
                with self.assertRaises(ValueError):
                    agg.validate_attempt(path, job)
        save(path/"config.json", original)

    def test_raw_query_pack_cap_and_trace_mismatch_rejected(self):
        job = self.jobs[0]
        path = write_attempt(self.root, job)
        original = [json.loads(line) for line in (path/"measurements.jsonl").read_text().splitlines()]
        for key, value in (("query_ids", [12, 11]), ("selected_indices", []), ("max_new_tokens", 3), ("trace_index", 1)):
            with self.subTest(key=key):
                rows = copy.deepcopy(original)
                rows[0][key] = value
                (path/"measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                with self.assertRaises(ValueError):
                    agg.validate_attempt(path, job)

    def test_cross_arm_budget_hardware_and_input_contract_rejected(self):
        paths = self.complete_all()
        selected = {j["arm"]: agg.validate_attempt(paths[j["id"]], j) for j in self.jobs if j["fraction"] == 1}
        agg.paired(selected)
        for change in ("budget", "hardware", "input"):
            arms = copy.deepcopy(selected)
            if change == "budget":
                arms["prefix"]["budget_bytes"] += 1
            else:
                arms["prefix"]["contract"]["hardware" if change == "hardware" else "input_digest"] = "changed"
            with self.subTest(change=change), self.assertRaises(ValueError):
                agg.paired(arms)

    def test_cache_event_bytes_and_false_overlap_hits_rejected(self):
        job = next(j for j in self.jobs if j["arm"] == "fix_all")
        path = write_attempt(self.root, job)
        original = [json.loads(line) for line in (path/"measurements.jsonl").read_text().splitlines()]
        for change in ("bytes", "false_hit"):
            rows = copy.deepcopy(original)
            event = rows[0]["stats"]["capacity_cache"]["events"][1]
            event["tensor_bytes" if change == "bytes" else "status"] = 801 if change == "bytes" else "hit"
            (path/"measurements.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.subTest(change=change), self.assertRaises(ValueError):
                agg.validate_attempt(path, job)

    def test_prefix_sink_hit_and_unknown_bypass(self):
        paths = self.complete_all()
        job = next(j for j in self.jobs if j["arm"] == "prefix")
        s = agg.summarize([agg.validate_attempt(paths[job["id"]], job)])
        self.assertEqual(s["sink_tokens_hit"], 1)
        self.assertEqual(s["document_tokens_hit"], 4)
        self.assertIsNone(s["cache_bypass_tensor_bytes"])
        self.assertIsNone(s["maximum_active_cpu_payload_bytes"])

    def test_exact_plan_and_trace_parity_without_importing_model_code(self):
        source = ast.parse((agg.HERE/"run_capacity.py").read_text(encoding="utf-8"))
        funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ("cohorts", "request_trace", "reference_working_set")]
        ns = {"random": random, "require": agg.require}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "run_capacity.py", "exec"), ns)
        docs = [{**d, "context_ids": list(range(d["context_tokens"]))} for d in self.docs]
        self.assertEqual(ns["cohorts"]("scbench", docs).keys(), agg.cohorts("scbench", self.docs).keys())
        _, rows = ns["request_trace"](docs, self.rows, "scbench", False)
        self.assertEqual(rows, agg.trace("scbench", self.docs, self.rows))
        self.assertEqual(ns["reference_working_set"](docs, rows, agg.BPT), agg.working_set(self.docs, rows))
        launch = ast.parse((agg.HERE/"launch_capacity.py").read_text(encoding="utf-8"))
        fn = next(n for n in launch.body if isinstance(n, ast.FunctionDef) and n.name == "make_jobs")
        ns["ARMS"] = agg.ARMS
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "launch_capacity.py", "exec"), ns)
        # Launcher smoke requires both SC tasks; duplicate a tiny choice doc.
        docs2 = docs+[{**docs[0], "document_id": "scbench_choice_eng/doc0"}]
        rows2 = self.rows+[{**r, "id": "choice-"+r["id"], "document_id": docs2[1]["document_id"]} for r in self.rows]
        actual = {j["id"]: (j["ids"], j["budget_bytes_expected"]) for j in ns["make_jobs"]("scbench", docs2, rows2) if not j["smoke"]}
        expected = {j["id"]: ([r["id"] for r in j["rows"]], j["budget"]) for j in agg.expected_jobs("scbench", docs2, rows2, self.root)}
        self.assertEqual(actual, expected)

    def test_incremental_document_reader_and_percentiles(self):
        path = self.root/"documents.json"
        save(path, [{"v": [1, 2]}, {"v": [3]}])
        self.assertEqual(list(agg.iter_json_array(path)), [{"v": [1, 2]}, {"v": [3]}])
        self.assertEqual(agg.percentile([1, 3], .5), 2)
        self.assertAlmostEqual(agg.percentile([1, 3], .95), 2.9)


if __name__ == "__main__":
    unittest.main()
