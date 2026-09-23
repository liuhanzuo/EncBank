"""Matched natural-EOS QA timing, local RTX 5090 only; uncached native reader.

Instrumentation wraps (never replaces) ExplicitPackReader and its native core.
Each query captures its selected chunks afresh. These are NOT cached steady-state
or write-once timings. CPU input validation, scoring, output serialization,
startup and warmup are excluded and labelled separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for folder in (ROOT / "COMem", ROOT / "exp", HERE):
    sys.path.insert(0, str(folder))

VERSION = "native-explicit-qa-cost-v1"
REQUIRED_GPU = "NVIDIA GeForce RTX 5090"
THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
TIME_FIELDS = ("cpu_prompt_tokenize_bm25_s", "h2d_s", "selected_capture_write_s",
               "lower_query_prefill_s", "upper_pack_prefill_s", "decode_s",
               "decode_forward_s", "output_text_decode_s", "prepared_input_generation_s",
               "ttft_from_prepared_input_s", "ttft_from_marked_prompt_s", "total_from_marked_prompt_s")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def input_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_job(plan_path, job_id):
    plan = read_json(plan_path)
    jobs = [j for j in plan["jobs"] if j["id"] == job_id]
    require(len(jobs) == 1, "Unknown/duplicate QA job")
    job = dict(jobs[0])
    require(job["task"] in {"qasper", "hotpotqa"} and job["arm"] in {"fix_all", "j0"}, "Invalid QA task/arm")
    require(job["max_new_tokens"] == {"qasper": 128, "hotpotqa": 32}[job["task"]], "Official cap changed")
    require(job["allow_later_eos"] is True, "Natural EOS is required")
    rows = [json.loads(s) for s in Path(job["input_file"]).read_text(encoding="utf-8").splitlines() if s.strip()]
    require(len(rows) == 200 and {r["index"] for r in rows} == set(range(200)), "Full 200-row source contract missing")
    require(len({r["id"] for r in rows}) == 200, "Duplicate source IDs")
    require(len(set(job["indices"])) == len(job["indices"]), "Duplicate planned indices")
    for row in rows:
        require(row["task"] == job["task"] and row["max_new_tokens"] == job["max_new_tokens"], "Input task/cap differs")
        require(row["chunk_size"] == 512 and row["context_token_count"] == len(row["context_ids"]), "Invalid context contract")
        require(row["query_ids"] and not row["fixed_generation_length"], "Full query/natural EOS required")
        selected = row["selected_indices"]
        require(len(set(selected)) == len(selected) and all(0 <= i < math.ceil(len(row["context_ids"])/512) for i in selected), "Invalid selected pack")
        count = 1 + len(row["query_ids"]) + sum(len(row["context_ids"][i*512:(i+1)*512]) for i in selected)
        require(count == row["pack"]["read_pack_tokens"], "Read pack count differs")
    rows = {r["index"]: r for r in rows}
    config = {"protocol": VERSION, "job_id": job_id, "task": job["task"], "arm": job["arm"],
              "phase": job["phase"], "indices": job["indices"], "repetitions": job["timed_repetitions"],
              "warmups": job["unmeasured_warmups"], "max_new_tokens": job["max_new_tokens"],
              "input": input_identity(job["input_file"]), "model": str(Path(plan["model"]).resolve()),
              "j": job["effective_j"], "dtype": "bfloat16", "attn_impl": "sdpa", "seed": 42,
              "thread_environment": THREAD_ENV, "torch_cpu_threads": 2, "torch_interop_threads": 16,
              "generation_cache": False, "selected_chunks_recaptured_per_query": True,
              "whole_document_write_once": False, "fixed_generation_length": False,
              "smoke_extra_reference_calls": len(job["indices"]) if job["phase"] == "smoke" else 0}
    return job, rows, config


@contextmanager
def replace_method(obj, name, value):
    # Restore an existing instance override exactly, otherwise remove our override.
    owned = name in obj.__dict__
    old = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def clock(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class NativeTrace:
    """Small scalar hooks, no retained logits and no altered decode loop."""
    def __init__(self, reader):
        self.reader = reader
        self.times = defaultdict(float)
        self.generated_ids = None
        self.decode_calls = 0
        self.first_token_at = self.native_finished_at = None
        self.selected_state_logical_bytes = 0
        self.write_depth = 0

    def timed(self, method, field):
        def wrapped(*args, **kwargs):
            start = clock(self.reader.device)
            result = method(*args, **kwargs)
            self.times[field] += clock(self.reader.device) - start
            return result
        return wrapped

    def selected(self, method):
        def wrapped(*args, **kwargs):
            outer = self.write_depth == 0
            self.write_depth += 1
            start = clock(self.reader.device) if outer else None
            try:
                result = method(*args, **kwargs)
            finally:
                self.write_depth -= 1
            if outer:
                self.times["selected_capture_write_s"] += clock(self.reader.device) - start
                # Logical live selected state, NOT a persistent serialized cache size.
                def count(value):
                    if hasattr(value, "numel"):
                        return value.numel() * value.element_size()
                    if isinstance(value, (tuple, list)):
                        return sum(count(v) for v in value)
                    return 0
                self.selected_state_logical_bytes += count(result)
                bottom = getattr(self.reader, "_bottom", None)
                if bottom:
                    self.selected_state_logical_bytes += sum(count((layer.keys, layer.values)) for layer in bottom["cache"].layers)
            return result
        return wrapped

    @contextmanager
    def install(self):
        from contextlib import ExitStack
        reader = self.reader
        with ExitStack() as stack:
            for name in (["build_bottom"] if hasattr(reader, "build_bottom") else ["write_chunk", "write_chunks"]):
                stack.enter_context(replace_method(reader, name, self.selected(getattr(reader, name))))
            for name, field in (("write_prefill", "lower_query_prefill_s"), ("read_prefill", "upper_pack_prefill_s")):
                stack.enter_context(replace_method(reader, name, self.timed(getattr(reader, name), field)))
            original_step = reader.decode_step
            def step(*args, **kwargs):
                start = clock(reader.device)
                if self.first_token_at is None:
                    # Native core has selected and appended the first token before this call.
                    self.first_token_at = start
                self.decode_calls += 1
                result = original_step(*args, **kwargs)
                self.times["decode_forward_s"] += clock(reader.device) - start
                return result
            stack.enter_context(replace_method(reader, "decode_step", step))
            original_core = reader._decode_from_pack
            def core(*args, **kwargs):
                result = original_core(*args, **kwargs)
                self.native_finished_at = clock(reader.device)
                self.generated_ids = list(result)
                return result
            stack.enter_context(replace_method(reader, "_decode_from_pack", core))
            stack.enter_context(replace_method(reader.tokenizer, "decode", self.timed(reader.tokenizer.decode, "output_text_decode_s")))
            yield self


def native_reference(reader, input_ids, row):
    """Uninstrumented official adapter; capture only its returned native IDs."""
    from reader_adapter import ExplicitPackReader
    original = reader._decode_from_pack
    result = {}
    def record(*args, **kwargs):
        ids = original(*args, **kwargs)
        result["ids"] = list(ids)
        return ids
    with replace_method(reader, "_decode_from_pack", record):
        text = ExplicitPackReader(reader).generate_from_ids(input_ids,
            context_token_count=row["context_token_count"], selected_indices=row["selected_indices"],
            chunk_size=row["chunk_size"], max_new_tokens=row["max_new_tokens"])
    return text, result["ids"]


def measure_native(reader, input_ids, row):
    from reader_adapter import ExplicitPackReader
    from prepare_qa_cost import native_decode_accounting
    trace = NativeTrace(reader)
    with trace.install():
        started = clock(reader.device)
        text = ExplicitPackReader(reader).generate_from_ids(input_ids,
            context_token_count=row["context_token_count"], selected_indices=row["selected_indices"],
            chunk_size=row["chunk_size"], max_new_tokens=row["max_new_tokens"])
        finished = clock(reader.device)
    require(trace.first_token_at is not None, "Caps > 1 must enter at least one decode forward")
    accounting = native_decode_accounting(trace.generated_ids, row["max_new_tokens"], row["eos_id"])
    require(trace.decode_calls == accounting["decode_forward_calls"], "Actual decode forwards differ from native EOS/cap accounting")
    require(getattr(reader, "_bottom", None) is None, "Reader lower cache leaked across queries")
    timings = dict(trace.times)
    timings.update(native_adapter_generation_s=finished-started,
                   native_adapter_ttft_s=trace.first_token_at-started,
                   decode_s=trace.native_finished_at-trace.first_token_at)
    return {"prediction": text, "generated_ids": trace.generated_ids, **accounting,
            "actual_decode_forward_calls": trace.decode_calls, "timings": timings,
            "selected_read_state_logical_tensor_bytes": trace.selected_state_logical_bytes,
            "selected_state_bytes_are_persistent_storage": False, "state_reset_verified": True}


def validate_result(result, row, config):
    from prepare_qa_cost import native_decode_accounting
    require(result["protocol"] == VERSION and result["job_id"] == config["job_id"], "Result protocol differs")
    require(result["index"] == row["index"] and result["id"] == row["id"] and result["pack"] == row["pack"], "Result input/pack differs")
    require(result["repetition"] in range(config["repetitions"]), "Invalid repetition")
    require(0 <= result["score"] <= 1 and math.isfinite(result["score"]), "Invalid score")
    expected = native_decode_accounting(result["generated_ids"], row["max_new_tokens"], row["eos_id"])
    require(all(result.get(k) == v for k, v in expected.items()), "Native accounting mismatch")
    require(result["actual_decode_forward_calls"] == expected["decode_forward_calls"], "Forward count mismatch")
    times = result["timings"]
    require(all(isinstance(times.get(k), (int, float)) and math.isfinite(times[k]) and times[k] >= 0 for k in TIME_FIELDS), "Missing/invalid phase timing")
    require(times["total_from_marked_prompt_s"] >= times["ttft_from_marked_prompt_s"], "TTFT exceeds total")
    hardware = result["hardware"]
    require(hardware["timing_eligible"] and hardware["device_name"] == REQUIRED_GPU and hardware["hostname"] == socket.gethostname(), "Wrong local timing hardware")
    require(hardware["torch_cpu_threads"] == 2 and hardware["torch_interop_threads"] == 16 and hardware["platform"] == "Windows", "Wrong local threads/platform")
    require(all(hardware.get(k) == v for k, v in {"omp_num_threads": "2", "mkl_num_threads": "2", "tokenizers_parallelism": "false"}.items()), "Wrong thread environment")
    admission = hardware["gpu_admission"]
    require(admission["initial_used_gib"] < 5 and admission["recheck_used_gib"] < 5 and not admission["other_python_compute_processes"], "Invalid GPU admission")
    mem = result["memory"]
    require(mem["peak_allocated_bytes"] >= mem["resident_model_baseline_allocated_bytes"] >= 0, "Invalid peak/baseline memory")
    require(mem["incremental_peak_allocated_bytes"] == mem["peak_allocated_bytes"] - mem["resident_model_baseline_allocated_bytes"], "Invalid memory increment")
    require(result["state_reset_verified"] and result["smoke_reference_equal"] is (True if config["phase"] == "smoke" else None), "Reference/reset verification missing")


def reusable_rows(attempts, config, inputs):
    """Retain valid per-repetition observations, including interrupted attempts."""
    found = {}
    for attempt in sorted(Path(attempts).glob("[0-9]*")):
        if not (attempt / "config.json").exists() or read_json(attempt / "config.json") != config:
            continue
        if not (attempt / "measurements.jsonl").exists():
            continue
        lines = (attempt / "measurements.jsonl").read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                require(number == len(lines)-1, "Corrupt nonterminal result line")
                continue  # An interrupted partial final write remains in its original attempt.
            require(result["index"] in config["indices"], "Unexpected recorded index")
            validate_result(result, inputs[result["index"]], config)
            key = result["index"], result["repetition"]
            if key not in found:
                found[key] = dict(result, reused_from=str(attempt))
    return found


def summarize(rows, config):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["index"]].append(row)
    require(set(grouped) == set(config["indices"]), "Missing sample")
    require(all(len(v) == config["repetitions"] for v in grouped.values()), "Missing repetitions")
    samples = []
    for index, records in sorted(grouped.items()):
        samples.append({"index": index, "id": records[0]["id"],
            "median_timings": {k: statistics.median(r["timings"][k] for r in records) for k in TIME_FIELDS},
            "mean_score": statistics.mean(r["score"] for r in records),
            "output_repeat_disagreement": len({tuple(r["generated_ids"]) for r in records}) > 1,
            "historical_prediction_disagreements": sum(not r["historical_prediction_equal"] for r in records)})
    def aggregates(values):
        return {"mean": statistics.mean(values), "median": statistics.median(values),
                "p95_nearest_rank": sorted(values)[math.ceil(.95*len(values))-1]}
    return {"protocol": VERSION, "job_id": config["job_id"], "phase": config["phase"],
            "unique_method_examples": len(samples), "timed_generations": len(rows),
            "score_percent_mean_all_repetitions": 100*statistics.mean(s["mean_score"] for s in samples),
            "sample_median_latency_aggregates": {k: aggregates([s["median_timings"][k] for s in samples]) for k in TIME_FIELDS},
            "output_repeat_disagreement_samples": sum(s["output_repeat_disagreement"] for s in samples),
            "historical_prediction_disagreement_repetitions": sum(s["historical_prediction_disagreements"] for s in samples),
            "sample_medians": samples, "outlier_exclusions": 0,
            "cost_scope": "Each query recomputes CPU prompt/tokenization/BM25 and captures selected chunks afresh; no whole-document write-once or steady cached-read claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "results/protocol/qa_cost/task_plan.json")
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    input_startup_started = time.perf_counter()
    job, inputs, config = load_job(args.plan, args.job)
    input_startup_s = time.perf_counter() - input_startup_started
    require(platform.system() == "Windows", "QA timing is local Windows only")
    require(all(os.environ.get(k) == v for k, v in THREAD_ENV.items()), "Required timing thread environment is absent")
    args.out.mkdir(parents=True, exist_ok=True)
    from run_driver import OutputLock
    with OutputLock(args.out):
        save_json(args.out / "config.json", config)
        reused = reusable_rows(args.out.parent, config, inputs)
        require(not (args.out / "measurements.jsonl").exists(), "A new attempt directory is required; prior records remain reusable")
        metadata = {"status": "waiting_for_gpu", "pid": os.getpid(), "started_at": utc_now(),
                    "job": args.job, "reused_repetitions": len(reused), "new_timed_generations": 0,
                    "warmup_generations": 0, "extra_smoke_reference_generations": 0,
                    "input_file_read_parse_validation_s_startup_excluded": input_startup_s,
                    "total_latency_input_boundary": "memory-resident official marked prompt; chat template, tokenizer, BM25 rerun per repetition; source-file IO excluded"}
        save_json(args.out / "status.json", metadata)
        try:
            import torch
            torch.set_num_threads(2)
            if torch.get_num_interop_threads() != 16:
                torch.set_num_interop_threads(16)
            # No CUDA/model initialization occurs before the original gate's two checks.
            from gpu_gate import acquire_gpu
            admission = acquire_gpu(need_gb=22, cap_gb=None, idle_slack_gb=5.0,
                                    max_wait=86400, tag="qa_cost_runner " + str(args.out))
            from serving_reuse import device_provenance  # Also imports the common Windows SDPA workaround.
            hardware = device_provenance("cuda:0")
            hardware["gpu_admission"] = admission
            hardware["torch_memory_cap_bytes"] = 28_000_000_000
            torch.cuda.set_per_process_memory_fraction(min(28e9/hardware["total_memory_bytes"], 1), device=0)
            import official_qa_driver as driver
            from comem import CoMem
            from s15_ruler_lower import CoMemLower
            torch.manual_seed(42)
            startup_start = clock("cuda:0")
            model, tok = driver.load_backbone(config["model"], "bfloat16", "sdpa", "cuda:0", "")
            model.eval()
            reader = CoMemLower(model, 12, tokenizer=tok) if job["arm"] == "fix_all" else CoMem(model, 0, tokenizer=tok)
            reader.write_sink = job["arm"] == "fix_all"
            startup_s = clock("cuda:0") - startup_start
            require(int(tok.eos_token_id) == inputs[0]["eos_id"] and int(reader._sink_prefix_id()) == inputs[0]["sink_id"], "Tokenizer sink/EOS differs")
            metadata.update(status="running", hardware=hardware, model_tokenizer_load_s=startup_s)
            save_json(args.out / "status.json", metadata)
            for _ in range(config["warmups"]):
                row = inputs[job["indices"][0]]
                ids = torch.tensor([row["context_ids"] + row["query_ids"]], dtype=torch.long, device="cuda:0")
                start = clock("cuda:0")
                pred, tokens = native_reference(reader, ids, row)
                elapsed = clock("cuda:0") - start
                save_json(args.out / "warmup.json", {"id": row["id"], "prediction": pred,
                    "generated_ids": tokens, "elapsed_s_excluded": elapsed, "measurement": False})
                metadata["warmup_generations"] += 1
                del ids
            results = []
            with (args.out / "measurements.jsonl").open("w", encoding="utf-8") as stream:
                for index in job["indices"]:
                    row = inputs[index]
                    for repetition in range(config["repetitions"]):
                        key = index, repetition
                        if key in reused:
                            result = reused[key]
                        else:
                            gc.collect()
                            clock("cuda:0")
                            baseline_allocated = torch.cuda.memory_allocated(0)
                            baseline_reserved = torch.cuda.memory_reserved(0)
                            torch.cuda.reset_peak_memory_stats(0)
                            prep_started = time.perf_counter()
                            ids, nctx, selected, pack = driver.tokenize_pack(tok, row, 512, "bm25", 12)
                            prep_s = time.perf_counter() - prep_started
                            validation_started = time.perf_counter()
                            require(ids[0].tolist() == row["context_ids"] + row["query_ids"] and nctx == row["context_token_count"] and selected == row["selected_indices"] and pack == row["pack"], "Runtime tokenizer/BM25 pack differs from exact prepared inputs")
                            validation_s = time.perf_counter() - validation_started
                            h2d_started = clock("cuda:0")
                            gpu_ids = ids.to("cuda:0")
                            h2d_s = clock("cuda:0") - h2d_started
                            result = measure_native(reader, gpu_ids, row)
                            peak_allocated = torch.cuda.max_memory_allocated(0)
                            peak_reserved = torch.cuda.max_memory_reserved(0)
                            timing = result["timings"]
                            timing.update(cpu_prompt_tokenize_bm25_s=prep_s, h2d_s=h2d_s,
                                prepared_input_generation_s=h2d_s + timing["native_adapter_generation_s"],
                                ttft_from_prepared_input_s=h2d_s + timing["native_adapter_ttft_s"],
                                ttft_from_marked_prompt_s=prep_s + h2d_s + timing["native_adapter_ttft_s"],
                                total_from_marked_prompt_s=prep_s + h2d_s + timing["native_adapter_generation_s"])
                            score_start = time.perf_counter()
                            score, scored = driver.score_prediction(result["prediction"], row)
                            scoring_s = time.perf_counter() - score_start
                            result.update(protocol=VERSION, job_id=job["id"], task=row["task"], arm=job["arm"],
                                index=index, id=row["id"], repetition=repetition, pack=pack, score=score,
                                scored_prediction=scored, hardware=hardware, timestamp=utc_now(),
                                input_contract=config["input"], max_new_tokens=row["max_new_tokens"],
                                input_validation_s_excluded=validation_s, scoring_s_excluded=scoring_s,
                                smoke_reference_equal=None, historical_prediction_equal=result["prediction"] == row["reference_outputs"][job["arm"]]["prediction"],
                                historical_score=row["reference_outputs"][job["arm"]]["official_score"],
                                memory={"resident_model_baseline_allocated_bytes": baseline_allocated,
                                        "resident_baseline_reserved_bytes": baseline_reserved,
                                        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                                        "incremental_peak_allocated_bytes": peak_allocated-baseline_allocated,
                                        "peak_region": "CPU prompt preparation through H2D and native generation/text decode; model resident; excludes smoke reference/warmup/scoring"})
                            metadata["new_timed_generations"] += 1
                            if job["phase"] == "smoke":
                                ref_started = clock("cuda:0")
                                ref_text, ref_ids = native_reference(reader, gpu_ids, row)
                                ref_elapsed = clock("cuda:0") - ref_started
                                metadata["extra_smoke_reference_generations"] += 1
                                result["smoke_reference_equal"] = ref_text == result["prediction"] and ref_ids == result["generated_ids"]
                                result["smoke_reference_s_excluded"] = ref_elapsed
                                require(result["smoke_reference_equal"], "Instrumentation differs from uncached official reference")
                            del gpu_ids, ids
                            validate_result(result, row, config)
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        results.append(result)
                        metadata.update(completed_repetitions=len(results), expected_repetitions=len(job["indices"])*config["repetitions"], updated_at=utc_now())
                        save_json(args.out / "status.json", metadata)
                        print(f"[{job['id']}] {len(results)}/{metadata['expected_repetitions']} index={index} rep={repetition} tokens={result['generated_tokens']} eos={result['terminated_by_eos']} f1={result['score']:.4f}", flush=True)
            summary = summarize(results, config)
            save_json(args.out / "summary.json", summary)
            metadata.update(status="complete", finished_at=utc_now(), expected_repetitions=len(results),
                unique_method_examples=len(job["indices"]), protocol=VERSION)
            save_json(args.out / "status.json", metadata)
            save_json(args.out / "COMPLETED.json", metadata)
        except BaseException as exc:
            metadata.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at=utc_now())
            save_json(args.out / "status.json", metadata)
            save_json(args.out / "FAILED.json", metadata)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
