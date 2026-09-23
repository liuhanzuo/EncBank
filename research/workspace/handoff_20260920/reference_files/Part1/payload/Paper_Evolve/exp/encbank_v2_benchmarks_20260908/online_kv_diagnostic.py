"""Local 5090-only direct KV inventory on existing completed serving cases."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (ROOT / "Encbank", ROOT / "exp", HERE):
    sys.path.insert(0, str(p))
from qa_cost_runner import read_json, save_json, require

PROTOCOL = "online-kv-inventory-v1"
ARMS = ("fix_all", "pub", "pub_sink", "j0", "pub_lora", "cacheblend16")
THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}


def source_attempt(arm, length):
    if arm == "cacheblend16":
        state = read_json(HERE / "results/local/bootstrap_cacheblend_serving/status.json")
        key = f"full/cacheblend16_{length}"
    else:
        state = read_json(HERE / "results/local/bootstrap_serving/status.json")
        key = f"{'pub_lora' if arm == 'pub_lora' else 'base'}/full/{arm}_{length}"
    row = state["jobs"][key]
    require(row["status"] == "complete", "Source cost job is not complete")
    path = Path(row["result"])
    marker = read_json(path / "COMPLETED.json")
    require(marker["status"] == "complete" and marker["cells"] == 12, "Source receipt incomplete")
    return path


def query_records(attempt, arm, length, generation):
    rows = [json.loads(s) for s in (attempt / f"queries_{length}_{arm}_disk_g{generation}.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    require([r["id"] for r in rows] == list(range(100)), "Expected full completed 100-query source")
    return rows


def resolve_cases(arm):
    cases = []
    for length in (32768, 131072):
        common = source_attempt("fix_all", length)
        reference = query_records(common, "fix_all", length, 16)
        longest = max(reference, key=lambda row: row["read_tokens"])["id"]
        indices = [0, longest if longest else 1]
        attempt = source_attempt(arm, length)
        workload = read_json(attempt / f"workload_{length}.json")["questions"]
        common_workload = read_json(common / f"workload_{length}.json")["questions"]
        source_config = read_json(attempt / "config.json")
        for generation in (16, 128):
            records = query_records(attempt, arm, length, generation)
            for index in indices:
                source = records[index]
                require(workload[index] == common_workload[index], "Cross-method query workload differs")
                require(source["selected_indices"] == reference[index]["selected_indices"] and source["read_tokens"] == reference[index]["read_tokens"], "Cross-method input pack differs")
                cases.append({"case_id": f"{arm}_{length}_q{index}_g{generation}", "arm": arm,
                    "context_tokens": length, "query_id": index, "G": generation,
                    "query": workload[index], "selected_indices": source["selected_indices"],
                    "read_tokens": source["read_tokens"], "query_tokens": source["query_tokens"],
                    "historical_generated_ids": source["generated_ids"],
                    "source_attempt": str(attempt), "source_config": source_config,
                    "store": str(attempt / f"store_{length}_{arm}")})
    require(len(cases) == 8 and len({c["case_id"] for c in cases}) == 8, "Expected 8 distinct method diagnostic cases")
    return cases


def store_metadata(folder):
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in Path(folder).iterdir() if p.is_file()}


def completed_attempt(folder, arm):
    from online_kv_probe import validate_inventory_record
    for path in sorted((Path(folder) / "attempts").glob("[0-9]*"), reverse=True):
        try:
            marker = read_json(path / "COMPLETED.json")
            cases = read_json(path / "cases.json")
            rows = [json.loads(s) for s in (path / "inventories.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
            require(marker["status"] == "complete" and marker["protocol"] == PROTOCOL and marker["arm"] == arm and marker["diagnostic_generations"] == len(rows) == len(cases) == 8, "Incomplete inventory scope")
            require({r["case_id"] for r in rows} == {c["case_id"] for c in cases}, "Missing cases")
            from cacheblend_serving_bootstrap import hardware_valid
            hardware_valid(marker["hardware"])
            for row in rows:
                require(row["hardware"] == marker["hardware"] and row["protocol"] == PROTOCOL and row["persistent_store_metadata_unchanged"], "Hardware or persistent store differs")
                require(all(t["device"] == "cuda:0" for phase in row["inventory"].values() for layer in phase["layers"] for t in layer["tensors"]), "Non-local GPU KV observation")
                validate_inventory_record(row, arm, row["split"], row["num_layers"])
            return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(platform.system() == "Windows", "Online KV GPU memory diagnostics require local Windows")
    require(all(os.environ.get(k) == v for k, v in THREAD_ENV.items()), "Correct diagnostic threads must be set before import")
    require(not args.out.exists() or not any(args.out.iterdir()), "New empty diagnostic attempt required")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = resolve_cases(args.arm)
    save_json(args.out / "cases.json", cases)
    state = {"protocol": PROTOCOL, "arm": args.arm, "pid": os.getpid(), "status": "waiting_for_gpu",
        "timing_eligible": False, "expected_diagnostic_generations": 8, "completed_diagnostic_generations": 0}
    save_json(args.out / "status.json", state)
    try:
        import torch
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 16:
            torch.set_num_interop_threads(16)
        from gpu_gate import acquire_gpu
        admission = acquire_gpu(need_gb=22, idle_slack_gb=5.0, max_wait=86400,
            tag="online_kv_diagnostic " + str(args.out))
        from serving_reuse import device_provenance, ReusableReader, validate_adapter_checkpoint, load_unmerged_adapter
        from cacheblend_serving_reuse import ReusableContextualCacheBlend
        from online_kv_probe import measure_query
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hardware = device_provenance("cuda:0")
        hardware["gpu_admission"] = admission
        hardware["torch_memory_cap_bytes"] = 28_000_000_000
        torch.cuda.set_per_process_memory_fraction(min(28e9/hardware["total_memory_bytes"], 1), device=0)
        torch.manual_seed(42)
        source = cases[0]["source_config"]
        require(all(c["source_config"]["model"] == source["model"] for c in cases), "Different source model IDs")
        tok = AutoTokenizer.from_pretrained(source["model"], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(source["model"], torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
        adapter = None
        if args.arm == "pub_lora":
            adapter = validate_adapter_checkpoint(source["adapter_ckpt"], source["adapter_completion_marker"])
            model = load_unmerged_adapter(model, source["adapter_ckpt"])
        reader = ReusableContextualCacheBlend(model, 12, tok, source["model"], .16) if args.arm == "cacheblend16" else ReusableReader(model, 12, tok, args.arm, source["model"], adapter)
        state.update(status="running", hardware=hardware, adapter=adapter)
        save_json(args.out / "status.json", state)
        active_store = None
        results = []
        with (args.out / "inventories.jsonl").open("w", encoding="utf-8") as output:
            for case in cases:
                if case["store"] != active_store:
                    reader.open_store(case["store"], "disk")
                    active_store = case["store"]
                before = store_metadata(active_store)
                query_ids = tok.encode(case["query"]["text"], add_special_tokens=False)
                require(len(query_ids) == case["query_tokens"], "Query tokenization differs from cost input")
                row = measure_query(reader, query_ids, selected_indices=case["selected_indices"], max_new_tokens=case["G"])
                require(row["read_tokens"] == case["read_tokens"] and row["selected_indices"] == case["selected_indices"], "Diagnostic pack differs")
                require(store_metadata(active_store) == before, "Diagnostic changed persistent store")
                row.update(protocol=PROTOCOL, arm=args.arm, case_id=case["case_id"], context_tokens=case["context_tokens"],
                    query_id=case["query_id"], hardware=hardware, split=reader.cm.resume_j, num_layers=reader.cm.num_layers,
                    source_attempt=case["source_attempt"], historical_generated_ids_equal=row["generated_ids"] == case["historical_generated_ids"],
                    persistent_store_metadata_unchanged=True, timestamp=datetime.now(timezone.utc).isoformat())
                output.write(json.dumps(row)+"\n")
                output.flush()
                os.fsync(output.fileno())
                results.append(row)
                state.update(completed_diagnostic_generations=len(results), updated_at=datetime.now(timezone.utc).isoformat())
                save_json(args.out / "status.json", state)
                print(f"[online_kv/{args.arm}] {len(results)}/8 {case['case_id']} prefill={row['inventory']['prefill_complete']['logical_tensor_bytes']} decode={row['inventory']['decode_complete']['logical_tensor_bytes']} bytes", flush=True)
                gc.collect()
        reader.close_store()
        state.update(status="complete", diagnostic_generations=len(results), observed_cache_boundaries=2*len(results),
            historical_output_disagreement_cases=sum(not r["historical_generated_ids_equal"] for r in results),
            completed_at=datetime.now(timezone.utc).isoformat())
        save_json(args.out / "status.json", state)
        save_json(args.out / "COMPLETED.json", state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        save_json(args.out / "status.json", state)
        save_json(args.out / "FAILED.json", state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
