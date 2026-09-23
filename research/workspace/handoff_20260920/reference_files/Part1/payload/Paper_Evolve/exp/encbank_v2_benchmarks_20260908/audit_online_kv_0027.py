"""Independent, stdlib-only validation of complete direct online-KV observations."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ARMS = ("fix_all", "pub", "pub_sink", "j0", "pub_lora", "cacheblend16")
PROTOCOL = "online-kv-inventory-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    started = time.perf_counter()
    state = read(HERE / "results/local/bootstrap_online_kv/status.json")
    result = {"protocol": PROTOCOL, "audited_at": datetime.now(timezone.utc).isoformat(),
        "measurement": "Direct local RTX 5090 active autoregressive KV tensor inventory",
        "timing_eligible": False, "complete_methods": {}, "pending_methods": [], "case_comparisons": {},
        "unit_definitions": {"MiB": 2**20, "GiB": 2**30},
        "scope": "Six methods, existing 32k/128k disk stores, common query IDs 0/max-read-pack, G16/128; 8 cases per method and 2 active-cache boundaries per case. Representative shape diagnostics, not a full benchmark average or transient peak.",
        "exclusions": ["model weights", "hidden states", "attention intermediates", "temporary staging copies", "persistent CPU/disk payloads", "allocator reservation", "whole-GPU resident memory"],
        "decode_boundary": "After G-1 decode forwards; final generated token has not been fed back. Active KV length is prefix+G-1.",
        "independent_validation": "Shapes and dtype -> element bytes; per-layer sum -> total; all observed tensors have distinct backing storages so backing-capacity sum exactly reconstructs unique total. Raw IDs compared directly to saved case/source cost outputs; no trust in a boolean alone.",
        "actual_failed_jobs": [], "queue_snapshot": {k: state.get(k) for k in ("status", "active_job", "child_pid", "error", "completed_diagnostic_generations", "completed_observed_cache_boundaries")}}
    paired = defaultdict(dict)
    source_rows = {}
    validated_phases = 0
    for arm in ARMS:
        job = state.get("jobs", {}).get(arm)
        if not job or job.get("status") != "complete":
            result["pending_methods"].append(arm)
            continue
        folder = Path(job["result"])
        marker = read(folder / "COMPLETED.json")
        require(marker["status"] == "complete" and marker["protocol"] == PROTOCOL and marker["arm"] == arm and marker["timing_eligible"] is False, "Wrong complete marker")
        require(marker["diagnostic_generations"] == 8 and marker["observed_cache_boundaries"] == 16, "Wrong method scope")
        rows = [json.loads(line) for line in (folder / "inventories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        cases = {c["case_id"]: c for c in read(folder / "cases.json")}
        require(len(rows) == len(cases) == 8 and {r["case_id"] for r in rows} == set(cases), "Incomplete/duplicate case inventory")
        hw = marker["hardware"]
        require(hw["device_name"] == "NVIDIA GeForce RTX 5090" and hw["device"] == "cuda:0" and hw["platform"] == "Windows", "Wrong actual hardware")
        require((hw["torch_cpu_threads"], hw["torch_interop_threads"], hw["omp_num_threads"], hw["mkl_num_threads"], hw["tokenizers_parallelism"]) == (2,16,"2","2","false"), "Wrong actual thread policy")
        admission = hw["gpu_admission"]
        require(admission["initial_used_gib"] < 5 and admission["recheck_used_gib"] < 5 and not admission["other_python_compute_processes"] and admission["comparison"] == "strictly_less_than", "Invalid GPU admission")
        views, logicals, physicals, output_matches = [], defaultdict(list), defaultdict(list), []
        for row in rows:
            case = cases[row["case_id"]]
            require(row["hardware"] == hw and row["protocol"] == PROTOCOL and row["timing_eligible"] is False, "Wrong row device/protocol")
            require(row["generated_tokens"] == case["G"] == len(row["generated_ids"]) and row["actual_decode_forward_calls"] == case["G"]-1, "Wrong native token/forward count")
            require(row["persistent_store_metadata_unchanged"] and row["state_reset_verified"] and row["document_or_query_capture_calls"] == 0, "Reuse/state invariance failed")
            source_file = Path(case["source_attempt"]) / f"queries_{case['context_tokens']}_{arm}_disk_g{case['G']}.jsonl"
            if source_file not in source_rows:
                original = [json.loads(s) for s in source_file.read_text(encoding="utf-8").splitlines() if s.strip()]
                require([r["id"] for r in original] == list(range(100)), "Original source query scope incomplete")
                source_rows[source_file] = original
            original = source_rows[source_file][case["query_id"]]
            require(original["selected_indices"] == case["selected_indices"] == row["selected_indices"] and original["read_tokens"] == case["read_tokens"] == row["read_tokens"] and original["query_tokens"] == case["query_tokens"] == row["query_tokens"], "Actual input/pack differs")
            require(case["historical_generated_ids"] == original["generated_ids"], "Historical case output differs from source")
            matches = row["generated_ids"] == original["generated_ids"]
            require(row["historical_generated_ids_equal"] == matches, "Wrong raw ID match flag")
            output_matches.append(matches)
            key = f"{case['context_tokens']}_q{case['query_id']}_g{case['G']}"
            pair_row = {"context_tokens": case["context_tokens"], "query_id": case["query_id"], "G": case["G"],
                "read_tokens": row["read_tokens"], "query_tokens": row["query_tokens"], "selected_indices": row["selected_indices"],
                "historical_generated_ids_equal": matches, "phases": {}}
            for phase, growth in (("prefill_complete", 0), ("decode_complete", case["G"]-1)):
                inv = row["inventory"][phase]
                layers = inv["layers"]
                require(row["num_layers"] == 36 and len(layers) == inv["populated_layer_entries"] == 36 and {layer["layer_index"] for layer in layers} == set(range(36)), "Missing or repeated model layer")
                require(inv["tensor_count"] == 72, "Each layer needs one K and one V")
                logical, capacities = [], []
                lengths_by_band = defaultdict(set)
                dtypes = set()
                for layer in layers:
                    expected_n = row["query_tokens"]+growth if arm in {"pub", "pub_sink", "pub_lora"} and layer["layer_index"] < 12 else row["read_tokens"]+growth
                    require(layer["sequence_length"] == expected_n and {t["kind"] for t in layer["tensors"]} == {"key", "value"}, "Wrong layer sequence/kind")
                    lengths_by_band[layer["band"]].add(expected_n)
                    layer_bytes = 0
                    for tensor in layer["tensors"]:
                        require(tensor["shape"] == [1,8,expected_n,128] and tensor["device"] == "cuda:0", "Unexpected actual tensor shape/device")
                        size = {"torch.bfloat16":2,"torch.float16":2,"torch.float32":4}[tensor["dtype"]]
                        dtypes.add(tensor["dtype"])
                        nbytes = math.prod(tensor["shape"])*size
                        require(nbytes == tensor["logical_bytes"], "Shape/dtype byte arithmetic differs")
                        require(tensor["backing_storage_bytes"] >= nbytes and tensor["storage_offset_elements"] >= 0, "Invalid tensor backing capacity")
                        logical.append(nbytes)
                        capacities.append(tensor["backing_storage_bytes"])
                        layer_bytes += nbytes
                    require(layer["logical_bytes"] == layer_bytes, "Wrong layer byte sum")
                require(inv["logical_tensor_bytes"] == sum(logical), "Wrong aggregate logical bytes")
                # All observed current-reader active tensors are distinct; if future
                # results alias, require exported group identities before reconstructing.
                require(inv["unique_backing_storages"] == 72 and inv["unique_backing_storage_bytes"] == sum(capacities), "Cannot independently reconstruct an aliased/mismatched unique-storage total")
                logicals[phase if phase == "prefill_complete" else f"decode_g{case['G']}"] += [inv["logical_tensor_bytes"]]
                physicals[phase if phase == "prefill_complete" else f"decode_g{case['G']}"] += [inv["unique_backing_storage_bytes"]]
                pair_row["phases"][phase] = {"logical_tensor_bytes": inv["logical_tensor_bytes"], "unique_backing_storage_bytes": inv["unique_backing_storage_bytes"],
                    "logical_mib": inv["logical_tensor_bytes"]/(2**20), "unique_mib": inv["unique_backing_storage_bytes"]/(2**20),
                    "logical_gib": inv["logical_tensor_bytes"]/(2**30), "unique_gib": inv["unique_backing_storage_bytes"]/(2**30),
                    "populated_layers":36, "kv_tensors":72, "distinct_storages":72, "dtypes":sorted(dtypes),
                    "sequence_lengths_by_band": {band:sorted(vals) for band,vals in lengths_by_band.items()}}
                validated_phases += 1
            paired[key][arm] = pair_row
            views.append({"case_id":row["case_id"], **pair_row})
        require(marker["historical_output_disagreement_cases"] == sum(not m for m in output_matches), "Wrong mismatch total")
        result["complete_methods"][arm] = {"result": str(folder), "diagnostic_generations":8,"observed_cache_boundaries":16,
            "all_raw_ids_match_prior_timing_output":all(output_matches),"raw_id_match_cases":sum(output_matches),
            "gate_admission": admission, "hardware": hw,
            "logical_bytes_ranges":{phase:[min(vals),max(vals)] for phase,vals in logicals.items()},
            "unique_backing_bytes_ranges":{phase:[min(vals),max(vals)] for phase,vals in physicals.items()},
            "logical_equals_unique_for_all_boundaries":all(logicals[k] == physicals[k] for k in logicals), "cases":views}
    for key, methods in paired.items():
        first = next(iter(methods.values()))
        require(all((m["read_tokens"],m["query_tokens"],m["selected_indices"]) == (first["read_tokens"],first["query_tokens"],first["selected_indices"]) for m in methods.values()), "Cross-method pack differs")
        result["case_comparisons"][key] = {"methods":methods,"same_pack_verified":True,"all_six_methods_complete":len(methods)==6}
    result["representative_128k_q0_g128"] = result["case_comparisons"].get("131072_q0_g128")
    result.update(complete_diagnostic_generations=8*len(result["complete_methods"]),complete_observed_cache_boundaries=validated_phases,
        entire_diagnostic_complete=len(result["complete_methods"])==6, audit_cpu_wall_s=time.perf_counter()-started,
        torch_imported=False, model_loaded=False)
    optional_probe = HERE / "logs/online_kv_optional_probe_0027.json"
    if optional_probe.exists():
        result["optional_subprocess_diagnostic"] = read(optional_probe)
    result["writeback_interpretation"] = {
        "active_kv": "For every same-pack observed case, V2, j0 and CacheBlend16 have equal logical and unique active KV bytes; pub/pub_sink/pub_lora use fewer bytes because the first 12 layers retain query-only KV.",
        "claim_limit": "This does not establish an online-KV memory advantage for V2. Persistent document storage, transient allocation peaks and full GPU residency are different quantities and require their own existing measurements.",
        "no_timing_or_quality_replacement": "Diagnostic instrumentation is timing-ineligible. Same-method historical output IDs match; this is not a new accuracy benchmark or a cross-method output-equality claim.",
    }
    out = HERE / "heartbeat_online_kv_20260909_0027.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(out), "complete_methods":list(result["complete_methods"]),"pending":result["pending_methods"],
        "generations":result["complete_diagnostic_generations"],"boundaries":validated_phases,"cpu_s":result["audit_cpu_wall_s"],
        "representative":{arm:{phase:p["logical_mib"] for phase,p in data["phases"].items()} for arm,data in (result["representative_128k_q0_g128"] or {}).get("methods",{}).items()}},indent=2))


if __name__ == "__main__":
    main()
