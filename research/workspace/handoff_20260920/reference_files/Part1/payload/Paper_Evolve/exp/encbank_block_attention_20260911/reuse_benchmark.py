"""Fixed ordered-pack cross-request cache reuse; called only inside the GPU guard.

No model loading, device admission, background process or automatic fallback lives
here. Each request owns fresh query KV; only document caches survive the stream.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import copy
import gc
import os
import tempfile
import time

from infra_protocol import digest, json_digest, now, save_json
from reuse_protocol import (VERSION, QUALITY_VERSION, recipe, source_identity,
                            summarize_reuse, validate_reuse_mode, workload)


def _readers(args, encbank):
    from native_infra_readers import FullRecomputeReader, NativeEncbankReader
    from native_prefix_reader import NativePrefixEncbankReader
    from sparse_reader import SparseEncbankReader
    if args.arm == "FULL":
        return FullRecomputeReader(encbank), None
    reference = SparseEncbankReader(encbank, fusion_layer=args.m,
        probe_mode="block" if args.arm in ("A", "B") else "dense",
        retain_ratio=1. if args.arm in ("D0", "NATIVE", "A") else args.rho,
        gradient_checkpointing=False).eval()
    return (NativePrefixEncbankReader(encbank), reference) if args.arm in ("D0", "NATIVE") else (reference, reference)


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _amp(torch, device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _timed(torch, device, fn):
    _sync(torch, device)
    start = time.perf_counter()
    value = fn()
    _sync(torch, device)
    return value, time.perf_counter() - start


def _tensors(value):
    if hasattr(value, "untyped_storage"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif hasattr(value, "raw_kv"):
        yield value.h_m
        for pair in value.raw_kv.values():
            yield pair.k
            yield pair.v


def _memory(value):
    tensors = list(_tensors(value))
    stores = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        stores[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return {"tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "unique_storage_bytes": sum(stores.values())}


def _signature(value):
    return tuple((id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.device), str(t.dtype))
                 for t in _tensors(value))


def _save_torch(torch, payload, path):
    start = time.perf_counter()
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"seconds": time.perf_counter() - start, "file_bytes": Path(path).stat().st_size}


def _hot_payload(entries):
    return [{"source_key": h.source_key, "token_count": h.token_count, "h_m": h.h_m,
             "raw_kv": {layer: (pair.k, pair.v) for layer, pair in h.raw_kv.items()}} for h in entries]


def _restore_hot(reader, payload, device):
    from sparse_reader import HotBlock, KVPair
    signature = reader._signature()
    return [HotBlock(h["source_key"], signature, h["token_count"], h["h_m"].to(device),
        {int(layer): KVPair(kv[0].to(device), kv[1].to(device)) for layer, kv in h["raw_kv"].items()})
        for h in payload]


def _resident(reader, sink, docs, entries):
    # NativePrefix keeps strong references to original h_j: include them too.
    values = [sink, *(entries if entries is not None else docs)]
    prefix = getattr(reader, "prefix", None)
    if prefix is not None:
        values += [t for pair in prefix.pairs.values() for t in (pair.k, pair.v)]
    return values


def _generate(torch, reader, sink, docs, prompt, probes, tokens, extra=None, capture_logits=False):
    logits, state = reader.prefill(sink, docs, prompt, probe_indices=probes, **(extra or {}))
    ids, captured, finite = [], [], True
    for index in range(tokens):
        if index:
            logits = reader.decode_step(ids[-1], state)
        finite = finite and bool(torch.isfinite(logits).all().item())
        ids.append(int(logits[0, -1].argmax().item()))
        if capture_logits:
            captured.append(logits[0, -1].detach().float().cpu().clone())
    return {"generated_ids": ids, "first_token": ids[0], "route_stats": copy.deepcopy(state.route_stats),
            "finite_logits": finite}, captured


def _quality_storage_roundtrip(args, torch, reader, sink, docs, keys, device):
    """Exercise the measured on-disk format and exact hot restore helper."""
    parent = Path(getattr(args, "out", args.model))
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reuse_roundtrip_", dir=parent) as temporary:
        folder = Path(temporary)
        original = {"sink": sink.detach().cpu().clone(),
                    "documents": [d.detach().cpu().clone() for d in docs]}
        cold_io = _save_torch(torch, original, folder / "cold.pt")
        loaded = torch.load(folder / "cold.pt", map_location="cpu", weights_only=False)
        equal = all(torch.equal(a, b) for a, b in zip(_tensors(original), _tensors(loaded)))
        entries, hot_io = None, None
        if args.arm in ("A", "B"):
            direct = reader.build_hot_cache(loaded["documents"], keys, cache_device="cpu")
            payload = _hot_payload(direct)
            hot_io = _save_torch(torch, payload, folder / "hot.pt")
            recovered = torch.load(folder / "hot.pt", map_location="cpu", weights_only=False)
            equal &= all(torch.equal(a, b) for a, b in zip(_tensors(payload), _tensors(recovered)))
            entries = _restore_hot(reader, recovered, device)
            equal &= all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(_tensors(direct), _tensors(entries)))
        restored_sink = loaded["sink"].to(device)
        restored_docs = loaded["documents"] if entries is not None else [d.to(device) for d in loaded["documents"]]
    return restored_sink, restored_docs, entries, {"passed": bool(equal),
        "cold_file_bytes": cold_io["file_bytes"], "hot_file_bytes": hot_io["file_bytes"] if hot_io else 0,
        "path": "real temporary files beneath the output directory; fsync then CPU load and benchmark hot restore helper",
        "temporary_files_removed": True}


def run_reuse_benchmark(args, torch, encbank, tokenizer, result, progress):
    """Return a complete result; caller retains the existing monitor and lease."""
    identity = validate_reuse_mode(args)
    if identity is None:
        raise ValueError("This runner requires a genuine reuse request stream")
    reader, unused_reference = _readers(args, encbank)
    del unused_reference
    device = encbank.device
    data = workload(tokenizer, encbank.config, args)
    chunks, prompts = data["chunks"], data["prompts"]
    probes = data["probe_indices"]
    writer = reader if hasattr(reader, "write_chunk") else encbank
    store_dir = args.out / "reuse_store"
    store_dir.mkdir(exist_ok=False)
    save_json(args.out / "reuse_input.json", data)
    result.update(reuse=identity, input={k: v for k, v in data.items() if k not in ("chunks", "prompts")},
        comparison_scope="fixed ordered pack, distinct sequential requests, trained matching adapters",
        implementation=getattr(reader, "implementation", "reference-sparse-reader"),
        quality_claim="implementation pilot only; retain separate trained-reader quality scores",
        full_recompute_scope="FULL computes every layer of the complete ordered token pack per request, uses normal within-request KV decode, same D0 upper LoRA",
        model_load_scope="excluded from latency totals; included in whole-worker peak memory and external telemetry")
    # No warm-up query touches this document pack. A small unrelated warm-up is
    # separately excluded, then all its state and prefix are discarded.
    progress("reuse_unrelated_warmup")
    warm_start = time.perf_counter()
    with torch.no_grad(), _amp(torch, device):
        warm_reader, warm_ref = _readers(args, encbank)
        warm_writer = warm_reader if hasattr(warm_reader, "write_chunk") else encbank
        ws = warm_writer.write_chunk([data["sink_id"]])
        wd = [warm_writer.write_chunk(c[:min(8, len(c))]) for c in chunks[:2]]
        _generate(torch, warm_reader, ws, wd, prompts[0][:4], list(range(min(4, len(prompts[0])))), 2)
    del ws, wd, warm_reader, warm_ref, warm_writer
    gc.collect()
    _sync(torch, device)
    result["warmup_s_excluded"] = time.perf_counter() - warm_start
    # Do not reset peak memory: writes, transfers, native prefix, model and
    # temporary caches must all remain represented in lifecycle high-water marks.
    stream_start = time.perf_counter()
    store = {"load_count": 1, "transfer_count": 1, "document_prefix_build_count": 0,
             "persistent_kind": "raw int64 tokens" if args.arm == "FULL" else "sink and h_j residuals",
             "os_file_cache": "uncontrolled; a just-written file may be cached by the OS",
             "path": str(store_dir), "workflow": "fixed ordered pack; no eviction, rearrangement or partial-prefix claim"}
    progress("reuse_write_once")
    adapter_digest = digest(args.adapter)
    keys = [json_digest({"tokens": c, "adapter": adapter_digest, "j": args.j}) for c in chunks]
    with torch.no_grad(), _amp(torch, device):
        def write_cold():
            return {"sink": writer.write_chunk([data["sink_id"]]).detach().cpu().clone(),
                    "documents": [writer.write_chunk(c).detach().cpu().clone() for c in chunks],
                    "input_sha256": data["token_sha256"]}
        cold, store["cold_compute_and_d2h_s"] = _timed(torch, device, write_cold)
        store["cold_tensor_bytes"] = _memory([cold["sink"], cold["documents"]])["tensor_bytes"]
        cold_io = _save_torch(torch, cold, store_dir / "cold.pt")
        store.update(cold_file_write_fsync_s=cold_io["seconds"], cold_file_bytes=cold_io["file_bytes"])
        if args.arm in ("A", "B"):
            progress("reuse_build_hot_once")
            hot, store["hot_build_and_d2h_s"] = _timed(torch, device,
                lambda: reader.build_hot_cache(cold["documents"], keys, cache_device="cpu"))
            store["hot_tensor_bytes"] = _memory(hot)["tensor_bytes"]
            hot_io = _save_torch(torch, _hot_payload(hot), store_dir / "hot.pt")
            store.update(hot_file_write_fsync_s=hot_io["seconds"], hot_file_bytes=hot_io["file_bytes"])
            del hot
    del cold
    progress("reuse_load_and_transfer_once")
    load_start = time.perf_counter()
    cold = torch.load(store_dir / "cold.pt", map_location="cpu", weights_only=False)
    if cold["input_sha256"] != data["token_sha256"]:
        raise ValueError("Loaded cache does not match the fixed input stream")
    hot_data = torch.load(store_dir / "hot.pt", map_location="cpu", weights_only=False) if args.arm in ("A", "B") else None
    store["file_load_once_s"] = time.perf_counter() - load_start
    with torch.no_grad(), _amp(torch, device):
        def transfer():
            sink = cold["sink"].to(device)
            if hot_data is not None:
                return sink, cold["documents"], _restore_hot(reader, hot_data, device)
            return sink, [d.to(device) for d in cold["documents"]], None
        (sink, docs, entries), store["h2d_once_s"] = _timed(torch, device, transfer)
        store["h2d_document_tensor_bytes"] = _memory([sink, entries if entries is not None else docs])["tensor_bytes"]
        if args.arm in ("D0", "NATIVE"):
            progress("reuse_build_native_prefix_once")
            _, store["native_prefix_build_s"] = _timed(torch, device, lambda: reader.build_prefix(sink, docs))
            store["document_prefix_build_count"] = reader.build_count
    del hot_data, cold
    resident = _resident(reader, sink, docs, entries)
    invariant = _signature(resident)
    store["resident_gpu_cache"] = _memory(resident)
    store["resident_cpu_hj_bytes"] = _memory(docs)["tensor_bytes"] if entries is not None else 0
    store["resident_scope"] = "actual retained document/sink tensors, plus h_j when retained by native prefix; excludes weights, query KV and transient attention concatenations"
    store["persistent_total_file_bytes"] = store["cold_file_bytes"] + store.get("hot_file_bytes", 0)
    store["setup_wall_s"] = time.perf_counter() - stream_start
    result["store"] = store
    save_json(args.out / "reuse_store_cost.json", store)
    records = []
    extra = {"hot_entries": entries, "document_keys": keys} if entries is not None else {}
    for index, prompt in enumerate(prompts):
        progress("reuse_request", query_index=index)
        if _signature(resident) != invariant:
            raise RuntimeError("A prior query changed shared document tensors")
        with torch.no_grad(), _amp(torch, device):
            _sync(torch, device)
            start = time.perf_counter()
            (logits, state), prefill_s = _timed(torch, device,
                lambda: reader.prefill(sink, docs, prompt, probe_indices=probes, **extra))
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("Nonfinite prefill logits")
            ids = [int(logits[0, -1].argmax().item())]
            _sync(torch, device)
            first = time.perf_counter()
            decode_steps = []
            for _ in range(args.generation_tokens - 1):
                step_start = time.perf_counter()
                logits = reader.decode_step(ids[-1], state)
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError("Nonfinite decode logits")
                ids.append(int(logits[0, -1].argmax().item()))
                _sync(torch, device)
                decode_steps.append(time.perf_counter() - step_start)
            end = time.perf_counter()
        route = copy.deepcopy(state.route_stats)
        if args.arm in ("D0", "NATIVE") and (not route.get("prefix_cache_hit") or reader.build_count != 1):
            raise RuntimeError("Native prefix was not reused exactly once across the fixed pack")
        if entries is not None and route.get("hot_hits") != len(chunks):
            raise RuntimeError("The reusable independent cache missed a fixed-pack document")
        if _signature(resident) != invariant:
            raise RuntimeError("Query generation changed shared document tensors")
        record = {"query_index": index, "prompt_sha256": json_digest(prompt), "generated_ids": ids,
            "generated_tokens": len(ids), "prefill_wall_s": prefill_s, "ttft_s": first - start,
            "decode_steps": len(ids) - 1, "decode_wall_s": end - first, "decode_step_wall_s": decode_steps,
            "decode_tps": (len(ids) - 1) / (end - first) if len(ids) > 1 else None,
            "query_e2e_s": end - start, "cumulative_e2e_s": end - stream_start,
            "cumulative_first_token_s": first - stream_start, "route_stats": route,
            "cache_file_load_s": 0., "document_h2d_bytes": 0,
            "peak_allocated_bytes_since_worker_start": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes_since_worker_start": torch.cuda.max_memory_reserved(device),
            "timing": "synchronized wall time including finite-logit checks and greedy token choice; no EOS early stop",
            "query_state_reused": False, "shared_document_cache_unchanged": True}
        records.append(record)
        del logits, state
        # No empty_cache/reset_peak/reload occurs between queries. Reporting and
        # private-state destruction are included in continuous lifecycle totals.
        save_json(args.out / f"reuse_request_{index:03d}.json", record)
    _sync(torch, device)
    lifecycle_s = time.perf_counter() - stream_start
    result.update(status="complete", phase="complete", completed_at=now(), requests=records,
        summary=summarize_reuse(records, store["setup_wall_s"], lifecycle_s),
        lifecycle_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        lifecycle_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        timing_eligible=True, external_monitor_validation_required=True,
        memory_scope="allocator high-water since guarded worker start, including model, warmup, cache construction and all requests; external monitor supplies driver/process incremental peaks")
    return result


def run_reuse_quality_probe(args, torch, encbank, tokenizer, chunks=None, prompts=None, source_ids=None,
                            probe_indices=None):
    """Bounded trained-model semantic probe; caller supplies the remote GPU guard.

    Pass real document chunks and two distinct prompts when available. Synthetic
    defaults support tiny CPU correctness tests only and cannot qualify timing.
    q0 -> q1 -> q0 verifies each path's fresh state and immutable cache reuse.
    """
    from native_infra_readers import NativeEncbankReader
    reader, reference = _readers(args, encbank)
    if chunks is None or prompts is None:
        data = workload(tokenizer, encbank.config, args)
        chunks, prompts = data["chunks"], data["prompts"][:2]
        real = False
    else:
        real = True
    if len(chunks) < 2 or len(prompts) != 2 or prompts[0] == prompts[1] or any(not x for x in chunks + prompts):
        raise ValueError("Quality probe needs at least two document blocks and two distinct nonempty prompts")
    if probe_indices is None:
        if real:
            raise ValueError("Real-data quality checks require each question's original probe_indices")
        probe_indices = [list(range(max(0, len(p) - 16), len(p))) for p in prompts]
    if (len(probe_indices) != 2 or any(not rows or min(rows) < 0 or max(rows) >= len(prompt)
            for rows, prompt in zip(probe_indices, prompts))):
        raise ValueError("Original question probes must fall entirely within the matching prompt")
    device, tokens = encbank.device, args.generation_tokens
    if not 1 <= tokens <= 32:
        raise ValueError("Quality probe generation is bounded to 1..32 tokens")
    if sum(map(len, chunks)) + 1 + max(map(len, prompts)) + tokens > encbank.config.max_position_embeddings:
        raise ValueError("Quality pack exceeds the unchanged shared model window")
    sink_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    keys = [json_digest(c) for c in chunks]
    writer = reader if hasattr(reader, "write_chunk") else encbank
    checks = dict(greedy_equal=True, selected_route_equal=True, repeat_query_equal=True,
                  cache_unchanged=True, finite_logits=True, storage_roundtrip=True)
    observations = []
    with torch.no_grad(), _amp(torch, device):
        direct_sink = writer.write_chunk([sink_id])
        direct_docs = [writer.write_chunk(c) for c in chunks]
        sink, docs, entries, roundtrip = _quality_storage_roundtrip(
            args, torch, reader, direct_sink, direct_docs, keys, device)
        checks["storage_roundtrip"] = roundtrip["passed"]
        if args.arm in ("D0", "NATIVE"):
            reader.build_prefix(sink, docs)
        resident = _resident(reader, sink, docs, entries)
        invariant = _signature(resident)
        extra = {"hot_entries": entries, "document_keys": keys} if entries is not None else {}
        paths = {"reuse": (reader, extra, sink, docs)}
        if reference is not None:
            paths["reference_cold"] = (reference, {}, direct_sink, direct_docs)
        if args.arm in ("D0", "NATIVE"):
            paths["native_cold"] = (NativeEncbankReader(encbank), {}, direct_sink, direct_docs)
        previous = {}
        for request_index, prompt_index in enumerate((0, 1, 0)):
            prompt = list(prompts[prompt_index])
            probes = list(probe_indices[prompt_index])
            record = {"request_index": request_index, "prompt_index": prompt_index, "paths": {}}
            logits_by_path = {}
            for name, (path_reader, arguments, path_sink, path_docs) in paths.items():
                out, captured = _generate(torch, path_reader, path_sink, path_docs, prompt, probes, tokens,
                                          extra=arguments, capture_logits=True)
                record["paths"][name] = out
                logits_by_path[name] = captured
                checks["finite_logits"] &= out["finite_logits"]
                if prompt_index in previous.get(name, {}):
                    old = previous[name][prompt_index]
                    checks["repeat_query_equal"] &= (old["generated_ids"] == out["generated_ids"]
                        and old["route_stats"]["selected_indices"] == out["route_stats"]["selected_indices"])
                previous.setdefault(name, {})[prompt_index] = out
            if args.arm == "FULL":
                # Independent direct-HF all-layer baseline on the exact full pack.
                packed = [sink_id] + [i for chunk in chunks for i in chunk] + prompt
                inputs = torch.tensor([packed], device=device, dtype=torch.long)
                expected, hf_logits = [], []
                output = encbank.model(input_ids=inputs, use_cache=True)
                past = output.past_key_values
                for index in range(tokens):
                    if index:
                        output = encbank.model(input_ids=torch.tensor([[expected[-1]]], device=device), past_key_values=past, use_cache=True)
                        past = output.past_key_values
                    value = output.logits[:, -1]
                    expected.append(int(value.argmax().item()))
                    hf_logits.append(value[0].float().cpu())
                record["paths"]["direct_hf"] = {"generated_ids": expected, "first_token": expected[0],
                    "route_stats": {"selected_indices": list(range(len(chunks)))},
                    "finite_logits": all(bool(torch.isfinite(t).all()) for t in hf_logits)}
                logits_by_path["direct_hf"] = hf_logits
                checks["finite_logits"] &= record["paths"]["direct_hf"]["finite_logits"]
                del output, past, inputs
            anchor = record["paths"]["reuse"]
            for name, out in record["paths"].items():
                checks["greedy_equal"] &= anchor["generated_ids"] == out["generated_ids"]
                checks["selected_route_equal"] &= anchor["route_stats"]["selected_indices"] == out["route_stats"]["selected_indices"]
                errors = [(a - b).abs() for a, b in zip(logits_by_path["reuse"], logits_by_path[name])]
                out["logit_error_vs_reuse"] = {"max_abs_by_step": [float(e.max()) if bool(torch.isfinite(e).all()) else None for e in errors],
                    "mean_abs_by_step": [float(e.mean()) if bool(torch.isfinite(e).all()) else None for e in errors],
                    "scope": "greedy trajectories; after any token divergence this is not same-input numerical parity"}
            checks["cache_unchanged"] &= _signature(resident) == invariant
            observations.append(record)
    return {"protocol": QUALITY_VERSION, "status": "complete", "passed": all(checks.values()) and real,
        "arm": "D0" if args.arm == "NATIVE" else args.arm, "checks": checks,
        "source_sha256": source_identity(), "adapter_sha256": digest(args.adapter),
        "model_config_sha256": digest(Path(args.model) / "config.json"), "recipe": recipe(args),
        "document_blocks": len(chunks), "document_tokens": sum(map(len, chunks)), "unique_queries": 2,
        "generation_tokens": tokens, "source_ids": source_ids, "real_input": real,
        "input_sha256": json_digest({"chunks": chunks, "prompts": prompts, "sink_id": sink_id,
                                     "probe_indices": probe_indices}),
        "probe_indices": probe_indices, "storage_roundtrip": roundtrip,
        "observations": observations, "finished_at": now(),
        "next_action": None if all(checks.values()) and real else "requires_path_quality_evaluation",
        "scope": "bounded implementation probe on one fixed pack, q0/q1/q0 fresh states, no answer targets; blocks are not independent source documents; not accuracy scoring or length-wide equivalence proof",
        "gate": "exact greedy IDs and selected indices, finite logits, repeated-query identity and no shared tensor mutation; no relaxed threshold"}
