"""One supervised local RTX5090 sparse-infra cell; never a quality benchmark.

Run via launch_sparse_infra.py so driver/process telemetry is observed outside
the CUDA process. No automatic shape reduction, retries, or RoPE extension.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for location in (ROOT / "exp", ROOT / "Encbank", ROOT / "exp/encbank_v2_benchmarks_20260908"):
    sys.path.insert(0, str(location))
from infra_protocol import (VERSION, GIB, GPU_NAME, INCREMENTAL_CAP_BYTES,
    ALLOCATOR_RESERVE_BYTES, digest, json_digest, now, read_json, save_json,
    snapshot, validate_idle, validate_shape, summarize_requests)
from infra_processes import MONITOR_POLICY_VERSION
from shared_state_protocol import validate_shared_mode
from same_math_protocol import validate_same_math_mode


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-kind", choices=("strong", "trained"), default="strong")
    ap.add_argument("--reader-implementation", choices=("reference", "decode_v2", "backend_v3"), default="reference")
    ap.add_argument("--validate-optimized-decode", action="store_true", help="Guarded tiny GPU parity checks before loading the measured 8B model")
    ap.add_argument("--validate-backend", action="store_true", help="Guarded tiny GPU backend parity before loading the measured model")
    ap.add_argument("--backend-model-parity", action="store_true", help="8B numerical diagnostic outside the profiler; diagnostic mode only")
    ap.add_argument("--profile-reader-only", action="store_true", help="Diagnostic profiler only; never an eligible infrastructure timing")
    ap.add_argument("--shared-state-diagnostic-only", action="store_true", help="One reference-prefill shared-state decode/oracle diagnostic; never formal timing")
    ap.add_argument("--shared-state-cpu-receipt", type=Path)
    ap.add_argument("--same-math-diagnostic-only", action="store_true")
    ap.add_argument("--same-math-cpu-receipt", type=Path)
    ap.add_argument("--arm", choices=("D0", "A", "B", "D1", "NATIVE", "FULL"), required=True)
    ap.add_argument("--cache-mode", choices=("cold_hj", "block_hot"), required=True)
    ap.add_argument("--document-tokens", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--generation-tokens", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--rho", type=float, default=.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.)
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--reuse-requests", type=int, default=0,
                    help="Separate fixed-pack stream with resident caches; 0 preserves the legacy cells")
    ap.add_argument("--reuse-quality-receipt", type=Path,
                    help="Passed remote quality checks for the trained reuse paths")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--gpu-wait-seconds", type=int, default=86400)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--supervisor-pid", type=int, required=True)
    ap.add_argument("--supervisor-heartbeat", type=Path, required=True)
    return ap


def require_supervisor(args):
    from infra_processes import validate_supervisor
    return validate_supervisor(args.supervisor_pid, args.supervisor_heartbeat,
                               HERE / "infra_sparse.py", args.out)


def start_supervisor_watchdog(args):
    """Exit only this owned worker if its external monitor disappears."""
    stop = threading.Event()
    def watch():
        while not stop.wait(2):
            try:
                require_supervisor(args)
            except Exception as exc:
                save_json(args.out / "WATCHDOG_ABORT.json", {"status": "monitor_lost", "error": str(exc), "time": now()})
                os._exit(75)
    thread = threading.Thread(target=watch, daemon=True, name="infra-supervisor-watchdog")
    thread.start()
    return stop


def durable_torch_save(torch, payload, path):
    path = Path(path)
    start = time.perf_counter()
    with path.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"seconds": time.perf_counter() - start, "file_bytes": path.stat().st_size}


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def timed_cuda(torch, function):
    torch.cuda.synchronize()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    value = function()
    end_event.record()
    end_event.synchronize()
    wall = time.perf_counter() - wall_start
    return value, {"wall_s": wall, "cuda_event_s": start_event.elapsed_time(end_event) / 1000.}


def synthetic_inputs(tokenizer, config, args):
    """Deterministic token IDs, not a semantic test or a length-padded QA score."""
    rng = random.Random(args.seed)
    special = set(tokenizer.all_special_ids)
    lo, hi = 128, min(int(config.vocab_size) - 1, 30000)
    if hi <= lo:
        raise ValueError("Vocabulary too small for the fixed synthetic workload")
    doc = []
    while len(doc) < args.document_tokens:
        token = rng.randrange(lo, hi)
        if token not in special:
            doc.append(token)
    seed_prompt = tokenizer.encode("Read the supplied records and explain their relationship in a short answer. ",
                                   add_special_tokens=False)
    prompt = (seed_prompt * ((args.prompt_tokens + len(seed_prompt) - 1) // len(seed_prompt)))[:args.prompt_tokens]
    chunks = [doc[i:i + args.chunk_size] for i in range(0, len(doc), args.chunk_size)]
    return chunks, prompt, list(range(max(0, len(prompt) - 16), len(prompt)))


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "result.json").exists():
        raise ValueError("Use a fresh attempt directory; never overwrite a measured result")
    lease = require_supervisor(args)
    if platform.system() != "Windows":
        raise ValueError("Formal measurements are restricted to the local Windows RTX5090")
    if args.cache_mode == "block_hot" and args.arm not in ("A", "B"):
        raise ValueError("Only independent-block A/B support a reusable per-document hot cache")
    if args.reader_implementation != "reference" and args.arm in ("NATIVE", "FULL"):
        raise ValueError("Decode optimization applies only to the sparse wrapper; native/full remain explicit references")
    if args.repetitions < 1 or not 0 < args.rho <= 1:
        raise ValueError("Repetitions and retain ratio are invalid")
    if args.profile_reader_only and (args.repetitions != 1 or args.generation_tokens != 4):
        raise ValueError("The bounded profiler uses one prefill plus three decode steps: repetitions=1, generation-tokens=4")
    if args.reader_implementation == "backend_v3" and not args.validate_backend:
        raise ValueError("The new backend requires guarded tiny GPU validation")
    if args.backend_model_parity and (not args.profile_reader_only or args.reader_implementation != "backend_v3"):
        raise ValueError("8B backend parity is diagnostic-only and requires backend_v3")
    shared_cpu = validate_shared_mode(args, worker=True)
    same_math_cpu = validate_same_math_mode(args, worker=True)
    reuse_contract = None
    if getattr(args, "reuse_requests", 0):
        from reuse_protocol import validate_reuse_mode
        reuse_contract = validate_reuse_mode(args)
    model_config = read_json(Path(args.model) / "config.json")
    window = validate_shape(model_config, args.document_tokens, args.prompt_tokens,
                            args.generation_tokens, args.j, args.m, args.chunk_size)
    identity = {"protocol": VERSION, "model": str(Path(args.model).resolve()),
                "model_config_sha256": digest(Path(args.model) / "config.json"),
                "adapter": str(Path(args.adapter).resolve()), "adapter_sha256": digest(args.adapter),
                "reader_sha256": digest(HERE / "sparse_reader.py"),
                "reader_implementation": args.reader_implementation,
                "optimized_reader_sha256": digest(HERE / "optimized_sparse_reader.py") if args.reader_implementation in ("decode_v2", "backend_v3") else None,
                "backend_reader_sha256": digest(HERE / "backend_sparse_reader.py") if args.reader_implementation == "backend_v3" else None,
                "backend_gpu_validation": args.validate_backend,
                "backend_validation_sha256": {name: digest(HERE / name) for name in ("backend_sparse_reader.py", "validate_backend_cpu.py", "test_backend_sparse_reader.py")} if args.validate_backend else None,
                "backend_model_parity": args.backend_model_parity,
                "backend_parity_sha256": digest(HERE / "backend_parity.py") if args.backend_model_parity else None,
                "shared_state_diagnostic_only": args.shared_state_diagnostic_only,
                "shared_state_sources_sha256": shared_cpu["source_sha256"] if shared_cpu else None,
                "shared_state_cpu_receipt_sha256": digest(args.shared_state_cpu_receipt) if shared_cpu else None,
                "shared_state_protocol_sha256": digest(HERE / "shared_state_protocol.py"),
                "same_math_diagnostic_only": args.same_math_diagnostic_only,
                "same_math_sources_sha256": same_math_cpu["source_sha256"] if same_math_cpu else None,
                "same_math_cpu_receipt_sha256": digest(args.same_math_cpu_receipt) if same_math_cpu else None,
                "same_math_protocol_sha256": digest(HERE / "same_math_protocol.py"),
                "optimized_gpu_validation": args.validate_optimized_decode,
                "profile_reader_only": args.profile_reader_only,
                "profiler_sha256": digest(HERE / "reader_profiler.py") if args.profile_reader_only else None,
                "optimized_validation_sha256": {name: digest(HERE / name) for name in ("validate_optimized_cpu.py", "test_optimized_sparse_reader.py")} if args.validate_optimized_decode else None,
                "infra_worker_sha256": digest(HERE / "infra_sparse.py"),
                "infra_protocol_sha256": digest(HERE / "infra_protocol.py"),
                "infra_processes_sha256": digest(HERE / "infra_processes.py"),
                "infra_launcher_sha256": digest(HERE / "launch_sparse_infra.py"),
                "monitor_policy_version": MONITOR_POLICY_VERSION,
                "encbank_sha256": digest(ROOT / "Encbank/encbank/model.py"),
                "native_reader_sha256": digest(HERE / "native_infra_readers.py") if args.arm in ("NATIVE", "FULL") else None,
                "adapter_kind": args.adapter_kind, "arm": args.arm, "cache_mode": args.cache_mode,
                "document_tokens": args.document_tokens, "prompt_tokens": args.prompt_tokens,
                "generation_tokens": args.generation_tokens, "chunk_size": args.chunk_size,
                "j": args.j, "m": args.m, "rho": args.rho, "rank": args.rank,
                "alpha": args.alpha, "seed": args.seed, "repetitions": args.repetitions}
    if reuse_contract is not None:
        identity["reuse"] = reuse_contract
    save_json(args.out / "config.json", identity)
    result = {"status": "starting", "identity": identity, "window": window, "started_at": now(),
              "pid": os.getpid(), "supervisor_lease": lease,
              "hostname": socket.gethostname(), "platform": platform.system(),
              "quality_claim": "none", "measurement_scope": "synthetic fixed-length engineering precheck",
              "fixed_generation_ignores_eos": True, "timing_eligible": False}
    if args.profile_reader_only:
        result["measurement_scope"] = "diagnostic profiler; no formal latency/throughput claim"
    elif args.shared_state_diagnostic_only:
        result["measurement_scope"] = "shared-prefill numerical diagnosis with same-QKV oracle; no formal latency/memory claim"
    elif args.same_math_diagnostic_only:
        result["measurement_scope"] = "fixed-MATH exact full-model logits and KV diagnosis; no formal latency/memory claim"
    def progress(phase, **changes):
        result.update(phase=phase, updated_at=now(), **changes)
        save_json(args.out / "progress.json", result)
    progress("waiting_for_gpu")
    watchdog = start_supervisor_watchdog(args)
    release = None
    torch = None
    try:
        from gpu_gate import acquire_gpu, _release_lock, _read_lock
        release = _release_lock
        waited_start = time.monotonic()
        while True:
            before = snapshot()
            if before["name"] != GPU_NAME:
                raise ValueError("GPU 0 is not the authorized local RTX5090")
            if not validate_idle(before, own_pid=os.getpid()):
                if time.monotonic() - waited_start >= args.gpu_wait_seconds:
                    raise TimeoutError("GPU still not idle; no model loaded")
                time.sleep(5)
                continue
            admission = acquire_gpu(22, None, 5., poll=5,
                max_wait=max(1, args.gpu_wait_seconds - int(time.monotonic() - waited_start)),
                tag="sparse-infra " + str(args.out))
            after = snapshot()
            owner = _read_lock()
            if owner and owner.get("pid") == os.getpid() and validate_idle(after, own_pid=os.getpid()):
                break
            release()
            time.sleep(2)
        # This is the fixed baseline before CUDA initialization/model allocation.
        progress("admitted", admission=admission, baseline_snapshot=after,
                 supplemental_prelock_snapshot=before, supervisor_pid=args.supervisor_pid)
        os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                          CUDA_VISIBLE_DEVICES="0")
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(2)
        torch.set_num_interop_threads(16)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        if properties.name != GPU_NAME:
            raise ValueError("CUDA device differs from admitted RTX5090")
        allocator_cap = INCREMENTAL_CAP_BYTES - ALLOCATOR_RESERVE_BYTES
        torch.cuda.set_per_process_memory_fraction(min(allocator_cap / properties.total_memory, 1.), 0)
        result["hardware"] = {"gpu": properties.name, "total_memory_bytes": properties.total_memory,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "torch_cpu_threads": 2,
            "torch_interop_threads": 16, "gpu_incremental_budget_bytes": INCREMENTAL_CAP_BYTES,
            "torch_allocator_cap_bytes": allocator_cap, "non_allocator_reserve_bytes": ALLOCATOR_RESERVE_BYTES,
            "autocast": "cuda bfloat16, including custom float32-master LoRA matmuls",
            "cap_scope": "allocator hard cap plus supervised sampled whole-GPU/process guard; not a driver hard cap",
            "admission": admission, "baseline_gpu_used_bytes": after["used_bytes"]}
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        import transformers.integrations.sdpa_attention as sdpa
        sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
        if args.validate_optimized_decode:
            progress("tiny_gpu_decode_validation")
            from validate_optimized_cpu import run_validation
            validation = run_validation(device="cuda", out_dir=args.out / "gpu_semantics")
            result["optimized_gpu_validation"] = validation
            if not validation.get("passed"):
                raise RuntimeError("Tiny GPU optimized-decode equivalence failed; no measured model loaded")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            # Tiny tests seed their fixture; restore the common measured recipe.
            torch.manual_seed(args.seed)
        if args.validate_backend:
            progress("tiny_gpu_backend_validation")
            from validate_backend_cpu import run_validation as validate_backend
            validation = validate_backend(device="cuda", out_dir=args.out / "gpu_backend_semantics")
            result["backend_gpu_validation"] = validation
            contract = validation.get("same_backend_contract", {})
            if (not contract.get("passed") or not validation.get("same_math_backend_bitwise_passed")
                    or any(contract.get(k, 0) for k in ("failures", "errors", "skipped"))):
                raise RuntimeError("Tiny GPU same-backend contract failed; no measured model loaded")
            # Cross-backend BF16 rounding/greedy changes in a random tiny model
            # remain in the original receipt (including passed=False). They do
            # not establish behavior of the measured 8B model: the outer pipeline
            # requires its separate strict numerical/route/profile proof first.
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(args.seed)
        if args.shared_state_diagnostic_only:
            progress("tiny_gpu_shared_state_validation")
            from validate_shared_state_cpu import run_validation as validate_shared
            validation = validate_shared(device="cuda", out_dir=args.out / "gpu_shared_state_semantics")
            result["shared_state_gpu_validation"] = validation
            if not validation.get("passed") or validation.get("source_sha256") != shared_cpu["source_sha256"]:
                raise RuntimeError("Shared-state GPU helper validation failed or source differs; no measured model loaded")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(args.seed)
        if args.same_math_diagnostic_only:
            progress("tiny_gpu_same_math_validation")
            from validate_same_math_cpu import run_validation as validate_same_math
            validation = validate_same_math(device="cuda", out_dir=args.out / "gpu_same_math_semantics")
            result["same_math_gpu_validation"] = validation
            if not validation.get("passed") or validation.get("source_sha256") != same_math_cpu["source_sha256"]:
                raise RuntimeError("Same-MATH GPU helper validation failed or source differs; no measured model loaded")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(args.seed)
        from train_8b_baseline import attach_lora, restore_flat
        from encbank.model import Encbank
        from sparse_reader import SparseEncbankReader, HotBlock, KVPair
        if args.reader_implementation == "decode_v2":
            from optimized_sparse_reader import OptimizedSparseEncbankReader
            SparseEncbankReader = OptimizedSparseEncbankReader
        elif args.reader_implementation == "backend_v3":
            from backend_sparse_reader import BackendAlignedSparseEncbankReader
            SparseEncbankReader = BackendAlignedSparseEncbankReader
        result["hardware"]["transformers"] = transformers.__version__
        progress("loading_model")
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to("cuda:0").eval()
        modules = attach_lora(model, args.j, args.rank, args.alpha, torch.float32)
        saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
        for key, expected in (("j", args.j), ("rank", args.rank), ("alpha", args.alpha)):
            if saved.get(key) != expected:
                raise ValueError(f"Adapter {key} differs from the requested recipe")
        if args.adapter_kind == "trained":
            recipe = saved.get("metadata", {}).get("recipe", {})
            source_arm = "D0" if args.arm in ("NATIVE", "FULL") else args.arm
            if recipe.get("arm") != source_arm or recipe.get("m") != args.m or recipe.get("rho") != args.rho:
                raise ValueError("Trained adapter must match arm, fusion depth and retention recipe")
        restore_flat(modules, saved["named"])
        result["adapter_metadata"] = {"kind": args.adapter_kind, "step": saved.get("step"),
            "comparison": "same strong Encbank LoRA engineering control" if args.adapter_kind == "strong"
            else "arm-specific trained adapter; quality evaluated separately"}
        del saved
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        encbank = Encbank(model, resume_j=args.j, tokenizer=tokenizer)
        if reuse_contract is not None:
            torch.cuda.synchronize()
            result["model_load_s_excluded"] = time.perf_counter() - load_start
            from reuse_benchmark import run_reuse_benchmark
            result = run_reuse_benchmark(args, torch, encbank, tokenizer, result, progress)
            save_json(args.out / "result.json", result)
            progress("complete")
            return 0
        if args.arm in ("NATIVE", "FULL"):
            from native_infra_readers import NativeEncbankReader, FullRecomputeReader
            reader = (NativeEncbankReader if args.arm == "NATIVE" else FullRecomputeReader)(encbank)
        else:
            reader = SparseEncbankReader(encbank, fusion_layer=args.m,
                probe_mode="block" if args.arm in ("A", "B") else "dense",
                retain_ratio=1. if args.arm in ("D0", "A") else args.rho,
                gradient_checkpointing=False).eval()
        writer = reader if hasattr(reader, "write_chunk") else encbank
        torch.cuda.synchronize()
        result["model_load_s_excluded"] = time.perf_counter() - load_start
        chunks, prompt, probes = synthetic_inputs(tokenizer, model.config, args)
        sink_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        result["input"] = {"token_sha256": json_digest({"chunks": chunks, "prompt": prompt}),
                            "source": "deterministic synthetic IDs, not QA data", "probe_indices": probes,
                            "document_tokens": sum(map(len, chunks)), "prompt_tokens": len(prompt)}
        # Short unrelated warmup, no document/query cache survives into the cell.
        progress("warmup")
        warm_start = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ws = writer.write_chunk([sink_id])
            wd = [writer.write_chunk(c[:16]) for c in chunks[:2]]
            wl, wstate = reader.prefill(ws, wd, prompt[:8], probe_indices=list(range(8)))
            reader.decode_step(int(wl[0, -1].argmax().item()), wstate)
        del ws, wd, wl, wstate
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result["warmup_s_excluded"] = time.perf_counter() - warm_start
        progress("writing_cold_cache")
        store_dir = args.out / "store"
        store_dir.mkdir()
        keys = [json_digest({"tokens": c, "model": identity["model"], "j": args.j}) for c in chunks]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            def write_documents():
                cold_sink = writer.write_chunk([sink_id]).detach().cpu().clone()
                cold_docs = [writer.write_chunk(c).detach().cpu().clone() for c in chunks]
                return {"identity": identity, "keys": keys, "sink": cold_sink, "documents": cold_docs}
            cold_payload, cold_compute = timed_cuda(torch, write_documents)
        cold_io = durable_torch_save(torch, cold_payload, store_dir / "cold.pt")
        cold_tensor_bytes = tensor_bytes([cold_payload["sink"], *cold_payload["documents"]])
        cold_dtype = str(cold_payload["documents"][0].dtype)
        store = {"cold_compute_and_d2h_wall_s": cold_compute["wall_s"],
            "cold_compute_and_d2h_cuda_event_s": cold_compute["cuda_event_s"],
            "cold_file_write_fsync_s": cold_io["seconds"],
            "cold_write_total_s": cold_compute["wall_s"] + cold_io["seconds"],
            "cold_tensor_bytes": cold_tensor_bytes, "cold_file_bytes": cold_io["file_bytes"],
            "cold_payload_dtype": cold_dtype, "cold_payload_semantics": "original token IDs" if args.arm == "FULL" else "h_j residuals plus sink",
            "hot_tensor_bytes": 0, "hot_file_bytes": 0, "hot_write_total_s": 0.,
            "os_file_cache": "uncontrolled; newly written files may already be cached; no claim of cold physical disk",
            "persistent_location": str(store_dir)}
        store["h2d_bytes_scope"] = "document cache and sink tensors only; prompt/token IDs and routing D2H are included in time but not this byte counter"
        if args.cache_mode == "block_hot":
            progress("building_hot_cache")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hot, hot_build = timed_cuda(torch, lambda: reader.build_hot_cache(
                    cold_payload["documents"], keys, cache_device="cpu"))
            hot_payload = {"identity": identity, "entries": [{"source_key": h.source_key,
                "token_count": h.token_count, "h_m": h.h_m,
                "raw_kv": {l: (p.k, p.v) for l, p in h.raw_kv.items()}} for h in hot]}
            hot_io = durable_torch_save(torch, hot_payload, store_dir / "hot.pt")
            store.update(hot_build_wall_s=hot_build["wall_s"], hot_build_cuda_event_s=hot_build["cuda_event_s"],
                hot_file_write_fsync_s=hot_io["seconds"], hot_write_total_s=hot_build["wall_s"] + hot_io["seconds"],
                hot_tensor_bytes=sum(h.tensor_bytes for h in hot), hot_file_bytes=hot_io["file_bytes"],
                hot_payload_semantics="every candidate block's raw K/V[j:m) plus full h_m; no h_m pruning before transfer")
            del hot, hot_payload
        del cold_payload
        result["store"] = store
        save_json(args.out / "store_cost.json", store)
        result["pre_request_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["pre_request_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        records = []
        for repetition in range(args.repetitions):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            progress("request", repetition=repetition)
            request_start = time.perf_counter()
            file_start = time.perf_counter()
            cold = torch.load(store_dir / "cold.pt", map_location="cpu", weights_only=False)
            if cold["identity"] != identity:
                raise ValueError("Cold cache identity differs")
            hot_data = None
            if args.cache_mode == "block_hot":
                hot_data = torch.load(store_dir / "hot.pt", map_location="cpu", weights_only=False)
                if hot_data["identity"] != identity:
                    raise ValueError("Hot cache identity differs")
            file_load_s = time.perf_counter() - file_start
            def transfer():
                sink_gpu = cold["sink"].to("cuda:0")
                if hot_data is None:
                    documents = [h.to("cuda:0") for h in cold["documents"]]
                    return sink_gpu, documents, None, tensor_bytes([cold["sink"], *cold["documents"]])
                signature = reader._signature()
                entries = [HotBlock(h["source_key"], signature, h["token_count"], h["h_m"].to("cuda:0"),
                    {int(l): KVPair(p[0].to("cuda:0"), p[1].to("cuda:0"))
                     for l, p in h["raw_kv"].items()}) for h in hot_data["entries"]]
                return sink_gpu, cold["documents"], entries, tensor_bytes([cold["sink"]]) + sum(h.tensor_bytes for h in entries)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                (sink_gpu, documents, entries, h2d_bytes), h2d_time = timed_cuda(torch, transfer)
                if args.same_math_diagnostic_only:
                    progress("same_math_diagnosis")
                    from optimized_sparse_reader import OptimizedSparseEncbankReader
                    from same_math_diagnostic import run_same_math_diagnostic
                    reference_reader = OptimizedSparseEncbankReader(encbank, fusion_layer=args.m,
                        probe_mode="dense", retain_ratio=1., gradient_checkpointing=False).eval()
                    receipt = run_same_math_diagnostic(reference_reader, reader, sink_gpu, documents,
                        prompt, probes, out_dir=args.out / "same_math_diagnostic", decode_steps=3)
                    result["same_math_receipt"] = receipt
                    if receipt.get("status") != "complete" or receipt.get("passed") is not True:
                        raise RuntimeError("Fixed-MATH exact model/KV diagnostic failed; inspect saved receipt")
                    store["request_h2d_bytes"] = h2d_bytes
                    result.update(status="complete", phase="same_math_complete", completed_at=now(),
                        requests=[], summary={}, store=store, timing_eligible=False,
                        external_monitor_validation_required=True)
                    save_json(args.out / "result.json", result)
                    progress("same_math_complete")
                    return 0
                if args.shared_state_diagnostic_only:
                    progress("shared_state_diagnosis")
                    from optimized_sparse_reader import OptimizedSparseEncbankReader
                    from shared_state_diagnostic import run_shared_state_diagnostic
                    reference_reader = OptimizedSparseEncbankReader(encbank, fusion_layer=args.m,
                        probe_mode="dense", retain_ratio=1., gradient_checkpointing=False).eval()
                    receipt = run_shared_state_diagnostic(reference_reader, reader, sink_gpu, documents,
                        prompt, probes, out_dir=args.out / "shared_state_diagnostic", decode_steps=3)
                    result["shared_state_receipt"] = receipt
                    if receipt.get("status") != "complete" or not receipt.get("protocol_checks_passed"):
                        raise RuntimeError("Shared-state diagnostic did not finish or violated its state/measurement contract")
                    store["request_h2d_bytes"] = h2d_bytes
                    result.update(status="complete", phase="shared_state_complete", completed_at=now(),
                        requests=[], summary={}, store=store, timing_eligible=False,
                        external_monitor_validation_required=True)
                    save_json(args.out / "result.json", result)
                    progress("shared_state_complete")
                    return 0
                if args.profile_reader_only:
                    from reader_profiler import run_reader_profile
                    extra = {} if entries is None else {"hot_entries": entries, "document_keys": keys}
                    if args.backend_model_parity:
                        progress("backend_model_parity")
                        from optimized_sparse_reader import OptimizedSparseEncbankReader
                        from backend_parity import run_backend_parity
                        reference_reader = OptimizedSparseEncbankReader(encbank, fusion_layer=args.m,
                            probe_mode="block" if args.arm in ("A", "B") else "dense",
                            retain_ratio=1. if args.arm in ("D0", "A") else args.rho,
                            gradient_checkpointing=False).eval()
                        parity = run_backend_parity(reference_reader, reader, sink_gpu, documents,
                            prompt, probes, hot_cache=entries, document_keys=keys,
                            decode_steps=3, out_dir=args.out / "model_backend_semantics")
                        result["backend_model_parity"] = parity
                        if not parity.get("passed"):
                            raise RuntimeError("Measured-model backend parity failed; inspect diagnostic receipt")
                        del reference_reader
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    progress("profiling_reader")
                    receipt = run_reader_profile(
                        lambda: reader.prefill(sink_gpu, documents, prompt, probe_indices=probes, **extra),
                        reader.decode_step, args.out / "profiler", decode_steps=3, device="cuda",
                        metadata={"identity": identity, "input": result["input"],
                                  "scope": "prefill and first three decode steps; file IO/H2D precede trace",
                                  "file_load_s_diagnostic": file_load_s,
                                  "h2d_timing_diagnostic": h2d_time, "h2d_bytes": h2d_bytes})
                    store["request_h2d_bytes"] = h2d_bytes
                    result.update(status="complete", phase="profile_complete", completed_at=now(),
                                  profiler_receipt=receipt, requests=[], summary={}, store=store,
                                  timing_eligible=False, external_monitor_validation_required=True)
                    save_json(args.out / "result.json", result)
                    progress("profile_complete")
                    return 0
                def prefill():
                    extra = {} if entries is None else {"hot_entries": entries, "document_keys": keys}
                    logits, state = reader.prefill(sink_gpu, documents, prompt, probe_indices=probes, **extra)
                    if not bool(torch.isfinite(logits).all().item()):
                        raise FloatingPointError("Nonfinite prefill logits")
                    token = int(logits[0, -1].argmax().item())
                    return logits, state, token
                (logits, state, token), prefill_time = timed_cuda(torch, prefill)
                first_token_at = time.perf_counter()
                ids, decode_events, decode_walls = [token], [], []
                for step in range(1, args.generation_tokens):
                    def decode_one(token=token):
                        next_logits = reader.decode_step(token, state)
                        if not bool(torch.isfinite(next_logits).all().item()):
                            raise FloatingPointError("Nonfinite decode logits")
                        return next_logits, int(next_logits[0, -1].argmax().item())
                    (logits, token), elapsed = timed_cuda(torch, decode_one)
                    ids.append(token)
                    decode_events.append(elapsed["cuda_event_s"])
                    decode_walls.append(elapsed["wall_s"])
                torch.cuda.synchronize()
                query_end = time.perf_counter()
            record = {"repetition": repetition, "generated_ids": ids, "generated_tokens": len(ids),
                "file_load_s": file_load_s, "h2d_wall_s": h2d_time["wall_s"],
                "h2d_cuda_event_s": h2d_time["cuda_event_s"], "h2d_bytes": h2d_bytes,
                "prefill_wall_s": prefill_time["wall_s"], "prefill_cuda_event_s": prefill_time["cuda_event_s"],
                "ttft_s": first_token_at - request_start, "decode_steps": len(ids) - 1,
                "decode_wall_s": query_end - first_token_at,
                "decode_step_wall_s": decode_walls, "decode_step_cuda_event_s": decode_events,
                "query_e2e_s": query_end - request_start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "route_stats": state.route_stats, "timing": "CUDA events + per-step synchronization and wall timers",
                "state_positions": {"query_position": int(state.query_position), "pack_position": int(state.pack_position)},
                "h2d_bytes_scope": store["h2d_bytes_scope"], "finite_logits_checks_included": True,
                "timing_overlap": "stage wall/CUDA event values are diagnostics; never sum them together"}
            records.append(record)
            save_json(args.out / f"request_{repetition:03d}.json", record)
            store["request_h2d_bytes"] = h2d_bytes
            del logits, state, cold, hot_data, sink_gpu, documents, entries
        result.update(status="complete", phase="complete", completed_at=now(), requests=records,
                      summary=summarize_requests(records), store=store,
                      timing_eligible=True, external_monitor_validation_required=True)
        result["summary"]["first_query_with_store_build_s"] = (store["cold_write_total_s"]
            + store["hot_write_total_s"] + records[0]["query_e2e_s"])
        result["summary"]["all_queries_with_store_build_s"] = (store["cold_write_total_s"]
            + store["hot_write_total_s"] + sum(r["query_e2e_s"] for r in records))
        save_json(args.out / "result.json", result)
        progress("complete")
        return 0
    except Exception as exc:
        is_oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        result.update(status="oom" if is_oom else "failed", timing_eligible=False,
                      error=str(exc), traceback=traceback.format_exc(), finished_at=now(),
                      original_shape_preserved=True, automatic_shorter_retry=False)
        if torch is not None and torch.cuda.is_initialized():
            result["failure_allocator"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        save_json(args.out / "result.json", result)
        save_json(args.out / "progress.json", result)
        print(result["traceback"], file=sys.stderr, flush=True)
        return 42 if is_oom else 1
    finally:
        watchdog.set()
        if release is not None:
            release()


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        exit_code = run(arguments)
    except Exception as error:
        arguments.out.mkdir(parents=True, exist_ok=True)
        save_json(arguments.out / "result.json", {"status": "invalid_configuration", "error": str(error),
            "traceback": traceback.format_exc(), "original_shape_preserved": True,
            "requested": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
            "timing_eligible": False, "time": now()})
        raise
    raise SystemExit(exit_code)
