"""Isolated reader profiling, never a source of publishable latency measurements.

Call only after the normal GPU admission/cap checks and after model/cache setup.
The caller retains ownership of its no_grad/autocast context, GPU, and cleanup.
This module does not move a model, change an attention backend, or launch work.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import warnings


FORMAT = "sparse-reader-profiler-v1"


def _number(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _attribute(event, name):
    try:
        return _number(getattr(event, name))
    except (AttributeError, RuntimeError):
        return None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _categories(name):
    value = name.lower()
    groups = []
    if "scaled_dot_product" in value or "sdpa" in value:
        groups.append("sdpa")
    if "flash_attention" in value or "flash_fwd" in value:
        groups.append("flash_named_operator")
    if "efficient_attention" in value or "mem_efficient" in value:
        groups.append("efficient_named_operator")
    if "scaled_dot_product_attention_math" in value:
        groups.append("math_named_operator")
    if name in ("aten::cat", "aten::concat", "aten::concatenate"):
        groups.append("cat")
    if name in ("aten::cos", "aten::sin", "aten::neg") or "rotary" in value or "rope" in value:
        groups.append("rope_related_primitive")
    if any(part in value for part in ("copy", "memcpy", "aten::to", "aten::_to_copy")):
        groups.append("copy_or_cast")
    if name in ("prefill", "decode") or name.startswith("decode_step_"):
        groups.append("phase")
    return groups


def inspect_chrome_trace(path):
    """Require device activity records, not merely a CUDA runtime launch call.

    Kineto Chrome traces use `kernel`, `gpu_memcpy`, and `gpu_memset` categories
    for GPU activities. Runtime API categories and CUDA-named CPU operators do
    not establish a GPU timeline. Preserve unexpected categories for review.
    """
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    categories = Counter()
    kernels = defaultdict(lambda: {"count": 0, "duration_sum_us": 0.0})
    transfers = defaultdict(lambda: {"count": 0, "duration_sum_us": 0.0})
    kernel_count = transfer_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        cat = str(event.get("cat", ""))
        categories[cat] += 1
        if event.get("ph") != "X":
            continue
        duration = _number(event.get("dur"))
        timestamp = _number(event.get("ts"))
        if duration is None or duration < 0 or timestamp is None:
            continue
        # Categories are case-insensitive exact tokens; do not match arbitrary
        # CPU event names containing 'cuda', 'kernel' or 'flash'.
        tokens = {x.strip().lower() for x in cat.split(",")}
        name = str(event.get("name", "<unnamed>"))
        if "kernel" in tokens:
            row = kernels[name]
            kernel_count += 1
        elif tokens.intersection({"gpu_memcpy", "gpu_memset"}):
            row = transfers[name]
            transfer_count += 1
        else:
            continue
        row["count"] += 1
        row["duration_sum_us"] += duration
    return {
        "event_count": len(events), "event_categories": dict(categories),
        "cuda_kernel_events": kernel_count, "cuda_transfer_events": transfer_count,
        "cuda_kernel_timeline_observed": kernel_count > 0,
        "cuda_any_activity_timeline_observed": kernel_count + transfer_count > 0,
        "kernels": [{"name": key, **value} for key, value in sorted(kernels.items())],
        "transfers": [{"name": key, **value} for key, value in sorted(transfers.items())],
        "duration_scope": "Sum of recorded device event durations; overlapping kernels may overlap. "
                          "This is not elapsed GPU wall time or request latency.",
    }


def _operator_rows(profiler, *, device_timeline_observed):
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        name = str(event.key)
        row = {
            "name": name, "count": int(event.count),
            "device_type": str(getattr(event, "device_type", "unknown")),
            "input_shapes": getattr(event, "input_shapes", None),
            "categories": _categories(name),
            "self_cpu_time_us": _attribute(event, "self_cpu_time_total"),
            "cpu_time_us": _attribute(event, "cpu_time_total"),
            "self_cpu_net_memory_bytes": _attribute(event, "self_cpu_memory_usage"),
            "cpu_net_memory_bytes": _attribute(event, "cpu_memory_usage"),
            "self_device_time_us": (_attribute(event, "self_device_time_total")
                                    if device_timeline_observed else None),
            "device_time_us": (_attribute(event, "device_time_total")
                               if device_timeline_observed else None),
            "profiler_self_device_net_memory_bytes": _attribute(event, "self_device_memory_usage"),
            "profiler_device_net_memory_bytes": _attribute(event, "device_memory_usage"),
        }
        rows.append(row)
    return rows


def _profiler_event_evidence(profiler):
    types = Counter()
    legacy = 0
    device_time_reported = False
    phases = Counter()
    for event in profiler.events():
        types[str(getattr(event, "device_type", "unknown"))] += 1
        legacy += int(bool(getattr(event, "is_legacy", False)))
        value = _attribute(event, "self_device_time_total")
        device_time_reported |= value is not None and value > 0
        if "sdpa" in _categories(str(event.name)):
            node, phase = event, "unattributed"
            # CPU ancestry is evidence for host operator phase, not a direct
            # CUDA stream correlation. Preserve unknown ancestry explicitly.
            for _ in range(200):
                if node is None:
                    break
                if str(node.name) in ("prefill", "decode"):
                    phase = str(node.name)
                    break
                node = getattr(node, "cpu_parent", None)
            phases[(phase, str(event.name))] += 1
    return {"event_device_type_counts": dict(types), "legacy_event_count": legacy,
            "profiler_reports_nonzero_device_time": device_time_reported,
            "backend_host_phase_counts": [{"phase": phase, "name": name, "count": count}
                                          for (phase, name), count in sorted(phases.items())],
            "interpretation": "Legacy CUDA event timing can exist without a CUPTI kernel timeline. "
                              "These flags alone never establish an exported device trace."}


def _allocator_snapshot(torch, target):
    if target.type != "cuda":
        return None
    values, errors = {}, {}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        try:
            values[name + "_bytes"] = int(getattr(torch.cuda, name)(target))
        except Exception as exc:
            values[name + "_bytes"] = None
            errors[name] = type(exc).__name__ + ": " + str(exc)
    try:
        retries = torch.cuda.memory_stats(target).get("num_alloc_retries")
        values["num_alloc_retries"] = None if retries is None else int(retries)
    except Exception as exc:
        values["num_alloc_retries"] = None
        errors["num_alloc_retries"] = type(exc).__name__ + ": " + str(exc)
    values["errors"] = errors
    return values


def run_reader_profile(prefill_fn, decode_step_fn, out_dir, *, decode_steps=3,
                       device="cuda", metadata=None):
    """Profile one prefill and bounded greedy cached decode, returning a receipt.

    `prefill_fn()` returns `(logits,state)`, and `decode_step_fn(int_token,state)`
    returns new logits. The reader graph, route and SDPA policy are unchanged.
    Logs contain diagnostics only. No accuracy or speed result is emitted.
    CUDA callers MUST already hold the original GPU gate and incremental cap.
    Exceptions produce a failed receipt and are then re-raised.
    """
    if not isinstance(decode_steps, int) or not 0 <= decode_steps <= 3:
        raise ValueError("This diagnostic is bounded to 0..3 decode steps")
    import torch
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or already-admitted CUDA profiling is supported")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no reader operation was run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(out_dir) / ("reader_profile_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "trace.json"
    operators_path = output / "operators.json"
    summary_path = output / "summary.json"
    supported = torch.profiler.supported_activities()
    requested = [torch.profiler.ProfilerActivity.CPU]
    cuda_supported = torch.profiler.ProfilerActivity.CUDA in supported
    if target.type == "cuda" and cuda_supported:
        requested.append(torch.profiler.ProfilerActivity.CUDA)
    receipt = {
        "format": FORMAT, "status": "running", "timing_eligible": False,
        "kind": "instrumented-backend-diagnostic-not-infra-benchmark",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device_requested": str(target),
        "supported_activities": sorted(str(a) for a in supported),
        "requested_activities": [str(a) for a in requested],
        "record_shapes": True, "profile_memory": True, "with_stack": False,
        "decode_steps_requested": decode_steps, "decode_steps_completed": 0,
        "generated_ids": [], "metadata": metadata or {},
        "trace_path": str(trace_path), "operators_path": str(operators_path),
        "summary_path": str(summary_path), "warnings": [], "errors": [],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundaries": "Model/cache loading, H2D and writing happen before this callback. "
                      "Stages include greedy token selection and completion sync. "
                      "Profiler overhead changes execution; never compare these durations as infra latency.",
        "memory_scope": "Operator-attributed net allocation/free deltas, not allocator peak, "
                        "reserved memory, NVML process usage or sampled device VRAM. Negative deltas are valid.",
        "rope_operator_scope": "cos/sin/neg are candidate RoPE primitives, not proven RoPE-only work. "
                               "The module does not inject hooks or change reader calls.",
        "timing_scope": "CPU operator intervals and attributed device kernel durations are different. "
                        "Rows and nested phase totals overlap; do not sum them as request duration.",
        "allocator_diagnostic_scope": "Before/after capture snapshots. Historical max counters are NOT reset "
                                      "and include earlier worker allocations. Profiler can retain tensors and add "
                                      "overhead; these observations do not replace non-profiled infra memory.",
    }
    # Validate caller metadata before executing the reader.
    json.dumps(receipt, allow_nan=False)
    failure = None
    profiler = None
    caught = []

    def synchronize():
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def token_from_logits(logits):
        if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("Reader logits must be a tensor shaped [1,T,V]")
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("Non-finite logits during reader profiling")
        return int(logits[0, -1].argmax().item())

    try:
        synchronize()
        receipt["allocator_before_capture"] = _allocator_snapshot(torch, target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.profiler.profile(activities=requested, record_shapes=True,
                                        profile_memory=True, with_stack=False) as profiler:
                with torch.profiler.record_function("prefill"):
                    logits, state = prefill_fn()
                    token = token_from_logits(logits)
                    receipt["generated_ids"].append(token)
                    synchronize()
                with torch.profiler.record_function("decode"):
                    for step in range(decode_steps):
                        with torch.profiler.record_function("decode_step_" + str(step)):
                            logits = decode_step_fn(token, state)
                            token = token_from_logits(logits)
                            receipt["generated_ids"].append(token)
                            receipt["decode_steps_completed"] += 1
                            synchronize()
                receipt["route_stats"] = getattr(state, "route_stats", None)
    except Exception as exc:
        failure = exc
        receipt["errors"].append({"phase": "reader_or_profiler", "type": type(exc).__name__, "message": str(exc)})
    receipt["warnings"] = [str(item.message) for item in caught]
    receipt["allocator_after_capture"] = _allocator_snapshot(torch, target)
    before_retries = (receipt.get("allocator_before_capture") or {}).get("num_alloc_retries")
    after_retries = (receipt.get("allocator_after_capture") or {}).get("num_alloc_retries")
    receipt["allocator_retry_delta"] = (after_retries - before_retries
                                        if before_retries is not None and after_retries is not None else None)
    trace_info = None
    rows = []
    if profiler is not None:
        try:
            profiler.export_chrome_trace(str(trace_path))
            trace_info = inspect_chrome_trace(trace_path)
            receipt["profiler_event_evidence"] = _profiler_event_evidence(profiler)
            rows = _operator_rows(profiler, device_timeline_observed=trace_info["cuda_kernel_timeline_observed"])
            _write_json(operators_path, {"rows": rows, "key_rows": [r for r in rows if r["categories"]],
                                         "scope": receipt["memory_scope"] + " " + receipt["timing_scope"]})
        except Exception as exc:
            receipt["errors"].append({"phase": "trace_export_or_aggregation", "type": type(exc).__name__, "message": str(exc)})
            if failure is None:
                failure = exc
    receipt["trace_evidence"] = trace_info
    observed = bool(trace_info and trace_info["cuda_kernel_timeline_observed"])
    legacy_reported = bool(receipt.get("profiler_event_evidence", {}).get("legacy_event_count"))
    receipt["cuda_timing_status"] = ("observed_device_kernel_timeline" if observed else
                                     "not_requested" if target.type == "cpu" else
                                     "legacy_device_timing_without_kernel_timeline" if legacy_reported else
                                     "missing_device_kernel_timeline")
    receipt["device_time_missing_reason"] = (None if observed else
        "No actual CUDA kernel activity records in exported trace. CUDA runtime launch calls, "
        "CPU operator names, or unsupported/fallback profiler timing are not GPU timeline evidence.")
    receipt["backend_operator_evidence"] = [
        {"name": row["name"], "device_type": row["device_type"], "count": row["count"],
         "input_shapes": row["input_shapes"]}
        for row in rows if "sdpa" in row["categories"]]
    receipt["key_operator_names"] = sorted({r["name"] for r in rows if r["categories"]})
    receipt["status"] = "complete" if failure is None else "failed"
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(summary_path, receipt)
    if failure is not None:
        raise failure
    return receipt


def _self_test(output):
    """Small real CPU profiler smoke, with no model, CUDA, queue or deployment."""
    import os
    import sys
    if sys.platform != "linux":
        raise RuntimeError("Run Torch self-test on the remote Linux CPU host")
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="2")
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(19)
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 7, 8)
    v = torch.randn_like(k)
    projection = torch.randn(16, 11)
    def calculation():
        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        h = attention.transpose(1, 2).reshape(1, 3, 16)
        return (h.cos() + h.sin()).clone() @ projection
    def prefill():
        return calculation(), SimpleNamespace(route_stats={"selected_indices": [0, 2]}, count=0)
    def decode(token, state):
        state.count += 1
        return calculation()
    with torch.no_grad():
        receipt = run_reader_profile(prefill, decode, output, device="cpu", metadata={"self_test": True})
    assert receipt["status"] == "complete" and receipt["decode_steps_completed"] == 3
    assert len(receipt["generated_ids"]) == 4
    assert receipt["cuda_timing_status"] == "not_requested"
    operators = json.loads(Path(receipt["operators_path"]).read_text())
    assert operators["rows"] and receipt["backend_operator_evidence"]
    assert all(r["self_device_time_us"] is None and r["device_time_us"] is None
               for r in operators["rows"])
    assert {"prefill", "decode", "decode_step_0", "decode_step_2"}.issubset(receipt["key_operator_names"])
    # A CPU/runtime launch event is not a CUDA kernel, even if its name sounds like one.
    synthetic = Path(output) / "synthetic_trace_category_fixture.json"
    _write_json(synthetic, {"synthetic_fixture": True, "traceEvents": [
        {"cat": "cuda_runtime", "ph": "X", "name": "cudaLaunchKernel", "ts": 0, "dur": 11},
        {"cat": "cpu_op", "ph": "X", "name": "aten::_scaled_dot_product_flash_attention", "ts": 0, "dur": 22}]})
    assert not inspect_chrome_trace(synthetic)["cuda_any_activity_timeline_observed"]
    _write_json(synthetic, {"synthetic_fixture": True, "traceEvents": [
        {"cat": "kernel", "ph": "X", "name": "flash_fwd_test", "ts": 0, "dur": 7},
        {"cat": "gpu_memcpy", "ph": "X", "name": "HtoD", "ts": 0, "dur": 2}]})
    checked = inspect_chrome_trace(synthetic)
    assert checked["cuda_kernel_events"] == 1 and checked["cuda_transfer_events"] == 1
    def failure_prefill():
        raise RuntimeError("expected reader failure for propagation test")
    failure_root = Path(output) / "failure_propagation_fixture"
    try:
        run_reader_profile(failure_prefill, decode, failure_root, device="cpu")
    except RuntimeError as exc:
        assert "expected reader failure" in str(exc)
    else:
        raise AssertionError("Reader failure was not propagated")
    failed_paths = sorted(failure_root.glob("reader_profile_*/summary.json"))
    failed = json.loads(failed_paths[-1].read_text())
    assert failed["status"] == "failed" and not failed["timing_eligible"]
    assert failed["decode_steps_completed"] == 0 and failed["errors"]
    result = {"passed": True, "kind": "CPU-profiler-functional-smoke-not-reader-equivalence",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "receipt": receipt["summary_path"], "trace_category_checks": "passed",
              "failure_propagation_check": "passed", "expected_failed_fixture": str(failed_paths[-1])}
    _write_json(Path(output) / "self_test.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _self_test(args.out)
