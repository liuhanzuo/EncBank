"""CPU-only protocol, NVIDIA telemetry parsing, and sparse-infra report helpers."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

GIB = 2 ** 30
MIB = 2 ** 20
VERSION = "sparse-comem-5090-infra-v1"
GPU_NAME = "NVIDIA GeForce RTX 5090"
INCREMENTAL_CAP_BYTES = 28 * GIB
ALLOCATOR_RESERVE_BYTES = GIB // 2
LABELS = {"D0": "CoMem control (sparse wrapper)", "A": "Block-all", "B": "Block-pruned",
          "D1": "Dense-pruned", "NATIVE": "Native CoMem + LoRA", "FULL": "Full recompute (same upper LoRA)"}


def now():
    return datetime.now(timezone.utc).isoformat()


def _replace_json_file(source, destination):
    """Atomically replace an existing Windows JSON even with shared readers.

    MoveFileExW (Python os.replace) can reject an open replacement target on
    Windows despite FILE_SHARE_DELETE. ReplaceFileW supports that use case.
    First creation still uses os.replace; never delete/unlink the destination.
    """
    if os.name != "nt":
        os.replace(source, destination)
        return
    import ctypes
    from ctypes import wintypes

    library = ctypes.WinDLL("kernel32", use_last_error=True)
    replace_file = library.ReplaceFileW
    replace_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                             wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID)
    replace_file.restype = wintypes.BOOL
    if replace_file(str(Path(destination).absolute()), str(Path(source).absolute()),
                    None, 0, None, None):
        return
    error = ctypes.get_last_error()
    if error in (2, 3) and not Path(destination).exists():
        # No replaced file exists (initial creation or a removed destination).
        # os.replace is itself atomic, including a destination created after
        # this check. Any sharing/permission failure reaches the bounded retry.
        os.replace(source, destination)
        return
    raise ctypes.WinError(error)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for trial in range(8):
        try:
            _replace_json_file(tmp, path)
            return
        except PermissionError:
            if trial == 7:
                raise
            time.sleep(.025 * (trial + 1))


@contextmanager
def shared_open_text(path, encoding="utf-8"):
    """Open a read-only text stream without blocking Windows atomic replace.

    A watchdog can be descheduled while its file handle remains open. Windows
    readers therefore share READ, WRITE and DELETE, so a writer can replace the
    directory entry while this stream continues to read the complete old file.
    Ownership transfers from Win32 HANDLE -> CRT fd. The text wrapper borrows
    the descriptor, and this context always closes it exactly once, even when
    wrapper construction or close fails.
    """
    if os.name != "nt":
        with Path(path).open("r", encoding=encoding) as stream:
            yield stream
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                            wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    # OPEN_EXISTING; FILE_ATTRIBUTE_NORMAL. No handle inheritance or writes.
    handle = create_file(str(Path(path).absolute()), 0x80000000, 0x1 | 0x2 | 0x4,
                         None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        close_handle(handle)
        raise
    stream = None
    try:
        stream = os.fdopen(descriptor, "r", encoding=encoding, closefd=False)
        yield stream
    finally:
        try:
            if stream is not None:
                stream.close()
        finally:
            os.close(descriptor)


def read_json(path, default=None):
    try:
        with shared_open_text(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            result.update(block)
    return result.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_memory(value):
    if value is None or value.strip().upper() in ("N/A", "NA", "[N/A]", "NOT SUPPORTED", ""):
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*(MiB|GiB|B)?\s*", value)
    if not match:
        raise ValueError(f"Unknown NVIDIA memory value: {value!r}")
    scale = {None: MIB, "MiB": MIB, "GiB": GIB, "B": 1}[match.group(2)]
    return int(float(match.group(1)) * scale)


def parse_nvidia_xml(xml_text, gpu_index=0):
    root = ET.fromstring(xml_text)
    gpus = root.findall("gpu")
    if not 0 <= gpu_index < len(gpus):
        raise ValueError("Requested GPU is absent from NVIDIA telemetry")
    gpu = gpus[gpu_index]
    used = parse_memory(gpu.findtext("fb_memory_usage/used"))
    total = parse_memory(gpu.findtext("fb_memory_usage/total"))
    if used is None or total is None:
        raise ValueError("NVIDIA total/used GPU memory is unavailable")
    processes = []
    processes_node = gpu.find("processes")
    if processes_node is None:
        raise ValueError("NVIDIA full process inventory is unavailable")
    for process in processes_node.findall("process_info"):
        pid_text = process.findtext("pid", "")
        if not pid_text.isdigit():
            raise ValueError("NVIDIA process inventory contains an unreadable PID")
        processes.append({"pid": int(pid_text), "name": process.findtext("process_name", ""),
                          "type": process.findtext("type", "unknown"),
                          "used_bytes": parse_memory(process.findtext("used_memory"))})
    return {"timestamp": now(), "monotonic_s": time.monotonic(), "gpu_index": gpu_index,
            "name": gpu.findtext("product_name"), "uuid": gpu.findtext("uuid"),
            "driver_version": root.findtext("driver_version"),
            "used_bytes": used, "total_bytes": total, "processes": processes,
            "source": "nvidia-smi -q -x; includes compute and graphics process rows"}


def snapshot():
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    snap = parse_nvidia_xml(result.stdout)
    empty = [p["pid"] for p in snap["processes"] if not p["name"].strip()]
    if empty and os.name == "nt":
        # Numeric observed PIDs only; do not inspect private process arguments.
        command = "Get-Process -Id " + ",".join(str(p) for p in empty) + " -ErrorAction SilentlyContinue | Select-Object Id,ProcessName | ConvertTo-Json -Compress"
        resolved = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if resolved.stdout.strip():
            records = json.loads(resolved.stdout)
            if isinstance(records, dict):
                records = [records]
            names = {int(p["Id"]): p["ProcessName"] for p in records}
            for process in snap["processes"]:
                if not process["name"].strip() and process["pid"] in names:
                    process["name"] = names[process["pid"]]
                    process["name_source"] = "Get-Process Id/ProcessName only"
    snap["unclassified_background_rows"] = [p for p in snap["processes"]
        if p["type"].upper() == "C+G" and not _known_model_name(p["name"])]
    return snap


def _known_model_name(name):
    return bool(re.search(r"python|ollama|llama|vllm|lm.?studio|triton|pytorch|torch|comfy|kobold", name, re.I))


def model_processes(snap, own_pid=None):
    """Preserve the authorized gate policy under WDDM's broad C+G labeling.

    Windows reports many ordinary desktop processes as C+G. Such rows are kept
    for review rather than asserted to be verified non-model processes. C-only,
    MPS, and known model runtimes (including G-type Python) block admission.
    """
    return [p for p in snap["processes"] if p["pid"] != own_pid and
            (p["type"].upper() in ("C", "M", "C+M", "M+C", "UNKNOWN") or _known_model_name(p["name"]))]


def validate_idle(snap, own_pid=None):
    if snap["name"] != GPU_NAME:
        raise ValueError(f"Formal infra is local {GPU_NAME} only, got {snap['name']!r}")
    return snap["used_bytes"] < 5 * GIB and not model_processes(snap, own_pid)


def case_id(arm, cache_mode, document_tokens, prompt_tokens, generation_tokens):
    return f"n{document_tokens}_q{prompt_tokens}_g{generation_tokens}_{arm}_{cache_mode}"


def make_jobs(lengths=(4096, 16384), prompt_tokens=64, generation_tokens=32,
              arms=("D0", "A", "B", "D1"), modes=("cold_hj", "block_hot")):
    jobs = []
    for n in lengths:
        for arm in arms:
            for mode in (("cold_hj", "block_hot") if arm in ("A", "B") else ("cold_hj",)):
                if mode not in modes:
                    continue
                jobs.append({"id": case_id(arm, mode, n, prompt_tokens, generation_tokens),
                             "arm": arm, "cache_mode": mode, "document_tokens": n,
                             "prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens})
    return jobs


def validate_shape(config, n, q, g, j, m, chunk_size):
    if any(isinstance(x, bool) or int(x) != x or x < 1 for x in (n, q, g, chunk_size)):
        raise ValueError("All token lengths must be positive integers")
    if config.get("model_type") != "qwen3":
        raise ValueError("The pilot is implemented for dense Qwen3 only")
    if not 0 <= j < m <= int(config["num_hidden_layers"]):
        raise ValueError("Require 0 <= j < m <= model depth")
    # One sink, all original document positions, complete query and generation.
    required = 1 + n + q + g
    limit = int(config["max_position_embeddings"])
    if required > limit:
        raise ValueError(f"Original logical positions require {required} tokens but checkpoint declares {limit}; "
                         "no automatic truncation, position compression, or RoPE extension")
    return {"required_total_positions": required, "declared_window": limit,
            "extension_applied": False, "length_measure": "actual document + query + generation + sink tokens"}


def summarize_requests(records):
    if not records:
        return {}
    means = {key: statistics.mean(r[key] for r in records)
             for key in ("file_load_s", "h2d_wall_s", "prefill_wall_s", "ttft_s",
                         "decode_wall_s", "query_e2e_s")}
    total_decode_steps = sum(r["decode_steps"] for r in records)
    total_decode_wall = sum(r["decode_wall_s"] for r in records)
    means.update(repetitions=len(records), decode_tokens_per_s=(total_decode_steps / total_decode_wall
                 if total_decode_wall > 0 else None),
                 request_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in records),
                 request_peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in records),
                 allocated_includes_model=True, reserved_includes_model=True,
                 total_decode_steps=total_decode_steps)
    return means


def _cell(value, scale=1, digits=2):
    if value is None:
        return "--"
    return f"{value / scale:.{digits}f}"


def render_report(folder, plan, states):
    """Write a reviewable empty/partial/full table without fabricating missing data."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in plan:
        record = states.get(job["id"], {})
        result = record.get("result") or {}
        monitor = record.get("monitor") or {}
        successful = record.get("status") == "complete"
        metrics = result.get("summary", {}) if successful else {}
        costs = result.get("store", {}) if successful else {}
        row = {**job, "label": LABELS[job["arm"]], "status": record.get("status", "pending"),
               "metrics": metrics, "store": costs, "monitor": monitor,
               "error": record.get("error") or result.get("error"), "attempt": record.get("attempt")}
        rows.append(row)
    save_json(folder / "INFRA_SUMMARY.json", {"protocol": VERSION, "generated_at": now(), "rows": rows,
        "quality_claim": "none; synthetic fixed-length engineering precheck",
        "pending_baselines": (["full recompute"] if not any(j["arm"] == "FULL" for j in plan) else []) + ["CoMem exact prefix reuse"],
        "128k_status": "pending a common validated model/window configuration; never auto-expanded"})
    text = ["# CoMem + sparse attention: local infrastructure precheck", "",
        "Synthetic fixed-length input; no accuracy comparison. Strong-adapter engineering runs are not the final trained comparison.",
        "TTFT includes file loading, real H2D transfers, prefill and first-token selection. Decode uses identical per-token synchronization instrumentation for every arm. E2E is one query; model load and one-time document/hot writes are shown separately.",
        "", "| Method | Doc tokens | Cache | Status | Cold / extra hot file MiB | Request alloc / reserved GiB | Sampled GPU increase GiB | TTFT s | Decode tok/s | E2E s |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    tex = ["% Generated infrastructure fragment. Requires booktabs. Engineering precheck only.",
        "\\begin{table}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Method / cache & $N$ & Store (MiB) & Alloc. (GiB) & TTFT (s) & Dec. (tok/s) & E2E (s) " + r"\\",
        "\\midrule"]
    for row in rows:
        m, c, mon = row["metrics"], row["store"], row["monitor"]
        store_str = _cell(c.get("cold_file_bytes"), MIB, 1) + " / " + _cell(c.get("hot_file_bytes"), MIB, 1)
        alloc = _cell(m.get("request_peak_allocated_bytes"), GIB)
        reserved = _cell(m.get("request_peak_reserved_bytes"), GIB)
        delta = _cell(mon.get("peak_incremental_gpu_bytes"), GIB)
        ttft, tps, e2e = [_cell(m.get(k)) for k in ("ttft_s", "decode_tokens_per_s", "query_e2e_s")]
        text.append(f"| {row['label']} | {row['document_tokens']} | {row['cache_mode']} | {row['status']} | {store_str} | {alloc} / {reserved} | {delta} | {ttft} | {tps} | {e2e} |")
        label = row["label"] + (" / hot" if row["cache_mode"] == "block_hot" else " / cold")
        if row["status"] != "complete":
            label += " (" + row["status"].replace("_", "-") + ")"
        total_file = (c.get("cold_file_bytes", 0) + c.get("hot_file_bytes", 0)) if c else None
        tex.append(f"{label} & {row['document_tokens']} & {_cell(total_file, MIB, 1)} & {alloc} & {ttft} & {tps} & {e2e} " + r"\\")
    pending = (["Full recompute"] if not any(j["arm"] == "FULL" for j in plan) else []) + ["CoMem exact prefix reuse"]
    for label in pending:
        text.append(f"| {label} | -- | pending | pending | -- | -- | -- | -- | -- | -- |")
        tex.append(f"{label} (pending) & -- & -- & -- & -- & -- & -- " + r"\\")
    tex += ["\\bottomrule", "\\end{tabular}",
        "\\caption{Synthetic fixed-length engineering precheck on the local RTX 5090. Store includes persistent cold and additional hot files. Allocator peaks include model weights. Missing and failed configurations remain explicit; full recompute and exact-prefix baselines are pending.}",
        "\\label{tab:sparse-infra-precheck}", "\\end{table}"]
    text += ["", "Cold and extra hot cache costs", "",
        "| Method / cache / N | Document write+fsync s | Extra hot build+fsync s | Cold tensor bytes | Extra hot tensor bytes | H2D bytes/query |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        c = row["store"]
        text.append(f"| {row['label']} / {row['cache_mode']} / {row['document_tokens']} | {_cell(c.get('cold_write_total_s'))} | {_cell(c.get('hot_write_total_s'))} | {c.get('cold_tensor_bytes', '--')} | {c.get('hot_tensor_bytes', '--')} | {c.get('request_h2d_bytes', '--')} |")
    text += ["", "Memory admission and limitations", "",
        "Each child passes the original shared gpu_gate.py before/after-lock strict used < 5 GiB check, plus full-process NVIDIA checks. The 28 GiB incremental budget includes weights; it is not a 28 GiB limit on total device usage.",
        "The PyTorch allocator is capped at 27.5 GiB, reserving 0.5 GiB for non-allocator memory. NVIDIA sampled process usage (when available) and total device increase against the fixed admission baseline are monitored against 28 GiB. This is not a hard bound on unsampled driver allocations. Windows WDDM may return N/A process memory; N/A is retained, never treated as zero. Total-device differences can be affected by desktop residency changes.",
        "OOM/cap violations preserve the original input shape and terminate only the owned worker. Foreign model activity invalidates the run. No automatic shorter-input retry is permitted.",
        "128K+ requires a common validated longer-window checkpoint/configuration; this runner refuses positions outside the existing declared window."]
    (folder / "INFRA_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (folder / "INFRA_TABLE.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return rows
