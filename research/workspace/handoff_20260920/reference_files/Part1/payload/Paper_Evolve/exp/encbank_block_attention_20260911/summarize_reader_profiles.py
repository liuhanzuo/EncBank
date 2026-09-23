"""Read actual diagnostic receipts; do not derive an infrastructure speed score."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from infra_protocol import read_json, save_json, now, GIB


def summarize(folder, completed_native_from=None):
    rows = []
    sources = [(folder, v, False) for v in ("reference", "decode_v2")]
    if completed_native_from is not None:
        sources.insert(0, (completed_native_from, "reference", True))
    for source_folder, variant, native_only in sources:
        status = read_json(source_folder / variant / "status.json", {})
        for case, job in status.get("jobs", {}).items():
            if native_only and (job.get("status") != "complete" or job.get("result", {}).get("identity", {}).get("arm") != "NATIVE"):
                continue
            attempt = Path(job["attempt"])
            result = read_json(attempt / "result.json", {})
            mon = read_json(attempt / "monitor.json", {})
            row = {"case": case, "implementation": variant, "status": job.get("status"),
                   "attempt": str(attempt), "timing_eligible": False, "source_folder": str(source_folder)}
            if job.get("status") == "complete":
                ident = result["identity"]
                receipt = result["profiler_receipt"]
                if (not ident.get("profile_reader_only") or result.get("timing_eligible")
                        or receipt.get("timing_eligible") or receipt.get("status") != "complete"
                        or mon.get("status") != "complete" or mon.get("exit_code") != 0
                        or mon.get("worker_lease") != result.get("supervisor_lease")):
                    raise ValueError(f"Invalid diagnostic receipt: {attempt}")
                disk_receipt = read_json(receipt["summary_path"])
                if disk_receipt != receipt or receipt.get("source_sha256") != ident.get("profiler_sha256"):
                    raise ValueError(f"Diagnostic helper/source differs: {attempt}")
                if len(receipt["generated_ids"]) != 4 or receipt["decode_steps_completed"] != 3:
                    raise ValueError(f"Incomplete bounded decode: {attempt}")
                operators = read_json(receipt["operators_path"])["rows"]
                backend = Counter()
                for op in operators:
                    name = op["name"]
                    if name.startswith("aten::_scaled_dot_product"):
                        shapes = op.get("input_shapes") or []
                        qshape = shapes[0] if shapes and isinstance(shapes[0], list) else []
                        qlen = qshape[-2] if len(qshape) >= 3 else None
                        backend[(name, "Q=1" if qlen == 1 else "Q>1" if isinstance(qlen, int) and qlen > 1 else "unknown")] += op["count"]
                row.update(arm=ident["arm"], cache_mode=ident["cache_mode"], document_tokens=ident["document_tokens"],
                    cuda_timing_status=receipt["cuda_timing_status"],
                    backend_counts=[{"operator": name, "query": query, "calls": calls} for (name, query), calls in sorted(backend.items())],
                    allocator={"before": receipt.get("allocator_before_capture"),
                               "after": receipt.get("allocator_after_capture"),
                               "retry_delta": receipt.get("allocator_retry_delta"),
                               "scope": receipt.get("allocator_diagnostic_scope")},
                    trace_evidence=receipt["trace_evidence"], warnings=receipt["warnings"],
                    receipt=receipt["summary_path"], sampled_increment_gib=mon["peak_incremental_gpu_bytes"] / GIB)
            rows.append(row)
    payload = {"created_utc": now(), "rows": rows, "timing_eligible": False,
               "scope": "Profiler-instrumented diagnosis; operator counts do not establish GPU time share"}
    save_json(folder / "PROFILE_SUMMARY.json", payload)
    text = ["# Reader backend 实际诊断", "", "仅用于定位算子、backend和临时分配；不把profile中的耗时或显存写入正式infra表。每项实际4K文档+64-token问题，1次prefill和3次cached decode。文件加载/H2D在trace外。", "",
            "| 方法 / 缓存 | 实现 | 状态 | CUDA kernel trace | Q>1 backend调用 | Q=1 backend调用 |", "|---|---|---|---|---|---|"]
    for row in rows:
        if row["status"] != "complete":
            text.append(f"| {row['case']} | {row['implementation']} | {row['status']} | -- | -- | -- |")
            continue
        def names(query):
            return "; ".join(f"{b['operator'].removeprefix('aten::_scaled_dot_product_')} ×{b['calls']}" for b in row["backend_counts"] if b["query"] == query) or "未记录"
        text.append(f"| {row['arm']} / {row['cache_mode']} | {row['implementation']} | complete | {row['cuda_timing_status']} | {names('Q>1')} | {names('Q=1')} |")
    text += ["", "Q=1也可能包含prefill中的单token sink/末tile，不能将这一列所有调用都称为decode；Q>1才明确属于prefill。原始input shapes、phase标记和operator记录保留。",
             "", "CUDA timeline缺失时，host侧实际backend operator仍是调度证据，但不能据CPU时间推断GPU kernel耗时占比。operator memory是带符号净分配归因；allocator前后状态与未重置的历史peak是另一口径。Profiler会改变执行开销，所有正式速度判断继续用独立COMPARISON/INFRA结果。"]
    if completed_native_from is not None:
        text += ["", f"原生Encbank复用此前完整诊断：{completed_native_from}。此前D0因监督进程心跳文件替换失败而被watchdog终止，无完整profile结果；未当作OOM。该失败保留原目录，修复Windows共享读和异常收尾后仅重试D0/B三项。只有监控实现改变，模型/reader/profiler保持原配置；不是跨版本正式速度比较。"]
    (folder / "PROFILE_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--completed-native-from", type=Path)
    args = parser.parse_args()
    print(f"Summarized {len(summarize(args.input, args.completed_native_from)['rows'])} diagnostic cells")
