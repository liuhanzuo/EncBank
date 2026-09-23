"""Read-only extraction of supervised fixed-pack reuse results, never legacy cells.

Only outputs REUSE_INFRA.md / REUSE_INFRA.csv. No jobs, model imports or GPU calls.
The queue's currently bound attempt is used; earlier attempts are never selected
for having better timings. Missing/ineligible results keep all numeric cells empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

GIB = 2 ** 30
REUSE_VERSION = "sparse-encbank-fixed-pack-reuse-v1"
GPU = "NVIDIA GeForce RTX 5090"
ORDER = ("D0", "A", "B", "D1", "FULL")
LABELS = {"D0": "D0 Encbank + LoRA（完整上层 prefix 复用）",
          "A": "A 块独立中层（全部块，hot 复用）",
          "B": "B 块独立中层 + 深层裁剪（hot 复用）",
          "D1": "D1 原中层 + 深层裁剪（中层 prefix 未复用）",
          "FULL": "Full recompute（D0 LoRA，每问全文 prefill）"}
METRICS = ("cold_storage_bytes", "additional_hot_storage_bytes", "resident_gpu_cache_tensor_bytes",
    "resident_gpu_cache_storage_bytes", "lifecycle_peak_allocated_bytes", "sampled_incremental_gpu_peak_bytes",
    "first_query_setup_inclusive_ttft_s", "later_queries_mean_ttft_s", "decode_tps",
    "cumulative_1_query_s", "cumulative_10_queries_s")
FIELDS = ("arm", "method", "status", "eligible", "reason", "document_tokens", "prompt_tokens",
          "generation_tokens", "requests_planned", "requests_observed", *METRICS, "input_sha256", "attempt")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _reject(row, status, reason):
    row.update(status=status, eligible=False, reason=reason)
    for metric in METRICS:
        row[metric] = None
    return row


def _row(job, queue, root):
    arm = "D0" if job["arm"] == "NATIVE" else job["arm"]
    row = dict.fromkeys(FIELDS)
    row.update(arm=arm, method=LABELS[arm], document_tokens=job["document_tokens"],
        prompt_tokens=job["prompt_tokens"], generation_tokens=job["generation_tokens"],
        requests_planned=queue.get("reuse_requests"), eligible=False)
    cell = queue.get("jobs", {}).get(job["id"], {})
    status = cell.get("status", "pending")
    if not cell.get("attempt"):
        return _reject(row, status, cell.get("error", "尚无当前尝试的结果"))
    attempt = Path(cell["attempt"])
    attempt = attempt if attempt.is_absolute() else root / attempt
    attempt = attempt.resolve()
    row["attempt"] = str(attempt)
    # Do not accept an external result inserted into an unrelated queue.
    if not attempt.is_relative_to(root.resolve()):
        return _reject(row, "invalid_attempt", "当前尝试不在所选队列目录内")
    result, monitor, config = (_json(attempt / name) for name in ("result.json", "monitor.json", "config.json"))
    if not all(isinstance(x, dict) for x in (result, monitor, config)):
        return _reject(row, status if status != "complete" else "incomplete_receipt", "结果、监控或配置文件尚未完整")
    if result.get("status") == "oom":
        return _reject(row, "oom", result.get("error", "原始形状 OOM；不提取部分计时"))
    if monitor.get("status") == "incremental_cap_exceeded":
        return _reject(row, "incremental_cap_exceeded", "新增显存超过预算；不提取部分计时")
    if (status != "complete" or result.get("status") != "complete"
            or result.get("timing_eligible") is not True):
        return _reject(row, status if status != "complete" else "ineligible", result.get("error", cell.get("error", "执行或计时资格未通过")))
    if (monitor.get("status") != "complete" or monitor.get("exit_code") != 0
            or monitor.get("samples", 0) <= 0 or monitor.get("interference")):
        return _reject(row, "invalid_monitor", monitor.get("error", "监控未完整成功、没有采样或有外部干扰"))
    identity = result.get("identity")
    if (not isinstance(identity, dict) or identity != config
            or (cell.get("result") and cell["result"].get("identity") != identity)):
        return _reject(row, "identity_mismatch", "结果 identity 与该尝试配置/队列记录不一致")
    if any(identity.get(k) != job.get(k) for k in ("arm", "cache_mode", "document_tokens", "prompt_tokens", "generation_tokens")):
        return _reject(row, "identity_mismatch", "尝试身份与计划方法/输入长度不同")
    reuse = identity.get("reuse", {})
    if (reuse.get("protocol") != REUSE_VERSION or reuse.get("workflow") != "fixed_ordered_pack"
            or reuse.get("requests") != queue.get("reuse_requests") or result.get("reuse") != reuse
            or identity.get("adapter_kind") != "trained" or identity.get("reader_implementation") != "reference"
            or identity.get("repetitions") != 1):
        return _reject(row, "not_matching_reuse", "不是所选协议的训练后、真实跨请求缓存复用结果")
    if (not reuse.get("quality_receipt_sha256") or not reuse.get("source_sha256")
            or monitor.get("monitor_policy_version") != identity.get("monitor_policy_version")
            or monitor.get("launcher_sha256") != identity.get("infra_launcher_sha256")):
        return _reject(row, "identity_mismatch", "质量、源码或监督器身份记录不完整/不一致")
    if (result.get("supervisor_lease") != monitor.get("worker_lease")
            or not result.get("supervisor_lease") or monitor.get("actual_worker_pid") != result.get("pid")):
        return _reject(row, "invalid_worker_identity", "结果与监控记录的实际 worker 不一致")
    hardware = result.get("hardware", {})
    baseline = monitor.get("baseline_gpu_used_bytes")
    peak = monitor.get("peak_incremental_gpu_bytes")
    if (hardware.get("gpu") != GPU or not _number(baseline) or baseline >= 5 * GIB
            or not _number(peak) or peak > 28 * GIB or monitor.get("baseline_is_fixed_before_cuda") is not True):
        return _reject(row, "invalid_resource_receipt", "5090、空闲准入或新增显存预算记录不成立")
    records, store, summary = result.get("requests", []), result.get("store", {}), result.get("summary", {})
    n = reuse["requests"]
    if len(records) != n or [r.get("query_index") for r in records] != list(range(n)):
        return _reject(row, "incomplete_requests", "不同问题的连续请求尚未完整")
    if len({r.get("prompt_sha256") for r in records}) != n or any(not r.get("prompt_sha256") for r in records):
        return _reject(row, "invalid_requests", "问题 hash 缺失或出现同一问题重复")
    for request in records:
        if (request.get("generated_tokens") != job["generation_tokens"]
                or request.get("decode_steps") != job["generation_tokens"] - 1
                or request.get("query_state_reused") is not False
                or request.get("shared_document_cache_unchanged") is not True
                or any(not _number(request.get(key), positive=True) for key in
                       ("ttft_s", "query_e2e_s", "cumulative_e2e_s", "cumulative_first_token_s"))
                or not _number(request.get("decode_wall_s"))):
            return _reject(row, "invalid_requests", "输出长度、计时或请求状态隔离记录不合格")
    if (store.get("load_count") != 1 or store.get("transfer_count") != 1
            or any(r.get("cache_file_load_s") != 0 or r.get("document_h2d_bytes") != 0 for r in records)):
        return _reject(row, "not_matching_reuse", "请求之间发生重新加载或文档传输")
    if arm == "D0" and (store.get("document_prefix_build_count") != 1
            or any(not r.get("route_stats", {}).get("prefix_cache_hit") for r in records)):
        return _reject(row, "not_matching_reuse", "D0 的合法完整 prefix 未持续命中")
    values = {"cold_storage_bytes": store.get("cold_file_bytes"),
              "additional_hot_storage_bytes": store.get("hot_file_bytes", 0),
              "resident_gpu_cache_tensor_bytes": store.get("resident_gpu_cache", {}).get("tensor_bytes"),
              "resident_gpu_cache_storage_bytes": store.get("resident_gpu_cache", {}).get("unique_storage_bytes"),
              "lifecycle_peak_allocated_bytes": result.get("lifecycle_peak_allocated_bytes"),
              "sampled_incremental_gpu_peak_bytes": peak,
              "first_query_setup_inclusive_ttft_s": records[0]["cumulative_first_token_s"],
              "later_queries_mean_ttft_s": sum(r["ttft_s"] for r in records[1:]) / (n - 1),
              "cumulative_1_query_s": records[0]["cumulative_e2e_s"],
              "cumulative_10_queries_s": summary.get("all_queries_with_store_build_s") if n == 10 else None}
    decode_s = sum(r["decode_wall_s"] for r in records)
    values["decode_tps"] = sum(r["decode_steps"] for r in records) / decode_s if decode_s > 0 else None
    mandatory = set(METRICS) - {"cumulative_10_queries_s", "decode_tps"}
    if (any(not _number(values[key]) for key in mandatory)
            or (n == 10 and not _number(values["cumulative_10_queries_s"], positive=True))
            or not _number(store.get("setup_wall_s"))
            or not _number(summary.get("all_queries_with_store_build_s"), positive=True)
            or records[0]["cumulative_first_token_s"] < store["setup_wall_s"]
            or summary["all_queries_with_store_build_s"] < records[-1]["cumulative_e2e_s"]
            or any(a["cumulative_e2e_s"] >= b["cumulative_e2e_s"] for a, b in zip(records, records[1:]))):
        return _reject(row, "invalid_accounting", "缓存成本、连续累计耗时或显存字段不完整/不一致")
    if not result.get("input", {}).get("token_sha256"):
        return _reject(row, "missing_input_identity", "缺少实际请求流 token hash")
    row.update(values, status="complete", eligible=True, reason="监控及身份核验通过",
               requests_observed=n, input_sha256=result["input"]["token_sha256"])
    return row


def collect(queue_dir):
    root = Path(queue_dir).resolve()
    plan, queue = _json(root / "plan.json"), _json(root / "status.json")
    if not isinstance(plan, list) or not isinstance(queue, dict) or queue.get("reuse_requests") not in (2, 10):
        raise ValueError("Select the new fixed-pack reuse queue with plan.json/status.json; no legacy fallback")
    if (not plan or any(job.get("arm") not in (*ORDER, "NATIVE") for job in plan)
            or len({(j.get("document_tokens"), j.get("prompt_tokens"), j.get("generation_tokens")) for j in plan}) != 1):
        raise ValueError("Summarize one common input/output shape per table")
    canonical = ["D0" if j["arm"] == "NATIVE" else j["arm"] for j in plan]
    if len(set(canonical)) != len(canonical):
        raise ValueError("D0/NATIVE are aliases; do not duplicate a method or select among attempts")
    by_arm = {("D0" if job["arm"] == "NATIVE" else job["arm"]): _row(job, queue, root) for job in plan}
    for arm in ORDER:
        if arm not in by_arm:
            row = dict.fromkeys(FIELDS)
            row.update(arm=arm, method=LABELS[arm])
            by_arm[arm] = _reject(row, "not_planned", "该方法不在当前计划中")
    rows = [by_arm[arm] for arm in ORDER]
    hashes = {r["input_sha256"] for r in rows if r["eligible"]}
    if len(hashes) > 1:
        for row in rows:
            if row["eligible"]:
                _reject(row, "cross_method_input_mismatch", "方法之间的实际请求流 token hash 不同，不能组成比较表")
    return rows


def render(queue_dir, out_dir=None):
    root, out = Path(queue_dir).resolve(), Path(out_dir or queue_dir).resolve()
    rows = collect(root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "REUSE_INFRA.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    def show(row, key, gib=False):
        value = row[key]
        return "" if value is None else f"{value / GIB if gib else value:.3f}"
    lines = ["# 训练后固定 pack 缓存复用：5090 infra", "",
        "只汇总当前队列绑定的完整、通过监控的尝试。空白表示尚无合格数据；失败/等待不是零成本。",
        "", "| 方法 | 状态 | 冷存储 GiB | 额外 hot 存储 GiB | GPU 常驻 cache GiB | 生命周期 allocated 峰值 GiB | 采样新增 GPU 峰值 GiB |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        keys = ("cold_storage_bytes", "additional_hot_storage_bytes", "resident_gpu_cache_storage_bytes",
                "lifecycle_peak_allocated_bytes", "sampled_incremental_gpu_peak_bytes")
        lines.append("| " + " | ".join([row["method"], row["status"], *[show(row, k, True) for k in keys]]) + " |")
    lines += ["", "| 方法 | 首问 TTFT 含准备 s | 后续问题平均 TTFT s | Decode token/s | 1问累计 s | 10问累计 s |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        keys = ("first_query_setup_inclusive_ttft_s", "later_queries_mean_ttft_s", "decode_tps", "cumulative_1_query_s", "cumulative_10_queries_s")
        lines.append("| " + " | ".join([row["method"], *[show(row, k) for k in keys]]) + " |")
    lines += ["", "首问 TTFT 和累计耗时包含一次缓存构建、落盘、加载、传输及 prefix 建立；1问累计取连续流首问结束，10问累计取完整流墙钟，不把准备成本重复乘10。",
        "Decode 吞吐按各问总 decode 步数 / 总 decode 时间计算，每问首 token 不计入 decode 分子。2问烟测不外推10问。",
        "", "GPU 常驻 cache 使用唯一 storage 字节，CSV另保留逻辑 tensor 字节。生命周期 allocated 峰值包含模型权重和临时张量；新增 GPU 峰值是外部监督器采样，不能视为瞬时显存上界。",
        "", "D0 复用完整上层 prefix；D1 当前尚未复用 dense 中层 prefix。Full recompute 每问全文 prefill、问内正常 KV decode，未跨请求复用 full-KV prefix；这不是最优 full-prefix serving 基线。所有方法均用各自训练后权重，FULL沿用D0权重。",
        "", "状态说明：", ""]
    lines += [f"- {row['arm']}: {row['reason']}" for row in rows]
    lines += ["", f"来源队列：`{root}`。未读取或混用 legacy 工程结果。", ""]
    (out / "REUSE_INFRA.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = render(args.queue, args.out)
    print(json.dumps({"rows": len(rows), "eligible": sum(r["eligible"] for r in rows),
                      "statuses": {r["arm"]: r["status"] for r in rows}}, ensure_ascii=False))
