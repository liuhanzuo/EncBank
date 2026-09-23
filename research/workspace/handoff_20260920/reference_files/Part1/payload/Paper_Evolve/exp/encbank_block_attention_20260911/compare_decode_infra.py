"""Compare two fresh matched local infra runs; no GPU or model imports."""
from __future__ import annotations

import argparse
from pathlib import Path
from infra_protocol import GIB, GPU_NAME, read_json, save_json, now


def collect(folder):
    status = read_json(folder / "status.json", {})
    output = {}
    for name, job in status.get("jobs", {}).items():
        if job.get("status") != "complete":
            raise ValueError(f"{folder}/{name}: {job.get('status')}")
        attempt = Path(job["attempt"])
        result = read_json(attempt / "result.json", {})
        monitor = read_json(attempt / "monitor.json", {})
        if (result.get("status") != "complete" or not result.get("timing_eligible")
                or monitor.get("status") != "complete" or monitor.get("exit_code") != 0
                or monitor.get("interference") or not monitor.get("samples")):
            raise ValueError(f"{attempt}: incomplete or invalid measurement")
        if monitor["peak_incremental_gpu_bytes"] > 28 * GIB:
            raise ValueError(f"{attempt}: sampled 28 GiB budget exceeded")
        if not monitor["baseline_gpu_used_bytes"] < 5 * GIB:
            raise ValueError(f"{attempt}: invalid original admission")
        requests = result["requests"]
        ident = result["identity"]
        if (monitor.get("launcher_sha256") != ident.get("infra_launcher_sha256")
                or monitor.get("monitor_policy_version") != ident.get("monitor_policy_version")
                or not result.get("supervisor_lease")
                or monitor.get("worker_lease") != result.get("supervisor_lease")
                or monitor.get("actual_worker_pid") != result.get("pid")):
            raise ValueError(f"{attempt}: monitor does not belong to this measured worker/source")
        if result.get("hardware", {}).get("gpu") != GPU_NAME or monitor.get("baseline_snapshot", {}).get("name") != GPU_NAME:
            raise ValueError(f"{attempt}: not the authorized RTX5090")
        if ident.get("optimized_gpu_validation") and not result.get("optimized_gpu_validation", {}).get("passed"):
            raise ValueError(f"{attempt}: missing passed GPU equivalence receipt")
        if len(requests) != ident["repetitions"]:
            raise ValueError(f"{attempt}: missing requests")
        for request in requests:
            if (len(request["generated_ids"]) != ident["generation_tokens"]
                    or request["decode_steps"] != ident["generation_tokens"] - 1):
                raise ValueError(f"{attempt}: incomplete generated sequence")
        output[name] = {"result": result, "monitor": monitor, "attempt": str(attempt)}
    if not output or status.get("status") != "complete":
        raise ValueError(f"{folder}: run is not complete")
    return output


def compare(reference, optimized, destination):
    left, right = collect(reference), collect(optimized)
    if left.keys() != right.keys():
        raise ValueError("The measured arm/cache/shape sets differ")
    checks, rows = [], []
    for name in left:
        a, b = left[name], right[name]
        ai, bi = a["result"]["identity"].copy(), b["result"]["identity"].copy()
        if ai.pop("reader_implementation") != "reference" or bi.pop("reader_implementation") != "decode_v2":
            raise ValueError("Unexpected implementation labels")
        if ai.pop("optimized_reader_sha256") is not None or not bi.pop("optimized_reader_sha256"):
            raise ValueError("Missing or unexpected optimized source identity")
        if ai != bi or a["result"]["input"] != b["result"]["input"]:
            raise ValueError(f"{name}: input, adapter, source or run settings differ")
        if not a["result"].get("hostname") or a["result"]["hostname"] != b["result"].get("hostname"):
            raise ValueError(f"{name}: measured hosts differ")
        if ai.get("optimized_gpu_validation") and a["result"]["optimized_gpu_validation"].get("source_sha256") != b["result"]["optimized_gpu_validation"].get("source_sha256"):
            raise ValueError(f"{name}: GPU-validated sources differ")
        for key in ("gpu", "total_memory_bytes", "torch", "cuda_runtime", "torch_cpu_threads",
                    "torch_interop_threads", "gpu_incremental_budget_bytes", "torch_allocator_cap_bytes",
                    "non_allocator_reserve_bytes", "autocast", "transformers"):
            av, bv = a["result"]["hardware"].get(key), b["result"]["hardware"].get(key)
            if av is None or av != bv:
                raise ValueError(f"{name}: hardware/runtime mismatch: {key}")
        for key in ("uuid", "driver_version"):
            av, bv = a["monitor"]["baseline_snapshot"].get(key), b["monitor"]["baseline_snapshot"].get(key)
            if av is None or av != bv:
                raise ValueError(f"{name}: physical GPU/driver mismatch: {key}")
        for ra, rb in zip(a["result"]["requests"], b["result"]["requests"]):
            tokens_equal = ra["generated_ids"] == rb["generated_ids"]
            route_equal = ra["route_stats"] == rb["route_stats"]
            checks.append({"case": name, "repetition": ra["repetition"],
                           "tokens_equal": tokens_equal, "route_equal": route_equal})
        for label, item in (("reference", a), ("decode_v2", b)):
            result, mon = item["result"], item["monitor"]
            metrics = result["summary"]
            rows.append({"case": name, "implementation": label, "attempt": item["attempt"],
                "arm": ai["arm"], "cache_mode": ai["cache_mode"], "document_tokens": ai["document_tokens"],
                "prompt_tokens": ai["prompt_tokens"], "generation_tokens": ai["generation_tokens"],
                "repetitions": ai["repetitions"], "ttft_s": metrics["ttft_s"],
                "decode_tokens_per_s": metrics["decode_tokens_per_s"], "query_e2e_s": metrics["query_e2e_s"],
                "ttft_min_max_s": [min(r["ttft_s"] for r in result["requests"]), max(r["ttft_s"] for r in result["requests"])],
                "decode_tps_min_max": [min(r["decode_steps"] / r["decode_wall_s"] for r in result["requests"]),
                                       max(r["decode_steps"] / r["decode_wall_s"] for r in result["requests"])],
                "request_peak_allocated_gib": metrics["request_peak_allocated_bytes"] / GIB,
                "request_peak_reserved_gib": metrics["request_peak_reserved_bytes"] / GIB,
                "sampled_increment_gib": mon["peak_incremental_gpu_bytes"] / GIB})
    passed = all(c["tokens_equal"] and c["route_equal"] for c in checks)
    payload = {"created_utc": now(), "parity_passed": passed, "checks": checks, "rows": rows,
        "scope": "Same synthetic input; repeated requests in one process per cell, no accuracy conclusion or independent-run uncertainty estimate"}
    save_json(destination / "COMPARISON.json", payload)
    lines = ["# 单 token decode 优化：5090 有界对照", "",
        f"生成 token 和完整 route 记录对齐：**{'全部通过' if passed else '存在差异，不能据此宣布等价加速'}**，共{len(checks)}对。",
        "", f"同一强 Encbank LoRA、同一输入/缓存/注意力图；问题token数{sorted(set(r['prompt_tokens'] for r in rows))}，固定输出token数{sorted(set(r['generation_tokens'] for r in rows))}，每单元同进程重复次数{sorted(set(r['repetitions'] for r in rows))}。TTFT/E2E取均值，decode TPS为总decode步数/总decode时间。不是QA准确率，也不是独立进程重复的置信区间。",
        "", "| 方法 / 缓存 | 实现 | TTFT s | Decode tok/s | E2E s | Alloc / reserved GiB | 采样增量 GiB |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['arm']} / {row['cache_mode']} / {row['document_tokens']} | {row['implementation']} | {row['ttft_s']:.3f} | {row['decode_tokens_per_s']:.2f} | {row['query_e2e_s']:.3f} | {row['request_peak_allocated_gib']:.2f} / {row['request_peak_reserved_gib']:.2f} | {row['sampled_increment_gib']:.2f} |")
    lines += ["", "只有decode算子组织改变，prefill及热缓存构建继承原实现。TTFT的波动不能归因于这次decode优化。D0是sparse wrapper实现对照，不能直接当作原生Encbank。此前原生Encbank测量另存，不与本次新增重复数混成同一组。",
        "", "沿用锁前/锁内严格<5GiB准入、27.5GiB allocator限额和28GiB设备采样增量监控。显存包含权重、排除既有桌面基线；WDDM采样峰值不能保证捕获瞬时峰值。原24单元工程矩阵完整保留。",
        "", "每次请求的最小/最大计时和完整配对检查见COMPARISON.json。"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.optimized, args.out)
    print(f"Compared {len(result['checks'])} request pairs; parity={result['parity_passed']}")
    raise SystemExit(0 if result["parity_passed"] else 1)
