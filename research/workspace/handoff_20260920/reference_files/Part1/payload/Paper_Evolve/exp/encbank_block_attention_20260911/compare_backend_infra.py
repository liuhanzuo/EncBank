"""Strict, CPU-only comparison of fresh backend runs and a matched native control.

Input layout: formal/{decode_v2,backend_v3,native}. No model/GPU imports. Profile
records never enter this report. Score roundoff is reported separately from
discrete routing/token parity; measured values remain visible on parity failure.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from compare_decode_infra import collect
from infra_protocol import GIB, now, read_json, save_json, summarize_requests


CASES = (("D0", "cold_hj"), ("B", "cold_hj"), ("B", "block_hot"))
HARDWARE = ("gpu", "total_memory_bytes", "torch", "cuda_runtime", "transformers",
            "torch_cpu_threads", "torch_interop_threads", "gpu_incremental_budget_bytes",
            "torch_allocator_cap_bytes", "non_allocator_reserve_bytes", "autocast")
VARIANT_FIELDS = {"reader_implementation", "backend_reader_sha256"}
NATIVE_FIELDS = VARIANT_FIELDS | {"arm", "optimized_reader_sha256", "native_reader_sha256"}
NATIVE_ROUTE_FIELDS = ("selected_indices", "candidate_blocks", "selected_blocks", "candidate_tokens",
    "selected_tokens", "target_retain_ratio", "actual_retain_ratio", "token_budget",
    "budget_overflow_tokens", "sink_tokens", "resume_j", "probe_indices", "original_query_start",
    "document_kv_tokens_by_layer", "document_kv_bytes", "unselected_late_kv_tokens")
BACKEND_VALIDATION_FILES = {"backend_sparse_reader.py", "validate_backend_cpu.py",
                          "test_backend_sparse_reader.py"}


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{label}: invalid negative/zero value")
    return value


def integer(value, label, *, positive=False):
    number(value, label, positive=positive)
    if not isinstance(value, int):
        raise ValueError(f"{label}: expected integer")
    return value


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_item(item, variant):
    r, mon, attempt = item["result"], item["monitor"], Path(item["attempt"])
    ident, requests = r["identity"], r["requests"]
    label = f"{variant}/{ident.get('arm')}/{ident.get('cache_mode')}"
    require(ident.get("profile_reader_only") is False and not r.get("profiler_receipt")
            and ident.get("backend_model_parity") is False and not r.get("backend_model_parity")
            and ident.get("profiler_sha256") is None and ident.get("backend_parity_sha256") is None,
            label + ": profiler/model-parity diagnostics are not formal measurements")
    require(ident.get("adapter_kind") == "strong", label + ": requires common strong adapter")
    require((ident.get("document_tokens"), ident.get("prompt_tokens"), ident.get("generation_tokens"),
             ident.get("repetitions")) == (4096, 64, 32, 3), label + ": requires 4K/Q64/G32/3 repeats")
    require(r.get("platform") == "Windows" and r.get("hostname"), label + ": missing local host")
    require(r.get("fixed_generation_ignores_eos") is True, label + ": fixed output protocol missing")
    require(read_json(attempt / "config.json") == ident, label + ": config/result identity differs")
    expected_impl = "reference" if variant == "native" else variant
    require(ident.get("reader_implementation") == expected_impl, label + ": wrong implementation")
    require(bool(ident.get("native_reader_sha256")) == (variant == "native"), label + ": native source identity")
    require(bool(ident.get("backend_reader_sha256")) == (variant == "backend_v3"), label + ": backend source identity")
    require(bool(ident.get("optimized_reader_sha256")) == (variant != "native"), label + ": decode source identity")
    validation, expected_sources = r.get("backend_gpu_validation", {}), ident.get("backend_validation_sha256")
    contract = validation.get("same_backend_contract", {})
    require(ident.get("backend_gpu_validation") is True and validation.get("device") == "cuda"
            and isinstance(validation.get("passed"), bool), label + ": guarded backend GPU receipt required")
    require(contract.get("passed") is True and validation.get("same_math_backend_bitwise_passed") is True,
            label + ": same-backend contract and bitwise algebraic checks required")
    integer(contract.get("tests_run"), label + " contract test count", positive=True)
    for key in ("failures", "errors", "skipped"):
        require(contract.get(key) == 0, label + ": same-backend contract contains " + key)
    require(isinstance(expected_sources, dict) and set(expected_sources) == BACKEND_VALIDATION_FILES,
            label + ": incomplete validated source recipe")
    sources = validation.get("source_sha256", {})
    require(isinstance(sources, dict) and sources, label + ": missing GPU-validated sources")
    for name, value in expected_sources.items():
        require(bool(value) and sources.get(name) == value, label + ": GPU source differs: " + name)
    for name, key in (("sparse_reader.py", "reader_sha256"), ("optimized_sparse_reader.py", "optimized_reader_sha256")):
        if ident.get(key):
            require(sources.get(name) == ident[key], label + ": GPU reader source differs: " + name)
    if variant == "backend_v3":
        require(sources["backend_sparse_reader.py"] == ident["backend_reader_sha256"], label + ": backend hash differs")
    if "tests_run" in validation:
        integer(validation["tests_run"], label + " GPU test count", positive=True)
    # Auto-dispatch tiny-model observations may differ between math and fused
    # backends. Preserve those failures; they are not the same-backend gate.
    for key in ("failures", "errors", "skipped"):
        integer(validation.get(key), label + " automatic tiny " + key)

    number(mon["baseline_gpu_used_bytes"], label + " baseline")
    require(mon["baseline_gpu_used_bytes"] < 5*GIB, label + ": baseline admission failed")
    number(mon["peak_incremental_gpu_bytes"], label + " sampled delta")
    require(mon["peak_incremental_gpu_bytes"] <= 28*GIB, label + ": sampled delta cap exceeded")
    if mon.get("peak_process_gpu_bytes") is not None:
        number(mon["peak_process_gpu_bytes"], label + " process peak")
        require(mon["peak_process_gpu_bytes"] <= 28*GIB, label + ": process cap exceeded")
    integer(mon["samples"], label + " monitor samples", positive=True)
    require(mon.get("cap_bytes") == 28*GIB and mon.get("baseline_is_fixed_before_cuda") is True,
            label + ": monitor cap/baseline recipe differs")
    hardware = r["hardware"]
    require(hardware.get("gpu_incremental_budget_bytes") == 28*GIB
            and hardware.get("torch_allocator_cap_bytes") == 27.5*GIB
            and hardware.get("non_allocator_reserve_bytes") == .5*GIB, label + ": allocator policy differs")
    require(len(requests) == 3, label + ": missing requests")
    for index, request in enumerate(requests):
        require(request.get("repetition") == index, label + ": repetition ordering differs")
        require(request.get("generated_tokens") == 32 and len(request["generated_ids"]) == 32
                and request["decode_steps"] == 31, label + ": output/decode count mismatch")
        for token in request["generated_ids"]:
            integer(token, label + " generated token")
        require(read_json(attempt / f"request_{index:03d}.json") == request, label + ": request artifact differs")
        for key in ("ttft_s", "decode_wall_s", "query_e2e_s"):
            number(request[key], label + " " + key, positive=True)
        for key in ("file_load_s", "h2d_wall_s", "prefill_wall_s"):
            number(request[key], label + " " + key)
        for key in ("decode_step_wall_s", "decode_step_cuda_event_s"):
            require(len(request[key]) == 31, label + ": incomplete per-token trace")
            for value in request[key]:
                number(value, label + " " + key, positive=True)
        require(math.isclose(request["query_e2e_s"], request["ttft_s"] + request["decode_wall_s"],
                             rel_tol=1e-9, abs_tol=1e-9), label + ": E2E boundary differs")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "h2d_bytes"):
            integer(request[key], label + " " + key)
        require(request["peak_allocated_bytes"] <= request["peak_reserved_bytes"] <= 27.5*GIB,
                label + ": invalid/exceeded allocator counters")
        route = request["route_stats"]
        for key in NATIVE_ROUTE_FIELDS:
            require(key in route, label + ": missing route field " + key)
        require(route["candidate_tokens"] == 4096 and route["original_query_start"] == 4096 + route["sink_tokens"],
                label + ": document/original position recipe differs")
        positions = request.get("state_positions", {})
        require(set(positions) == {"query_position", "pack_position"}, label + ": missing final positions")
        for key, value in positions.items():
            integer(value, label + " " + key)
        # Position mismatches are retained as parity failures below, not repaired.
        scores = route.get("block_scores", [])
        if scores:
            require(len(scores) == route["candidate_blocks"], label + ": incomplete block scores")
            for score in scores:
                number(score, label + " probe score")

    metrics = summarize_requests(requests)
    for key, value in metrics.items():
        require(r["summary"].get(key) == value, label + ": summary does not match requests: " + key)
    store = r["store"]
    for key in ("cold_tensor_bytes", "cold_file_bytes", "hot_tensor_bytes", "hot_file_bytes", "request_h2d_bytes"):
        integer(store[key], label + " " + key)
    require(store["cold_tensor_bytes"] > 0 and store["cold_file_bytes"] > 0, label + ": missing cold cache")
    require(all(q["h2d_bytes"] == store["request_h2d_bytes"] for q in requests), label + ": H2D counters differ")
    location = Path(store["persistent_location"])
    require(location.resolve() == (attempt / "store").resolve(), label + ": cache belongs to another attempt")
    for prefix in ("cold", "hot"):
        size = store[prefix + "_file_bytes"]
        if size:
            file = location / (prefix + ".pt")
            require(file.is_file() and file.stat().st_size == size, label + ": cache file byte receipt differs")
    hot = ident["cache_mode"] == "block_hot"
    require((store["hot_file_bytes"] > 0) == hot and (store["hot_tensor_bytes"] > 0) == hot,
            label + ": hot storage mode differs")
    item["validated_metrics"] = metrics
    return item


def matching_environment(a, b, *, native=False):
    ar, br = a["result"], b["result"]
    excluded = NATIVE_FIELDS if native else VARIANT_FIELDS
    require({k:v for k,v in ar["identity"].items() if k not in excluded}
            == {k:v for k,v in br["identity"].items() if k not in excluded}, "Mismatched model/adapter/source/recipe")
    require(ar["input"] == br["input"] and ar["hostname"] == br["hostname"], "Mismatched input or measured host")
    for key in HARDWARE:
        require(ar["hardware"].get(key) is not None and ar["hardware"].get(key) == br["hardware"].get(key),
                "Mismatched runtime/hardware: " + key)
    for key in ("uuid", "driver_version"):
        av, bv = a["monitor"]["baseline_snapshot"].get(key), b["monitor"]["baseline_snapshot"].get(key)
        require(av is not None and av == bv, "Mismatched physical GPU/driver: " + key)
    require(ar["backend_gpu_validation"]["source_sha256"] == br["backend_gpu_validation"]["source_sha256"],
            "Mismatched GPU-validated source set")


def paired_checks(a, b, case, *, native=False):
    checks = []
    for qa, qb in zip(a["result"]["requests"], b["result"]["requests"]):
        ra, rb = qa["route_stats"], qb["route_stats"]
        route_keys = set(NATIVE_ROUTE_FIELDS) if native else (set(ra) | set(rb)) - {"block_scores"}
        different = sorted(key for key in route_keys if ra.get(key) != rb.get(key))
        sa, sb = ra.get("block_scores", []), rb.get("block_scores", [])
        score_delta = max((abs(x-y) for x,y in zip(sa,sb)), default=0.) if sa and sb and len(sa)==len(sb) else None
        expected_query = 64 + 31
        positions_ok = (qa["state_positions"] == qb["state_positions"]
            and qa["state_positions"] == {"query_position":expected_query,
                "pack_position":ra["original_query_start"]+expected_query})
        checks.append(dict(case=case, repetition=qa["repetition"],
            tokens_equal=qa["generated_ids"] == qb["generated_ids"],
            discrete_route_equal=not different, route_differences=different,
            positions_equal_and_expected=positions_ok, block_score_max_abs=score_delta))
    return checks


def checks_passed(checks):
    return bool(checks) and all(c["tokens_equal"] and c["discrete_route_equal"] and c["positions_equal_and_expected"] for c in checks)


def row_for(item, variant):
    r, mon, m = item["result"], item["monitor"], item["validated_metrics"]
    ident, store = r["identity"], r["store"]
    return dict(implementation=variant, arm=ident["arm"], cache_mode=ident["cache_mode"], attempt=item["attempt"],
        document_tokens=4096, prompt_tokens=64, generation_tokens=32, repetitions=3,
        cold_file_bytes=store["cold_file_bytes"], extra_hot_file_bytes=store["hot_file_bytes"],
        total_cache_file_bytes=store["cold_file_bytes"]+store["hot_file_bytes"],
        cold_tensor_bytes=store["cold_tensor_bytes"], extra_hot_tensor_bytes=store["hot_tensor_bytes"],
        request_h2d_bytes=store["request_h2d_bytes"],
        tiny_auto_passed=r["backend_gpu_validation"]["passed"],
        tiny_auto_failures=r["backend_gpu_validation"]["failures"],
        tiny_auto_errors=r["backend_gpu_validation"]["errors"],
        tiny_auto_skipped=r["backend_gpu_validation"]["skipped"],
        tiny_same_backend_contract_passed=r["backend_gpu_validation"]["same_backend_contract"]["passed"],
        tiny_same_math_bitwise_passed=r["backend_gpu_validation"]["same_math_backend_bitwise_passed"],
        ttft_s=m["ttft_s"], decode_tokens_per_s=m["decode_tokens_per_s"], query_e2e_s=m["query_e2e_s"],
        ttft_min_max_s=[min(q["ttft_s"] for q in r["requests"]),max(q["ttft_s"] for q in r["requests"])],
        decode_tps_min_max=[min(q["decode_steps"]/q["decode_wall_s"] for q in r["requests"]),
                            max(q["decode_steps"]/q["decode_wall_s"] for q in r["requests"])],
        request_peak_allocated_bytes=m["request_peak_allocated_bytes"],
        request_peak_reserved_bytes=m["request_peak_reserved_bytes"],
        sampled_increment_gpu_bytes=mon["peak_incremental_gpu_bytes"], sampled_process_gpu_bytes=mon.get("peak_process_gpu_bytes"))


def compare(formal_root, destination):
    formal_root, destination = Path(formal_root), Path(destination)
    groups = {}
    for variant in ("decode_v2", "backend_v3", "native"):
        items = collect(formal_root / variant)
        expected = {("NATIVE", "cold_hj")} if variant == "native" else set(CASES)
        found = {}
        for item in items.values():
            validate_item(item, variant)
            key = (item["result"]["identity"]["arm"], item["result"]["identity"]["cache_mode"])
            require(key not in found, "Duplicate arm/cache cell")
            found[key] = item
        require(set(found) == expected, "Missing/unexpected arm/cache cells: " + variant)
        groups[variant] = found
    reference, backend = groups["decode_v2"], groups["backend_v3"]
    native = groups["native"][("NATIVE", "cold_hj")]
    anchor = reference[("D0", "cold_hj")]
    checks, native_checks, rows, ratios = [], [], [row_for(native,"native")], []
    # Require one runtime/source recipe across every cell. arm/cache may differ
    # across methods; pair-specific equality below still checks them strictly.
    for variant, cells in groups.items():
        for item in cells.values():
            if variant != "native":
                left = anchor["result"]["identity"].copy()
                right = item["result"]["identity"].copy()
                for key in VARIANT_FIELDS | {"arm", "cache_mode"}:
                    left.pop(key, None); right.pop(key, None)
                require(left == right, "Cross-cell recipe/source mismatch")
            # Environment checks against a same-arm/cache view avoid excusing
            # real pair mismatches while allowing the planned matrix structure.
            shadow = dict(item, result=dict(item["result"], identity=dict(item["result"]["identity"],
                arm="D0", cache_mode="cold_hj")))
            matching_environment(anchor, shadow, native=(variant == "native"))
    for case in CASES:
        a,b = reference[case],backend[case]
        matching_environment(a,b)
        name = "/".join(case)
        current = paired_checks(a,b,name)
        checks.extend(current)
        ra,rb = row_for(a,"decode_v2"),row_for(b,"backend_v3")
        rows.extend([ra,rb])
        ratios.append(dict(case=name, equivalent_on_checked_requests=checks_passed(current),
            ttft_speedup=ra["ttft_s"]/rb["ttft_s"], decode_throughput_ratio=rb["decode_tokens_per_s"]/ra["decode_tokens_per_s"],
            e2e_speedup=ra["query_e2e_s"]/rb["query_e2e_s"],
            allocated_delta_bytes=rb["request_peak_allocated_bytes"]-ra["request_peak_allocated_bytes"],
            reserved_delta_bytes=rb["request_peak_reserved_bytes"]-ra["request_peak_reserved_bytes"]))
    for variant in ("decode_v2","backend_v3"):
        d0 = groups[variant][("D0","cold_hj")]
        matching_environment(d0,native,native=True)
        native_checks.extend(paired_checks(d0,native,variant+"/D0_vs_native",native=True))
    backend_pass, native_pass = checks_passed(checks), checks_passed(native_checks)
    payload = dict(created_utc=now(), parity_passed=backend_pass and native_pass,
        backend_parity_passed=backend_pass, native_control_parity_passed=native_pass,
        checks=checks, native_control_checks=native_checks, rows=rows, backend_ratios=ratios,
        scope="4K synthetic documents, Q64/G32; 3 requests per process, not independent-run uncertainty or QA accuracy",
        ratio_scope="Numeric observed ratios remain visible if parity fails; they are not claims of equivalent speed then",
        validation_policy="Same-MATH contract and bitwise checks are required. Tiny auto-dispatch pass/failure remains an observation; real 32-token output/route/position parity is checked independently.",
        memory_scope="Includes model; allocator request peaks and sampled GPU delta are distinct. WDDM N/A stays null.")
    save_json(destination / "COMPARISON.json",payload)
    lines=["# GQA 后端对照：5090 非 profiler 测量", "",
        f"Backend 配对：**{'通过' if backend_pass else '存在差异'}**（9 对）；D0 / native 控制：**{'通过' if native_pass else '存在差异'}**（6 对）。",
        "", "同一强 Encbank LoRA、4K 文档、64-token prompt、固定32-token输出（31次cached decode），每单元同进程重复3次。TTFT/E2E为均值，decode TPS为总步数/总decode时间。",
        "", "所有单元要求同MATH contract和逐位代数检查通过；tiny自动后端观察的失败仍原样列出，不写成全验证通过。真实32-token输出/route/位置另行严格配对。",
        "", "| 方法 / 缓存 | 实现 | Cold + extra hot MiB | TTFT s | Decode tok/s | E2E s | Alloc / reserved GiB | 采样增量 GiB | Tiny auto观察 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        observed = "通过" if row["tiny_auto_passed"] else f"未通过(f={row['tiny_auto_failures']},e={row['tiny_auto_errors']})"
        lines.append(f"| {row['arm']} / {row['cache_mode']} | {row['implementation']} | {row['cold_file_bytes']/2**20:.3f} + {row['extra_hot_file_bytes']/2**20:.3f} | {row['ttft_s']:.3f} | {row['decode_tokens_per_s']:.2f} | {row['query_e2e_s']:.3f} | {row['request_peak_allocated_bytes']/GIB:.3f} / {row['request_peak_reserved_bytes']/GIB:.3f} | {row['sampled_increment_gpu_bytes']/GIB:.3f} | {observed} |")
    lines += ["", "| 同方法 decode_v2 → backend_v3 | TTFT 比值 | Decode TPS 比值 | E2E 比值 | 离散输出/route/位置 |",
        "|---|---:|---:|---:|---|"]
    for ratio in ratios:
        lines.append(f"| {ratio['case']} | {ratio['ttft_speedup']:.3f}× | {ratio['decode_throughput_ratio']:.3f}× | {ratio['e2e_speedup']:.3f}× | {'通过' if ratio['equivalent_on_checked_requests'] else '存在差异'} |")
    lines += ["", "比值大于1表示本次观测改善。若输出/route/位置不同，保留测量但不能宣布等价加速。Probe浮点score的max_abs逐对记录在JSON，未要求浮点分数逐位相同。",
        "", "缓存列为实际持久化文件字节，extra hot需加到cold；JSON另列tensor与请求H2D字节。临时KV展开的影响必须看allocated/reserved和采样显存，不能只看persistent KV。",
        "", "Native为本轮匹配的独立参考；B与native attention图不同，不能从合成输出或速度表推断质量等价。三次同进程重复不提供独立运行置信区间。Profiler与8B额外parity运行已排除在此正式表之外。",
        "", "所有单元沿用锁前/锁内<5GiB准入、27.5GiB allocator限额、28GiB采样增量/可得进程显存监控。WDDM采样不是瞬时显存硬上界。"]
    (destination / "COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.input,args.out)
    print(f"Compared 9 backend and 6 native control pairs; parity={result['parity_passed']}")
    raise SystemExit(0 if result["parity_passed"] else 1)
