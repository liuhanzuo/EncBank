"""Summarize one saved shared-prefill numerical diagnosis; no Torch/GPU imports.

This report never enters INFRA_REPORT and never upgrades the older numerical
failure. A completed diagnostic and a local oracle are not model equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PREVIOUS = HERE / "results/infra_backend_v3/diagnostic/n4096_q64_g4_D0_cold_hj/attempts/0001/result.json"
RECIPE = ("model", "model_config_sha256", "adapter", "adapter_sha256", "reader_sha256",
          "optimized_reader_sha256", "backend_reader_sha256", "reader_implementation",
          "adapter_kind", "arm", "cache_mode", "document_tokens", "prompt_tokens",
          "generation_tokens", "chunk_size", "j", "m", "rho", "rank", "alpha", "seed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locate(source):
    source = Path(source).resolve()
    if source.is_file():
        return source.parent, None
    if (source / "result.json").is_file():
        return source, None
    state = read(source / "status.json")
    jobs = state.get("jobs", {})
    if len(jobs) != 1:
        raise ValueError("Expected one bounded diagnostic, not a formal matrix")
    return Path(next(iter(jobs.values()))["attempt"]), state


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fmt(value):
    return f"{value:.7g}" if finite(value) else "不可得"


def backend_evidence(branch):
    evidence = branch.get("backend_evidence", {})
    counts = Counter()
    for item in evidence.get("operator_events", []):
        counts[item["name"]] += item["count"]
    return {"operators": dict(counts), "cuda_kernel_timeline_observed":
            evidence.get("cuda_kernel_timeline_observed", False),
            "cuda_kernel_count": evidence.get("cuda_kernel_count"),
            "cuda_kernel_names": evidence.get("cuda_kernel_names", []),
            "evidence_scope": evidence.get("evidence_scope"), "errors": evidence.get("errors", [])}


def brief(row):
    return {key: row.get(key) for key in ("step", "layer", "point", "numerical")} if row else None


def oracle_receipt_comparison(embedded):
    if not embedded.get("receipt_path"):
        return {"exact_equal": False, "semantic_equal": False, "differences": {}, "torch_version_repr_only": False}
    saved = read(embedded["receipt_path"])
    differences = {k: {"embedded": embedded.get(k), "saved": saved.get(k)}
        for k in set(embedded) | set(saved) if embedded.get(k) != saved.get(k)}
    # TorchVersion subclasses str; the oracle's _safe(type(...) is str) can
    # serialize its repr while the outer JSON encoder writes its string value.
    # Accept only that exact single-field representation, retaining the evidence.
    representation = (set(differences) == {"torch_version"}
        and type(embedded.get("torch_version")) is str
        and saved.get("torch_version") == repr(embedded["torch_version"]))
    return {"exact_equal": not differences, "semantic_equal": not differences or representation,
            "differences": differences, "torch_version_repr_only": representation}


def summarize(source, destination=None, previous=DEFAULT_PREVIOUS):
    attempt, outer = locate(source)
    destination = Path(destination) if destination is not None else Path(source)
    if destination.is_file():
        destination = destination.parent
    result, monitor = read(attempt / "result.json"), read(attempt / "monitor.json")
    ident, shared = result["identity"], result.get("shared_state_receipt", {})
    old_result = read(previous)
    old = old_result.get("backend_model_parity", {})
    check = {}
    check["diagnostic_only"] = (ident.get("shared_state_diagnostic_only") is True
        and ident.get("profile_reader_only") is False and ident.get("backend_model_parity") is False
        and result.get("timing_eligible") is False and result.get("requests") == []
        and result.get("summary") == {})
    check["fixed_shape"] = (ident.get("arm"), ident.get("cache_mode"), ident.get("document_tokens"),
        ident.get("prompt_tokens"), ident.get("generation_tokens"), ident.get("repetitions")) == (
            "D0", "cold_hj", 4096, 64, 4, 1)
    check["result_complete"] = result.get("status") == "complete"
    check["outer_complete"] = outer is None or outer.get("status") == "complete"
    check["shared_protocol_complete"] = (shared.get("status") == "complete"
        and shared.get("protocol_checks_passed") is True and shared.get("formal_timing_eligible") is False
        and shared.get("prefill_count") == 1 and shared.get("decode_steps") == 3)
    sources = ident.get("shared_state_sources_sha256", {}) or {}
    validation = result.get("shared_state_gpu_validation", {})
    check["shared_helper_source_bound"] = bool(sources) and shared.get("source_sha256") == sources.get("shared_state_diagnostic.py")
    check["gpu_helper_validation_bound"] = (validation.get("passed") is True
        and validation.get("device") == "cuda" and validation.get("source_sha256") == sources)
    check["saved_shared_receipt_equal"] = bool(shared.get("receipt_path")) and read(shared["receipt_path"]) == shared
    check["monitor_complete"] = monitor.get("status") == "complete" and monitor.get("exit_code") == 0
    check["monitor_worker_bound"] = (bool(result.get("supervisor_lease"))
        and monitor.get("worker_lease") == result.get("supervisor_lease")
        and monitor.get("actual_worker_pid") == result.get("pid")
        and monitor.get("launcher_sha256") == ident.get("infra_launcher_sha256")
        and monitor.get("monitor_policy_version") == ident.get("monitor_policy_version"))
    check["local_5090"] = (result.get("platform") == "Windows"
        and result.get("hardware", {}).get("gpu") == "NVIDIA GeForce RTX 5090"
        and monitor.get("baseline_snapshot", {}).get("name") == "NVIDIA GeForce RTX 5090")
    baseline, peak = monitor.get("baseline_gpu_used_bytes"), monitor.get("peak_incremental_gpu_bytes")
    check["monitor_admission_and_budget"] = (finite(baseline) and 0 <= baseline < 5*2**30
        and finite(peak) and 0 <= peak <= 28*2**30 and bool(monitor.get("samples"))
        and not monitor.get("interference"))
    check["previous_reader_model_recipe_equal"] = all(ident.get(k) == old_result["identity"].get(k) for k in RECIPE)
    check["previous_input_equal"] = result.get("input") == old_result.get("input")
    check["previous_remains_failed"] = old_result.get("status") == "failed" and old.get("passed") is False
    rows = shared.get("steps", [])
    check["three_decode_steps"] = [r.get("step") for r in rows] == [1, 2, 3]
    check["centered_metrics_present"] = all(finite(r.get("numerical", {}).get(k))
        for r in rows for k in ("max_abs", "rms", "centered_rms", "signed_mean_difference")) and len(rows) == 3
    capture = shared.get("same_input_oracle", {})
    oracle = capture.get("oracle", {})
    check["oracle_local_explanation_eligible"] = oracle.get("oracle_can_explain_captured_call") is True
    check["oracle_actual_capture_matches"] = oracle.get("captured_vs_repeat_auto_match") is True
    oracle_comparison = oracle_receipt_comparison(oracle)
    check["saved_oracle_receipt_semantics_equal"] = oracle_comparison["semantic_equal"]
    branches = {name: {"status": value.get("status"), "versus_fp64": value.get("versus_fp64"),
        "versus_fp64_rounded_bf16": value.get("versus_fp64_rounded_bf16"),
        "actual_backend": backend_evidence(value), "error": value.get("error")}
        for name, value in oracle.get("branches", {}).items()}
    check["three_oracle_branches"] = set(branches) == {"gqa_math", "repeat_math", "repeat_automatic"}
    observations = shared.get("layer_observations", [])
    ranked = sorted((r for r in observations if finite(r.get("numerical", {}).get("max_abs"))),
                    key=lambda r: r["numerical"]["max_abs"], reverse=True)
    hidden_rows = [brief(r) for r in observations if r.get("point") == "hidden_output" and r.get("layer") in (0, 11, 15, 35)]
    old_steps = [{"phase": s["phase"], "max_abs": s["numerical"].get("max_abs"),
        "rms": s["numerical"].get("rms"), "centered_rms": s["numerical"].get("centered_rms"),
        "greedy_equal": s["numerical"].get("greedy_equal"), "passed": s.get("passed")}
        for s in old.get("stages", [])]
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
        "report_checks_passed": all(check.values()), "checks": check,
        "numerical_equivalence_asserted": False, "formal_timing_eligible": False,
        "accuracy_claim": False, "current": {"attempt": str(attempt), "status": result.get("status"),
            "shared_status": shared.get("status"), "structure": shared.get("structure"),
            "protocol_checks_passed": shared.get("protocol_checks_passed"), "steps": rows,
            "first_difference": shared.get("first_difference"), "largest_difference": shared.get("largest_difference"),
            "largest_difference_by_point": shared.get("largest_difference_by_point"),
            "five_largest_layer_points": [brief(r) for r in ranked[:5]],
            "hidden_output_layer_trace": hidden_rows,
            "prefill_greedy_id": shared.get("prefill_greedy_id"), "errors": shared.get("errors", [])},
        "same_qkv_oracle": {"step": capture.get("step"), "layer": capture.get("layer"),
            "capture_checks": {k: capture.get(k) for k in ("normal_attention_vs_first_pass", "query_q_vs_first_pass", "new_query_k_vs_first_pass")},
            "candidate_replay": shared.get("oracle_replay"), "status": oracle.get("status"),
            "oracle_can_explain_captured_call": oracle.get("oracle_can_explain_captured_call"),
            "captured_vs_repeat_auto_match": oracle.get("captured_vs_repeat_auto_match"),
            "captured_vs_repeat_auto_summary_differences": oracle.get("captured_vs_repeat_auto_summary_differences"),
            "input_shape": oracle.get("input_shape"), "captured_flags": oracle.get("captured_flags"),
            "effective_input_dtypes": oracle.get("effective_input_dtypes"),
            "estimated_temporary_bytes": oracle.get("estimated_temporary_bytes"),
            "same_math_bitwise_equal": oracle.get("same_math_bitwise_equal"),
            "oracle_finite": oracle.get("oracle_finite"), "all_branch_outputs_finite": oracle.get("all_branch_outputs_finite"),
            "branches": branches, "receipt_path": oracle.get("receipt_path")},
        "oracle_saved_receipt_comparison": oracle_comparison,
        "previous_separate_prefills": {"source": str(Path(previous).resolve()), "receipt_path": old.get("receipt_path"),
            "passed": old.get("passed"), "thresholds_unchanged": old.get("thresholds"), "steps": old_steps,
            "scope": "Two independently computed prefills; raw errors are not interchangeable with the shared-prefill experiment. Missing centered RMS is unavailable."},
        "monitor": {k: monitor.get(k) for k in ("status", "exit_code", "samples", "actual_worker_pid",
            "baseline_gpu_used_bytes", "peak_incremental_gpu_bytes", "peak_process_gpu_bytes", "maximum_observed_sample_gap_s", "interference")},
        "limits": ["One D0 cold 4K diagnostic, not benchmark accuracy or formal speed/memory.",
            "Shared-prefill decode isolates a starting-state variable; it does not repair the independent-prefill path.",
            "Local same-QKV FP64 attention oracle does not recompute projections, RoPE, all layers or the model in FP64.",
            "Centered RMS removes the mean logit difference for explanation only; old acceptance thresholds are unchanged.",
            "All diagnostic allocation and profiler overhead are excluded from formal infrastructure comparisons."],
        "evidence": {"result": str(attempt/"result.json"), "monitor": str(attempt/"monitor.json"),
            "shared_receipt": shared.get("receipt_path"), "previous_result": str(Path(previous).resolve())}}
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"SHARED_STATE_DIAGNOSTIC.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    valid = "完整" if payload["report_checks_passed"] else "存在缺项或失败"
    lines = ["# 共享 reference-prefill 状态：单项数值诊断", "",
        f"诊断证据检查：**{valid}**。本报告不宣称模型数值等价，不更改此前失败，也不生成正式速度/显存或准确率结论。", "",
        "同一 strong CoMem LoRA、D0/cold、4K文档、64-token prompt；仅计算一次 reference prefill，再从共享的初始缓存分别追加 query KV，candidate 消费 reference token。", "",
        f"共享缓存结构检查：{shared.get('protocol_checks_passed')}；monitor：{monitor.get('status')} / exit {monitor.get('exit_code')}。", "",
        "| 共享起点 decode | Max abs | RMS | Signed mean | Centered RMS | Reference / candidate ID |",
        "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        n=r.get("numerical", {})
        lines.append(f"| {r['step']} | {fmt(n.get('max_abs'))} | {fmt(n.get('rms'))} | {fmt(n.get('signed_mean_difference'))} | {fmt(n.get('centered_rms'))} | {r.get('reference_greedy_id')} / {r.get('candidate_greedy_id')} |")
    for label, key in (("首个差异", "first_difference"), ("最大差异", "largest_difference")):
        r=shared.get(key)
        lines += ["", f"{label}：step {r['step']}，第 {r['layer']} 层（从0计），`{r['point']}`，max_abs={fmt(r['numerical'].get('max_abs'))}。" if r else f"{label}：未记录。"]
    if hidden_rows:
        lines += ["", "后续层的 hidden-output RMS 差异如下（层从0计；误差不保证随层数或步数单调）：", "",
            "| Decode step | Layer 0 | Layer 11 | Layer 15 | Layer 35 | Layer 35 相对RMS |", "|---|---:|---:|---:|---:|---:|"]
        for step in (1, 2, 3):
            values = {r["layer"]: r["numerical"] for r in hidden_rows if r["step"] == step}
            lines.append(f"| {step} | "+" | ".join(fmt(values.get(layer, {}).get("rms")) for layer in (0, 11, 15, 35))
                +f" | {fmt(values.get(35, {}).get('relative_rms'))} |")
        lines += ["", "相对RMS为差值RMS/reference activation RMS。不同激活量级的 max_abs 不能直接解释为准确率损失；这里仅记录首层 attention 之后的差异传播。"]
    lines += ["", "各自 prefill 的原失败作为不同实验单列：", "",
        "| 原独立 prefill 实验 | Max abs | RMS | Centered RMS | 原判定 |", "|---|---:|---:|---:|---|"]
    for r in old_steps:
        lines.append(f"| {r['phase']} | {fmt(r['max_abs'])} | {fmt(r['rms'])} | {fmt(r['centered_rms'])} | {'通过' if r['passed'] else '未通过'} |")
    lines += ["", f"原始来源：`{Path(previous).resolve()}`。原 max_abs≤0.15、RMS≤0.02 门槛与失败保留；旧 receipt 没有 centered RMS 时明确不可得，不能反推。", "",
        f"同 Q/K/V oracle 位于 step {capture.get('step')}、第 {capture.get('layer')} 层。实际捕获输出匹配：{oracle.get('captured_vs_repeat_auto_match')}；局部解释资格：{oracle.get('oracle_can_explain_captured_call')}。", "",
        "| 同输入 SDPA 分支 | Vs FP64 max abs | Vs FP64 RMS | Vs FP64→BF16 RMS | 与FP64→BF16逐值相等 | 实际 backend host 算子 | CUDA kernel证据 |", "|---|---:|---:|---:|---:|---|---|"]
    for name, value in branches.items():
        n=value.get("versus_fp64") or {}; rounded=value.get("versus_fp64_rounded_bf16") or {}; e=value["actual_backend"]
        backend=", ".join(f"{k}×{v}" for k,v in e["operators"].items() if k!="aten::scaled_dot_product_attention") or "不可得"
        lines.append(f"| {name} | {fmt(n.get('max_abs'))} | {fmt(n.get('rms'))} | {fmt(rounded.get('rms'))} | {rounded.get('exact_elements', '不可得')}/{rounded.get('elements', '不可得')} | {backend} | {e['cuda_kernel_timeline_observed']} ({e['cuda_kernel_count']}) |")
    math_rows=[branches.get(k, {}).get("versus_fp64_rounded_bf16") or {} for k in ("gqa_math", "repeat_math")]
    auto=branches.get("repeat_automatic", {}).get("versus_fp64_rounded_bf16") or {}
    if all(r.get("elements", 0)>0 and r.get("exact_elements")==r.get("elements") for r in math_rows) and auto.get("rms", 0)>0:
        lines += ["", "此处两个 MATH 输出都恰好等于 FP64 oracle 再舍入到 BF16 的结果；efficient 的误差略大。三者相同的最大绝对误差不代表 RMS 并列。该判断只针对本次捕获的首差异层输入。"]
    lines += ["", "FP64 oracle 只计算同一份 effective Q/K/V 的局部 attention，保留完整 K/V 和原 mask/scale；未用 FP64 重算投影、RoPE 或整模型。三个分支的误差不能代替完整模型验证。", "",
        "Centered RMS 仅用于区分共同 logit 偏移与其他误差。监控采样、额外重放与局部 profiler 都属于诊断开销，不能进入正式 INFRA 表。", ""]
    if oracle_comparison["torch_version_repr_only"]:
        diff=oracle_comparison["differences"]["torch_version"]
        lines += [f"回执序列化说明：oracle 独立 JSON 的 torch_version 为 `{diff['saved']}`，嵌入回执为 `{diff['embedded']}`；仅此字段因 TorchVersion 的 repr 多了引号。汇总记录原差异，并只对这一明确形式作字符串规范化，其余内容完全一致；未修改任何原回执。", ""]
    failed=[name for name, ok in check.items() if not ok]
    if failed:
        lines += ["未通过的证据检查："+", ".join(failed)+"。", ""]
    (destination/"SHARED_STATE_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    args=parser.parse_args()
    report=summarize(args.input, args.out, args.previous)
    print("Diagnostic report checks:", report["report_checks_passed"])
    raise SystemExit(0 if report["report_checks_passed"] else 1)
