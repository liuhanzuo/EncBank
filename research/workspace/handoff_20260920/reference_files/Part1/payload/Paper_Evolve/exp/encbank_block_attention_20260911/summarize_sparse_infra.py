"""CPU-only final table using sampled incremental VRAM, plus cache-parity diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

from infra_protocol import GIB, MIB, LABELS, now, read_json, save_json


def number(value, unit=1, digits=2):
    if value is None:
        return "--"
    scaled = value / unit
    if 0 < abs(scaled) < .1:
        digits = max(digits, 3)
    return f"{scaled:.{digits}f}"


def parity_pair(left, right, comparison):
    if not left or not right or left.get("status") != "complete" or right.get("status") != "complete":
        return {"comparison": comparison, "status": "pending"}
    a, b = left.get("result", {}), right.get("result", {})
    ia, ib = a.get("identity", {}), b.get("identity", {})
    same = (a.get("input", {}).get("token_sha256") == b.get("input", {}).get("token_sha256")
            and ia.get("adapter_sha256") == ib.get("adapter_sha256")
            and ia.get("model_config_sha256") == ib.get("model_config_sha256")
            and all(ia.get(key) == ib.get(key) for key in ("j", "m", "rho", "document_tokens", "prompt_tokens", "generation_tokens", "seed")))
    if not same:
        return {"comparison": comparison, "status": "incomparable_configuration"}
    pairs = []
    left_rows = {r["repetition"]: r for r in a.get("requests", [])}
    right_rows = {r["repetition"]: r for r in b.get("requests", [])}
    for index in sorted(set(left_rows) & set(right_rows)):
        x, y = left_rows[index], right_rows[index]
        pairs.append({"repetition": index, "generated_ids_equal": x["generated_ids"] == y["generated_ids"],
            "selected_indices_equal": x["route_stats"].get("selected_indices") == y["route_stats"].get("selected_indices"),
            "left_generated_ids": x["generated_ids"], "right_generated_ids": y["generated_ids"]})
    if not pairs:
        status = "pending"
    else:
        status = "passed_token_diagnostic" if all(p["generated_ids_equal"] and p["selected_indices_equal"] for p in pairs) else "failed_token_diagnostic"
    return {"comparison": comparison, "status": status, "pairs": pairs,
            "claim_scope": "greedy-token/route check on this synthetic input; not a distribution-equivalence or benchmark-quality proof"}


def summarize(folder):
    folder = Path(folder)
    plan = read_json(folder / "plan.json", [])
    state = read_json(folder / "status.json", {})
    jobs = state.get("jobs", {})
    if not plan:
        raise ValueError("No infrastructure plan found")
    grouped = {}
    for job in plan:
        group = (job["document_tokens"], job["prompt_tokens"], job["generation_tokens"])
        grouped.setdefault(group, {})[(job["arm"], job["cache_mode"])] = jobs.get(job["id"])
    diagnostics = []
    for shape, rows in grouped.items():
        for arm in ("A", "B"):
            diagnostics.append({"shape": shape, "arm": arm, **parity_pair(rows.get((arm, "cold_hj")),
                rows.get((arm, "block_hot")), "cold vs independent hot")})
        diagnostics.append({"shape": shape, "arm": "D0/NATIVE", **parity_pair(rows.get(("D0", "cold_hj")),
            rows.get(("NATIVE", "cold_hj")), "sparse wrapper D0 vs native Encbank")})
    save_json(folder / "CACHE_PARITY.json", {"generated_at": now(), "diagnostics": diagnostics})
    lines = ["# Cache reuse token diagnostics", "", "| Shape (document/query/output) | Comparison | Status |", "|---|---|---|"]
    for check in diagnostics:
        lines.append(f"| {'/'.join(map(str, check['shape']))} | {check['arm']}: {check['comparison']} | {check['status']} |")
    lines += ["", "These compare deterministic generated IDs and selected block sets. Floating-point distributions were not recorded; greedy equality does not establish distribution equality. A missing pair stays pending, and a failed pair is not described as quality-preserving reuse."]
    (folder / "CACHE_PARITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex_path = folder / "INFRA_TABLE.tex"
    old = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    if "Alloc. (GiB)" in old:
        (folder / "INFRA_TABLE_ALLOCATOR.tex").write_text(old, encoding="utf-8")
    tex = ["% Generated final fragment; requires booktabs. Synthetic engineering precheck, no accuracy claim.",
           r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Method / cache & $N$ & Store (MiB) & $\Delta$VRAM (GiB) & TTFT (s) & Dec. (tok/s) & E2E (s) \\", r"\midrule"]
    labels = {"D0": "D0 (wrapper)", "A": "Block-all", "B": "Block-pruned", "D1": "Dense-pruned",
              "NATIVE": "Encbank + LoRA", "FULL": "Full recompute"}
    for job in plan:
        receipt = jobs.get(job["id"], {})
        result, monitor = receipt.get("result", {}), receipt.get("monitor", {})
        good = receipt.get("status") == "complete"
        metrics, store = (result.get("summary", {}), result.get("store", {})) if good else ({}, {})
        label = labels[job["arm"]] + (" / hot" if job["cache_mode"] == "block_hot" else " / cold")
        if not good:
            label += " (" + receipt.get("status", "pending").replace("_", "-") + ")"
        storage = sum(store.get(k, 0) for k in ("cold_file_bytes", "hot_file_bytes")) if good else None
        delta = monitor.get("peak_incremental_gpu_bytes") if good else None
        tex.append(f"{label} & {job['document_tokens']} & {number(storage, MIB, 1)} & {number(delta, GIB)} & "
                   f"{number(metrics.get('ttft_s'))} & {number(metrics.get('decode_tokens_per_s'))} & {number(metrics.get('query_e2e_s'))} " + r"\\")
    for arm, label in (("NATIVE", "Native Encbank + LoRA"), ("FULL", "Full recompute"), ("EXACT_PREFIX", "Encbank exact prefix")):
        if not any(job["arm"] == arm for job in plan):
            tex.append(label + r" (pending) & -- & -- & -- & -- & -- & -- \\")
    tex += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Synthetic fixed-length engineering measurements on one local RTX 5090. Store includes cold and additional hot files. $\Delta$VRAM is the sampled device-memory increase against the fixed pre-CUDA baseline, including model weights; it is not an instantaneous upper bound and may include desktop residency changes. Exact allocator allocated/reserved peaks and one-time writing costs are reported separately. Full recompute retains the same upper-layer LoRA. Missing baselines and failures remain explicit.}",
        r"\label{tab:sparse-infra-precheck}", r"\end{table}"]
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    # FULL uses raw token storage despite the shared CLI's cold_hj enum.
    # Preserve a small nonzero token file rather than rounding it to 0.0 MiB.
    report_path = folder / "INFRA_REPORT.md"
    if report_path.is_file():
        lines = report_path.read_text(encoding="utf-8").splitlines()
        for job in plan:
            receipt = jobs.get(job["id"], {})
            if job["arm"] != "FULL" or receipt.get("status") != "complete":
                continue
            store = receipt["result"]["store"]
            prefix = f"| {LABELS['FULL']} | {job['document_tokens']} |"
            for index, line in enumerate(lines):
                if line.startswith(prefix):
                    cells = line.split("|")
                    cells[3] = " raw_tokens "
                    cells[5] = f" {store['cold_file_bytes']/MIB:.3f} / {store['hot_file_bytes']/MIB:.1f} "
                    lines[index] = "|".join(cells)
        report_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.input)
