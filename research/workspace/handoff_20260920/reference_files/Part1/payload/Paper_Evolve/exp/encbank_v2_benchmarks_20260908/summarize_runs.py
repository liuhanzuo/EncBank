"""Report actual completed predictions and missing jobs without inferring completion."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_from(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    obj = read_json(path)
    return obj.get("rows", obj.get("records", [])) if isinstance(obj, dict) else obj


def values_for(job, output):
    arm, benchmark = job["arm"], job["benchmark"]
    if job["layout"] == "ruler":
        rows = rows_from(output)
        return [(f"{r['task']}/{r['length']}", int(r["i"]), float(r[f"{arm}_recall"])) for r in rows]
    if job["layout"] == "official_qa":
        if not (output / "COMPLETED.json").exists():
            raise ValueError("No COMPLETED.json")
        rows = rows_from(output / "predictions.jsonl")
        return [("category_" + str(r["category"]) if benchmark == "locomo" else r["task"],
                 int(r["index"]), float(r["score"])) for r in rows]
    meta = read_json(output / "COMPLETED.json")
    native = output / "attempts" / Path(meta["output_dir"]).parent.name / "native"
    if benchmark == "babilong":
        from prepare_babilong import score_prediction
        values = []
        for cell in job["cells"]:
            matches = list(native.glob(f"{cell['task']}_{cell['length']}*.csv"))
            if len(matches) != 1:
                raise ValueError(f"Expected one BABILong result for {cell['key']}")
            for i, row in enumerate(rows_from(matches[0])):
                values.append((cell["key"], i, score_prediction(row["output"], row, cell["task"])))
        return values
    if benchmark == "longeval":
        values = []
        for cell in job["cells"]:
            matches = list(native.glob(f"longeval_{cell['length']}*.json"))
            if len(matches) != 1:
                raise ValueError(f"Expected one LongEval result for {cell['key']}")
            values.extend((cell["key"], int(row["sample_index"]), float(row["correct"])) for row in rows_from(matches[0]))
        return values
    values = []
    for path in native.glob("*.jsonl"):
        values.extend((row["task"], int(row["index"]), float(row["score"])) for row in rows_from(path))
    return values


def summarize(plan, result_root=None, reuse_root=None):
    grouped, diagnostics = defaultdict(dict), []
    completed, missing, invalid = [], [], []
    for job in plan["jobs"]:
        output = Path(result_root) / job["relative_output"] if result_root else Path(job["output"])
        if not output.exists():
            missing.append(job["id"])
            continue
        try:
            values = values_for(job, output)
            actual = {(key, index) for key, index, value in values}
            expected = {(cell["key"], index) for cell in job["cells"] for index in cell["indices"]}
            if actual != expected or len(actual) != len(values):
                raise ValueError(f"IDs/count differ: unique={len(actual)}, rows={len(values)}, expected={len(expected)}")
            if any(not 0 <= value <= 1 for _, _, value in values):
                raise ValueError("Score outside 0..1")
            for key, index, value in values:
                group = grouped[(job["benchmark"], job["arm"], key)]
                if index in group:
                    raise ValueError(f"Duplicate source ID {key}/{index}")
                group[index] = value
            completed.append(job["id"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append(job["id"])
            diagnostics.append({"job": job["id"], "error": str(exc)})
    reused_n = 0
    if reuse_root:
        for source in plan.get("reused_results", []):
            rows = rows_from(Path(reuse_root) / source["relative_path"])
            wanted = {(c["task"], c["length"], i) for c in source["cells"] for i in c["indices"]}
            selected = [r for r in rows if (r["task"], r["length"], r["i"]) in wanted]
            if len(selected) != len(wanted):
                raise ValueError(f"Incomplete reusable source {source['relative_path']}")
            for arm in source["arms"]:
                for row in selected:
                    group = grouped[("ruler", arm, f"{row['task']}/{row['length']}")]
                    if row["i"] in group:
                        raise ValueError("Duplicate reusable source ID")
                    group[row["i"]] = float(row[f"{arm}_recall"])
                    reused_n += 1
    expected_cells = Counter()
    for job in plan["jobs"]:
        for cell in job["cells"]:
            expected_cells[(job["benchmark"], job["arm"], cell["key"])] += cell["expected_n"]
    for source in plan.get("reused_results", []):
        for arm in source["arms"]:
            for cell in source["cells"]:
                expected_cells[(source["benchmark"], arm, cell["key"])] += cell["expected_n"]
    table = []
    for (benchmark, arm, key), expected in sorted(expected_cells.items()):
        values = grouped[(benchmark, arm, key)]
        table.append({"benchmark": benchmark, "arm": arm, "cell": key, "n": len(values),
                      "expected_n": expected, "complete": len(values) == expected,
                      "score_percent": 100 * sum(values.values()) / len(values) if values else None})
    family_summaries = []
    for benchmark, arm in sorted({(row['benchmark'], row['arm']) for row in table}):
        cells = [row for row in table if row['benchmark'] == benchmark and row['arm'] == arm]
        if not all(row['complete'] for row in cells):
            continue
        summary = {'benchmark': benchmark, 'arm': arm, 'n': sum(row['n'] for row in cells),
                   'cells': [row['cell'] for row in cells], 'scope': 'complete requested cells in this plan'}
        if benchmark == 'locomo':
            answerable = [row for row in cells if row['cell'] != 'category_5']
            total = sum(row['n'] for row in answerable)
            summary['answerable_micro_f1_percent'] = sum(row['n'] * row['score_percent'] for row in answerable) / total if total else None
            summary['adversarial_accuracy_percent'] = next((row['score_percent'] for row in cells if row['cell'] == 'category_5'), None)
        elif benchmark == 'infinitebench':
            summary['task_scores_percent'] = {row['cell']: row['score_percent'] for row in cells}
        elif benchmark == 'oracle_support':
            summary['diagnostic_scores_percent'] = {row['cell']: row['score_percent'] for row in cells}
            summary['scope'] = 'eligible annotated-support subset only; report tasks/categories separately; natural-pack pairing requires a separate checked join'
        else:
            summary['macro_over_requested_cells_percent'] = sum(row['score_percent'] for row in cells) / len(cells)
        family_summaries.append(summary)
    return {"mode": plan["mode"], "planned_jobs": len(plan["jobs"]), "completed_jobs": len(completed),
            "missing_jobs": missing, "incomplete_or_invalid_jobs": invalid, "diagnostics": diagnostics,
            "reused_predictions": reused_n, "observed_predictions": sum(row["n"] for row in table),
            "all_complete": not missing and not invalid and all(row["complete"] for row in table),
            "family_summaries": family_summaries, "cells": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(read_json(args.plan), args.result_root, args.reuse_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cells", "missing_jobs", "incomplete_or_invalid_jobs", "diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
