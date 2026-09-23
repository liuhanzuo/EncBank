"""Join completed oracle accuracy with existing natural-pack controls on CPU.

No model generation, partial-shard scoring, or cross-task macro is performed.
Old natural outputs record source fields and pack metadata, not raw input IDs;
their pairing evidence is explicitly limited to the recorded protocol/fields.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def score(row):
    value = float(row["score"])
    if row.get("status") != "ok" or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Prediction status/score is not a valid completed accuracy result")
    return value


def check_options(config, arm, source_benchmark=None):
    expected = {"j": 12, "selector": "bm25", "topk": 12, "chunk_size": 512,
                "seed": 42, "dtype": "bfloat16", "attn_impl": "sdpa", "adapter": ""}
    options = config["options"]
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError(f"Mismatched {key}: {options.get(key)!r} != {value!r}")
    if config.get("arm", options.get("arm")) != arm:
        raise ValueError("Reader arm differs from planned arm")
    if source_benchmark is not None and config["benchmark"] != source_benchmark:
        raise ValueError("Natural result belongs to another source benchmark")


def check_source(row, fixture, oracle=False):
    sample = fixture["sample"]
    for key, value in sample.items():
        if key == "task" and oracle:
            if row.get("source_task") != value:
                raise ValueError("Oracle source task changed")
        elif row.get(key) != value:
            raise ValueError(f"Source field changed: {key}")
    if oracle:
        for key in ("source_benchmark", "task", "input_ids_sha256", "fixture_row_sha256"):
            if row.get(key) != fixture[key]:
                raise ValueError(f"Oracle fixture identity changed: {key}")
        if row.get("pack") != fixture["oracle_pack"] or row.get("natural_pack") != fixture["natural_pack"]:
            raise ValueError("Oracle intervention or natural reference pack changed")
    elif row.get("pack") != fixture["natural_pack"]:
        raise ValueError("Natural result pack differs from prepared natural pack")
    score(row)


def completed_rows(plan, root):
    """Read canonical, whole jobs only; return diagnostics instead of partial means."""
    found, configs, errors = {}, {}, []
    for job in plan["jobs"]:
        if job.get("layout") != "official_qa":
            continue
        output = Path(root) / job["relative_output"]
        if not (output / "COMPLETED.json").exists():
            continue
        try:
            config = read_json(output / "run_config.json")
            config["_canonical_output"] = str(output.resolve())
            rows = read_rows(output / "predictions.jsonl")
            actual = {(r["task"], int(r["index"])) for r in rows}
            expected = {(c["key"], int(i)) for c in job["cells"] for i in c["indices"]}
            if actual != expected or len(rows) != len(actual):
                raise ValueError("Completed job IDs/counts differ from plan")
            for row in rows:
                key = (job["benchmark"], job["arm"], row["id"])
                if key in found:
                    raise ValueError("Duplicate canonical source ID")
                score(row)
            for row in rows:
                key = (job["benchmark"], job["arm"], row["id"])
                found[key], configs[key] = row, config
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"job": job["id"], "error": str(exc)})
    return found, configs, errors


def natural_is_planned(main_plan, arm, fixture):
    return any(j["benchmark"] == fixture["source_benchmark"] and j["arm"] == arm
               and any(c["key"] == fixture["source_task"] and fixture["index"] in c["indices"]
                       for c in j["cells"]) for j in main_plan["jobs"])


def natural_generation_key(fixture, contexts):
    context_ids = contexts[fixture["context_key"]]["context_ids"]
    tokens = context_ids + fixture["query_ids"]
    content = {"tokens": [tokens], "generation": {
        "context_token_count": len(context_ids),
        "selected_indices": fixture["natural_pack"]["selected_indices"],
        "chunk_size": 512, "max_new_tokens": fixture["sample"]["max_new_tokens"]}}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), len(tokens)


def verify_natural_cache(connection, fixture, row, contexts):
    key, token_n = natural_generation_key(fixture, contexts)
    record = connection.execute("SELECT prediction,n_tokens FROM generations WHERE key=?", (key,)).fetchone()
    if record is None or json.loads(record[0]) != row["pred"] or record[1] != token_n:
        raise ValueError("Natural durable generation key/input tokens/prediction do not match the fixture")
    return key


def paired_summary(fixtures, oracle_plan, main_plan, oracle_root, main_root, contexts=None):
    oracle, oracle_configs, errors = completed_rows(oracle_plan, oracle_root)
    natural, natural_configs, natural_errors = completed_rows(main_plan, main_root)
    errors += natural_errors
    arms = list(oracle_plan["arms"])
    groups = defaultdict(list)
    for fixture in fixtures:
        groups[fixture["task"]].append(fixture)
    if len({(r["source_benchmark"], r["id"]) for r in fixtures}) != len(fixtures):
        raise ValueError("Duplicate eligible fixture IDs")
    cells, records, valid_oracle = [], [], {}
    connections = {}
    for task, items in sorted(groups.items()):
        for arm in arms:
            oracle_values, pairs = [], []
            generated_n, derived_n = 0, 0
            planned = [natural_is_planned(main_plan, arm, f) for f in items]
            if any(planned) and not all(planned):
                raise ValueError("Natural control availability changes within an eligible task subset")
            for fixture in items:
                oracle_key = ("oracle_support", arm, fixture["id"])
                natural_key = (fixture["source_benchmark"], arm, fixture["id"])
                o, n = oracle.get(oracle_key), natural.get(natural_key)
                try:
                    if n is not None:
                        check_options(natural_configs[natural_key], arm, fixture["source_benchmark"])
                        check_source(n, fixture)
                        if "model" in oracle_plan and natural_configs[natural_key]["options"]["model"] != oracle_plan["model"]:
                            raise ValueError("Natural model differs from the approved oracle model")
                        if contexts is not None:
                            db = Path(natural_configs[natural_key]["_canonical_output"]) / "generations.sqlite3"
                            if str(db) not in connections:
                                connections[str(db)] = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
                            verify_natural_cache(connections[str(db)], fixture, n, contexts)
                    generated_planned = any(j["arm"] == arm and any(
                        c["key"] == fixture["task"] and fixture["index"] in c["indices"]
                        for c in j["cells"]) for j in oracle_plan["jobs"])
                    reuse_allowed = (not generated_planned and all(planned)
                        and fixture["oracle_pack"] == fixture["natural_pack"])
                    if not generated_planned and not reuse_allowed:
                        raise ValueError("Eligible oracle row is neither scheduled nor eligible for identical-pack reuse")
                    origin = "new_oracle_generation"
                    if o is not None:
                        check_options(oracle_configs[oracle_key], arm)
                        check_source(o, fixture, oracle=True)
                        if "model" in oracle_plan and oracle_configs[oracle_key]["options"]["model"] != oracle_plan["model"]:
                            raise ValueError("Oracle model differs from its approved plan")
                        if n is not None and oracle_configs[oracle_key]["options"]["model"] != natural_configs[natural_key]["options"]["model"]:
                            raise ValueError("Natural and oracle model paths differ")
                        generated_n += 1
                    elif reuse_allowed and n is not None:
                        o = n
                        origin = "derived_from_identical_natural_pack"
                        derived_n += 1
                    if o is not None:
                        oracle_values.append(score(o))
                        valid_oracle[(arm, fixture["source_benchmark"], fixture["id"])] = score(o)
                    if o is not None and n is not None:
                        pair = {"id": fixture["id"], "index": fixture["index"], "task": task,
                            "arm": arm, "natural_score": score(n), "oracle_score": score(o),
                            "oracle_origin": origin,
                            "natural_exact_generation_cache_verified": contexts is not None,
                            "difference_pp": 100 * (score(o) - score(n)),
                            "natural_all_required_visible": fixture["natural_all_required_visible"],
                            "read_pack_token_delta": fixture["read_pack_token_delta"]}
                        pairs.append(pair)
                        records.append(pair)
                except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
                    errors.append({"task": task, "arm": arm, "id": fixture["id"], "error": str(exc)})
            oracle_complete = len(oracle_values) == len(items)
            paired_complete = all(planned) and len(pairs) == len(items)
            cell = {"task": task, "arm": arm, "expected_n": len(items),
                "oracle_observed_n": len(oracle_values), "oracle_complete": oracle_complete,
                "oracle_generated_n": generated_n, "oracle_derived_from_identical_natural_n": derived_n,
                "oracle_f1_percent": 100 * sum(oracle_values) / len(items) if oracle_complete else None,
                "natural_control_planned": all(planned), "paired_observed_n": len(pairs),
                "paired_complete": paired_complete,
                "natural_f1_percent": sum(p["natural_score"] for p in pairs) * 100 / len(items) if paired_complete else None,
                "oracle_minus_natural_pp": sum(p["difference_pp"] for p in pairs) / len(items) if paired_complete else None}
            cell["status"] = ("paired_complete" if paired_complete else
                "oracle_only_no_natural_control_planned" if oracle_complete and not all(planned) else
                "waiting_for_oracle_or_main_control")
            cells.append(cell)
    visibility = []
    for task, items in sorted(groups.items()):
        differences = []
        for fixture in items:
            a = valid_oracle.get(("fix_all", fixture["source_benchmark"], fixture["id"]))
            b = valid_oracle.get(("fix_none", fixture["source_benchmark"], fixture["id"]))
            if a is not None and b is not None:
                differences.append(100 * (a - b))
        complete = len(differences) == len(items)
        visibility.append({"task": task, "expected_n": len(items), "observed_n": len(differences),
            "complete": complete, "oracle_v2_minus_fix_none_pp": sum(differences) / len(items) if complete else None})
    for connection in connections.values():
        connection.close()
    return {"scope": "eligible annotated-support subsets; task/category results remain separate",
        "pairing_evidence": ("source fields, configuration, packs and original durable generation-cache keys verify the reconstructed full token input, generation arguments and exact stored prediction" if contexts is not None else "source/config/pack metadata only; exact generation-cache verification unavailable"),
        "timing_data": False, "eligible_samples": len(fixtures), "cells": cells,
        "oracle_visibility_comparison": visibility, "paired_records": records,
        "diagnostics": errors,
        "all_planned_pairs_complete": bool(cells) and not errors and all(
            c["oracle_complete"] and (c["paired_complete"] or not c["natural_control_planned"]) for c in cells)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "data/oracle_support/inputs.jsonl")
    parser.add_argument("--contexts", type=Path, default=HERE / "data/oracle_support/contexts.json")
    parser.add_argument("--oracle-plan", type=Path, default=HERE / "oracle_support_full_plan.json")
    parser.add_argument("--main-plan", type=Path, default=HERE / "full_plan.json")
    parser.add_argument("--oracle-root", type=Path, default=HERE / "results/remote/outputs/oracle_support/full")
    parser.add_argument("--main-root", type=Path, default=HERE / "results/remote/outputs/benchmarks_v2/full")
    parser.add_argument("--out", type=Path, default=HERE / "results/oracle_support_paired.json")
    args = parser.parse_args()
    report = paired_summary(read_rows(args.inputs), read_json(args.oracle_plan),
        read_json(args.main_plan), args.oracle_root, args.main_root, read_json(args.contexts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(args.out)
    print(json.dumps({"all_planned_pairs_complete": report["all_planned_pairs_complete"],
        "eligible_samples": report["eligible_samples"], "paired_rows": len(report["paired_records"]),
        "diagnostic_errors": len(report["diagnostics"])}))


if __name__ == "__main__":
    main()
