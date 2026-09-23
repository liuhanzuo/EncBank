"""CPU-only aggregation of the four-arm SFT pilot; no torch/model imports.

The final development set and the fixed small trajectory set are reported
separately. No difference between their unpaired means is ever called a gain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


ARMS = {"beacon4": ("beacon", 4), "encbank": ("encbank", 1),
        "beacon8": ("beacon", 8), "pool4": ("pool", 4)}
DEFAULT_ROOT = Path(__file__).resolve().parent / "results/remote"
INPUT_FIELDS = ("document_id", "source", "split", "question", "references",
                "selected_chunk_indices", "selected_context_tokens", "original_context_tokens",
                "prompt_tokens", "answer_truncated", "answer_ce_tokens")
RECIPE_REQUIRED = ("model", "init_adapter_sha256", "train_sha256", "dev_sha256", "objective",
                   "selection", "seed", "steps", "grad_accum", "chunk_size", "max_chunks",
                   "max_question_tokens", "max_answer_tokens", "max_new_tokens", "eval_every",
                   "eval_limit", "final_eval_limit", "train_examples", "dev_examples")
PROTOCOL_REQUIRED = ("decoding", "max_new_tokens", "eos_token_ids", "enable_thinking",
                     "answer_ce", "ce_reference_index", "ce_targets", "hardware_timing")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize(records: list[dict]) -> dict:
    tokens = sum(row["answer_ce_tokens"] for row in records)
    return {"examples": len(records), "documents": len({row["document_id"] for row in records}),
            "exact_match": sum(row["exact_match"] for row in records) / len(records) if records else None,
            "token_f1": sum(row["token_f1"] for row in records) / len(records) if records else None,
            "answer_ce": sum(row["answer_ce_sum"] for row in records) / tokens if tokens else None,
            "answer_ce_tokens": tokens,
            "selected_context_tokens": sum(row["selected_context_tokens"] for row in records),
            "prompt_tokens": sum(row["prompt_tokens"] for row in records),
            "source_counts": dict(Counter(row["source"] for row in records))}


def load_evaluation(path: Path, expected_count: int) -> dict:
    data = read_json(path)
    records, summary = data.get("records", []), data.get("summary", {})
    if not isinstance(records, list) or not records:
        raise ValueError("No completed evaluation records")
    if len(records) != expected_count:
        raise ValueError(f"Evaluation has {len(records)} records, expected {expected_count}")
    identifiers = [row.get("id") for row in records]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Missing or duplicate evaluation IDs")
    for row in records:
        missing = [key for key in INPUT_FIELDS if key not in row]
        if missing:
            raise ValueError(f"Record {row['id']} missing input fields: {missing}")
        if row["split"] != "dev" or not row["document_id"] or not row["references"]:
            raise ValueError(f"Invalid development source/reference identity: {row['id']}")
        for key in ("exact_match", "token_f1"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid score {key}: {row['id']}")
        if not isinstance(row["answer_ce_tokens"], int) or row["answer_ce_tokens"] < 1:
            raise ValueError(f"Missing CE token denominator: {row['id']}")
        if not finite(row.get("answer_ce_sum")) or row["answer_ce_sum"] < 0:
            raise ValueError(f"Invalid CE numerator: {row['id']}")
        if not finite(row.get("answer_ce")) or not math.isclose(row["answer_ce"], row["answer_ce_sum"] / row["answer_ce_tokens"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Inconsistent record CE: {row['id']}")
    calculated = summarize(records)
    for key in ("examples", "exact_match", "token_f1", "answer_ce", "answer_ce_tokens"):
        if not finite(summary.get(key)) or not math.isclose(summary[key], calculated[key], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Summary does not match records: {key}")
    protocol = summary.get("protocol", {})
    missing = [key for key in PROTOCOL_REQUIRED if key not in protocol]
    if missing or protocol.get("answer_ce") is not True or summary.get("score_scale") != "0-to-1":
        raise ValueError(f"Missing/wrong evaluation protocol: {missing}")
    return {"records": records, "ids": identifiers, "metrics": calculated, "protocol": protocol}


def read_training_log(path: Path) -> tuple[dict[int, dict], list[str]]:
    if not path.exists():
        return {}, []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    steps, warnings = {}, []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                warnings.append("Ignored an incomplete last training-log line while the writer may be active")
                continue
            raise ValueError(f"Malformed training-log line {i + 1}")
        step = row.get("step")
        if not isinstance(step, int) or step < 1:
            raise ValueError("Invalid training-log step")
        if step in steps:
            # A resume may roll back unsaved steps. A new lower step invalidates
            # the old tail, not merely the one duplicated line.
            for old in [key for key in steps if key >= step]:
                del steps[old]
            warnings.append(f"Discarded superseded log tail at resumed step {step}")
        for key in ("cursor", "raw_tokens", "target_tokens"):
            if not isinstance(row.get(key), int) or row[key] < 1:
                raise ValueError(f"Invalid training counter {key} at step {step}")
        if not finite(row.get("loss")):
            raise ValueError(f"Nonfinite training loss at step {step}")
        steps[step] = row
    return steps, warnings


def _same_inputs(reference: dict, candidate: dict, ids: list[str]) -> list[str]:
    problems = []
    if reference["protocol"] != candidate["protocol"]:
        problems.append("evaluation protocols differ")
    left = {row["id"]: row for row in reference["records"]}
    right = {row["id"]: row for row in candidate["records"]}
    for key in ids:
        if key not in left or key not in right:
            problems.append(f"missing paired ID {key}")
            continue
        mismatches = [field for field in INPUT_FIELDS if left[key][field] != right[key][field]]
        if mismatches:
            problems.append(f"input mismatch for {key}: {','.join(mismatches)}")
    return problems


def paired_difference(reference: dict, candidate: dict, ids: list[str]) -> dict:
    problems = _same_inputs(reference, candidate, ids)
    if problems:
        return {"comparable": False, "issues": problems}
    ref_by_id = {row["id"]: row for row in reference["records"]}
    cand_by_id = {row["id"]: row for row in candidate["records"]}
    left = [ref_by_id[key] for key in ids]
    right = [cand_by_id[key] for key in ids]
    a, b = summarize(left), summarize(right)
    signs = Counter("win" if y["token_f1"] > x["token_f1"] else "loss" if y["token_f1"] < x["token_f1"] else "tie"
                    for x, y in zip(left, right))
    return {"comparable": True, "ids": ids, "examples": len(ids), "documents": a["documents"],
            "reference": a, "candidate": b,
            "delta_exact_match_pp": 100 * (b["exact_match"] - a["exact_match"]),
            "delta_token_f1_pp": 100 * (b["token_f1"] - a["token_f1"]),
            "delta_answer_ce": b["answer_ce"] - a["answer_ce"],
            "f1_pairs": {key: signs[key] for key in ("win", "tie", "loss")}}


def inspect_arm(directory: Path, arm: str) -> tuple[dict, dict]:
    public = {"path": str(directory.resolve()), "state": "waiting", "issues": [], "warnings": [], "evaluations": {}}
    internal = {"evaluations": {}, "train": {}, "recipe": None, "model": None}
    if not (directory / "metadata.json").exists():
        public["issues"].append("metadata.json not available")
        return public, internal
    try:
        metadata = read_json(directory / "metadata.json")
        recipe = metadata["recipe"]
        missing = [key for key in RECIPE_REQUIRED if key not in recipe]
        if missing:
            raise ValueError(f"Incomplete recipe: {missing}")
        if (recipe.get("mode"), recipe.get("ratio")) != ARMS[arm]:
            raise ValueError("Directory arm differs from recorded mode/ratio")
        if recipe["steps"] < 1 or recipe["grad_accum"] < 1 or recipe["eval_every"] < 1:
            raise ValueError("Invalid training schedule")
        internal["recipe"] = recipe
        internal["model"] = metadata.get("model")
        public["recipe"] = recipe
        public["trainable_parameters"] = metadata.get("trainable_parameters")
        public["gpu"] = metadata.get("gpu")
        public["formal_inference_timing"] = False
        status = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        public["status"] = status
        internal["train"], warnings = read_training_log(directory / "train.jsonl")
        public["warnings"].extend(warnings)
        steps = internal["train"]
        public["latest_logged_step"] = max(steps, default=0)
        for step, row in steps.items():
            if row["cursor"] != step * recipe["grad_accum"]:
                raise ValueError(f"Training example budget inconsistent at step {step}")
            previous = steps.get(step - 1)
            if previous and any(row[key] < previous[key] for key in ("raw_tokens", "target_tokens")):
                raise ValueError(f"Training token counters decrease at step {step}")
        for path in sorted(directory.glob("eval_step*.json")):
            match = re.fullmatch(r"eval_step(\d+)\.json", path.name)
            if not match:
                continue
            step = int(match.group(1))
            count = min(recipe["dev_examples"], recipe["final_eval_limit"] if step == recipe["steps"] else recipe["eval_limit"])
            try:
                evaluation = load_evaluation(path, count)
                if evaluation["protocol"]["max_new_tokens"] != recipe["max_new_tokens"]:
                    raise ValueError("Generation limit differs from training recipe")
                internal["evaluations"][step] = evaluation
                public["evaluations"][str(step)] = {key: evaluation[key] for key in ("ids", "metrics", "protocol")}
            except (ValueError, KeyError, TypeError) as error:
                public["issues"].append(f"{path.name}: {error}")
        required_steps = sorted({0, recipe["steps"], *range(recipe["eval_every"], recipe["steps"], recipe["eval_every"])})
        missing_evals = [step for step in required_steps if step not in internal["evaluations"]]
        public["missing_evaluation_steps"] = missing_evals
        target = recipe["steps"]
        training_done = status.get("complete") is True and status.get("phase") == "complete" and status.get("step") == target and status.get("target_steps") == target
        if training_done:
            if set(steps) != set(range(1, target + 1)):
                public["issues"].append("Completed status lacks an exact 1..target training-log lineage")
            else:
                last = steps[target]
                for status_key, log_key in (("training_examples_processed", "cursor"), ("raw_tokens", "raw_tokens"), ("target_tokens", "target_tokens")):
                    if status.get(status_key) != last[log_key]:
                        public["issues"].append(f"Completed status counter differs from log: {status_key}")
        public["state"] = "invalid" if public["issues"] else "complete" if training_done and not missing_evals else "running"
    except (ValueError, KeyError, TypeError) as error:
        public["state"] = "invalid"
        public["issues"].append(str(error))
    return public, internal


def aggregate(root: Path) -> dict:
    result = {"format": "beacon-sft-report-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "root": str(root.resolve()), "complete": False, "arms": {}, "comparison_issues": [],
              "checkpoints": [], "paired_initial_to_final_fixed_ids": {},
              "limitations": ["Internal document-heldout QASPER development results, not final LongBench test results.",
                              "Initial/scheduled small-set means are never subtracted from full final-set means. Initial-to-final gains use identical IDs only.",
                              "CE is summed answer negative log-likelihood divided by supervised target tokens, not an unweighted mean of example CE.",
                              "Inputs are checked through source fingerprints, declared preparation/decoding protocol, IDs, question/reference strings, selected chunk indices and token counts. Actual prompt IDs/template fingerprints are not stored by the current evaluator.",
                              "No significance or confidence-interval claim; questions within a document are not treated as independent CI samples.",
                              "Synchronized files are captured independently; a transient status/log mismatch calls for a fresh snapshot, not a claim that training failed.",
                              "Remote training seconds/allocations are operational metadata, not formal inference speed or memory results."]}
    if (root / "SYNC_MANIFEST.json").exists():
        manifest = read_json(root / "SYNC_MANIFEST.json")
        result["snapshot"] = {key: manifest.get(key) for key in ("source_host", "source_root", "snapshot_started_utc", "snapshot_finished_utc", "consistency")}
        result["snapshot"]["checkpoints_copied"] = False
    internals = {}
    for arm in ARMS:
        result["arms"][arm], internals[arm] = inspect_arm(root / arm, arm)
    reference_recipe = internals["encbank"]["recipe"]
    comparable_recipes = reference_recipe is not None
    if reference_recipe:
        reference_shared = {key: value for key, value in reference_recipe.items() if key not in {"mode", "ratio"}}
        for arm, item in internals.items():
            candidate = item["recipe"]
            if candidate is None:
                comparable_recipes = False
                result["comparison_issues"].append(f"{arm}: recipe unavailable")
            else:
                shared = {key: value for key, value in candidate.items() if key not in {"mode", "ratio"}}
                keys = sorted(key for key in set(shared) | set(reference_shared) if shared.get(key) != reference_shared.get(key))
                if keys:
                    comparable_recipes = False
                    result["comparison_issues"].append(f"{arm}: shared recipe differs from Encbank: {keys}")
                if item["model"] != internals["encbank"]["model"]:
                    comparable_recipes = False
                    result["comparison_issues"].append(f"{arm}: recorded backbone configuration differs from Encbank")
    else:
        result["comparison_issues"].append("Encbank reference recipe unavailable")
    all_steps = sorted({step for item in internals.values() for step in item["evaluations"]})
    for step in all_steps:
        stage = {"step": step, "comparable": False, "issues": [], "by_arm": {}, "deltas_vs_encbank": {}}
        for arm, item in internals.items():
            if step in item["evaluations"]:
                stage["by_arm"][arm] = item["evaluations"][step]["metrics"]
            else:
                stage["issues"].append(f"{arm}: completed evaluation unavailable")
        if not comparable_recipes:
            stage["issues"].append("Shared recipes not verified equal")
        budgets = {}
        for arm, item in internals.items():
            if step == 0:
                budgets[arm] = {"cursor": 0, "raw_tokens": 0, "target_tokens": 0}
            elif step in item["train"]:
                budgets[arm] = {key: item["train"][step][key] for key in ("cursor", "raw_tokens", "target_tokens")}
            else:
                stage["issues"].append(f"{arm}: training-budget record unavailable at this checkpoint")
        stage["training_budgets"] = budgets
        if budgets and any(value != next(iter(budgets.values())) for value in budgets.values()):
            stage["issues"].append("Training raw/target token or example budgets differ")
        if not stage["issues"]:
            reference = internals["encbank"]["evaluations"][step]
            ids = reference["ids"]
            for arm, item in internals.items():
                candidate = item["evaluations"][step]
                if candidate["ids"] != ids:
                    stage["issues"].append(f"{arm}: ordered evaluation IDs differ")
                    continue
                paired = paired_difference(reference, candidate, ids)
                if not paired["comparable"]:
                    stage["issues"].extend(f"{arm}: {issue}" for issue in paired["issues"])
                elif arm != "encbank":
                    stage["deltas_vs_encbank"][arm] = paired
            stage["comparable"] = not stage["issues"]
            if stage["issues"]:
                stage["deltas_vs_encbank"] = {}
        result["checkpoints"].append(stage)
    for arm, item in internals.items():
        recipe = item["recipe"]
        if recipe is None or 0 not in item["evaluations"] or recipe["steps"] not in item["evaluations"]:
            result["paired_initial_to_final_fixed_ids"][arm] = {"comparable": False, "issues": ["Initial or final evaluation unavailable"]}
            continue
        initial, final = item["evaluations"][0], item["evaluations"][recipe["steps"]]
        comparison = paired_difference(initial, final, initial["ids"])
        comparison.update(initial_step=0, final_step=recipe["steps"], full_final_examples=final["metrics"]["examples"])
        result["paired_initial_to_final_fixed_ids"][arm] = comparison
    final_step = reference_recipe["steps"] if reference_recipe else None
    final_cross = next((stage for stage in result["checkpoints"] if stage["step"] == final_step), {})
    result["complete"] = (all(arm["state"] == "complete" for arm in result["arms"].values()) and comparable_recipes
                          and final_cross.get("comparable") is True
                          and all(item.get("comparable") is True for item in result["paired_initial_to_final_fixed_ids"].values())
                          and all(stage["comparable"] for stage in result["checkpoints"]))
    return result


def render_report(result: dict) -> str:
    lines = ["# Beacon + Encbank SFT pilot", "", f"Report complete: **{str(result['complete']).lower()}**. Scores are internal QASPER development results.", "",
             "EM/F1 below use percent; CE is token-weighted nats per supervised answer token. No inference-speed or GPU-memory claim is made.", "",
             "| Arm | State | Latest logged step | Issues |", "|---|---|---:|---|"]
    for name, arm in result["arms"].items():
        lines.append(f"| {name} | {arm['state']} | {arm.get('latest_logged_step', 0)} | {'; '.join(arm['issues']) or '—'} |")
    for stage in result["checkpoints"]:
        lines.extend(["", f"## Checkpoint {stage['step']}", "", f"Four-arm comparison valid: **{str(stage['comparable']).lower()}**.", "",
                      "| Arm | Questions | Documents | EM (%) | F1 (%) | Answer CE |", "|---|---:|---:|---:|---:|---:|"])
        for name, metrics in stage["by_arm"].items():
            lines.append(f"| {name} | {metrics['examples']} | {metrics['documents']} | {100 * metrics['exact_match']:.2f} | {100 * metrics['token_f1']:.2f} | {metrics['answer_ce']:.4f} |")
        if stage["issues"]:
            lines.extend(["", "Comparison withheld: " + "; ".join(stage["issues"])])
    lines.extend(["", "## Initial-to-final changes on the same fixed IDs", "",
                  "Final full-set means are not used in this subtraction. Each row selects exactly that arm's initial small-set IDs from its final records.", "",
                  "| Arm | Paired questions | Documents | ΔEM (pp) | ΔF1 (pp) | ΔCE |", "|---|---:|---:|---:|---:|---:|"])
    for name, item in result["paired_initial_to_final_fixed_ids"].items():
        if item.get("comparable"):
            lines.append(f"| {name} | {item['examples']} | {item['documents']} | {item['delta_exact_match_pp']:+.2f} | {item['delta_token_f1_pp']:+.2f} | {item['delta_answer_ce']:+.4f} |")
        else:
            lines.append(f"| {name} | pending | — | — | — | — |")
    lines.extend(["", "## Scope and checks", ""])
    lines.extend(f"- {text}" for text in result["limitations"])
    if result["comparison_issues"]:
        lines.extend(["", "Shared-recipe issues: " + "; ".join(result["comparison_issues"])])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Four-arm output directory, sync root with LATEST.json, or an explicit snapshot; defaults to results/remote")
    parser.add_argument("--output", type=Path, help="Default: reports/ outside immutable sync snapshots")
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "LATEST.json").exists():
        from sync_sft import latest_snapshot
        input_root = latest_snapshot(root)
        default_output = root / "reports"
    else:
        input_root = root
        default_output = (root.parent.parent / "reports" if root.parent.name == "snapshots" and (root / "SYNC_MANIFEST.json").exists()
                          else root / "reports")
    result = aggregate(input_root)
    output = args.output or default_output
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("aggregate.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
                          ("REPORT.md", render_report(result))):
        path = output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"complete": result["complete"], "states": {key: value["state"] for key, value in result["arms"].items()},
                      "report": str((output / "REPORT.md").resolve())}))


if __name__ == "__main__":
    main()
