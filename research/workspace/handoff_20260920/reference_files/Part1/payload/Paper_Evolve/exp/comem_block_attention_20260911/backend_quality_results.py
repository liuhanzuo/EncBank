"""Strict, standard-library aggregation for the paired Qasper-train QA pilot.

Scores are fractions internally and percentages in explicit *_percent fields.
Quality differences are observations, not an equivalence or speed gate. Failed
or partial runs never receive the complete-run metrics object.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import string
from collections import Counter

BRANCHES = ("decode_v2", "backend_v3")
RECORD_SCHEMA = "backend-quality-record-v1"
SUMMARY_SCHEMA = "backend-quality-summary-v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_json(value, name="record"):
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name}: nonfinite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), f"{name}: non-string key")
            _finite_json(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item, name)
    else:
        _require(value is None or isinstance(value, (str, int, bool)),
                 f"{name}: non-JSON value")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sources(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and bool(key) and _sha(item) for key, item in value.items())


def _ids(value, *, nonempty=False):
    return isinstance(value, list) and (bool(value) or not nonempty) and all(
        type(item) is int and item >= 0 for item in value)


def _normalize(text):
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_prediction(prediction, references):
    """Same normalization and best-reference scoring as evaluate_sft.py."""
    predicted = _normalize(prediction).split()
    f1, em = [], []
    for reference in references:
        expected = _normalize(reference).split()
        em.append(float(predicted == expected))
        if not predicted or not expected:
            f1.append(float(predicted == expected))
        else:
            overlap = sum((Counter(predicted) & Counter(expected)).values())
            precision, recall = overlap / len(predicted), overlap / len(expected)
            f1.append(2 * precision * recall / (precision + recall) if overlap else 0.)
    return {"token_f1": max(f1), "exact_match": max(em)}


def _validate_record(record, ordered_ids, run_id, recipe_sha256, source_sha256):
    _require(isinstance(record, dict), "record must be an object")
    _finite_json(record)
    for field, expected in (("schema", RECORD_SCHEMA), ("run_id", run_id),
            ("recipe_sha256", recipe_sha256), ("source_sha256", source_sha256)):
        _require(record.get(field) == expected, f"record {field} mismatch")
    ordinal = record.get("ordinal")
    _require(type(ordinal) is int and 0 <= ordinal < len(ordered_ids), "invalid ordinal")
    _require(record.get("id") == ordered_ids[ordinal], "ID/order mismatch")
    _require(record.get("branch") in BRANCHES, "invalid branch")
    _require(record.get("status") in ("complete", "failed"), "invalid record status")
    for field in ("document_id", "source"):
        _require(isinstance(record.get(field), str) and bool(record[field]), f"invalid {field}")
    refs = record.get("references")
    _require(isinstance(refs, list) and refs and all(isinstance(x, str) for x in refs),
             "references must be a nonempty list of strings")
    for field in ("selected_chunk_indices", "prompt_ids", "probe_indices"):
        _require(_ids(record.get(field), nonempty=True), f"invalid {field}")
    for field in ("selected_chunk_indices", "probe_indices"):
        _require(len(record[field]) == len(set(record[field])), f"duplicate {field}")
    _require(all(i < len(record["prompt_ids"]) for i in record["probe_indices"]),
             "probe index outside prompt")
    _require(type(record.get("candidate_tokens")) is int and record["candidate_tokens"] > 0,
             "invalid candidate_tokens")
    _require(type(record.get("max_new_tokens")) is int and 0 < record["max_new_tokens"] <= 128,
             "invalid max_new_tokens")
    if "question" in record:
        _require(isinstance(record["question"], str), "question must be text")
    for field in ("formal_inference_timing", "formal_inference_memory"):
        _require(record.get(field, False) is False, f"{field} cannot be true")
    if record["status"] == "failed":
        _require(bool(record.get("error")), "failed record lacks error")
        if "generated_ids" in record:
            _require(_ids(record["generated_ids"]), "invalid partial generated_ids")
        return
    _require(isinstance(record.get("prediction"), str), "prediction must be text")
    generated, stops = record.get("generated_ids"), record.get("stop_token_ids")
    _require(_ids(generated, nonempty=True), "invalid generated_ids")
    _require(_ids(stops, nonempty=True) and len(stops) == len(set(stops)), "invalid stop_token_ids")
    _require(len(generated) <= record["max_new_tokens"], "generation exceeds limit")
    finish = record.get("finish_reason")
    if finish == "eos":
        _require(generated[-1] in stops and record.get("eos_token_id") == generated[-1],
                 "EOS receipt mismatch")
        _require(not any(x in stops for x in generated[:-1]), "generation continues after EOS")
    elif finish == "max_new_tokens":
        _require(len(generated) == record["max_new_tokens"] and record.get("eos_token_id") is None,
                 "limit receipt mismatch")
        _require(not any(x in stops for x in generated), "limit generation contains EOS")
    else:
        raise ValueError("invalid finish_reason")
    if "shared_writer_hj_unchanged" in record:
        _require(record["shared_writer_hj_unchanged"] is True, "shared writer input changed")
    expected_scores = score_prediction(record["prediction"], refs)
    for field, expected in expected_scores.items():
        actual = record.get(field)
        _require(type(actual) in (int, float) and math.isfinite(actual) and 0 <= actual <= 1,
                 f"invalid {field}: expected fraction in [0,1]")
        _require(math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12),
                 f"{field} does not match prediction/references")
    route = record.get("route_stats")
    _require(isinstance(route, dict), "missing route_stats")
    selected = route.get("selected_indices")
    _require(_ids(selected, nonempty=True) and len(selected) == len(set(selected)),
             "invalid route selected_indices")
    blocks = route.get("candidate_blocks")
    _require(type(blocks) is int and blocks == len(record["selected_chunk_indices"]),
             "route candidate_blocks mismatch")
    _require(all(i < blocks for i in selected), "route index outside candidates")
    _require(route.get("selected_blocks") == len(selected), "route selected_blocks mismatch")
    _require(route.get("candidate_tokens") == record["candidate_tokens"],
             "route candidate_tokens mismatch")
    _require(type(route.get("selected_tokens")) is int and
             0 < route["selected_tokens"] <= record["candidate_tokens"], "invalid selected_tokens")
    scores = route.get("block_scores")
    _require(isinstance(scores, list) and len(scores) == blocks and all(
        type(x) in (int, float) and math.isfinite(x) for x in scores), "invalid block_scores")


def _paired_input(left, right):
    fields = ("document_id", "source", "references", "selected_chunk_indices", "prompt_ids",
              "probe_indices", "candidate_tokens", "max_new_tokens")
    for field in fields:
        _require(left[field] == right[field], f"paired {field} mismatch for {left['id']}")
    if "question" in left or "question" in right:
        _require(left.get("question") == right.get("question"), "paired question mismatch")
    if left["status"] == right["status"] == "complete":
        _require(left["stop_token_ids"] == right["stop_token_ids"], "paired stop IDs mismatch")


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _observations(pairs):
    per_question = []
    metrics = {branch: {"n": len(pairs)} for branch in BRANCHES}
    for branch in BRANCHES:
        rows = [pair[branch] for pair in pairs]
        for name in ("token_f1", "exact_match"):
            value = _mean([row[name] for row in rows])
            metrics[branch][name] = value
            metrics[branch][name + "_percent"] = None if value is None else 100 * value
        metrics[branch]["finish_reasons"] = {
            reason: sum(row["finish_reason"] == reason for row in rows)
            for reason in ("eos", "max_new_tokens")}
        metrics[branch]["mean_generated_tokens"] = _mean([len(row["generated_ids"]) for row in rows])
    for pair in pairs:
        left, right = (pair[branch] for branch in BRANCHES)
        ls, rs = left["route_stats"], right["route_stats"]
        per_question.append({"id": left["id"], "ordinal": left["ordinal"],
            "token_f1_delta": right["token_f1"] - left["token_f1"],
            "exact_match_delta": right["exact_match"] - left["exact_match"],
            "prediction_equal": left["prediction"] == right["prediction"],
            "normalized_prediction_equal": _normalize(left["prediction"]) == _normalize(right["prediction"]),
            "generated_ids_equal": left["generated_ids"] == right["generated_ids"],
            "route_selected_indices_equal": ls["selected_indices"] == rs["selected_indices"],
            "reference_route_selected_indices": ls["selected_indices"],
            "candidate_route_selected_indices": rs["selected_indices"],
            "block_scores_max_abs_difference": max(abs(a-b) for a, b in
                zip(ls["block_scores"], rs["block_scores"])),
            "finish_reason_equal": left["finish_reason"] == right["finish_reason"]})
    paired = {"n": len(pairs), "delta_direction": "backend_v3 minus decode_v2"}
    for name in ("token_f1", "exact_match"):
        values = [row[name + "_delta"] for row in per_question]
        value = _mean(values)
        paired[name] = {"mean_delta": value,
            "mean_delta_percentage_points": None if value is None else 100 * value,
            "wins": sum(x > 0 for x in values), "ties": sum(x == 0 for x in values),
            "losses": sum(x < 0 for x in values)}
    for field in ("prediction_equal", "normalized_prediction_equal", "generated_ids_equal",
                  "route_selected_indices_equal", "finish_reason_equal"):
        count = sum(row[field] for row in per_question)
        paired[field] = {"equal": count, "different": len(pairs) - count}
    paired["block_scores_max_abs_difference"] = max(
        (row["block_scores_max_abs_difference"] for row in per_question), default=None)
    return {"branches": metrics, "paired": paired, "per_question": per_question}


def aggregate_paired(records, ordered_ids, *, run_id, recipe_sha256, source_sha256):
    """Return complete/pending/partial/failed/invalid without hiding partial data."""
    records, ordered_ids = list(records), list(ordered_ids)
    result = {"schema": SUMMARY_SCHEMA, "run_id": run_id, "recipe_sha256": recipe_sha256,
        "source_sha256": source_sha256, "ordered_ids": ordered_ids, "status": "invalid",
        "complete": False, "expected_pairs": len(ordered_ids), "records_seen": len(records),
        "scope": "Qasper-train document-held-out pilot; not official Qasper test",
        "formal_inference_timing": False, "formal_inference_memory": False,
        "score_scale": "fractions [0,1]; *_percent uses 0..100; deltas candidate minus reference",
        "errors": [], "failures": [], "missing": [], "metrics": None}
    by_id = {}
    try:
        _require(ordered_ids and all(isinstance(i, str) and i for i in ordered_ids)
                 and len(set(ordered_ids)) == len(ordered_ids), "invalid ordered_ids")
        _require(isinstance(run_id, str) and run_id, "invalid run_id")
        _require(_sha(recipe_sha256) and _sources(source_sha256), "invalid recipe/source hashes")
        seen_ordinals = {branch: [] for branch in BRANCHES}
        for index, record in enumerate(records):
            try:
                _validate_record(record, ordered_ids, run_id, recipe_sha256, source_sha256)
                branch, ordinal = record["branch"], record["ordinal"]
                _require(ordinal not in seen_ordinals[branch], "duplicate branch/ID")
                _require(not seen_ordinals[branch] or ordinal > seen_ordinals[branch][-1],
                         "branch records out of order")
                seen_ordinals[branch].append(ordinal)
                by_id.setdefault(record["id"], {})[branch] = record
                if record["status"] == "failed":
                    result["failures"].append({"id": record["id"], "branch": branch, "error": record["error"]})
            except (ValueError, KeyError, TypeError) as exc:
                result["errors"].append(f"record {index}: {exc}")
        for identity in ordered_ids:
            pair = by_id.get(identity, {})
            for branch in BRANCHES:
                if branch not in pair:
                    result["missing"].append({"id": identity, "branch": branch})
            if len(pair) == 2:
                try:
                    _paired_input(pair[BRANCHES[0]], pair[BRANCHES[1]])
                except ValueError as exc:
                    result["errors"].append(str(exc))
    except (ValueError, TypeError) as exc:
        result["errors"].append(str(exc))
    result["complete_records"] = sum(row["status"] == "complete" for pair in by_id.values() for row in pair.values())
    result["completed_pairs"] = sum(len(pair) == 2 and all(row["status"] == "complete" for row in pair.values())
                                    for pair in by_id.values())
    if result["errors"]:
        return result
    pairs = [by_id[identity] for identity in ordered_ids if identity in by_id and
             len(by_id[identity]) == 2 and all(row["status"] == "complete" for row in by_id[identity].values())]
    if result["failures"]:
        result["status"] = "failed"
    elif result["missing"]:
        result["status"] = "partial" if records else "pending"
    else:
        result["status"], result["complete"] = "complete", True
    observations = _observations(pairs)
    if result["complete"]:
        result["metrics"] = observations
    else:
        result["partial_observations"] = {"scope": "Available complete matched pairs only; not full-run scores",
                                          **observations}
    return result


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    _finite_json(value)
    return value


def validate_complete_result(path_or_result, expected_recipe, *, expected_source_sha256=None):
    """Return True or raise ValueError. The directory form rechecks raw records.

    An object must contain metadata (or top-level recipe/identity), status,
    summary and records. expected_recipe may be a subset, but mode/ordered_ids
    are mandatory. The queue should also pass its pinned expected source map.
    """
    try:
        if isinstance(path_or_result, (str, Path)):
            path = Path(path_or_result)
            if path.is_dir():
                metadata = _read_json(path / "metadata.json")
                status = _read_json(path / "status.json")
                summary = _read_json(path / "summary.json")
                with (path / "records.jsonl").open(encoding="utf-8") as stream:
                    records = [json.loads(line) for line in stream if line.strip()]
                value = {"metadata": metadata, "status": status, "summary": summary, "records": records}
            else:
                value = _read_json(path)
        else:
            value = path_or_result
        _require(isinstance(value, dict), "completion result must be an object")
        _finite_json(value)
        metadata = value.get("metadata", value)
        recipe = metadata["recipe"]
        _require(isinstance(expected_recipe, dict) and "mode" in expected_recipe and
                 "ordered_ids" in expected_recipe, "expected recipe needs mode and ordered_ids")
        for key, expected in expected_recipe.items():
            _require(recipe.get(key) == expected, f"expected recipe {key} mismatch")
        _require(recipe["mode"] in ("smoke", "full"), "invalid quality mode")
        count = 8 if recipe["mode"] == "smoke" else 99
        _require(len(recipe["ordered_ids"]) == count, "mode/ordered ID count mismatch")
        for key, expected in (("seed", 42), ("max_new_tokens", 128), ("j", 12), ("m", 16),
                ("probe_mode", "dense"), ("retain_ratio", 1.), ("rank", 32), ("alpha", 32.),
                ("branches", list(BRANCHES)), ("decoding", "independent-free-greedy-natural-eos"),
                ("teacher_forced_ce", False)):
            _require(recipe.get(key) == expected, f"pilot recipe {key} mismatch")
        recipe_sha, source, run_id = (metadata[key] for key in ("recipe_sha256", "source_sha256", "run_id"))
        _require(canonical_hash(recipe) == recipe_sha, "recipe hash does not match recipe")
        _require(_sources(source), "invalid source hash map")
        if expected_source_sha256 is not None:
            _require(source == expected_source_sha256, "expected source hash map mismatch")
        _require(source.get(Path(__file__).name) == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "quality results helper source mismatch")
        status = value["status"]
        if isinstance(status, str):
            status = {"status": status, "run_id": value.get("run_id")}
        _require(status.get("status") == "complete" and status.get("run_id") == run_id,
                 "worker status is not complete for this run")
        _require(status.get("records_completed") == count * 2, "worker completed record count mismatch")
        for item in (metadata, status, value["summary"]):
            _require(item.get("formal_inference_timing", False) is False and
                     item.get("formal_inference_memory", False) is False, "quality cannot be formal infra")
        calculated = aggregate_paired(value["records"], recipe["ordered_ids"], run_id=run_id,
            recipe_sha256=recipe_sha, source_sha256=source)
        _require(calculated["complete"], f"records are not complete: {calculated['status']} {calculated['errors']}")
        recorded = value["summary"]
        for key, expected in calculated.items():
            _require(recorded.get(key) == expected, f"summary {key} differs from raw records")
        profiles = recorded.get("profiles")
        _require(isinstance(profiles, dict) and set(profiles) == set(BRANCHES), "missing paired backend profiles")
        for branch, receipt in profiles.items():
            _require(receipt.get("status") == "complete" and receipt.get("prefix_equal") is True,
                     f"{branch} first-example profile failed")
            prefix, actual = receipt.get("normal_generation_prefix"), receipt.get("profile_generated_ids")
            _require(_ids(prefix, nonempty=True) and prefix == actual, "profile prefix receipt mismatch")
            first = next(row for row in value["records"] if row["branch"] == branch)
            _require(prefix == first["generated_ids"][:len(prefix)], "profile does not match first answer")
            profile = receipt.get("profile", {})
            _require(profile.get("status") == "complete", "profile body not complete")
            _require(profile.get("timing_eligible") is False, "profile cannot be a formal timing result")
            evidence = profile.get("backend_operator_evidence")
            _require(isinstance(evidence, list) and evidence and all(
                isinstance(row, dict) and isinstance(row.get("name"), str) and
                type(row.get("count")) is int and row["count"] > 0 for row in evidence),
                "missing actual backend operator evidence")
            pm = profile.get("metadata", {})
            for key, expected in (("run_id", run_id), ("id", recipe["ordered_ids"][0]),
                                  ("branch", branch), ("recipe_sha256", recipe_sha)):
                _require(pm.get(key) == expected, f"profile {key} mismatch")
            _require(receipt.get("formal_inference_timing") is False and
                     receipt.get("formal_inference_memory") is False, "profile is not diagnostic only")
        return True
    except (OSError, KeyError, TypeError, AttributeError, IndexError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete or invalid backend quality result: {exc}") from exc
