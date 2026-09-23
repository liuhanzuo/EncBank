"""Audit and summarize the four-arm Qasper-train pilot without loading torch.

Examples:
  python aggregate_sparse.py --input /.../outputs/sparse_comem_20260911
  python aggregate_sparse.py --input mirror --data-dir mirror/data --metadata-only

Default completion requires a local nonempty last.pt, but never deserializes it.
Run on the source server and copy SUMMARY.json/REPORT.md for a weight-free mirror.
--metadata-only audits available records without asserting checkpoint completion.
Remote paths can be remapped using repeated --path-map REMOTE_PREFIX=LOCAL_PREFIX.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import string

ARMS = ("D0", "A", "B", "D1")
PROTOCOL = "Held-out Qasper-train pilot (planned 99 dev examples); not official Qasper test or LongBench."


class Pending(Exception):
    """An artifact has not arrived yet; this is not a fabricated zero score."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite numeric value in artifact")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def read_json(path):
    if not path.is_file():
        raise Pending(f"Missing {path.name}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    finite_tree(value)
    return value


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def resolve_path(value, maps):
    text = str(value).replace("\\", "/")
    for remote, local in sorted(maps, key=lambda pair: -len(pair[0])):
        remote = remote.replace("\\", "/").rstrip("/")
        if text == remote or text.startswith(remote + "/"):
            return Path(local) / text[len(remote):].lstrip("/")
    return Path(value)


def read_prepared(recipe, split, data_dir, maps):
    path = Path(data_dir) / f"{split}.jsonl" if data_dir else resolve_path(recipe[split], maps)
    if not path.is_file():
        raise Pending(f"Prepared {split} unavailable: provide --data-dir or --path-map")
    require(sha256(path) == recipe[f"{split}_sha256"], f"Prepared {split} hash differs from training recipe")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    finite_tree(rows)
    by_id = {row["id"]: row for row in rows}
    require(len(by_id) == len(rows) and bool(rows), f"Duplicate/empty prepared {split} IDs")
    requested = recipe[f"{split}_ids"]
    require(len(set(requested)) == len(requested) and bool(requested), f"Duplicate/empty recipe {split} IDs")
    require(all(ident in by_id for ident in requested), f"Unknown prepared {split} IDs")
    selected = [by_id[ident] for ident in requested]
    for row in selected:
        require(row["source"] == "allenai/qasper:train:v0.3" and row["split"] == split,
                "Only document-held-out official-Qasper-train pilot is supported")
        require(isinstance(row["question"], str) and row["question"].strip(), "Missing prepared question")
        require(isinstance(row["references"], list) and bool(row["references"])
                and all(isinstance(ref, str) for ref in row["references"]), "Invalid prepared references")
        require(row["answer_ids"] and row["prompt_ids"] and not row.get("answer_truncated"), "Invalid/shortened target")
        require(row["document_chunks"] and all(row["document_chunks"]), "Empty document block")
    return selected


def normalize(text):
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction, reference):
    predicted, expected = normalize(prediction).split(), normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected))


def mean(values):
    values = list(values)
    require(bool(values), "Cannot average empty records")
    return sum(values) / len(values)


def close(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), f"{name} mismatch")


def route_audit(route, prepared, recipe, model):
    chunks = prepared["document_chunks"]
    candidate = sum(map(len, chunks))
    selected = route["selected_indices"]
    require(selected == sorted(set(selected)) and all(type(i) is int and 0 <= i < len(chunks) for i in selected),
            "Invalid selected block indices")
    kept = sum(len(chunks[i]) for i in selected)
    rho, j, m, layers = recipe["retain_ratio"], recipe["j"], recipe["m"], model["num_hidden_layers"]
    sink = route["sink_tokens"]
    require(sink == 1, "Trainer must use the declared single sink token")
    require(route["selection_source"] == ("all-blocks" if rho == 1 else "prompt-attention-mass"),
            "Selection is not the declared prompt-only route")
    require(route["probe_indices"] == prepared["probe_indices"] and route["probe_rows_fallback"] is False,
            "Routing must use prepared question-only probe rows")
    require(route["probe_mode"] == recipe["probe_mode"] and route["resume_j"] == j
            and route["fusion_layer"] == m, "Route configuration mismatch")
    require(route["candidate_blocks"] == len(chunks) and route["selected_blocks"] == len(selected)
            and route["candidate_tokens"] == candidate and route["selected_tokens"] == kept,
            "Route token/block accounting mismatch")
    close(route["target_retain_ratio"], rho, "Target rho")
    close(route["actual_retain_ratio"], kept / candidate, "Realized rho")
    budget = candidate if rho == 1 else math.floor(candidate * rho)
    require(route["token_budget"] == budget and kept <= budget and route["budget_overflow_tokens"] == 0,
            "Route exceeds document token budget")
    require(rho != 1 or selected == list(range(len(chunks))), "Full-retention arm dropped blocks")
    expected_layers = {str(layer): (candidate if layer < m else kept) + sink for layer in range(j, layers)}
    require(route["document_kv_tokens_by_layer"] == expected_layers, "Actual document KV layer lengths differ")
    require(route["unselected_late_kv_tokens"] == 0, "Unselected late KV is still materialized")
    head_dim = model.get("head_dim") or model["hidden_size"] // model["num_attention_heads"]
    bytes_per_layer_token = 2 * model["num_key_value_heads"] * head_dim * 2  # K/V, BF16.
    expected_bytes = bytes_per_layer_token * sum(expected_layers.values())
    require(route["document_kv_bytes"] == expected_bytes, "Tensor KV bytes differ from BF16 shape accounting")
    baseline = bytes_per_layer_token * (layers - j) * (candidate + sink)
    return dict(candidate_tokens=candidate, selected_tokens=kept, actual_retain_ratio=kept / candidate,
                analytic_document_kv_bytes=expected_bytes, tensor_document_kv_bytes=route["document_kv_bytes"],
                dense_comem_document_kv_bytes=baseline,
                document_kv_reduction_fraction=1 - expected_bytes / baseline)


def summarize(records):
    count = len(records)
    require(count > 0, "Empty evaluation")
    tokens = sum(row["answer_ce_tokens"] for row in records)
    result = dict(examples=count, token_f1=mean(row["token_f1"] for row in records),
                  exact_match=mean(row["exact_match"] for row in records),
                  answer_ce=sum(row["answer_ce"] * row["answer_ce_tokens"] for row in records) / tokens,
                  answer_ce_tokens=tokens, score_scale="0-to-1")
    keys = ("candidate_tokens", "selected_tokens", "actual_retain_ratio", "analytic_document_kv_bytes",
            "tensor_document_kv_bytes", "dense_comem_document_kv_bytes", "document_kv_reduction_fraction")
    result.update({f"mean_{key}": mean(row[key] for row in records) for key in keys})
    result["pooled_token_retain_ratio"] = sum(r["selected_tokens"] for r in records) / sum(r["candidate_tokens"] for r in records)
    result["pooled_document_kv_reduction_fraction"] = 1 - sum(r["tensor_document_kv_bytes"] for r in records) / sum(r["dense_comem_document_kv_bytes"] for r in records)
    return result


def evaluation_audit(path, prepared_rows, recipe, model, limit):
    value = read_json(path)
    records, summary = value["records"], value["summary"]
    expected = sorted(prepared_rows, key=lambda row: hashlib.sha256(f"{recipe['seed']}:{row['id']}".encode()).hexdigest())[:limit]
    require([row["id"] for row in records] == [row["id"] for row in expected], "Evaluation dev IDs/order differ")
    require(summary["score_scale"] == "0-to-1" and summary["decoding"] == "greedy-natural-eos"
            and summary["max_new_tokens"] == recipe["max_new_tokens"]
            and summary["formal_inference_timing"] is False, "Wrong score scale/decoding protocol")
    checked = []
    for row, prepared in zip(records, expected):
        require(row["references"] == prepared["references"] and row["document_id"] == prepared["document_id"]
                and row["source"] == prepared["source"] and row["probe_indices"] == prepared["probe_indices"],
                "Question identity/references/probe source differs from hashed prepared data")
        require("question" not in row or row["question"] == prepared["question"], "Evaluation question differs")
        require(isinstance(row["prediction"], str), "Prediction must be a string")
        f1 = max(token_f1(row["prediction"], ref) for ref in row["references"])
        em = max(float(normalize(row["prediction"]) == normalize(ref)) for ref in row["references"])
        close(row["token_f1"], f1, "Recomputed F1")
        close(row["exact_match"], em, "Recomputed EM")
        require(type(row["answer_ce"]) in (int, float) and row["answer_ce"] >= 0
                and type(row["answer_ce_tokens"]) is int and row["answer_ce_tokens"] == len(prepared["answer_ids"]),
                "Invalid answer CE or target length")
        generated = row["generated_ids"]
        require(isinstance(generated, list) and 1 <= len(generated) <= recipe["max_new_tokens"]
                and all(type(token) is int and 0 <= token < model["vocab_size"] for token in generated), "Invalid generated IDs")
        require(row["finish_reason"] in ("eos", "max_new_tokens") and
                (row["finish_reason"] != "max_new_tokens" or len(generated) == recipe["max_new_tokens"]), "Invalid generation termination")
        stats = route_audit(row["route_stats"], prepared, recipe, model)
        ce_stats = route_audit(row["ce_route_stats"], prepared, recipe, model)
        require(stats == ce_stats and row["route_stats"]["selected_indices"] == row["ce_route_stats"]["selected_indices"],
                "Teacher-forced CE and prompt generation selected different blocks")
        require(row["candidate_tokens"] == stats["candidate_tokens"], "Candidate input count mismatch")
        checked.append(dict(id=row["id"], token_f1=f1, exact_match=em, answer_ce=row["answer_ce"],
                            answer_ce_tokens=row["answer_ce_tokens"], **stats))
    result = summarize(checked)
    require(summary["examples"] == len(checked), "Evaluation example count mismatch")
    for key in ("token_f1", "exact_match", "answer_ce"):
        close(summary[key], result[key], f"Reported mean {key}")
    return checked, result


def training_audit(directory, state, recipe, train):
    steps, accum = recipe["steps"], recipe["grad_accum"]
    require(state["step"] == steps and state["target_steps"] == steps and state["cursor"] == steps * accum,
            "Incomplete/different optimizer and example budget")
    path = directory / "train.jsonl"
    if not path.is_file():
        raise Pending("Training consumption log not yet available")
    logs = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    finite_tree(logs)
    require([row["step"] for row in logs] == list(range(1, steps + 1)), "Missing/duplicate training steps")
    consumed = []
    for epoch in range(math.ceil(steps * accum / len(train))):
        order = list(range(len(train)))
        random.Random(recipe["seed"] + epoch).shuffle(order)
        consumed.extend(train[index] for index in order)
    raw, targets = 0, 0
    for index, log in enumerate(logs):
        batch = consumed[index * accum:(index + 1) * accum]
        require([row["id"] for row in log["routes"]] == [row["id"] for row in batch], "Training example order mismatch")
        raw += sum(sum(map(len, row["document_chunks"])) + len(row["prompt_ids"]) + len(row["answer_ids"]) for row in batch)
        targets += sum(len(row["answer_ids"]) for row in batch)
        require(log["cursor"] == (index + 1) * accum and log["raw_tokens"] == raw and log["target_tokens"] == targets,
                "Training input/target-token accounting mismatch")
        require(log["loss"] >= 0 and log["grad_norm"] >= 0, "Invalid training loss/gradient")
    require(state["raw_tokens"] == raw and state["target_tokens"] == targets, "Final consumed-token budget mismatch")
    return dict(steps=steps, examples_consumed=steps * accum, raw_tokens=raw, target_tokens=targets)


def paired_delta(before, after, include_records=False):
    lookup = {row["id"]: row for row in after}
    require(all(row["id"] in lookup for row in before), "Paired evaluation IDs are not a subset of final IDs")
    matching = [lookup[row["id"]] for row in before]
    old, new = summarize(before), summarize(matching)
    result = dict(examples=len(before), ids=[row["id"] for row in before],
                  token_f1_delta=new["token_f1"] - old["token_f1"],
                  exact_match_delta=new["exact_match"] - old["exact_match"],
                  answer_ce_delta=new["answer_ce"] - old["answer_ce"],
                  before=old, after_on_identical_ids=new)
    if include_records:
        result["records"] = [{"id": left["id"], **{f"{key}_delta": right[key] - left[key]
                             for key in ("token_f1", "exact_match", "answer_ce")}}
                             for left, right in zip(before, matching)]
    return result


def aggregate(root, data_dir=None, maps=(), metadata_only=False, include_pairs=False):
    root = Path(root)
    output = dict(schema="sparse-comem-pilot-summary-v1", generated_utc=datetime.now(timezone.utc).isoformat(),
                  input=str(root.resolve()), protocol=PROTOCOL, status="pending", comparison_valid=False,
                  formal_inference_timing=False, formal_inference_memory=False,
                  checkpoint_validation="not_requested_metadata_only" if metadata_only else "nonempty_file_without_deserialization",
                  notes=["F1/EM are 0-to-1 in JSON and percentages in Markdown; CE is weighted by answer tokens.",
                         "Initial-to-final deltas use identical initial dev IDs, not 25 initial versus 99 final means.",
                         "Document KV bytes are BF16 tensor/shape accounting including the sink, not GPU peak memory or latency.",
                         "Weights, query/answer KV, temporary tensors and additional hot/cold cache storage are excluded.",
                         "Results retain source GPU metadata; copying artifacts locally does not create a local GPU benchmark."],
                  arms={}, errors=[])
    validated = {}
    for arm in ARMS:
        item = dict(status="pending", reason="Waiting for training artifacts")
        output["arms"][arm] = item
        directory = root / "train" / arm
        try:
            state = read_json(directory / "status.json")
            item.update(phase=state.get("phase"), step=state.get("step"), target_steps=state.get("target_steps"))
            if state.get("phase") == "failed":
                raise ValueError(f"Trainer failed: {state.get('error', 'inspect worker.log')}")
            if state.get("complete") is not True or state.get("phase") != "complete":
                raise Pending("Training/final evaluation has not completed")
            metadata = read_json(directory / "metadata.json")
            recipe, model = metadata["recipe"], metadata["model"]
            require(recipe["arm"] == arm and state["arm"] == arm and recipe["smoke"] is False, "Wrong arm or smoke run")
            require(recipe["probe_mode"] == ("block" if arm in ("A", "B") else "dense")
                    and recipe["retain_ratio"] == (1.0 if arm in ("D0", "A") else recipe["rho"]), "Arm configuration mismatch")
            require(0 < recipe["j"] < recipe["m"] <= model["num_hidden_layers"] and 0 < recipe["rho"] <= 1, "Invalid depth/rho")
            require(metadata["formal_inference_timing"] is False and metadata["formal_inference_memory"] is False,
                    "Training artifacts must not be advertised as formal inference measurements")
            gradient = state["gradient_check"]
            tensors = (model["num_hidden_layers"] - recipe["j"]) * 7 * 2
            require(type(gradient["reader_norm"]) in (int, float) and gradient["reader_norm"] > 0
                    and gradient["frozen_base_has_grad"] is False
                    and gradient["trainable_tensors_with_grad"] == tensors
                    and gradient["total_trainable_tensors"] == tensors, "Reader gradient completion check failed")
            checkpoint = directory / "last.pt"  # Never follow remote state['checkpoint'] in a local mirror.
            checkpoint_present = checkpoint.is_file() and checkpoint.stat().st_size > 0
            if not checkpoint_present and not metadata_only:
                raise Pending("Nonempty last.pt unavailable; aggregate on source server or use --metadata-only")
            train = read_prepared(recipe, "train", data_dir, maps)
            dev = read_prepared(recipe, "dev", data_dir, maps)
            require(not ({row["document_id"] for row in train} & {row["document_id"] for row in dev}), "Train/dev document leakage")
            budget = training_audit(directory, state, recipe, train)
            final_rows, final = evaluation_audit(directory / f"eval_step{recipe['steps']}.json", dev, recipe, model, recipe["final_eval_limit"])
            item.update(status="metadata_only" if metadata_only else "complete", reason=None,
                        checkpoint_present=checkpoint_present, source_gpu=metadata["gpu"], budget=budget, final=final)
            initial_path = directory / "eval_step0.json"
            if initial_path.is_file():
                initial_rows, initial = evaluation_audit(initial_path, dev, recipe, model, recipe["eval_limit"])
                item["initial"] = initial
                item["initial_to_final_matched"] = paired_delta(initial_rows, final_rows, include_pairs)
            else:
                item["initial_to_final_matched"] = None
                item["initial_note"] = "No initial evaluation artifact; no initial-to-final delta claimed."
            # Match all data/training/decoding settings, allowing only the actual arm treatment.
            match_recipe = {key: value for key, value in recipe.items()
                            if key not in {"arm", "probe_mode", "retain_ratio", "train", "dev", "init_adapter", "model"}}
            validated[arm] = dict(recipe=match_recipe, model=model, budget=budget, rows=final_rows,
                                  identities=[{key: row[key] for key in ("id", "question", "references", "prompt_ids", "document_chunks", "answer_ids", "probe_indices")} for row in dev])
        except Pending as error:
            item.update(status="pending", reason=str(error))
        except (ValueError, KeyError, TypeError, AttributeError, OSError, IndexError, ZeroDivisionError) as error:
            item.update(status="invalid", reason=str(error))
            output["errors"].append(f"{arm}: {error}")
    if len(validated) == len(ARMS):
        baseline = validated["D0"]
        for arm in ARMS[1:]:
            for key in ("recipe", "model", "budget", "identities"):
                if validated[arm][key] != baseline[key]:
                    output["errors"].append(f"{arm} versus D0: unmatched {key}")
        if not output["errors"] and not metadata_only:
            output.update(status="complete", comparison_valid=True,
                          final_vs_D0={arm: paired_delta(baseline["rows"], validated[arm]["rows"], include_pairs) for arm in ARMS[1:]})
        elif not output["errors"]:
            output["status"] = "metadata_only"
    if output["errors"]:
        output["status"] = "invalid"
    return output


def render_report(summary):
    lines = ["# CoMem + Sparse Attention pilot", "", PROTOCOL, "",
             f"Status: **{summary['status']}**. Matched four-arm comparison validated: **{summary['comparison_valid']}**.", "",
             "This report contains no 5090 speed or GPU peak-memory result. A local copy retains its source GPU provenance.", "",
             "| Arm | Status | F1 (%) | EM (%) | Answer CE | Candidate / selected tokens (mean) | Document KV (MiB, tensor) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, value in summary["arms"].items():
        final = value.get("final")
        if final:
            lines.append(f"| {arm} | {value['status']} | {final['token_f1'] * 100:.2f} | {final['exact_match'] * 100:.2f} | {final['answer_ce']:.4f} | {final['mean_candidate_tokens']:.1f} / {final['mean_selected_tokens']:.1f} | {final['mean_tensor_document_kv_bytes'] / 2**20:.3f} |")
        else:
            lines.append(f"| {arm} | {value['status']} | — | — | — | — | — |")
    lines += ["", "Rows can finish independently; do not interpret an incomplete or invalid set as the matched four-arm result.", ""]
    for arm, value in summary["arms"].items():
        if value.get("reason"):
            lines.append(f"- {arm}: {value['reason']}")
        if value.get("source_gpu"):
            lines.append(f"- {arm} source GPU: {value['source_gpu']}; consumed raw / supervised tokens: {value['budget']['raw_tokens']} / {value['budget']['target_tokens']}.")
    if summary["errors"]:
        lines += ["", "Validation errors; no matched conclusion:", ""] + [f"- {error}" for error in summary["errors"]]
    pairs = summary.get("final_vs_D0")
    if pairs:
        lines += ["", "Final versus D0 on identical examples:", "", "| Arm | Paired N | F1 delta (pp) | EM delta (pp) | CE delta |", "| --- | ---: | ---: | ---: | ---: |"]
        lines += [f"| {arm} | {delta['examples']} | {delta['token_f1_delta'] * 100:+.2f} | {delta['exact_match_delta'] * 100:+.2f} | {delta['answer_ce_delta']:+.4f} |" for arm, delta in pairs.items()]
    if any(value.get("initial_to_final_matched") for value in summary["arms"].values()):
        lines += ["", "Initial to final on the identical initial-evaluation subset (not a comparison of unequal dev sets):", "", "| Arm | Paired N | F1 delta (pp) | EM delta (pp) | CE delta |", "| --- | ---: | ---: | ---: | ---: |"]
        for arm, value in summary["arms"].items():
            delta = value.get("initial_to_final_matched")
            if delta:
                lines.append(f"| {arm} | {delta['examples']} | {delta['token_f1_delta'] * 100:+.2f} | {delta['exact_match_delta'] * 100:+.2f} | {delta['answer_ce_delta']:+.4f} |")
    lines += ["", "Accounting and scope:", ""] + [f"- {note}" for note in summary["notes"]]
    lines += ["", "KV formula: `2 × KV_heads × head_dim × 2 BF16 bytes × [(m-j)(candidate+sink) + (L-m)(selected+sink)]`.",
              "Analytical bytes must equal the logged tensor bytes. This is live document-KV accounting, not allocator peak, resident cache capacity, or measured acceleration.", ""]
    return "\n".join(lines)


def write_outputs(summary, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in (("SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("REPORT.md", render_report(summary))):
        path = destination / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Default: --input / aggregate")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--path-map", action="append", default=[], metavar="REMOTE=LOCAL")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include-pairs", action="store_true")
    args = parser.parse_args()
    maps = []
    for entry in args.path_map:
        if "=" not in entry or not all(entry.split("=", 1)):
            parser.error("--path-map requires nonempty REMOTE_PREFIX=LOCAL_PREFIX")
        maps.append(tuple(entry.split("=", 1)))
    result = aggregate(args.input, args.data_dir, maps, args.metadata_only, args.include_pairs)
    destination = args.out or args.input / "aggregate"
    write_outputs(result, destination)
    print(json.dumps(dict(status=result["status"], comparison_valid=result["comparison_valid"],
                          arms={arm: value["status"] for arm, value in result["arms"].items()}, output=str(destination))))
    raise SystemExit(2 if result["status"] == "invalid" else 0)


if __name__ == "__main__":
    main()
