"""Independent finite SmolLM2 formal accuracy run: 350 x five, reuse smoke70.

--validate-only imports/cross-checks the 70 immutable smoke predictions on CPU.
--execute consumes verified records and generates only missing same-fixture rows.
Each failed attempt is retained; valid formal records survive explicit restart.
"""
from collections import Counter, defaultdict
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from non_qwen_download import ALLOWED_ROOT, REVISION, REPO
from non_qwen_smoke_driver import REMOTE, HERE, alive, write, now, score, load_fixtures
from non_qwen_smoke_bootstrap import completion_ok

SMOKE = REMOTE / "outputs/non_qwen_smol_20260909/smoke/attempts/0001"
FORMAL = REMOTE / "outputs/non_qwen_smol_20260909/formal"
ARMS = ("pub", "pub_sink", "fix_all", "j0", "fix_none")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def released_smoke():
    bootstrap_path = REMOTE / "outputs/bootstrap_non_qwen_smoke/status.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    audit = json.loads((SMOKE / "non_qwen_FINAL_AUDIT.json").read_text())
    assert completion_ok(SMOKE) and bootstrap["status"] == "completed" and bootstrap["child_exit_code"] == 0
    assert audit["status"] == "complete" and audit["smoke_generations"] == 70 and audit["extra_native_reference_sequences"] == 4
    assert audit["all_raw_reference_ids_equal"] and audit["all_generated_step_logits_pass"]
    historical = {bootstrap["pid"], audit["bootstrap_pid"], audit["model_child_pid"]}
    assert not any(alive(pid) for pid in historical), "smoke processes have not all naturally exited"
    return {"bootstrap_status": str(bootstrap_path), "final_audit": str(SMOKE / "non_qwen_FINAL_AUDIT.json"),
            "historical_pids": sorted(historical), "all_exited": True, "smoke_rows": 70, "excluded_native_references": 4}

def validate_prediction(row, fixture, arm, tokenizer):
    assert row["arm"] == arm and row["effective_j"] == (0 if arm == "j0" else 8)
    assert row["task"] == fixture["task"] and row["index"] == fixture["index"] and row["source_id"] == fixture["id"]
    assert row["model_revision"] == REVISION and row["dtype"] == "bfloat16" and row["device"] == "RTX3090"
    for field in ("fixture_sha256", "pack_ids_sha256", "query_ids_sha256", "context_ids_sha256",
                  "selected_ids_sha256", "source_prompt_sha256", "max_new_tokens", "eos_token_ids",
                  "ordered_selected_indices", "suppress_first_eos", "answers"):
        assert row[field] == fixture[field], (arm, fixture["id"], field)
    assert row.get("source_row_sha256") == fixture.get("source_row_sha256")
    assert row["actual_read_pack_tokens"] == fixture["read_pack_tokens"]
    ids = row["ids"]
    assert isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < len(tokenizer) for x in ids)
    assert row["generated_tokens"] == len(ids) <= row["max_new_tokens"]
    assert row["decoder_forwards"] == len(ids) - 1
    assert ids[0] not in fixture["eos_token_ids"] and not any(x in fixture["eos_token_ids"] for x in ids[:-1])
    eos = ids[-1] in fixture["eos_token_ids"]
    assert row["stopped_on_eos"] == eos and (eos or len(ids) == row["max_new_tokens"])
    assert row["first_eos_suppressed"] == (row["first_raw_argmax"] in fixture["eos_token_ids"])
    decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
    assert decoded == row["decoded_output"]
    actual_score = score(decoded, fixture)
    assert abs(actual_score - row["official_score"]) < 1e-12
    return True

def output_name(arm, fixture):
    return f"{arm}__{fixture['task']}__{fixture['index']:03d}.json"

def validate_record(record, fixture, arm, tokenizer):
    assert record["schema"] == "non-qwen-formal-prediction-v1" and record["seed"] == 42
    assert record["model_revision"] == REVISION and record["attn_impl"] == "sdpa"
    validate_prediction(record["prediction"], fixture, arm, tokenizer)
    provenance = record["provenance"]
    if provenance["kind"] == "verified_smoke":
        assert record["prediction"]["smoke_only"] is True
        path = Path(provenance["path"])
        assert path.resolve().parent == (SMOKE / "records").resolve()
        assert path.name == output_name(arm, fixture)
        assert sha(path) == provenance["sha256"]
        assert json.loads(path.read_text()) == record["prediction"]
    else:
        assert provenance["kind"] == "formal_generation" and provenance["attempt"] >= 1
        assert record["prediction"]["smoke_only"] is False
        rawpath = Path(provenance["raw_file"])
        expected = FORMAL / "attempts" / f"{provenance['attempt']:04d}" / "raw" / output_name(arm, fixture)
        assert rawpath.resolve() == expected.resolve()
        assert sha(rawpath) == provenance["raw_sha256"]
        assert json.loads(rawpath.read_text()) == record["prediction"]

def ingest_existing(out, all_rows, tokenizer):
    indexed = {(f["task"], f["index"]): f for f in all_rows}
    expected = {(arm, task, index) for arm in ARMS for task, index in indexed}
    assert len(indexed) == 350 and len(expected) == 1750
    smoke_files = sorted((SMOKE / "records").glob("*.json"))
    assert len(smoke_files) == 70
    for path in smoke_files:
        prediction = json.loads(path.read_text())
        key = (prediction["arm"], prediction["task"], prediction["index"])
        assert key in expected
        fixture = indexed[key[1:]]
        validate_prediction(prediction, fixture, key[0], tokenizer)
        assert path.name == output_name(key[0], fixture)
        target = out / "records" / path.name
        record = {"schema": "non-qwen-formal-prediction-v1", "model_revision": REVISION,
                  "seed": 42, "attn_impl": "sdpa", "prediction": prediction,
                  "provenance": {"kind": "verified_smoke", "path": str(path), "sha256": sha(path)}}
        if target.exists():
            assert json.loads(target.read_text()) == record, "existing smoke reuse changed"
        else:
            write(target, record)
    records = {}
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text())
        row = record["prediction"]
        key = (row["arm"], row["task"], row["index"])
        assert key in expected and key not in records
        assert path.name == output_name(key[0], indexed[key[1:]])
        validate_record(record, indexed[key[1:]], key[0], tokenizer)
        records[key] = record
    assert sum(r["provenance"]["kind"] == "verified_smoke" for r in records.values()) == 70
    return records

def summary(records):
    grouped = defaultdict(list)
    for (arm, task, _), record in records.items():
        grouped[(arm, task)].append(record["prediction"]["official_score"])
    cells = []
    for arm in ARMS:
        for task in ("qasper", "niah_single_2", "niah_multikey_1", "variable_tracking"):
            scores = grouped[(arm, task)]
            n = 200 if task == "qasper" else 50
            cells.append({"arm": arm, "task": task, "n": len(scores), "expected_n": n,
                          "complete": len(scores) == n,
                          "score_percent": 100 * sum(scores) / n if len(scores) == n else None})
    return {"formal_rows": len(records), "expected_rows": 1750,
            "verified_smoke_reused": sum(r["provenance"]["kind"] == "verified_smoke" for r in records.values()),
            "formal_generations": sum(r["provenance"]["kind"] == "formal_generation" for r in records.values()),
            "excluded_native_references": 4, "completed_cells": sum(c["complete"] for c in cells), "cells": cells}

def finalize_verified_records(out, base, records):
    """Recover the last-record/marker crash window, after full CPU validation."""
    totals = summary(records)
    assert totals["formal_rows"] == 1750 and totals["verified_smoke_reused"] == 70
    assert totals["formal_generations"] == 1680 and totals["completed_cells"] == 20
    marker = out / "non_qwen_FORMAL_COMPLETE.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        assert previous["status"] == "complete"
        for field in ("formal_rows", "expected_rows", "verified_smoke_reused", "formal_generations", "excluded_native_references", "completed_cells"):
            assert previous[field] == totals[field]
        assert previous["model_revision"] == REVISION and previous["j"] == 8
        assert len(previous["cells"]) == len(totals["cells"]) == 20
        for old, actual in zip(previous["cells"], totals["cells"]):
            assert all(old[k] == actual[k] for k in ("arm", "task", "n", "expected_n", "complete"))
            assert abs(old["score_percent"] - actual["score_percent"]) < 1e-9
        write(out / "status.json", previous)
        return previous
    result = {**base, **totals, "status": "complete", "child_pid": None,
              "active_arm": None, "active_task": None, "active_index": None,
              "completed_utc": now(), "recovered_from_validated_records_without_gpu": True,
              "new_this_attempt": 0, "previously_verified_rows": 1750}
    write(marker, result)
    write(out / "status.json", result)
    return result

def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.out.resolve() == FORMAL.resolve(), "formal output remains in its independent task directory"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ("" if args.validate_only else "1")
    assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    from non_qwen_explicit_reader import NonQwenExplicitReader
    config = AutoConfig.from_pretrained(ALLOWED_ROOT, local_files_only=True, trust_remote_code=False)
    ready, _, _ = load_fixtures(args.fixtures, config)
    all_rows = [json.loads(line) for info in ready["files"].values() for line in Path(info["path"]).read_text().splitlines()]
    assert len(all_rows) == 350
    tokenizer = AutoTokenizer.from_pretrained(ALLOWED_ROOT, local_files_only=True, trust_remote_code=False)
    args.out.mkdir(parents=True, exist_ok=True)
    # This CPU filesystem lock protects record ingestion/resume from duplicate
    # parent dispatches as well as simultaneous validate-only commands.
    record_lock = (args.out / "execution.lock").open("a")
    fcntl.flock(record_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = ingest_existing(args.out, all_rows, tokenizer)
    base = {"protocol": "non-qwen-smollm2-formal-v1", "model_revision": REVISION, "j": 8,
            "dtype": "bfloat16", "attn_impl": "sdpa", "seed": 42, "arms": list(ARMS),
            "fixtures": str(args.fixtures), "source_smoke": str(SMOKE), "accuracy_only": True,
            "timing_eligible": False, "same_native_template_EOS_pack_caps": True,
            "status": "prepared", "new_this_attempt": 0, "updated_utc": now(), **summary(records)}
    if len(records) == 1750:
        finalize_verified_records(args.out, base, records)
        assert not torch.cuda.is_initialized()
        print(json.dumps({"status": "complete", "verified_records": 1750, "model_loaded": False, "gpu_started": False}), flush=True)
        return 0
    if args.validate_only:
        assert not torch.cuda.is_initialized()
        base.update(all_350_fixtures_validated=True, all_70_smoke_predictions_redecoded_rescored=True,
                    remaining_generations=1750 - len(records), model_loaded=False, gpu_started=False)
        write(args.out / "non_qwen_FORMAL_CPU_READY.json", base)
        print(json.dumps({"status": "prepared", "formal_records": len(records), "verified_smoke_reused": 70,
                          "remaining_generations": 1750 - len(records)}), flush=True)
        return 0
    base["predecessor"] = released_smoke()
    from remote_queue import gpu_idle
    initial_idle, initial = gpu_idle(1, 512)
    assert initial_idle, initial
    # Same physical GPU1 lock used by the completed isolated smoke.
    gpu_lock_path = REMOTE / "outputs/non_qwen_smol_20260909/gpu1_smoke.lock"
    gpu_lock = gpu_lock_path.open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    second_idle, second = gpu_idle(1, 512)
    assert second_idle, second
    attempts = args.out / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = 1
    while (attempts / f"{attempt:04d}").exists():
        attempt += 1
    attempt_dir = attempts / f"{attempt:04d}"
    attempt_dir.mkdir()
    base.update(status="running", attempt=attempt, child_pid=os.getpid(), started_utc=now(),
                previously_verified_rows=len(records), initial_gpu_check=initial, locked_gpu_check=second)
    def save():
        base.update(summary(records), updated_utc=now())
        write(attempt_dir / "status.json", base)
        write(args.out / "status.json", base)
    save()
    try:
        download = json.loads((ALLOWED_ROOT / "non_qwen_DOWNLOAD_READY.json").read_text())
        assert download["status"] == "complete" and download["revision"] == REVISION
        assert all((ALLOWED_ROOT / r["name"]).stat().st_size == r["bytes"] and r["integrity_verified"] for r in download["files"])
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(ALLOWED_ROOT, local_files_only=True, trust_remote_code=False,
                    dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        third_idle, third = gpu_idle(1, 512)
        assert third_idle, third
        base["pre_to_gpu_check"] = third
        assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"
        model = model.to("cuda:0")
        assert next(model.parameters()).dtype == torch.bfloat16
        for arm in ARMS:
            reader = NonQwenExplicitReader(model, arm, j=8, model_binding=REPO + "@" + REVISION, tokenizer=tokenizer)
            for fixture in all_rows:
                key = (arm, fixture["task"], fixture["index"])
                if key in records:
                    continue  # Already individually verified; never recapture/generate it.
                base.update(active_arm=arm, active_task=fixture["task"], active_index=fixture["index"])
                output = reader.generate_fixture(fixture)
                prediction = {**output, "task": fixture["task"], "index": fixture["index"], "source_id": fixture["id"],
                              "answers": fixture["answers"], "official_score": score(output["decoded_output"], fixture),
                              "model_revision": REVISION, "dtype": "bfloat16", "device": "RTX3090", "smoke_only": False,
                              "source_row_sha256": fixture.get("source_row_sha256"), "source_prompt_sha256": fixture["source_prompt_sha256"],
                              "context_ids_sha256": fixture["context_ids_sha256"], "selected_ids_sha256": fixture["selected_ids_sha256"]}
                # Save a raw result before validation so a failed validation never
                # discards the actual generation; only valid records are reusable.
                rawpath = attempt_dir / "raw" / output_name(arm, fixture)
                write(rawpath, prediction)
                validate_prediction(prediction, fixture, arm, tokenizer)
                record = {"schema": "non-qwen-formal-prediction-v1", "model_revision": REVISION,
                          "seed": 42, "attn_impl": "sdpa", "prediction": prediction,
                          "provenance": {"kind": "formal_generation", "attempt": attempt, "raw_file": str(rawpath), "raw_sha256": sha(rawpath)}}
                target = args.out / "records" / output_name(arm, fixture)
                assert not target.exists()
                write(target, record)
                records[key] = record
                base["new_this_attempt"] += 1
                save()
                print(json.dumps({"formal_rows": len(records), "new_this_attempt": base["new_this_attempt"],
                                  "arm": arm, "task": fixture["task"], "index": fixture["index"]}), flush=True)
        assert len(records) == 1750
        final = summary(records)
        assert final["verified_smoke_reused"] == 70 and final["formal_generations"] == 1680 and final["completed_cells"] == 20
        base.update(status="complete", child_pid=None, active_arm=None, active_task=None, active_index=None, completed_utc=now())
        save()
        write(args.out / "non_qwen_FORMAL_COMPLETE.json", base)
        return 0
    except Exception:
        base.update(status="failed", child_pid=None, error=traceback.format_exc(), failed_utc=now())
        save()
        print(base["error"], flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
