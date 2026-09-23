"""Append only the two formerly incomplete cells to the fixed 18-cell audit."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import torch
from transformers import AutoTokenizer
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
from non_qwen_formal_driver import FORMAL, REMOTE, HERE, ARMS, alive, validate_record, score
from non_qwen_download import ALLOWED_ROOT
from remote_queue import gpu_idle

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

def main():
    audit_path = HERE / "non_qwen_formal_20260909_0631.json"
    first = json.loads(audit_path.read_text())
    assert first["complete_cells_audited"] == 18 and first["validated_complete_cell_records"] == 1650
    initial_evidence = HERE / "non_qwen_formal_20260909_0631_initial18.json"
    assert not initial_evidence.exists()
    save(initial_evidence, first)
    initial_snapshot = json.loads(Path(first["snapshot"]).read_text())
    completion_time = datetime.now(timezone.utc).isoformat()
    bootstrap = json.loads((REMOTE / "outputs/bootstrap_non_qwen_formal/status.json").read_text())
    marker = json.loads((FORMAL / "non_qwen_FORMAL_COMPLETE.json").read_text())
    assert bootstrap["status"] == "completed" and bootstrap["child_exit_code"] == 0
    assert marker["status"] == "complete" and marker["formal_rows"] == 1750
    assert marker["verified_smoke_reused"] == 70 and marker["formal_generations"] == 1680 and marker["completed_cells"] == 20
    process_states = {"bootstrap3166138_alive": alive(3166138), "child3166145_alive": alive(3166145)}
    assert not any(process_states.values())
    assert len(list((FORMAL / "records").glob("*.json"))) == 1750
    # Confirm previously audited canonical records are unchanged; no repeat
    # decoding/scoring or input/retrieval reconstruction for those18 cells.
    old_keys = {(c["arm"], c["task"]) for c in first["cells"]}
    old_unchanged = 0
    for old in initial_snapshot["records"]:
        row = old["record"]["prediction"]
        if (row["arm"], row["task"]) in old_keys:
            assert sha(Path(old["path"])) == old["sha256"]
            old_unchanged += 1
    assert old_unchanged == 1650
    assert {(p["arm"], p["task"]) for p in first["pending_cells"]} == {
        ("fix_none", "niah_multikey_1"), ("fix_none", "variable_tracking")}
    tokenizer = AutoTokenizer.from_pretrained(ALLOWED_ROOT, local_files_only=True, trust_remote_code=False)
    fixture_root = REMOTE / "outputs/diagnostics/non_qwen_fixtures_0531_attempt1"
    new_cells, new_snapshot_rows = [], []
    for task in ("niah_multikey_1", "variable_tracking"):
        fixtures = [json.loads(line) for line in (fixture_root / f"non_qwen_{task}.jsonl").read_text().splitlines()]
        assert len(fixtures) == 50
        scores, provenance, lengths = [], Counter(), []
        for index, f in enumerate(fixtures):
            assert f["index"] == index
            path = FORMAL / "records" / f"fix_none__{task}__{index:03d}.json"
            record = json.loads(path.read_text())
            new_snapshot_rows.append({"path": str(path), "sha256": sha(path), "record": record})
            validate_record(record, f, "fix_none", tokenizer)
            prediction = record["prediction"]
            decoded = tokenizer.decode(prediction["ids"], skip_special_tokens=True).strip()
            scores.append(score(decoded, f))
            lengths.append(prediction["generated_tokens"])
            provenance[record["provenance"]["kind"]] += 1
        new_cells.append({"arm": "fix_none", "task": task, "n": 50, "score_percent": 100 * sum(scores) / 50,
                          "scores_by_source_index": scores, "source_indices": list(range(50)),
                          "provenance_counts": dict(provenance), "output_tokens_mean": sum(lengths) / 50,
                          "output_tokens_range": [min(lengths), max(lengths)],
                          "official_rescore_equal": True, "input_pack_source_and_raw_provenance_valid": True})
    idle, reading = gpu_idle(1, 512)
    completion = {"captured_start_utc": completion_time, "captured_end_utc": datetime.now(timezone.utc).isoformat(),
                  "bootstrap": bootstrap, "marker": marker, "processes": process_states,
                  "GPU1_idle": idle, "GPU1_reading": reading, "newly_audited_records": new_snapshot_rows,
                  "previous1650_records_unchanged": True, "new_complete_cells": 2, "newly_audited_n": 100}
    completion_path = REMOTE / "outputs/diagnostics/non_qwen_formal_0631/completion_snapshot.json"
    assert not completion_path.exists()
    save(completion_path, completion)
    first["cells"].extend(new_cells)
    order = {name: i for i, name in enumerate(ARMS)}
    tasks = {name: i for i, name in enumerate(("qasper", "niah_single_2", "niah_multikey_1", "variable_tracking"))}
    first["cells"].sort(key=lambda c: (order[c["arm"]], tasks[c["task"]]))
    indexed = {(c["arm"], c["task"]): c for c in first["cells"]}
    for differences in first["differences"]:
        task = differences["task"]
        if task not in ("niah_multikey_1", "variable_tracking"):
            continue
        v2, other = indexed[("fix_all", task)], indexed[("fix_none", task)]
        a, b = v2["scores_by_source_index"], other["scores_by_source_index"]
        differences["comparisons"]["fix_none"] = {"score_percent": other["score_percent"],
            "V2_minus_comparator_pp": v2["score_percent"] - other["score_percent"],
            "paired_V2_higher": sum(x > y + 1e-12 for x, y in zip(a, b)),
            "paired_equal": sum(abs(x - y) <= 1e-12 for x, y in zip(a, b)),
            "paired_V2_lower": sum(x < y - 1e-12 for x, y in zip(a, b))}
    provenance = Counter(first["validated_provenance"])
    for c in new_cells:
        provenance.update(c["provenance_counts"])
    assert provenance == {"verified_smoke": 70, "formal_generation": 1680}
    assert len(marker["cells"]) == 20
    for recorded in marker["cells"]:
        validated = indexed[(recorded["arm"], recorded["task"])]
        assert recorded["n"] == validated["n"] and abs(recorded["score_percent"] - validated["score_percent"]) < 1e-9
    assert not torch.cuda.is_initialized()
    first.update(snapshot_all_formal_complete=True, initial_snapshot_records=first["snapshot_records"], snapshot_records=1750,
                 complete_cells_audited=20, validated_complete_cell_records=1750, validated_provenance=dict(provenance),
                 pending_cells=[], completion_snapshot=str(completion_path), completion_snapshot_start_utc=completion_time,
                 completion_snapshot_end_utc=completion["captured_end_utc"], initial_audit_evidence=str(initial_evidence),
                 appended_complete_cells=2, appended_validated_records=100, previous1650_records_unchanged=True,
                 previous18_cells_not_rescored=True, source350_not_rebuilt_on_append=True,
                 completion_processes=process_states, completion_bootstrap_status=bootstrap["status"],
                 completion_child_exit_code=bootstrap["child_exit_code"], completion_GPU1_idle=idle, completion_GPU1_reading=reading,
                 completed_utc=marker["completed_utc"])
    save(audit_path, first)
    writeback = {k: first[k] for k in ("snapshot_start_utc", "initial_snapshot_records", "completion_snapshot_start_utc",
        "snapshot_records", "snapshot_all_formal_complete", "complete_cells_audited", "validated_complete_cell_records",
        "validated_provenance", "excluded_native_references", "cells", "pending_cells", "differences", "metric", "input_budget", "limitations",
        "appended_complete_cells", "appended_validated_records", "previous18_cells_not_rescored", "source350_not_rebuilt_on_append")}
    save(HERE / "non_qwen_formal_20260909_0631_writeback.json", writeback)
    print(json.dumps({"status": "all20_complete", "new_cells": [{k: c[k] for k in ("arm", "task", "n", "score_percent")} for c in new_cells],
                      "completed_utc": marker["completed_utc"], "snapshot_utc": completion_time,
                      "all_pids_naturally_exited": True, "GPU1_idle": idle, "GPU1_reading": reading}), flush=True)

if __name__ == "__main__":
    main()
