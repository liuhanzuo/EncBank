"""One fixed formal snapshot, remote CPU source/pack/provenance/score audit."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2"
assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
from non_qwen_formal_driver import FORMAL, REMOTE, HERE, SMOKE, ARMS, alive, validate_record
from non_qwen_smoke_driver import load_fixtures, write, score
from non_qwen_download import ALLOWED_ROOT, REVISION
from remote_queue import gpu_idle

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def capture():
    directory = REMOTE / "outputs/diagnostics/non_qwen_formal_0631"
    directory.mkdir(exist_ok=True)
    path = directory / "snapshot.json"
    if path.exists():
        snapshot = json.loads(path.read_text())
        print(json.dumps({"resuming_same_fixed_snapshot": str(path), "snapshot_records": snapshot["listed_record_count"]}), flush=True)
        return path, snapshot
    start = now()
    filelist = sorted((FORMAL / "records").glob("*.json"))
    records = []
    for file in filelist:
        raw = file.read_bytes()
        records.append({"path": str(file), "sha256": hashlib.sha256(raw).hexdigest(), "record": json.loads(raw)})
    bootstrap = json.loads((REMOTE / "outputs/bootstrap_non_qwen_formal/status.json").read_text())
    status = json.loads((FORMAL / "status.json").read_text())
    marker = FORMAL / "non_qwen_FORMAL_COMPLETE.json"
    idle, reading = gpu_idle(1, 512)
    snapshot = {"captured_start_utc": start, "captured_end_utc": now(), "records": records,
                "listed_record_count": len(filelist), "bootstrap": bootstrap, "formal_status": status,
                "complete_marker": json.loads(marker.read_text()) if marker.exists() else None,
                "processes": {"bootstrap3166138_alive": alive(3166138), "child3166145_alive": alive(3166145)},
                "GPU1_idle": idle, "GPU1_reading": reading,
                "child_log_tail": (REMOTE / "outputs/bootstrap_non_qwen_formal/child0001.log").read_text(errors="replace").splitlines()[-5:]}
    write(path, snapshot)
    counts = Counter((r["record"]["prediction"]["arm"], r["record"]["prediction"]["task"]) for r in records)
    print(json.dumps({"snapshot_records": len(records), "complete_cells": sum(n == (200 if t == "qasper" else 50) for (_, t), n in counts.items()),
                      "snapshot_start": start, "GPU1_idle": idle}), flush=True)
    return path, snapshot

def main():
    snapshot_path, snapshot = capture()
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from non_qwen_explicit_reader import validate_fixture
    # This CPU-only helper sets Torch2/interop16 once at import. Do not call the
    # interop setter a second time; the first failed audit attempt is retained.
    from non_qwen_prepare_fixtures import prepare_row, BOUNDARY
    assert torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 16
    config = AutoConfig.from_pretrained(ALLOWED_ROOT, local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(ALLOWED_ROOT, local_files_only=True, trust_remote_code=False)
    fixture_root = REMOTE / "outputs/diagnostics/non_qwen_fixtures_0531_attempt1"
    ready, _, _ = load_fixtures(fixture_root, config)
    assert ready["protocol"]["revision"] == REVISION and ready["protocol"]["j"] == 8
    for source in ready["sources"]:
        assert Path(source["path"]).stat().st_size == source["bytes"] and sha(Path(source["path"])) == source["sha256"]
    fixtures = {}
    for info in ready["files"].values():
        for line in Path(info["path"]).read_text().splitlines():
            f = json.loads(line)
            fixtures[(f["task"], f["index"])] = f
    assert len(fixtures) == 350
    qa = [json.loads(line) for line in (HERE / "data/longbench/qasper.jsonl").read_text().splitlines()]
    prompts = json.loads((HERE / "protocol_sources/longbench/config/dataset2prompt.json").read_text())
    caps = json.loads((HERE / "protocol_sources/longbench/config/dataset2maxlen.json").read_text())
    input_checks = Counter()
    for key, f in fixtures.items():
        task, index = key
        validate_fixture(f, config)
        if task == "qasper":
            original = qa[index]
            assert digest(original) == f["source_row_sha256"]
            assert f["id"] == str(original.get("_id", original.get("id", index)))
            assert f["answers"] == original["answers"] and f["max_new_tokens"] == caps[task]
            marked = prompts[task].format(**dict(original, context=original["context"] + BOUNDARY))
            rendered = tok.apply_chat_template([{"role": "user", "content": marked}], tokenize=False, add_generation_prompt=True)
            context, query = rendered.split(BOUNDARY)
            assert context == f["context_text"] and query == f["query_text"]
            assert rendered.replace(BOUNDARY, "") == f["source_prompt"]
        else:
            assert f["source_target_tokens"] == 16384 and f["template_kind"] == "completion"
            assert len(tok.encode(f["source_prompt"], add_special_tokens=False)) == f["source_token_count"]
            expected_seed = 42 + int(hashlib.sha256((REVISION + "/" + task + "/16k").encode()).hexdigest()[:12], 16)
            assert f["stable_base_seed"] == expected_seed
            marker = "\nQuestion:" if task == "variable_tracking" else "\nWhat"
            boundary = f["source_prompt"].rfind(marker)
            assert boundary > 0
            assert f["context_text"] == f["source_prompt"][:boundary]
            assert f["query_text"] == f["source_prompt"][boundary:]
            if task == "variable_tracking":
                target = re.search(r"assigned the value (\d+)", f["query_text"]).group(1)
                assignments = re.findall(r"VAR ([A-Z]{5}) = (?:VAR ([A-Z]{5})|(\d+))", f["context_text"])
                closure = {a for a, b, n in assignments if n == target}
                for _ in range(5):
                    closure |= {a for a, b, n in assignments if b in closure}
                assert len(closure) == 5 and closure == set(f["answers"])
                assert f["max_new_tokens"] == 60 and f["selector"] == "iter_bm25"
            else:
                target = re.search(r"for ([a-z-]+) mentioned", f["query_text"]).group(1)
                answers = re.findall(r"One of the special magic numbers for " + re.escape(target) + r" is: (\d+)\.", f["context_text"])
                assert answers == f["answers"]
                assert f["max_new_tokens"] == 48 and f["selector"] == "bm25"
        rebuilt = prepare_row(tok, task, f["id"], index, f["context_text"], f["query_text"],
                              f["retrieval_question"], f["answers"], f["max_new_tokens"], f["selector"], f["source_prompt"])
        for field in ("context_ids", "query_ids", "context_chunk_lengths", "ordered_selected_indices",
                      "selected_chunk_lengths", "read_pack_tokens", "conservative_read_plus_cap",
                      "context_ids_sha256", "query_ids_sha256", "selected_ids_sha256", "pack_ids_sha256"):
            assert rebuilt[field] == f[field], (key, field)
        input_checks[task] += 1
    cells = defaultdict(list)
    seen = set()
    for entry in snapshot["records"]:
        record = entry["record"]
        row = record["prediction"]
        key = row["arm"], row["task"], row["index"]
        assert key not in seen and row["arm"] in ARMS
        seen.add(key)
        cells[key[:2]].append(entry)
    audited, pending, paired = [], [], {}
    provenance = Counter()
    validated_records = 0
    for arm in ARMS:
        for task in ("qasper", "niah_single_2", "niah_multikey_1", "variable_tracking"):
            entries = sorted(cells[(arm, task)], key=lambda x: x["record"]["prediction"]["index"])
            n = 200 if task == "qasper" else 50
            if len(entries) != n:
                pending.append({"arm": arm, "task": task, "snapshot_n": len(entries), "expected_n": n, "score_percent": None})
                continue
            assert [e["record"]["prediction"]["index"] for e in entries] == list(range(n))
            scores, cell_provenance, output_lengths = [], Counter(), []
            for entry in entries:
                record = entry["record"]
                row = record["prediction"]
                fixture = fixtures[(task, row["index"])]
                assert sha(Path(entry["path"])) == entry["sha256"]  # fixed canonical files must not mutate
                validate_record(record, fixture, arm, tok)
                decoded = tok.decode(row["ids"], skip_special_tokens=True).strip()
                scores.append(score(decoded, fixture))
                cell_provenance[record["provenance"]["kind"]] += 1
                output_lengths.append(row["generated_tokens"])
                validated_records += 1
            provenance.update(cell_provenance)
            result = {"arm": arm, "task": task, "n": n, "score_percent": 100 * sum(scores) / n,
                      "scores_by_source_index": scores, "source_indices": list(range(n)),
                      "provenance_counts": dict(cell_provenance), "output_tokens_mean": sum(output_lengths) / n,
                      "output_tokens_range": [min(output_lengths), max(output_lengths)],
                      "official_rescore_equal": True, "input_pack_source_and_raw_provenance_valid": True}
            audited.append(result)
            paired[(arm, task)] = result
    differences = []
    for task in ("qasper", "niah_single_2", "niah_multikey_1", "variable_tracking"):
        v2 = paired.get(("fix_all", task))
        if v2 is None:
            continue
        diff = {"task": task, "n": v2["n"], "V2": v2["score_percent"], "comparisons": {}}
        for comparator in ("j0", "pub", "pub_sink", "fix_none"):
            other = paired.get((comparator, task))
            if other is None:
                diff["comparisons"][comparator] = None
                continue
            a, b = v2["scores_by_source_index"], other["scores_by_source_index"]
            diff["comparisons"][comparator] = {"score_percent": other["score_percent"],
                  "V2_minus_comparator_pp": v2["score_percent"] - other["score_percent"],
                  "paired_V2_higher": sum(x > y + 1e-12 for x, y in zip(a, b)),
                  "paired_equal": sum(abs(x - y) <= 1e-12 for x, y in zip(a, b)),
                  "paired_V2_lower": sum(x < y - 1e-12 for x, y in zip(a, b))}
        differences.append(diff)
    marker = snapshot["complete_marker"]
    all_complete = len(audited) == 20
    if all_complete:
        assert len(snapshot["records"]) == validated_records == 1750
        assert provenance == {"verified_smoke": 70, "formal_generation": 1680}
        assert marker is not None and marker["status"] == "complete" and marker["formal_rows"] == 1750
        assert marker["verified_smoke_reused"] == 70 and marker["formal_generations"] == 1680 and marker["completed_cells"] == 20
        for declared in marker["cells"]:
            actual = paired[(declared["arm"], declared["task"])]
            assert declared["n"] == actual["n"] and abs(declared["score_percent"] - actual["score_percent"]) < 1e-9
    assert not torch.cuda.is_initialized()
    result = {"status": "audit_complete", "snapshot_all_formal_complete": all_complete, "snapshot": str(snapshot_path),
              "snapshot_start_utc": snapshot["captured_start_utc"], "snapshot_end_utc": snapshot["captured_end_utc"],
              "snapshot_records": len(snapshot["records"]), "complete_cells_audited": len(audited),
              "validated_complete_cell_records": validated_records, "expected_records": 1750,
              "validated_provenance": dict(provenance), "expected_provenance": {"verified_smoke": 70, "formal_generation": 1680},
              "excluded_native_references": 4, "input_source_template_token_pack_rebuilt": dict(input_checks),
              "cells": audited, "pending_cells": pending, "differences": differences,
              "protocol": ready["protocol"], "input_budget": ready["tasks"], "snapshot_processes": snapshot["processes"],
              "snapshot_bootstrap_status": snapshot["bootstrap"]["status"], "snapshot_formal_status": snapshot["formal_status"]["status"],
              "snapshot_GPU1_idle": snapshot["GPU1_idle"], "snapshot_GPU1_reading": snapshot["GPU1_reading"],
              "remote_cpu": {"device": "cpu", "threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                             "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "cuda_initialized": False},
              "metric": "Qasper official LongBench answer F1; RULER original string_match_all recall, percent",
              "limitations": ["within-checkpoint paired comparison; native tokenizer and new RULER draw differ from Qwen",
                              "24-layer1.7B SmolLM versus8B Qwen does not isolate architecture from scale/training",
                              "default native window8192; 16k is synthetic source target, not dense native context",
                              "no remote timing/peak-memory inference or formal significance claim",
                              "partial cells excluded, smoke means never substituted for formal means"],
              "no_models_loaded": True, "no_generations_rerun": True}
    write(HERE / "non_qwen_formal_20260909_0631.json", result)
    write(HERE / "non_qwen_formal_20260909_0631_writeback.json", {k: result[k] for k in (
        "snapshot_start_utc", "snapshot_records", "snapshot_all_formal_complete", "complete_cells_audited",
        "validated_complete_cell_records", "validated_provenance", "excluded_native_references",
        "cells", "pending_cells", "differences", "metric", "input_budget", "limitations")})
    print(json.dumps({"status": "audit_complete", "cells": len(audited), "validated_records": validated_records,
                      "scores": [{k: cell[k] for k in ("arm", "task", "n", "score_percent")} for cell in audited], "pending": pending}), flush=True)

if __name__ == "__main__":
    main()
