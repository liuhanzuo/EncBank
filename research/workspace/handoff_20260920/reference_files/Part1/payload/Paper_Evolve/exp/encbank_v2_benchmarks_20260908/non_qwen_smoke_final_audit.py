"""Light remote CPU audit of the one completed non-Qwen smoke; no model load."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from non_qwen_smoke_driver import alive, write, REMOTE
from non_qwen_smoke_bootstrap import completion_ok
from remote_queue import gpu_idle

def main():
    out = REMOTE / "outputs/non_qwen_smol_20260909/smoke/attempts/0001"
    bootstrap = json.loads((REMOTE / "outputs/bootstrap_non_qwen_smoke/status.json").read_text())
    done = json.loads((out / "non_qwen_SMOKE_COMPLETE.json").read_text())
    assert completion_ok(out) and bootstrap["status"] == "completed" and bootstrap["child_exit_code"] == 0
    assert not alive(bootstrap["pid"]) and not alive(3160516)
    records = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    refs = [json.loads(p.read_text()) for p in sorted((out / "references").glob("*.json"))]
    assert len(records) == 70 and len(refs) == 4
    counts = Counter((r["arm"], r["task"]) for r in records)
    assert len(counts) == 20
    assert all(n == (8 if task == "qasper" else 2) for (arm, task), n in counts.items())
    assert len({(r["arm"], r["task"], r["index"]) for r in records}) == 70
    grouped, quality = defaultdict(list), defaultdict(list)
    for row in records:
        assert row["generated_tokens"] == len(row["ids"]) <= row["max_new_tokens"]
        assert row["decoder_forwards"] == len(row["ids"]) - 1
        assert row["suppress_first_eos"] is True and row["ids"][0] != 2
        assert (row["ids"][-1] == 2) == row["stopped_on_eos"]
        assert isinstance(row["decoded_output"], str) and 0 <= row["official_score"] <= 1
        assert row["dtype"] == "bfloat16" and row["device"] == "RTX3090"
        assert row["actual_read_pack_tokens"] + row["max_new_tokens"] <= 8192
        grouped[(row["task"], row["index"])].append(row)
        quality[(row["arm"], row["task"])].append(row["official_score"])
    for rows in grouped.values():
        assert len(rows) == 5
        for key in ("fixture_sha256", "pack_ids_sha256", "query_ids_sha256", "context_ids_sha256", "selected_ids_sha256", "source_prompt_sha256"):
            assert len({r[key] for r in rows}) == 1
        assert all(r["ordered_selected_indices"] == rows[0]["ordered_selected_indices"] for r in rows)
    reference_steps = 0
    for ref in refs:
        assert ref["equal_ids"] and ref["j0_ids"] == ref["native_ids"]
        all_steps = ref["all_steps"]
        assert all_steps["all_generated_step_logits_pass"] and all_steps["same_step_count"]
        assert all_steps["j0_steps"] == len(ref["j0_ids"])
        assert all(x["finite"] and x["same_shape"] and x["allclose"] for x in all_steps["steps"])
        assert Path(ref["raw_step_logits_file"]).is_file()
        reference_steps += all_steps["j0_steps"]
    idle, reading = gpu_idle(1, 512)
    result = {"status": "complete", "audited_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "finite real-BF16 non-Qwen smoke only; not formal benchmark/infra",
              "result_dir": str(out), "bootstrap_pid": bootstrap["pid"], "model_child_pid": 3160516,
              "bootstrap_naturally_exited": True, "model_child_naturally_exited": True,
              "child_exit_code": 0, "smoke_generations": 70, "extra_native_reference_sequences": 4,
              "complete_arm_task_cells": 20, "five_arm_same_fixture_pairs": 14,
              "native_reference_generated_steps": reference_steps, "all_generated_step_logits_pass": True,
              "max_native_j0_logit_abs_error": max(s["max_abs_error"] for r in refs for s in r["all_steps"]["steps"]),
              "all_raw_reference_ids_equal": True, "officially_scored_smoke_rows": 70,
              "smoke_scores_not_formal_results": True,
              "smoke_scores_by_arm_task": [{"arm": arm, "task": task, "n": len(scores), "score_percent": 100 * sum(scores) / len(scores)} for (arm, task), scores in sorted(quality.items())],
              "generated_tokens": sum(len(r["ids"]) for r in records),
              "reference_comparisons": refs, "current_GPU1_idle": idle, "current_GPU1_check": reading,
              "automatic_formal_benchmark_started": False, "timing_eligible": False, "gpu_model_loaded_by_audit": False}
    write(out / "non_qwen_FINAL_AUDIT.json", result)
    print(json.dumps({k: result[k] for k in ("status", "smoke_generations", "extra_native_reference_sequences", "all_raw_reference_ids_equal", "all_generated_step_logits_pass", "max_native_j0_logit_abs_error", "current_GPU1_idle")}))

if __name__ == "__main__":
    main()
