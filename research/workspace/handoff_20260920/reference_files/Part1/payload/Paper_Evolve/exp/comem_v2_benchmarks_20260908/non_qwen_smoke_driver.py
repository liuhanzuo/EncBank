"""Prepared real-weight smoke: 14 fixtures x five arms + four stock references.

Default --validate-only never loads weights. --execute-gpu-smoke is for the
parent's later coordinated dispatch after BABI16 has completed and exited.
This script does not create a queue or modify any Qwen/shared state.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import string
import sys
import traceback

HERE = Path(__file__).resolve().parent
REMOTE = Path("/data/liuhanzuo/comem_v2_20260908")
from non_qwen_download import ALLOWED_ROOT, REPO, REVISION

def now():
    return datetime.now(timezone.utc).isoformat()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)

def alive(pid):
    if not pid:
        return False
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False

def predecessor_complete():
    from native_priority_receipt import completed
    path = REMOTE / "outputs/bootstrap_babi16_priority/status.json"
    state = json.loads(path.read_text())
    assert state["status"] == "completed" and state["completed_jobs"] == 6 and state["completed_predictions"] == 600
    assert len(state["jobs"]) == 6
    pids = [state.get("pid"), state.get("queue_pid")]
    for job in state["jobs"].values():
        assert job["status"] == "complete" and job["exit_code"] == 0
        pids.append(job.get("model_pid"))
        queue = json.loads(Path(job["queue_state"]).read_text())
        pids.append(queue.get("queue_pid"))
        pids += [j.get("pid") for j in queue["jobs"].values() if not j.get("external_dependency")]
    assert not any(alive(p) for p in pids), "BABI16 processes have not all naturally exited"
    plan = json.loads((HERE / "babi16_priority_full_plan.json").read_text())
    assert len(plan["jobs"]) == 6 and all(completed(job) for job in plan["jobs"])
    return {"status_file": str(path), "complete_jobs": 6, "complete_predictions": 600,
            "all_recorded_processes_exited": True, "native_completion_receipts_verified": 6}

def score(prediction, sample):
    if sample["task"] != "qasper":
        return sum(a.lower() in prediction.lower() for a in sample["answers"]) / len(sample["answers"])
    source = HERE / "protocol_sources/longbench/metrics.py"
    names = {"normalize_answer", "f1_score", "qa_f1_score"}
    nodes = [x for x in ast.parse(source.read_text()).body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes} == names
    namespace = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return max(float(namespace["qa_f1_score"](prediction, a)) for a in sample["answers"])

def load_fixtures(directory, config):
    from non_qwen_explicit_reader import validate_fixture
    ready = json.loads((directory / "non_qwen_FIXTURES_READY.json").read_text())
    assert ready["status"] == "ready" and ready["samples"] == 350
    assert ready["protocol"]["revision"] == REVISION and ready["protocol"]["j"] == 8
    all_rows = {}
    for task, info in ready["files"].items():
        path = Path(info["path"])
        assert path.resolve().parent == directory.resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == (200 if task == "qasper" else 50)
        assert [x["index"] for x in rows] == list(range(len(rows)))
        for row in rows:
            validate_fixture(row, config)
        all_rows[task] = rows
    selected = [all_rows[task][i] for task, indices in ready["smoke_indices"].items() for i in indices]
    refs = [all_rows[item["task"]][item["index"]] for item in ready["native_reference_examples"]]
    assert len(selected) == 14 and len(refs) == 4
    return ready, selected, refs

def smoke_actions(arms, selected, refs):
    """Four native gates first; these j0 rows count toward the 70 smoke rows."""
    keys = {(x["task"], x["index"]) for x in selected}
    ref_keys = {(x["task"], x["index"]) for x in refs}
    assert len(selected) == len(keys) == 14 and len(refs) == len(ref_keys) == 4
    assert ref_keys.issubset(keys)
    assert set(arms) == {"pub", "pub_sink", "fix_all", "j0", "fix_none"}
    actions = [("native_j0_gate", "j0", fixture) for fixture in refs]
    actions += [("generation", arm, fixture) for arm in arms for fixture in selected
                if not (arm == "j0" and (fixture["task"], fixture["index"]) in ref_keys)]
    assert len(actions) == 70
    return actions

def compare_reference_logits(actual, expected):
    import torch
    steps = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        same_shape = a.shape == b.shape
        steps.append({"step": index, "finite": finite, "same_shape": same_shape,
                      "max_abs_error": float((a.float() - b.float()).abs().max()) if same_shape and finite else None,
                      "allclose": bool(same_shape and finite and torch.allclose(a.float(), b.float(), atol=.002, rtol=.002))})
    return {"j0_steps": len(actual), "native_steps": len(expected), "same_step_count": len(actual) == len(expected),
            "atol": .002, "rtol": .002, "steps": steps,
            "all_generated_step_logits_pass": bool(actual) and len(actual) == len(expected) and all(x["allclose"] for x in steps)}

def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-gpu-smoke", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=ALLOWED_ROOT)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.model_dir.resolve() == ALLOWED_ROOT
    assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ("" if args.validate_only else "1")
    download = json.loads((args.model_dir / "non_qwen_DOWNLOAD_READY.json").read_text())
    assert download["status"] == "complete" and download["revision"] == REVISION and download["repo"] == REPO
    assert not download["authenticated"] and not download["additional_terms_accepted"]
    assert len(download["files"]) == 9 and all(x["integrity_verified"] for x in download["files"])
    for item in download["files"]:
        assert (args.model_dir / item["name"]).stat().st_size == item["bytes"]
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    from non_qwen_explicit_reader import ARMS, NonQwenExplicitReader, native_reference
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    assert config.model_type == "llama" and config.num_hidden_layers == 24
    assert config.num_attention_heads == config.num_key_value_heads == 32 and config.head_dim == 64
    assert config.max_position_embeddings == 8192 and config.rope_parameters["rope_type"] == "default"
    assert config.rope_parameters["rope_theta"] == 130000
    ready, selected, refs = load_fixtures(args.fixtures, config)
    ordered_actions = smoke_actions(ARMS, selected, refs)
    proof = {"created_utc": now(), "model_dir": str(args.model_dir), "revision": REVISION,
             "fixtures": str(args.fixtures), "all_350_fixtures_validated": True,
             "smoke_examples": [{"task": s["task"], "index": s["index"], "id": s["id"], "fixture_sha256": s["fixture_sha256"]} for s in selected],
             "smoke_arms": list(ARMS), "smoke_generations": 70, "extra_stock_reference_sequences": 4,
             "pretrained_benchmark_complete": False, "official_qasper_scoring": "LongBench original qa_f1_score max over references",
             "ruler_scoring": "original string_match_all recall", "all_arms_first_step_eos_suppression": True,
             "driver_entry_version": "native-first-all-step-logits-v2",
             "native_gates_first": [x[0] for x in ordered_actions[:4]] == ["native_j0_gate"] * 4,
             "native_reference_success_policy": "exact raw IDs; equal step counts; all steps finite, same shape, allclose atol=0.002 rtol=0.002; fail immediately",
             "all_generated_step_logit_gate_required": True}
    if args.validate_only:
        assert not torch.cuda.is_initialized()
        proof.update(status="prepared", ready_for_root_coordinated_smoke=True, real_weights_loaded=False,
                     gpu_started=False, cuda_initialized=False, device="cpu", torch_threads=2, torch_interop_threads=16)
        write(args.out / "non_qwen_SMOKE_CPU_READY.json", proof)
        print(json.dumps({"status": "prepared", "fixtures_validated": 350, "smoke_generations": 70, "extra_references": 4}), flush=True)
        return 0
    # This branch is not entered by the preparation task. Parent dispatch must
    # first reserve GPU1 after the finite BABI16 predecessor and its children exit.
    from remote_queue import gpu_idle
    proof["predecessor"] = predecessor_complete()
    initial_idle, initial = gpu_idle(1, 512)
    assert initial_idle, ("GPU1 is busy before lock", initial)
    lockpath = REMOTE / "outputs/non_qwen_smol_20260909/gpu1_smoke.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    second_idle, second = gpu_idle(1, 512)
    assert second_idle, ("GPU1 is busy after lock", second)
    assert not args.out.exists(), "preserve prior real smoke attempt; use a new attempt directory"
    args.out.mkdir(parents=True)
    proof.update(status="running", gpu=1, child_pid=os.getpid(), started_utc=now(),
                 initial_gpu_check=initial, locked_gpu_check=second, required_gpu="NVIDIA GeForce RTX 3090",
                 max_idle_mib=512, no_other_compute_process=True, accuracy_only=True,
                 actual_rows=0, actual_extra_references=0)
    write(args.out / "status.json", proof)
    try:
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                    dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        third_idle, third = gpu_idle(1, 512)
        assert third_idle, ("GPU1 changed during CPU weight load", third)
        proof["pre_to_gpu_check"] = third
        assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"
        model = model.to("cuda:0")
        assert next(model.parameters()).dtype == torch.bfloat16
        proof.update(real_weights_loaded=True, gpu_started=True, actual_dtype="bfloat16", actual_device="cuda:0")
        write(args.out / "status.json", proof)
        rows, reference_proofs = [], []
        for phase, arm, fixture in ordered_actions:
            proof.update(phase=phase, active_arm=arm, active_task=fixture["task"], active_index=fixture["index"])
            write(args.out / "status.json", proof)
            reader = NonQwenExplicitReader(model, arm, j=8, model_binding=REPO + "@" + REVISION, tokenizer=tokenizer)
            key = (fixture["task"], fixture["index"])
            output = reader.generate_fixture(fixture, capture_logits=phase == "native_j0_gate")
            logits = output.pop("step_logits", None)
            record = {**output, "task": fixture["task"], "index": fixture["index"], "source_id": fixture["id"],
                      "answers": fixture["answers"], "official_score": score(output["decoded_output"], fixture),
                      "model_revision": REVISION, "dtype": "bfloat16", "device": "RTX3090", "smoke_only": True,
                      "source_row_sha256": fixture.get("source_row_sha256"), "source_prompt_sha256": fixture["source_prompt_sha256"],
                      "context_ids_sha256": fixture["context_ids_sha256"], "selected_ids_sha256": fixture["selected_ids_sha256"]}
            write(args.out / "records" / f"{arm}__{fixture['task']}__{fixture['index']:03d}.json", record)
            rows.append(record)
            proof["actual_rows"] = len(rows)
            write(args.out / "status.json", proof)
            print(json.dumps({"phase": phase, "arm": arm, "task": fixture["task"], "index": fixture["index"], "completed": len(rows)}), flush=True)
            if phase != "native_j0_gate":
                continue
            native = native_reference(model, fixture, capture_logits=True)
            reference_logits = native.pop("step_logits")
            comparison = {"task": key[0], "index": key[1], "fixture_sha256": fixture["fixture_sha256"],
                          "native_ids": native["ids"], "j0_ids": record["ids"], "equal_ids": record["ids"] == native["ids"],
                          "extra_reference_sequences": 1, "native_decoder_forwards": native["decoder_forwards"],
                          "first_logits_max_abs_error": float((logits[0].float() - reference_logits[0].float()).abs().max()),
                          "first_logits_allclose_atol_rtol_0_002": torch.allclose(logits[0].float(), reference_logits[0].float(), atol=.002, rtol=.002)}
            comparison["all_steps"] = compare_reference_logits(logits, reference_logits)
            tensor_path = args.out / "references" / f"{key[0]}__{key[1]:03d}_step_logits.pt"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"j0_step_logits": logits, "native_step_logits": reference_logits}, tensor_path)
            comparison["raw_step_logits_file"] = str(tensor_path)
            write(args.out / "references" / f"{key[0]}__{key[1]:03d}.json", comparison)
            reference_proofs.append(comparison)
            proof["actual_extra_references"] = len(reference_proofs)
            write(args.out / "status.json", proof)
            assert comparison["equal_ids"] and comparison["all_steps"]["all_generated_step_logits_pass"], "native/j0 gate failed; preserve evidence and stop before remaining smoke"
        assert len(rows) == 70 and len(reference_proofs) == 4
        assert all(r["equal_ids"] and r["all_steps"]["all_generated_step_logits_pass"] for r in reference_proofs), "preserve failed native/j0 comparisons; do not pass on accuracy"
        proof.update(status="complete", completed_utc=now(), child_pid=None, all_raw_reference_ids_equal=True,
                     first_step_logit_checks_passed=True, all_generated_step_logits_passed=True,
                     phase=None, active_arm=None, active_task=None, active_index=None,
                     formally_scored_smoke_rows=70, extra_reference_sequences=4,
                     smoke_is_not_formal_n200_n50_quality=True)
        write(args.out / "non_qwen_SMOKE_COMPLETE.json", proof)
        write(args.out / "status.json", proof)
        return 0
    except Exception:
        proof.update(status="failed", error=traceback.format_exc(), child_pid=None, failed_utc=now())
        write(args.out / "status.json", proof)
        print(proof["error"], flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
