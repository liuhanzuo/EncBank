"""Matched four-arm, answer-only LoRA SFT for the SparseCoMem pilot.

Runs on one explicitly reserved remote GPU. All arms start from the same strong
CoMem LoRA and consume the same prepared QA examples. This is an accuracy and
training entry point, never a formal latency or inference-memory benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "comem_v2_benchmarks_20260908"))
sys.path.insert(0, str(HERE.parent / "beacon_comem_20260909"))
sys.path.insert(0, str(HERE.parents[1] / "COMem"))
from prepare_data import FORMAT, atomic_json, digest, read_rows


def configuration_for_arm(arm, rho):
    if arm not in {"D0", "A", "B", "D1"} or not 0 < rho <= 1:
        raise ValueError("Invalid arm or retain ratio")
    return {"probe_mode": "block" if arm in {"A", "B"} else "dense",
            "retain_ratio": 1.0 if arm in {"A", "D0"} else rho}


def validate_prepared(train, dev):
    if not train or not dev:
        raise ValueError("Both prepared splits must be nonempty")
    for split, rows in (("train", train), ("dev", dev)):
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate prepared example IDs")
        for row in rows:
            if row.get("format") != FORMAT or row.get("split") != split or row.get("source") != "allenai/qasper:train:v0.3":
                raise ValueError("Use prepare_data.py output from official-train Qasper only")
            if row.get("answer_truncated") or not row["answer_ids"] or not row["prompt_ids"]:
                raise ValueError("Targets must be complete and prompt nonempty")
            if not row["document_chunks"] or any(not chunk for chunk in row["document_chunks"]):
                raise ValueError("Empty candidate documents")
            probe = row.get("probe_indices")
            if not probe or probe != sorted(set(probe)) or min(probe) < 0 or max(probe) >= len(row["prompt_ids"]):
                raise ValueError("Invalid prompt-only probe positions")
    if {row["document_id"] for row in train} & {row["document_id"] for row in dev}:
        raise ValueError("Prepared training/development documents overlap")


def example_at(examples, cursor, seed):
    epoch, offset = divmod(cursor, len(examples))
    order = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(order)
    return examples[order[offset]]


def total_tokens(row):
    return sum(map(len, row["document_chunks"])) + len(row["prompt_ids"]) + len(row["answer_ids"])


def select_smoke_examples(rows):
    ordered = sorted(rows, key=lambda row: (total_tokens(row), row["id"]))
    result = [ordered[-1], ordered[len(ordered) // 2]]
    if len(rows) > 1 and result[0]["id"] == result[1]["id"]:
        result[1] = ordered[0]
    return result


@contextmanager
def evaluation_mode(reader):
    previous = reader.training
    reader.eval()
    try:
        yield
    finally:
        reader.train(previous)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "init-adapter", "train", "dev", "out", "device"):
        ap.add_argument(f"--{key}", required=True)
    ap.add_argument("--arm", required=True, choices=["D0", "A", "B", "D1"])
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--rho", type=float, default=.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-limit", type=int, default=25)
    ap.add_argument("--final-eval-limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--dev-limit", type=int)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--skip-initial-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def run(args):
    from remote_gpu_guard import validate_worker_lease
    lease_path = os.environ.get('SPARSE_GPU_LEASE_PATH')
    if not lease_path:
        raise RuntimeError('A live remote controller GPU lease is required')
    admissions = [dict(stage='before_torch_import',
                       **validate_worker_lease(lease_path, require_idle=True))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'gpu_admission.json', {'checks': admissions})
    import torch
    import torch.nn.functional as F
    from train_8b_baseline import attach_lora, restore_flat, flat_state, atomic_save
    from evaluate_sft import token_f1, exact_match, eos_token_ids
    from comem.model import CoMem
    from sparse_reader import SparseCoMemReader
    if not args.device.startswith("cuda") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one reserved remote GPU explicitly")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError("This entry point is single-GPU; reserve exactly one GPU")
    if min(args.steps, args.grad_accum, args.save_every, args.eval_every,
           args.eval_limit, args.final_eval_limit, args.max_new_tokens) < 1:
        raise ValueError("Step and evaluation budgets must be positive")
    if args.smoke and (args.steps != 1 or args.grad_accum != 2 or args.train_limit is not None):
        raise ValueError("Smoke requires --steps 1 --grad-accum 2 and no --train-limit")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    admissions.append(dict(stage='immediately_before_cuda',
                           **validate_worker_lease(lease_path, require_idle=True)))
    atomic_json(out / 'gpu_admission.json', {'checks': admissions})
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Fixed recipe requires BF16")
    if (out / "last.pt").exists() and not args.resume:
        raise ValueError("Existing checkpoint requires explicit --resume")
    train, dev = read_rows(args.train), read_rows(args.dev)
    validate_prepared(train, dev)
    if args.smoke:
        train = select_smoke_examples(train)
    elif args.train_limit is not None:
        if args.train_limit < 1:
            raise ValueError("train-limit must be positive")
        train = train[:args.train_limit]
    if args.dev_limit is not None:
        if args.dev_limit < 1:
            raise ValueError("dev-limit must be positive")
        dev = dev[:args.dev_limit]
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    admissions.append(dict(stage='before_model_to_cuda',
                           **validate_worker_lease(lease_path, require_idle=False)))
    atomic_json(out / 'gpu_admission.json', {'checks': admissions})
    model = model.to(device).eval()
    validate_worker_lease(lease_path, require_idle=False)
    if model.config.model_type != "qwen3" or model.config.attention_dropout != 0:
        raise ValueError("Validated for dense Qwen3 with zero attention dropout")
    if not 0 < args.j < args.m <= model.config.num_hidden_layers:
        raise ValueError("Require 0 < j < m <= L")
    longest = max(sum(map(len, row["document_chunks"])) + len(row["prompt_ids"])
                  + max(len(row["answer_ids"]), args.max_new_tokens) + 1 for row in train + dev)
    if longest > model.config.max_position_embeddings:
        raise ValueError("Prepared inputs/generation exceed declared model window; no automatic shortening")
    modules = attach_lora(model, args.j, args.rank, args.alpha, torch.float32)
    saved = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    for key, value in {"j": args.j, "rank": args.rank, "alpha": args.alpha}.items():
        if saved.get(key) != value:
            raise ValueError(f"Strong adapter {key} does not match this recipe")
    restore_flat(modules, saved["named"])
    del saved
    comem = CoMem(model, resume_j=args.j)
    arm = configuration_for_arm(args.arm, args.rho)
    reader = SparseCoMemReader(comem, fusion_layer=args.m,
        gradient_checkpointing=True, **arm)
    reader.train()  # The reader enables checkpointing only in training mode.
    sink_id = tokenizer.bos_token_id
    if sink_id is None:
        sink_id = tokenizer.eos_token_id
    with torch.no_grad():
        sink = comem.write_chunk([int(sink_id)]).detach()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(.9, .95), weight_decay=0., foreach=False)
    recipe = {key: value for key, value in vars(args).items()
              if key not in {"out", "device", "resume", "stop_after", "skip_initial_eval"}}
    recipe.update(train_sha256=digest(args.train), dev_sha256=digest(args.dev),
        init_adapter_sha256=digest(args.init_adapter), reader_sha256=digest(HERE / "sparse_reader.py"),
        objective="answer-only CE; prompt-only route; fixed writer; strong reader LoRA continuation",
        train_ids=[row["id"] for row in train], dev_ids=[row["id"] for row in dev], **arm)
    metadata = {"recipe": recipe, "model": model.config.to_dict(), "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(device),
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "formal_inference_timing": False,
        "formal_inference_memory": False, "trainable_parameters": sum(p.numel() for p in parameters),
        "smoke_examples": [{"id": r["id"], "total_tokens": total_tokens(r)} for r in train] if args.smoke else None,
        "gpu_admission": admissions,
        "score_protocol": "max-reference normalized token F1 on held-out Qasper-train pilot; not official test leaderboard"}
    step, cursor, raw_tokens, target_tokens, training_seconds = 0, 0, 0, 0, 0.
    last_loss, first_gradient = None, None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["metadata"]["recipe"] != recipe:
            raise ValueError("Resume recipe/data/reader code differs")
        restore_flat(modules, saved["named"])
        optimizer.load_state_dict(saved["optimizer"])
        step, cursor = saved["step"], saved["cursor"]
        raw_tokens, target_tokens = saved["raw_tokens"], saved["target_tokens"]
        training_seconds = saved["training_seconds"]
        last_loss, first_gradient = saved.get("last_loss"), saved.get("gradient_check")
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_cuda"], device)
        del saved
    atomic_json(out / "metadata.json", metadata)

    def status(phase):
        atomic_json(out / "status.json", {"phase": phase, "complete": phase == "complete",
            "step": step, "target_steps": args.steps, "cursor": cursor, "loss": last_loss,
            "raw_tokens": raw_tokens, "target_tokens": target_tokens,
            "training_seconds": training_seconds, "gradient_check": first_gradient,
            "checkpoint": str(out / "last.pt"), "arm": args.arm, "gpu": metadata["gpu"],
            "physical_gpu": metadata["physical_gpu"], "formal_inference_timing": False,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def save(phase):
        state = {"step": step, "cursor": cursor, "raw_tokens": raw_tokens,
            "target_tokens": target_tokens, "training_seconds": training_seconds,
            "last_loss": last_loss, "gradient_check": first_gradient, "metadata": metadata,
            "named": flat_state(modules), "j": args.j, "rank": args.rank, "alpha": args.alpha}
        atomic_save(out / "last.pt", {**state, "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device)})
        if step % args.eval_every == 0 or phase == "complete":
            atomic_save(out / f"step{step}.pt", state)
        status(phase)

    def forward(row):
        validate_worker_lease(lease_path, require_idle=False)
        with torch.no_grad():
            memories = [comem.write_chunk(chunk).detach() for chunk in row["document_chunks"]]
        result = reader.forward_answer(sink, memories, row["prompt_ids"], row["answer_ids"],
            probe_indices=row["probe_indices"])
        logits = result["logits"]
        if logits.shape[:2] != (1, len(row["answer_ids"])):
            raise ValueError("Reader answer logits must be directly aligned with answer IDs")
        labels = torch.tensor(row["answer_ids"], device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits[0].float(), labels)
        return loss, result.get("route_stats", {})

    def evaluate(limit):
        selected = sorted(dev, key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['id']}".encode()).hexdigest())[:limit]
        path = out / f"eval_step{step}.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if [r["id"] for r in old.get("records", [])] != [r["id"] for r in selected]:
                raise ValueError("Existing evaluation IDs differ from requested set")
            return
        status("evaluating")
        records = []
        stops = eos_token_ids(tokenizer, model)
        record_path = out / f"eval_step{step}.records.jsonl"
        with record_path.open("w", encoding="utf-8", buffering=1) as stream, evaluation_mode(reader):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for row in selected:
                    validate_worker_lease(lease_path, require_idle=False)
                    memories = [comem.write_chunk(chunk).detach() for chunk in row["document_chunks"]]
                    logits, state = reader.prefill(sink, memories, row["prompt_ids"],
                        probe_indices=row["probe_indices"])
                    generated, finish_reason = [], "max_new_tokens"
                    route_stats = getattr(state, "route_stats", None)
                    if route_stats is None and isinstance(state, dict):
                        route_stats = state.get("route_stats")
                    for index in range(args.max_new_tokens):
                        if index and index % 8 == 0:
                            validate_worker_lease(lease_path, require_idle=False)
                        last = logits.reshape(-1, logits.shape[-1])[-1]
                        if not bool(torch.isfinite(last).all()):
                            raise FloatingPointError("Nonfinite generation logits")
                        token = int(last.argmax())
                        generated.append(token)
                        if token in stops:
                            finish_reason = "eos"
                            break
                        if index + 1 < args.max_new_tokens:
                            logits = reader.decode_step(token, state)
                    del state, logits, memories
                    visible = generated[:-1] if finish_reason == "eos" else generated
                    prediction = tokenizer.decode(visible, skip_special_tokens=True).strip()
                    loss, ce_route = forward(row)
                    record = {"id": row["id"], "document_id": row["document_id"],
                        "prediction": prediction, "references": row["references"],
                        "token_f1": max(token_f1(prediction, ref) for ref in row["references"]),
                        "exact_match": max(exact_match(prediction, ref) for ref in row["references"]),
                        "answer_ce": float(loss), "answer_ce_tokens": len(row["answer_ids"]),
                        "generated_ids": generated, "finish_reason": finish_reason,
                        "route_stats": route_stats, "ce_route_stats": ce_route,
                        "candidate_tokens": sum(map(len, row["document_chunks"])),
                        "source": row["source"], "probe_indices": row["probe_indices"]}
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    del loss
        count = len(records)
        ce_tokens = sum(row["answer_ce_tokens"] for row in records)
        summary = {"examples": count, "token_f1": sum(r["token_f1"] for r in records) / count,
            "exact_match": sum(r["exact_match"] for r in records) / count,
            "answer_ce": sum(r["answer_ce"] * r["answer_ce_tokens"] for r in records) / ce_tokens,
            "score_scale": "0-to-1", "metric_protocol": metadata["score_protocol"],
            "decoding": "greedy-natural-eos", "max_new_tokens": args.max_new_tokens,
            "formal_inference_timing": False}
        atomic_json(path, {"summary": summary, "records": records})
        print(json.dumps({"event": "evaluation", "step": step, "summary": summary}), flush=True)

    status("initialized")
    if step == 0 and not args.skip_initial_eval:
        evaluate(args.eval_limit)
    stop = min(args.steps, args.stop_after if args.stop_after is not None else args.steps)
    if stop < step:
        raise ValueError("Stop step precedes resumed step")
    print(json.dumps({"event": "train_start", "arm": args.arm, "step": step,
        "target_steps": args.steps, "train_examples": len(train), "physical_gpu": metadata["physical_gpu"]}), flush=True)
    with (out / "train.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
        while step < stop:
            validate_worker_lease(lease_path, require_idle=False)
            status("training")
            optimizer.zero_grad(set_to_none=True)
            multiplier = ((step + 1) / max(1, args.warmup) if step < args.warmup
                else max(.1, (args.steps - step) / max(1, args.steps - args.warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            torch.cuda.synchronize(device)
            started, losses, routes = time.monotonic(), [], []
            for _ in range(args.grad_accum):
                row = example_at(train, cursor, args.seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, route = forward(row)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite answer loss at step {step + 1}")
                (loss / args.grad_accum).backward()
                losses.append(float(loss.detach()))
                routes.append({"id": row["id"], **route})
                raw_tokens += total_tokens(row)
                target_tokens += len(row["answer_ids"])
                cursor += 1
                del loss
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError(f"Nonfinite gradient at step {step + 1}")
            if first_gradient is None:
                norm = math.sqrt(sum(float(p.grad.float().square().sum()) for p in parameters if p.grad is not None))
                if not math.isfinite(norm) or norm == 0:
                    raise RuntimeError("Answer gradients do not reach reader LoRA")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen base has gradients")
                first_gradient = {"reader_norm": norm, "frozen_base_has_grad": False,
                    "trainable_tensors_with_grad": sum(p.grad is not None for p in parameters),
                    "total_trainable_tensors": len(parameters)}
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            validate_worker_lease(lease_path, require_idle=False)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            step += 1
            training_seconds += elapsed
            last_loss = sum(losses) / len(losses)
            record = {"step": step, "loss": last_loss, "seconds": elapsed,
                "training_seconds": training_seconds, "cursor": cursor,
                "raw_tokens": raw_tokens, "target_tokens": target_tokens,
                "grad_norm": grad_norm, "routes": routes}
            stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == stop:
                save("trained_pending_evaluation" if step == args.steps else "training")
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(args.final_eval_limit if step == args.steps else args.eval_limit)
    if step == args.steps:
        evaluate(args.final_eval_limit)
    save("complete" if step == args.steps else "paused")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        path = Path(args.out) / "status.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        atomic_json(path, {**previous, "phase": "failed", "complete": False,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "arm": args.arm, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "automatic_oom_shortening": False})
        raise


if __name__ == "__main__":
    main()
