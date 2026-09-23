"""Single-GPU answer-only SFT for Beacon residuals and matched CoMem controls.

No teacher, no pretraining, no implicit GPU selection. Each process owns one
explicitly assigned GPU. Training runtime is operational data, not paper timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # initialize torch.utils.checkpoint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "comem_v2_benchmarks_20260908"))
from train_8b_baseline import attach_lora, restore_flat, atomic_json, atomic_save, digest
from beacon_comem import BeaconCoMem, BeaconMemory


class SFTModel(BeaconCoMem):
    """Shared inference interface; raw/pool controls use the same frozen writer."""

    def __init__(self, model, *, mode, split, compression_ratio, sink_token_id,
                 writer_id, gradient_checkpointing=True):
        self.mode = mode
        if mode not in {"beacon", "comem", "pool"}:
            raise ValueError("Unknown SFT arm")
        if mode == "comem" and compression_ratio != 1:
            raise ValueError("Uncompressed control must have ratio 1")
        super().__init__(model, split, compression_ratio, sink_token_id,
                         sink_token_id, writer_id,
                         gradient_checkpointing=gradient_checkpointing)
        if mode != "beacon":
            self.beacon_embedding.requires_grad_(False)

    def configuration(self):
        return {**super().configuration(), "mode": self.mode}

    def write_chunk(self, token_ids):
        if self.mode == "beacon":
            return super().write_chunk(token_ids)
        self._sync_reader_location()
        ids = self.reader._as_ids(token_ids)
        hidden = self.reader.write_chunk(ids)
        if self.mode == "pool":
            # Contiguous equal-size groups; the final short group is not padded.
            hidden = torch.cat([part.mean(dim=1, keepdim=True)
                                for part in hidden.split(self.ratio, dim=1)], dim=1)
        return BeaconMemory(hidden, ids.shape[1], self.ratio,
                            self.reader.resume_j, self.writer_id)

    def answer_loss(self, document_chunks, prompt_ids, answer_ids):
        if not prompt_ids or not answer_ids:
            raise ValueError("Need nonempty prompt and supervised answer")
        memories = [self.write_chunk(c) for c in document_chunks]
        # Position P-1 predicts answer[0]; answer tokens alone are supervised.
        query_ids = list(prompt_ids) + list(answer_ids[:-1])
        query = self.reader.write_chunk(query_ids)
        sink = self.reader.write_chunk([self.sink_token_id])
        # CoMem's checkpoint loop checks the input's requires_grad once. A tiny
        # dummy sink leaf enables identical checkpointing for detached controls;
        # it is never optimized or saved, and no frozen parameter receives grad.
        if self.reader.grad_checkpoint and torch.is_grad_enabled() and self.mode != "beacon":
            sink = sink.detach().requires_grad_(True)
        logits = self.reader.read_core(sink, self._hidden_list(memories), query,
                                       logits_tail=len(answer_ids))
        labels = torch.tensor(answer_ids, device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits[0].float(), labels)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_split(train_rows, dev_rows):
    train_docs = {r["document_id"] for r in train_rows}
    dev_docs = {r["document_id"] for r in dev_rows}
    if train_docs & dev_docs:
        raise ValueError("Training and development documents overlap")
    if not train_docs or not dev_docs:
        raise ValueError("Empty training/development split")


def example_at(examples, cursor, seed):
    epoch, offset = divmod(cursor, len(examples))
    order = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(order)
    return examples[order[offset]]


def lr_multiplier(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.1, (total - step) / max(1, total - warmup))


def trainable_state(net):
    return {n: p.detach().cpu().clone() for n, p in net.named_parameters() if p.requires_grad}


def restore_trainable(net, saved):
    current = {n: p for n, p in net.named_parameters() if p.requires_grad}
    if set(current) != set(saved):
        raise ValueError("Trainable checkpoint keys differ")
    with torch.no_grad():
        for n, p in current.items():
            if p.shape != saved[n].shape or not torch.isfinite(saved[n]).all():
                raise ValueError(f"Invalid trainable tensor: {n}")
            p.copy_(saved[n].to(device=p.device, dtype=p.dtype))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--mode", choices=["beacon", "comem", "pool"], required=True)
    ap.add_argument("--ratio", type=int, required=True)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beacon-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=7)
    ap.add_argument("--max-question-tokens", type=int, default=256)
    ap.add_argument("--max-answer-tokens", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-limit", type=int, default=32)
    ap.add_argument("--final-eval-limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--skip-initial-eval", action="store_true")
    args = ap.parse_args()
    if not args.device.startswith("cuda") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Select a remote GPU explicitly using CUDA_VISIBLE_DEVICES and --device")
    if min(args.steps, args.grad_accum, args.save_every, args.eval_every) < 1:
        raise ValueError("Step budgets must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("The fixed recipe requires BF16 support")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise ValueError("Existing checkpoint: use --resume")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    from evaluate_sft import prepare_examples, evaluate_examples

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train_rows, dev_rows = read_rows(args.train), read_rows(args.dev)
    validate_split(train_rows, dev_rows)
    prep = dict(chunk_size=args.chunk_size, max_chunks=args.max_chunks,
                max_question_tokens=args.max_question_tokens,
                max_answer_tokens=args.max_answer_tokens, seed=args.seed)
    train_data = prepare_examples(train_rows, tokenizer, **prep)
    dev_data = prepare_examples(dev_rows, tokenizer, **prep)
    if not train_data or not dev_data:
        raise ValueError("No valid prepared examples")
    if {x.document_id for x in train_data} & {x.document_id for x in dev_data}:
        raise ValueError("Prepared document splits overlap")
    atomic_json(out / "preparation.json", {
        "train": getattr(train_data, "preparation_summary", {"examples": len(train_data)}),
        "dev": getattr(dev_data, "preparation_summary", {"examples": len(dev_data)}),
        "train_skipped": getattr(train_data, "skipped", []),
        "dev_skipped": getattr(dev_data, "skipped", [])})
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    if model.config.model_type != "qwen3" or model.config.attention_dropout != 0:
        raise ValueError("Validated for dense Qwen3 with attention_dropout=0")
    torch.manual_seed(args.seed)
    modules = attach_lora(model, args.j, args.rank, args.alpha, torch.float32)
    initial = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    for key, value in {"j": args.j, "rank": args.rank, "alpha": args.alpha}.items():
        if initial.get(key) != value:
            raise ValueError(f"Initialization adapter {key} mismatch")
    restore_flat(modules, initial["named"])
    del initial
    sink = tokenizer.bos_token_id
    sink = int(tokenizer.eos_token_id if sink is None else sink)
    net = SFTModel(model, mode=args.mode, split=args.j, compression_ratio=args.ratio,
                   sink_token_id=sink, writer_id=f"{out.name}:step0")
    net.eval()  # no dropout; eval() does not disable autograd or LoRA updates
    reader_params = [p for name, p in net.named_parameters()
                     if p.requires_grad and name != "beacon_embedding"]
    groups = [{"params": reader_params, "lr": args.lr, "base_lr": args.lr}]
    if net.beacon_embedding.requires_grad:
        groups.append({"params": [net.beacon_embedding], "lr": args.beacon_lr,
                       "base_lr": args.beacon_lr})
    parameters = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    recipe = {k: v for k, v in vars(args).items()
              if k not in {"out", "device", "resume", "stop_after", "skip_initial_eval"}}
    recipe.update(train_sha256=digest(args.train), dev_sha256=digest(args.dev),
                  init_adapter_sha256=digest(args.init_adapter),
                  objective="answer-only CE; no teacher or pretraining",
                  selection="question-only BM25, selected chunks in original order",
                  train_examples=len(train_data), dev_examples=len(dev_data))
    metadata = {"recipe": recipe, "model": model.config.to_dict(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "trainable_parameters": sum(p.numel() for p in parameters),
                "trainable_beacon_parameters": net.beacon_embedding.numel() if args.mode == "beacon" else 0,
                "precision": "BF16 base, FP32 LoRA/embedding masters, BF16 autocast",
                "formal_inference_timing": False}
    step, cursor, raw_tokens, target_tokens, training_seconds = 0, 0, 0, 0, 0.0
    last_loss, first_gradient = None, None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["metadata"]["recipe"] != recipe:
            raise ValueError("Resume recipe/data/init differs")
        restore_trainable(net, saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        step, cursor = saved["step"], saved["cursor"]
        raw_tokens, target_tokens = saved["raw_tokens"], saved["target_tokens"]
        training_seconds = saved["training_seconds"]
        last_loss, first_gradient = saved.get("last_loss"), saved.get("gradient_check")
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_cuda"], device)
        del saved
    net.writer_id = f"{out.name}:step{step}"
    atomic_json(out / "metadata.json", metadata)

    def save(phase):
        state = {"step": step, "cursor": cursor, "raw_tokens": raw_tokens,
                 "target_tokens": target_tokens, "training_seconds": training_seconds,
                 "last_loss": last_loss, "gradient_check": first_gradient,
                 "metadata": metadata, "configuration": net.configuration(),
                 "trainable": trainable_state(net)}
        atomic_save(out / "last.pt", {**state, "optimizer": optimizer.state_dict(),
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device)})
        if step % args.eval_every == 0 or phase == "complete":
            atomic_save(out / f"step{step}.pt", state)
        atomic_json(out / "status.json", {"phase": phase, "step": step,
            "target_steps": args.steps, "complete": phase == "complete", "loss": last_loss,
            "training_examples_processed": cursor, "raw_tokens": raw_tokens,
            "target_tokens": target_tokens, "training_seconds": training_seconds,
            "peak_training_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "gradient_check": first_gradient, "checkpoint": str(out / "last.pt"),
            "gpu": metadata["gpu"], "physical_gpu": metadata["physical_gpu"],
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def evaluate(count):
        path = out / f"eval_step{step}.json"
        selected = sorted(dev_data, key=lambda ex: hashlib.sha256(
            f"{args.seed}:{ex.id}".encode()).hexdigest())[:count]
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if ([r['id'] for r in previous.get('records', [])] != [ex.id for ex in selected]
                    or previous.get('summary', {}).get('examples') != len(selected)
                    or previous.get('summary', {}).get('answer_ce') is None):
                raise ValueError("Existing evaluation is incomplete or uses different examples")
            return
        net.writer_id = f"{out.name}:step{step}"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = evaluate_examples(net, tokenizer, selected,
                max_new_tokens=args.max_new_tokens, answer_ce=True,
                output_path=out / f"eval_step{step}.records.jsonl")
        atomic_json(path, result)
        print(json.dumps({"event": "evaluation", "step": step,
                          "summary": result.get("summary", {})}, ensure_ascii=False), flush=True)

    if step == 0 and not args.skip_initial_eval:
        evaluate(args.eval_limit)
    stop = min(args.steps, args.stop_after or args.steps)
    if stop < step:
        raise ValueError("Stop step precedes resumed step")
    print(json.dumps({"event": "train_start", "step": step, "stop": stop,
                      "physical_gpu": metadata["physical_gpu"], "recipe": recipe}), flush=True)
    with (out / "train.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        while step < stop:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_multiplier(step, args.steps, args.warmup)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            losses = []
            for _ in range(args.grad_accum):
                ex = example_at(train_data, cursor, args.seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net.answer_loss(ex.document_chunks, ex.prompt_ids, ex.answer_ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                (loss / args.grad_accum).backward()
                losses.append(float(loss.detach()))
                raw_tokens += sum(len(c) for c in ex.document_chunks) + len(ex.prompt_ids) + len(ex.answer_ids)
                target_tokens += len(ex.answer_ids)
                cursor += 1
                del loss
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"Nonfinite gradient at step {step + 1}")
            if first_gradient is None:
                beacon_norm = (float(net.beacon_embedding.grad.norm())
                               if net.beacon_embedding.grad is not None else None)
                reader_norm = math.sqrt(sum(float(p.grad.float().square().sum())
                                            for p in reader_params if p.grad is not None))
                if reader_norm == 0 or (args.mode == "beacon" and not beacon_norm):
                    raise RuntimeError("Answer loss did not reach all intended trainable components")
                if any(p.grad is not None for p in net.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen base has gradients")
                first_gradient = {"beacon_norm": beacon_norm, "reader_norm": reader_norm,
                                  "frozen_base_has_grad": False}
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            training_seconds += elapsed
            step += 1
            net.writer_id = f"{out.name}:step{step}"
            last_loss = sum(losses) / len(losses)
            row = {"step": step, "loss": last_loss, "seconds": elapsed,
                   "training_seconds": training_seconds, "cursor": cursor,
                   "raw_tokens": raw_tokens, "target_tokens": target_tokens,
                   "grad_norm": grad_norm}
            log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == stop:
                save("trained_pending_evaluation" if step == args.steps else "training")
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(args.final_eval_limit if step == args.steps else args.eval_limit)
    # Resume can enter with all optimizer steps done but final evaluation missing.
    if step == args.steps:
        evaluate(args.final_eval_limit)
    save("complete" if step == args.steps else "paused_after_smoke")


if __name__ == "__main__":
    main()
