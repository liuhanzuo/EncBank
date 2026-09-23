"""Isolated, explicitly budgeted full-conversation SFT on a remote RTX 3090.

Consumes PreparedConversation.to_dict() JSONL, never truncates/retrieves text,
and never changes the running Qasper trainer. Runtime is operational training
data, not formal inference timing. Importing this module initializes no CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "encbank_v2_benchmarks_20260908"))
from official_sft import PreparedAssistantTurn, PreparedConversation
from official_sft_objective import conversation_loss
from train_8b_baseline import attach_lora, restore_flat, atomic_json, atomic_save, digest
from evaluate_sft import exact_match, token_f1, eos_token_ids

ARMS = {"encbank": ("encbank", 1), "beacon4": ("beacon", 4),
        "beacon8": ("beacon", 8), "pool4": ("pool", 4)}


def require(value, message):
    if not value:
        raise ValueError(message)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def trainable_state(net):
    # Keep resume helpers independent of model/transformers imports for CPU
    # fault injection. Same strict parameter-only state as the existing trainer.
    return {name: p.detach().cpu().clone() for name, p in net.named_parameters() if p.requires_grad}


def restore_trainable(net, saved):
    current = {name: p for name, p in net.named_parameters() if p.requires_grad}
    require(set(current) == set(saved), "Trainable checkpoint keys differ")
    with torch.no_grad():
        for name, parameter in current.items():
            require(parameter.shape == saved[name].shape and bool(torch.isfinite(saved[name]).all()), "Invalid trainable tensor: "+name)
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))


def decode_prepared(row):
    """Validate the serialized objective while retaining extra grouping metadata."""
    require(isinstance(row, dict), "Prepared row must be an object")
    names = {f.name for f in fields(PreparedConversation)}
    require(names <= set(row), f"Missing PreparedConversation fields: {names-set(row)}")
    values = {key: row[key] for key in names}
    for key in ("query_ids", "label_ids", "target_indices"):
        values[key] = tuple(values[key])
    values["document_chunks"] = tuple(tuple(c) for c in values["document_chunks"])
    turn_names = {f.name for f in fields(PreparedAssistantTurn)}
    turns = []
    for item in values["turns"]:
        require(turn_names <= set(item), "Incomplete prepared assistant turn")
        turn = {key: item[key] for key in turn_names}
        for key in ("prompt_ids", "references", "target_indices", "training_target_ids"):
            turn[key] = tuple(turn[key])
        turns.append(PreparedAssistantTurn(**turn))
    values["turns"] = tuple(turns)
    ex = PreparedConversation(**values)
    require(ex.id == ex.conversation_id and isinstance(ex.id, str) and bool(ex.id), "Invalid conversation ID")
    require(ex.split in {"train", "dev"} and bool(ex.document_id) and bool(ex.source), "Invalid source/split identity")
    require(len(ex.query_ids) == len(ex.label_ids) and bool(ex.query_ids), "Query/label lengths differ")
    require(bool(ex.document_chunks) and all(0 < len(c) <= 512 for c in ex.document_chunks), "Invalid full document chunks")
    require(all(len(c) == 512 for c in ex.document_chunks[:-1]), "Interior document chunk is not complete")
    require(all(type(t) is int and t >= 0 for c in ex.document_chunks for t in c), "Invalid document token ID")
    require(all(type(t) is int and t >= 0 for t in ex.query_ids), "Invalid query token ID")
    require(all(type(t) is int and (t == -100 or t >= 0) for t in ex.label_ids), "Invalid label ID")
    indices = tuple(i for i, t in enumerate(ex.label_ids) if t != -100)
    require(indices == ex.target_indices and bool(indices), "Target mask/index mismatch")
    require(len(indices) == ex.target_token_count, "Target budget mismatch")
    require(sum(map(len, ex.document_chunks)) == ex.document_token_count, "Document budget mismatch")
    require(ex.reader_template_token_count == len(ex.query_ids)+1, "Reader template budget mismatch")
    require(ex.raw_uncompressed_read_pack_tokens == 1+ex.document_token_count+len(ex.query_ids), "Raw pack budget mismatch")
    require(ex.assistant_turn_count == len(ex.turns) and bool(ex.turns), "Assistant turn count mismatch")
    require(tuple(i for turn in ex.turns for i in turn.target_indices) == indices, "Turns do not partition labels")
    for index, turn in enumerate(ex.turns):
        require(turn.assistant_turn_index == index and bool(turn.prompt_ids), "Invalid assistant order/prompt")
        require(all(type(t) is int and t >= 0 for t in turn.prompt_ids), "Invalid generation prompt token ID")
        require(bool(turn.answer.strip()) and bool(turn.references), "Empty assistant/reference")
        require(all(isinstance(r, str) for r in turn.references), "Invalid reference string")
        require(tuple(ex.label_ids[i] for i in turn.target_indices) == turn.training_target_ids, "Turn target IDs differ")
        require(turn.target_token_count == len(turn.target_indices), "Turn target count differs")
        require(math.isclose(turn.conversation_token_weight, turn.target_token_count/len(indices)), "Turn weight differs")
    # The serializer stores already shifted labels. Check every observable next
    # token; the final target may be outside query_ids by design.
    require(all(i+1 == len(ex.query_ids) or ex.label_ids[i] == ex.query_ids[i+1] for i in indices), "Labels are not shifted exactly once")
    protocol = ex.protocol
    require(protocol.get("format") == "official-full-conversation-qwen3-encbank-v1", "Unknown preparation protocol")
    require(protocol.get("selector") == "all_original_chunks_in_order" and protocol.get("truncated") is False, "Retrieval/truncated rows are not full-conversation SFT")
    require(protocol.get("chunk_size") == 512 and protocol.get("official_pretraining_LM_branch") == "not included", "Unexpected preparation branch")
    require(protocol.get("min_original_template_tokens", 0) <= ex.original_template_token_count <= protocol.get("max_original_template_tokens", 0), "Original template violates recorded length bounds")
    extras = {key: value for key, value in row.items() if key not in names}
    return PreparedRecord(ex, extras)


class PreparedRecord:
    def __init__(self, prepared, extra_metadata):
        self.prepared, self.extra_metadata = prepared, extra_metadata

    def __getattr__(self, key):
        return getattr(self.prepared, key)


class PreparedDataset:
    """Byte-indexed JSONL; only a few long tokenized conversations stay in RAM."""
    def __init__(self, path, split):
        self.path, self.split = Path(path), split
        self.index, self.cache = {}, OrderedDict()
        self.protocol = None
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                ex = decode_prepared(json.loads(line))
                require(ex.split == split, f"Wrong split in {self.path}: {ex.id}")
                require(ex.id not in self.index, f"Duplicate conversation identity: {ex.id}")
                if self.protocol is None:
                    self.protocol = ex.protocol
                require(ex.protocol == self.protocol, "Prepared protocol varies within a split")
                metadata = {key: ex.extra_metadata.get(key) for key in ("normalized_group_id", "overlap_group_id", "tokenizer_fingerprint", "tokenizer_sha256")}
                self.index[ex.id] = {"offset": offset, "length": len(line), "document_id": ex.document_id,
                    "context_sha256": ex.context_sha256, "source": ex.source, "target_tokens": ex.target_token_count,
                    "raw_pack_tokens": ex.raw_uncompressed_read_pack_tokens,
                    "original_template_tokens": ex.original_template_token_count,
                    "document_tokens": ex.document_token_count, "assistant_turns": ex.assistant_turn_count,
                    "assistant_end_ids": list({t.training_target_ids[-1] for t in ex.turns}),
                    "max_token_id": max(max(ex.query_ids), max(max(c) for c in ex.document_chunks), max(ex.label_ids),
                                        max(max(t.prompt_ids) for t in ex.turns)), **metadata}
        require(bool(self.index), f"Empty {split} prepared dataset")
        self.ids = sorted(self.index)

    def get(self, cid):
        if cid in self.cache:
            value = self.cache.pop(cid)
        else:
            item = self.index[cid]
            with self.path.open("rb") as stream:
                stream.seek(item["offset"])
                value = decode_prepared(json.loads(stream.read(item["length"])))
            require(value.id == cid, "Prepared byte index changed")
        self.cache[cid] = value
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return value


def validate_splits(train, dev):
    require(train.protocol == dev.protocol, "Train/dev preparation protocol differs")
    require(not set(train.ids) & set(dev.ids), "Conversation IDs overlap across splits")
    require(all(isinstance(v.get("overlap_group_id"), str) and v["overlap_group_id"]
                for ds in (train, dev) for v in ds.index.values()),
            "Official training requires complete overlap_group_id metadata, not just exact-context split")
    for key in ("document_id", "context_sha256", "normalized_group_id", "overlap_group_id"):
        left = {v[key] for v in train.index.values() if v.get(key)}
        right = {v[key] for v in dev.index.values() if v.get(key)}
        require(not left & right, f"Train/dev {key} overlap")
    # Any supplied grouping field must be present on all rows, not just a subset.
    for key in ("normalized_group_id", "overlap_group_id"):
        values = [v.get(key) for ds in (train, dev) for v in ds.index.values()]
        require(not any(values) or all(isinstance(v, str) and v for v in values), f"Partial {key} coverage")


def validate_tokenizer_fingerprint(train, dev, model_path):
    from prepare_official_pool import tokenizer_receipt
    receipt = tokenizer_receipt(Path(model_path))
    expected = receipt["tokenizer_fingerprint"]
    for dataset in (train, dev):
        for cid, row in dataset.index.items():
            require(row.get("tokenizer_fingerprint") == expected, f"Prepared tokenizer differs from loaded local tokenizer files: {cid}")
    return receipt


def select_ids(train, dev, path=None):
    selection = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    result = []
    for name, dataset in (("train_ids", train), ("dev_ids", dev)):
        ids = selection.get(name, dataset.ids)
        require(isinstance(ids, list) and bool(ids) and len(ids) == len(set(ids)), f"Invalid {name}")
        require(set(ids) <= set(dataset.ids), f"Unknown {name}")
        result.append(ids)
    return (*result, selection)


def sample_schedule(ids, count, seed, order):
    require(bool(ids) and count > 0 and order in {"seeded_shuffle", "selection_order"}, "Invalid sample schedule")
    result, epoch = [], 0
    while len(result) < count:
        current = list(ids)
        if order == "seeded_shuffle":
            random.Random(seed+epoch).shuffle(current)
        result.extend(current[:count-len(result)])
        epoch += 1
    return result


def budget_at(dataset, schedule, cursor):
    require(0 <= cursor <= len(schedule), "Cursor outside sample schedule")
    keys = ("target_tokens", "raw_pack_tokens", "original_template_tokens", "document_tokens", "assistant_turns")
    result = {key: 0 for key in keys}
    sources = Counter()
    for cid in schedule[:cursor]:
        item = dataset.index[cid]
        for key in keys:
            result[key] += item[key]
        sources[item["source"]] += 1
    return {**result, "conversations": cursor, "by_source_conversations": dict(sorted(sources.items()))}


def capture_rng(device=None):
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device is not None else None}


def restore_rng(state, device=None):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    require((state["torch_cuda"] is None) == (device is None), "RNG device branch differs")
    if device is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def fingerprint_tensors(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_checkpoint(net, optimizer, progress, metadata, device=None):
    tensors = trainable_state(net)
    return {"format": "official-sft-resume-v1", "progress": dict(progress), "metadata": metadata,
            "trainable": tensors, "model_state_id": fingerprint_tensors(tensors),
            "configuration": net.configuration() if callable(getattr(net, "configuration", None)) else None,
            "optimizer": optimizer.state_dict(), "rng": capture_rng(device)}


def restore_checkpoint(saved, net, optimizer, metadata, *, grad_accum, device=None):
    require(saved.get("format") == "official-sft-resume-v1", "Not a full resume checkpoint")
    require(saved["metadata"]["recipe"] == metadata["recipe"], "Resume recipe/data/selection/init differs")
    require(saved["metadata"]["model_signature"] == metadata["model_signature"], "Backbone/tokenizer signature differs")
    require(saved["model_state_id"] == fingerprint_tensors(saved["trainable"]), "Saved trainable fingerprint differs")
    progress = dict(saved["progress"])
    require(progress["cursor"] == progress["step"]*grad_accum, "Checkpoint is inside an optimizer step")
    restore_trainable(net, saved["trainable"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"], device)
    return progress


def reconcile_train_log(path, saved_step):
    """Remove only uncommitted trailing steps after a crash; checkpoint is truth."""
    path = Path(path)
    if not path.exists():
        require(saved_step == 0, "Committed training log is missing")
        return
    kept, discarded = [], []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            row = json.loads(line)
            valid = row["step"] == len(kept)+1 and row["step"] <= saved_step
        except (ValueError, KeyError, TypeError):
            valid = False
        if valid and not discarded:
            kept.append(line if line.endswith("\n") else line+"\n")
        else:
            discarded.append(line)
    require(len(kept) == saved_step, "Training log has missing/corrupt committed rows")
    if discarded:
        recovery = path.with_name(path.name+f".discarded-{time.time_ns()}")
        recovery.write_text("".join(discarded), encoding="utf-8")
        temporary = path.with_name(path.name+".recovering")
        temporary.write_text("".join(kept), encoding="utf-8")
        os.replace(temporary, path)


def evaluation_ids(dataset, selected, seed, ce_limit, per_source):
    ranked = sorted(selected, key=lambda cid: (object_digest([seed, cid]), cid))
    ce_ids = ranked[:ce_limit]
    by_source = defaultdict(list)
    for cid in ranked:
        by_source[dataset.index[cid]["source"]].append(cid)
    generation_ids = [cid for source in sorted(by_source) for cid in by_source[source][:per_source]]
    require(ce_ids and generation_ids, "Evaluation must contain CE and true generation")
    return ce_ids, generation_ids


def task_type(ex, prompt_text):
    declared = ex.extra_metadata.get("task_type")
    if declared in {"summarization", "qa", "needle"}:
        return declared, "prepared_task_type"
    if ex.source.endswith("/booksum"):
        return "summarization", "booksum_source"
    if ex.source.endswith("/needle"):
        return "needle", "needle_source"
    summary_request = (r"(?:^|\n)(?:Question:\s*)?(?:Please\s+)?(?:summari[sz]e\b|"
        r"(?:give|provide|write|generate|create)\b[^\n?]{0,80}\b(?:summary|summari[sz]ation)\b|"
        r"(?:can|could)\s+you\s+(?:please\s+)?summari[sz]e\b)")
    if re.search(summary_request, prompt_text, re.I):
        return "summarization", "summary_instruction_in_first_turn_prompt"
    return "qa", "question_source"


def rouge_l(prediction, reference):
    if not prediction.strip() or not reference.strip():
        return float(not prediction.strip() and not reference.strip())
    # The package splits sentences on periods and rejects an empty result.
    if not prediction.replace(".", "").strip() or not reference.replace(".", "").strip():
        return 0.0
    from rouge import Rouge
    previous_limit = sys.getrecursionlimit()
    # Its LCS reconstruction is recursive; full BookSum references can exceed
    # Python's default depth. This changes neither tokens nor the ROUGE formula.
    required_limit = max(previous_limit, len(prediction.split())+len(reference.split())+200)
    try:
        sys.setrecursionlimit(required_limit)
        return float(Rouge(metrics=["rouge-l"]).get_scores(prediction, reference)[0]["rouge-l"]["f"])
    finally:
        sys.setrecursionlimit(previous_limit)


def greedy_record(net, tokenizer, ex, qa_max_new_tokens, summary_max_new_tokens):
    turn = ex.turns[0]
    prompt_text = tokenizer.decode(turn.prompt_ids, skip_special_tokens=False)
    kind, classification = task_type(ex, prompt_text)
    maximum = summary_max_new_tokens if kind == "summarization" else qa_max_new_tokens
    stops = eos_token_ids(tokenizer, net)
    memories = [net.encode_chunk(chunk, cache_device="cpu") for chunk in ex.document_chunks]
    state = net.start_query(turn.prompt_ids, memories)
    generated, finish = [], "max_new_tokens"
    for index in range(maximum):
        logits = state.logits.reshape(-1, state.logits.shape[-1])[-1]
        require(bool(torch.isfinite(logits).all()), "Nonfinite generation logits")
        token = int(logits.argmax())
        generated.append(token)
        if token in stops:
            finish = "eos"
            break
        if index+1 < maximum:
            net.decode_token(token, state)
    del state, memories
    visible = generated[:-1] if finish == "eos" else generated
    prediction = tokenizer.decode(visible, skip_special_tokens=True).strip()
    record = {"id": turn.id, "conversation_id": ex.id, "document_id": ex.document_id,
        "source": ex.source, "source_file": ex.source_file, "source_index": ex.source_index,
        "split": ex.split, "assistant_turn_index": 0, "task_type": kind,
        "task_type_rule": classification, "prompt_text": prompt_text, "prompt_ids": list(turn.prompt_ids),
        "references": list(turn.references), "prediction": prediction, "generated_ids": generated,
        "generated_tokens": len(generated), "finish_reason": finish, "max_new_tokens": maximum,
        "generation_completed": True, "document_tokens": ex.document_token_count,
        "context_sha256": ex.context_sha256, "extra_metadata": ex.extra_metadata}
    if kind == "summarization":
        record["rouge_l"] = max(rouge_l(prediction, ref) for ref in turn.references)
    else:
        record["exact_match"] = max(exact_match(prediction, ref) for ref in turn.references)
        record["token_f1"] = max(token_f1(prediction, ref) for ref in turn.references)
    return record


def evaluation_signature(step, state_id, recipe_digest, ce_ids, generation_ids, qa_max, summary_max):
    return {"step": step, "model_state_id": state_id, "recipe_digest": recipe_digest,
            "ce_conversation_ids": list(ce_ids), "generation_conversation_ids": list(generation_ids),
            "generation_turn": "first_assistant_only", "qa_max_new_tokens": qa_max,
            "summary_max_new_tokens": summary_max}


def validate_evaluation(result, signature):
    require(result.get("complete") is True and result.get("signature") == signature, "Evaluation incomplete or belongs to another checkpoint/protocol")
    ce, generated = result.get("ce_records", []), result.get("generation_records", [])
    require([r.get("conversation_id") for r in ce] == signature["ce_conversation_ids"], "CE record IDs differ")
    require([r.get("conversation_id") for r in generated] == signature["generation_conversation_ids"], "Generation record IDs differ")
    for row in ce:
        require(row.get("target_tokens", 0) > 0 and math.isfinite(row.get("ce_sum", float("nan"))), "Invalid conversation CE")
    for row in generated:
        require(row.get("generation_completed") is True and bool(row.get("generated_ids")), "Real generation did not finish")
        require(row.get("assistant_turn_index") == 0 and row.get("references"), "Missing first-turn generation/reference")
        require(row.get("finish_reason") in {"eos", "max_new_tokens"}, "Invalid generation ending")
        require(row.get("generated_tokens") == len(row["generated_ids"]), "Generation token count differs")
        scores = ["rouge_l"] if row.get("task_type") == "summarization" else ["exact_match", "token_f1"]
        require(all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in scores), "Missing/invalid generated quality score")


def summarize_evaluation(ce_records, generation_records):
    result = {"ce_by_source": {}, "generation_by_source_and_task": {}, "quality_aggregate": None,
        "note": "No cross-task quality aggregate; CE is auxiliary, first-turn generation is a small development diagnostic.",
        "rouge_l_implementation": "python rouge package, rouge-l F1; no stemming", "score_scale": "0-to-1"}
    for source in sorted({r["source"] for r in ce_records}):
        records = [r for r in ce_records if r["source"] == source]
        total = sum(r["target_tokens"] for r in records)
        result["ce_by_source"][source] = {"conversations": len(records), "target_tokens": total,
            "token_weighted_ce": sum(r["ce_sum"] for r in records)/total,
            "conversation_mean_ce": sum(r["conversation_ce"] for r in records)/len(records)}
    for source, kind in sorted({(r["source"], r["task_type"]) for r in generation_records}):
        records = [r for r in generation_records if r["source"] == source and r["task_type"] == kind]
        names = ("rouge_l",) if kind == "summarization" else ("exact_match", "token_f1")
        result["generation_by_source_and_task"][source+":"+kind] = {"examples": len(records),
            "length_limited": sum(r["finish_reason"] == "max_new_tokens" for r in records),
            **{k: sum(r[k] for r in records)/len(records) for k in names}}
    return result


def evaluate_checkpoint(path, signature, ce_fn, generation_fn):
    """Commit only complete evaluations. A crash reruns the bounded fixed set."""
    path = Path(path)
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_evaluation(result, signature)
        return result
    partial = path.with_name(path.name+".partial.jsonl")
    ce_records, generation_records = [], []
    with partial.open("w", encoding="utf-8", buffering=1) as stream:
        for cid in signature["ce_conversation_ids"]:
            record = ce_fn(cid)
            ce_records.append(record)
            stream.write(json.dumps({"kind": "ce", "record": record}, ensure_ascii=False)+"\n")
        for cid in signature["generation_conversation_ids"]:
            record = generation_fn(cid)
            generation_records.append(record)
            stream.write(json.dumps({"kind": "generation", "record": record}, ensure_ascii=False)+"\n")
    result = {"complete": True, "signature": signature, "ce_records": ce_records,
        "generation_records": generation_records, "summary": summarize_evaluation(ce_records, generation_records),
        "completed_utc": utc_now(), "formal_inference_timing": False}
    validate_evaluation(result, signature)
    atomic_json(path, result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "init-adapter", "train-prepared", "dev-prepared", "out"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expected-gpu-substring", default="3090")
    ap.add_argument("--selection-json")
    ap.add_argument("--sample-order", choices=["seeded_shuffle", "selection_order"], default="seeded_shuffle")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beacon-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head-chunk-size", type=int, default=128)
    ap.add_argument("--mlp-chunk-tokens", type=int, default=0,
        help="Token-block complete MLP with recomputation; 0 keeps the original execution")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-ce-limit", type=int, default=5)
    ap.add_argument("--eval-per-source", type=int, default=1)
    ap.add_argument("--qa-max-new-tokens", type=int, default=128)
    ap.add_argument("--summary-max-new-tokens", type=int, default=1024)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--skip-initial-eval", action="store_true")
    ap.add_argument("--validate-only", action="store_true", help="CPU input/budget validation, no model or GPU")
    args = ap.parse_args()
    require(min(args.steps, args.grad_accum, args.save_every, args.eval_every, args.head_chunk_size,
                args.eval_ce_limit, args.eval_per_source, args.qa_max_new_tokens, args.summary_max_new_tokens) > 0, "All budgets must be positive")
    require(args.stop_after is None or args.stop_after > 0, "stop-after must be positive")
    require(args.mlp_chunk_tokens >= 0, "mlp-chunk-tokens must be nonnegative")
    train, dev = PreparedDataset(args.train_prepared, "train"), PreparedDataset(args.dev_prepared, "dev")
    validate_splits(train, dev)
    tokenizer_files = validate_tokenizer_fingerprint(train, dev, args.model)
    train_ids, dev_ids, selection = select_ids(train, dev, args.selection_json)
    schedule = sample_schedule(train_ids, args.steps*args.grad_accum, args.seed, args.sample_order)
    final_budget = budget_at(train, schedule, len(schedule))
    ce_ids, gen_ids = evaluation_ids(dev, dev_ids, args.seed, args.eval_ce_limit, args.eval_per_source)
    recipe = {k: v for k, v in vars(args).items() if k not in {"out", "device", "resume", "stop_after", "skip_initial_eval", "validate_only", "expected_gpu_substring"}}
    # Preserve legacy recipes when the new scheduling option is not enabled.
    if args.mlp_chunk_tokens == 0:
        recipe.pop("mlp_chunk_tokens")
    else:
        from chunked_mlp import EXECUTION_VERSION
        recipe["mlp_execution"] = EXECUTION_VERSION
    recipe.update(train_prepared_sha256=digest(args.train_prepared), dev_prepared_sha256=digest(args.dev_prepared),
        init_adapter_sha256=digest(args.init_adapter), selection_sha256=digest(args.selection_json) if args.selection_json else None,
        selected_train_ids=train_ids, selected_dev_ids=dev_ids, schedule_sha256=object_digest(schedule),
        final_budget=final_budget, preparation_protocol=train.protocol,
        tokenizer_fingerprint=tokenizer_files["tokenizer_fingerprint"],
        eval_ce_ids=ce_ids, eval_generation_ids=gen_ids,
        objective="mean complete-assistant-token CE within each conversation, then equal mean across accumulated conversations",
        writer="all original 512-token source chunks; no retrieval or target truncation",
        mixed_pretraining_replay=False)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    input_report = {"recipe": recipe, "selection_metadata": selection,
        "prepared_extra_fields_preserved": True,
        "split_check": "document/context hashes plus provided normalized/overlap groups; no claim of original-book identity",
        "train_source_counts": dict(Counter(train.index[i]["source"] for i in train_ids)),
        "dev_source_counts": dict(Counter(dev.index[i]["source"] for i in dev_ids))}
    if args.validate_only:
        atomic_json(out/"input_validation.json", input_report)
        print(json.dumps({"validated": True, "final_budget": final_budget, "cuda_initialized": torch.cuda.is_initialized()}), flush=True)
        return
    require(platform.system() != "Windows", "Training is remote-only; local Windows GPU is reserved for formal inference")
    require(args.device.startswith("cuda") and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Explicit CUDA_VISIBLE_DEVICES and CUDA device required")
    require(not (out/"last.pt").exists() or bool(args.resume), "Existing checkpoint requires --resume")
    require(args.resume or not (out/"train.jsonl").exists(), "Existing log requires an explicit resume checkpoint")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    random.seed(args.seed); torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    device = torch.device(args.device); torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    require(args.expected_gpu_substring in gpu and torch.cuda.is_bf16_supported(), "Expected remote GPU with BF16 support required")
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from train_sft import SFTModel, lr_multiplier
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    from rouge import Rouge  # fail before loading/training if metric dependency is absent
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True, trust_remote_code=False)
    native_ends = eos_token_ids(tokenizer)
    require(all(set(item["assistant_end_ids"]) <= native_ends for ds in (train, dev) for item in ds.index.values()),
            "Prepared assistant endings differ from native tokenizer EOS")
    initial = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
    require(all(initial.get(k) == v for k, v in {"j": args.j, "rank": args.rank, "alpha": args.alpha}.items()), "Original Encbank adapter dimensions differ")
    require(initial.get("step") == 4000 and initial.get("path") == "base", "Initialize from the original published-path 4000-step Encbank adapter")
    require(all(bool(torch.isfinite(t).all()) for t in initial["named"].values()), "Initialization adapter contains nonfinite values")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    require(model.config.model_type == "qwen3" and model.config.attention_dropout == 0, "Validated only for dense Qwen3 without attention dropout")
    require(max(v["max_token_id"] for ds in (train, dev) for v in ds.index.values()) < model.config.vocab_size, "Prepared IDs exceed model vocabulary")
    modules = attach_lora(model, args.j, args.rank, args.alpha, torch.float32)
    restore_flat(modules, initial["named"]); del initial
    if args.mlp_chunk_tokens:
        from chunked_mlp import install_chunked_mlp
        install_chunked_mlp(model, args.mlp_chunk_tokens)
    sink = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    mode, ratio = ARMS[args.arm]
    net = SFTModel(model, mode=mode, split=args.j, compression_ratio=ratio,
        sink_token_id=int(sink), writer_id=f"{out.name}:step0")
    net.eval()
    reader_params = [p for name, p in net.named_parameters() if p.requires_grad and name != "beacon_embedding"]
    groups = [{"params": reader_params, "lr": args.lr, "base_lr": args.lr}]
    if mode == "beacon":
        groups.append({"params": [net.beacon_embedding], "lr": args.beacon_lr, "base_lr": args.beacon_lr})
    params = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0, foreach=False)
    model_signature = {"config": model.config.to_dict(), "tokenizer_template_sha256": object_digest(tokenizer.chat_template),
                       "tokenizer_files": tokenizer_files["files"]}
    metadata = {"recipe": recipe, "model_signature": model_signature, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": gpu, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "precision": "BF16 backbone, FP32 LoRA/beacon masters, BF16 autocast", "formal_inference_timing": False,
        "trainable_parameters": sum(p.numel() for p in params)}
    progress = {"step": 0, "cursor": 0, "training_seconds": 0.0, "peak_training_allocated_gib": 0.0, "last_loss": None,
                "gradient_check": None, "phase": "ready", "budget": budget_at(train, schedule, 0)}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        progress = restore_checkpoint(saved, net, optimizer, metadata, grad_accum=args.grad_accum, device=device)
        del saved
    require(progress["step"] <= args.steps, "Resumed step exceeds total budget")
    require(progress["budget"] == budget_at(train, schedule, progress["cursor"]), "Resumed raw/target budget differs")
    reconcile_train_log(out/"train.jsonl", progress["step"])
    atomic_json(out/"metadata.json", metadata)
    atomic_json(out/"input_validation.json", input_report)
    net.writer_id = f"{out.name}:step{progress['step']}"
    current_state_id = None

    def save(phase):
        nonlocal current_state_id
        progress["phase"] = phase
        saved = make_checkpoint(net, optimizer, progress, metadata, device)
        current_state_id = saved["model_state_id"]
        if phase == "complete":
            signature = evaluation_signature(progress["step"], current_state_id, object_digest(recipe), ce_ids, gen_ids,
                                             args.qa_max_new_tokens, args.summary_max_new_tokens)
            require(progress["step"] == args.steps, "Training budget not reached")
            validate_evaluation(json.loads((out/f"eval_step{progress['step']}.json").read_text(encoding="utf-8")), signature)
        atomic_save(out/"last.pt", saved)
        if phase in {"evaluating", "evaluating_final", "complete"}:
            weights = {k: v for k, v in saved.items() if k not in {"optimizer", "rng"}}
            weights["format"] = "official-sft-weights-v1"
            atomic_save(out/f"step{progress['step']}.pt", weights)
        atomic_json(out/"status.json", {**progress, "target_steps": args.steps, "complete": phase == "complete",
            "updated_utc": utc_now(), "checkpoint": str(out/"last.pt"), "model_state_id": current_state_id,
            "gpu": gpu, "physical_gpu": metadata["physical_gpu"], "formal_inference_timing": False,
            "peak_training_allocated_gib": progress["peak_training_allocated_gib"]})

    def evaluate(final=False):
        save("evaluating_final" if final else "evaluating")
        signature = evaluation_signature(progress["step"], current_state_id, object_digest(recipe), ce_ids, gen_ids,
                                         args.qa_max_new_tokens, args.summary_max_new_tokens)
        state_rng = capture_rng(device)
        def ce_fn(cid):
            ex = dev.get(cid)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ce = float(conversation_loss(net, ex.document_chunks, ex.query_ids, ex.labels, head_chunk_size=args.head_chunk_size))
            require(math.isfinite(ce), "Nonfinite development CE")
            return {"conversation_id": cid, "document_id": ex.document_id, "source": ex.source,
                    "assistant_turns": ex.assistant_turn_count, "target_tokens": ex.target_token_count,
                    "conversation_ce": ce, "ce_sum": ce*ex.target_token_count}
        def gen_fn(cid):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return greedy_record(net, tokenizer, dev.get(cid), args.qa_max_new_tokens, args.summary_max_new_tokens)
        try:
            result = evaluate_checkpoint(out/f"eval_step{progress['step']}.json", signature, ce_fn, gen_fn)
        finally:
            restore_rng(state_rng, device)
        print(json.dumps({"event": "evaluation", "step": progress["step"], "summary": result["summary"]}), flush=True)

    save("ready")
    if progress["step"] == 0 and not args.skip_initial_eval:
        evaluate()
    stop = min(args.steps, args.stop_after or args.steps)
    require(stop >= progress["step"], "Smoke stop precedes resumed step")
    with (out/"train.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        while progress["step"] < stop:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"]*lr_multiplier(progress["step"], args.steps, args.warmup)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device); started = time.monotonic()
            losses, batch_ids = [], []
            next_cursor = progress["cursor"]
            for _ in range(args.grad_accum):
                cid = schedule[next_cursor]; ex = train.get(cid)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = conversation_loss(net, ex.document_chunks, ex.query_ids, ex.labels, head_chunk_size=args.head_chunk_size)
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                (loss/args.grad_accum).backward()
                losses.append(float(loss.detach())); batch_ids.append(cid); next_cursor += 1
                del loss
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params), "Nonfinite gradients")
            if progress["gradient_check"] is None:
                reader_norm = math.sqrt(sum(float(p.grad.float().square().sum()) for p in reader_params if p.grad is not None))
                beacon_norm = float(net.beacon_embedding.grad.norm()) if net.beacon_embedding.grad is not None else None
                require(reader_norm > 0 and (mode != "beacon" or beacon_norm and beacon_norm > 0), "CE did not reach reader/beacon")
                require(not any(p.grad is not None for p in net.parameters() if not p.requires_grad), "Frozen backbone received gradients")
                progress["gradient_check"] = {"reader_norm": reader_norm, "beacon_norm": beacon_norm, "frozen_base_has_grad": False}
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            optimizer.step()
            torch.cuda.synchronize(device); seconds = time.monotonic()-started
            progress.update(step=progress["step"]+1, cursor=next_cursor, last_loss=sum(losses)/len(losses),
                            training_seconds=progress["training_seconds"]+seconds, budget=budget_at(train, schedule, next_cursor),
                            peak_training_allocated_gib=max(progress["peak_training_allocated_gib"], torch.cuda.max_memory_allocated(device)/2**30))
            net.writer_id = f"{out.name}:step{progress['step']}"
            row = {**progress, "phase": "training", "seconds": seconds, "grad_norm": norm,
                   "conversation_ids": batch_ids, "learning_rates": [g["lr"] for g in optimizer.param_groups]}
            log.write(json.dumps(row)+"\n"); log.flush()
            print(json.dumps(row), flush=True)
            if progress["step"] % args.save_every == 0 or progress["step"] == stop:
                save("training")
            if progress["step"] % args.eval_every == 0 and progress["step"] < args.steps:
                evaluate()
                save("training")
    if progress["step"] == args.steps:
        # This is deliberately outside the loop: a checkpoint saved immediately
        # before final evaluation resumes generation even with zero steps left.
        evaluate(final=True)
        save("complete")
    else:
        save("paused_smoke")
    print(json.dumps({"event": "done", "phase": progress["phase"], "step": progress["step"], "complete": progress["phase"] == "complete"}), flush=True)


if __name__ == "__main__":
    main()
