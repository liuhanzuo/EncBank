"""CPU-only fixed support-intervention inputs; no model weights are loaded."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import official_qa_driver as qa
from diagnose_gold_support import coverage, span_tokens
from diagnose_locomo_evidence import turn_spans, evidence_ids

HERE = Path(__file__).resolve().parent
CHUNK, TOPK, SEED = 512, 12, 42
ARMS = ("fix_all", "j0", "fix_none")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def make_pack(context_ids, query_ids, selected):
    chunks = [context_ids[i:i + CHUNK] for i in range(0, len(context_ids), CHUNK)]
    require(selected == sorted(set(selected)) and all(0 <= i < len(chunks) for i in selected), "Invalid chronological pack")
    return {"input_tokens": len(context_ids) + len(query_ids), "context_tokens": len(context_ids),
        "query_tokens": len(query_ids), "context_chunks": len(chunks), "selected_indices": selected,
        "read_pack_tokens": 1 + len(query_ids) + sum(len(chunks[i]) for i in selected),
        "truncation": "none", "context_query_boundary": "explicit; independently tokenized; no padding"}


def forced_indices(required, ranked, natural):
    k = len(natural)
    if len(required) > k:
        return None
    selected = set(required)
    for i in ranked:
        if len(selected) == k:
            break
        selected.add(i)
    require(len(selected) == k, "Could not fill same-k pack")
    return sorted(selected)


def minimum_union(facts):
    """Exactly minimize chunks covering one complete occurrence of every fact."""
    possibilities = {frozenset()}
    for fact in facts:
        candidates = {frozenset(s["chunks"]) for s in fact["span_coverage"] if s["token_n"] > 0}
        if not candidates:
            return None
        merged = {previous | candidate for previous in possibilities for candidate in candidates}
        # A strict superset can never improve a later union, so discard it.
        possibilities = {s for s in merged if not any(t < s for t in merged)}
    return sorted(min(possibilities, key=lambda s: (len(s), sorted(s)))) if facts else None


def source_samples():
    options = SimpleNamespace(tasks=["hotpotqa"], data_dir=HERE / "data/longbench", max_samples=-1,
        num_shards=1, shard_index=0, categories=[1, 2, 3, 4, 5], seed=SEED,
        locomo_data=HERE / "data/locomo/locomo10.json")
    return [("longbench", s) for s in qa.longbench_samples(options)] + [("locomo", s) for s in qa.locomo_samples(options)]


def locomo_spans(context, conversation):
    turns = turn_spans(context, conversation)
    headers, turn_sessions, cursor = {}, {}, 0
    for session in sorted(int(k.split("_")[1]) for k in conversation if re.fullmatch(r"session_\d+", k)):
        fragment = "\nDATE: " + conversation[f"session_{session}_date_time"] + "\nCONVERSATION:\n"
        start = context.find(fragment, cursor)
        require(start >= 0, "Missing session DATE header")
        headers[session] = (start, start + len(fragment))
        cursor = start + len(fragment)
        for turn in conversation[f"session_{session}"]:
            if turn.get("dia_id"):
                require(turn["dia_id"] not in turn_sessions, "Duplicate evidence dialogue ID")
                turn_sessions[turn["dia_id"]] = session
    return turns, headers, turn_sessions


def prepare(model, destination):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    require(tok.is_fast, "Fast tokenizer required for exact offsets")
    mappings = {r["id"]: r for r in json.loads((HERE / "data/gold_support/gold_support_mapping.json").read_text(encoding="utf-8"))["records"]}
    hotpot = {str(r["_id"]): r for r in [json.loads(line) for line in (HERE / "data/longbench/hotpotqa.jsonl").read_text(encoding="utf-8").splitlines()]}
    locomo = json.loads((HERE / "data/locomo/locomo10.json").read_text(encoding="utf-8"))
    contexts, cache, inputs, roster = {}, {}, [], []
    verified_contexts = set()
    for number, (source_benchmark, sample) in enumerate(source_samples(), 1):
        identity = {k: sample[k] for k in ("id", "index", "task")}
        record = dict(identity, source_benchmark=source_benchmark, category=sample.get("category"))
        if source_benchmark == "longbench" and not mappings[sample["id"]]["all_support_aligned"]:
            roster.append(dict(record, status="support_alignment_unknown"))
            continue
        labels = evidence_ids(sample.get("evidence", [])) if source_benchmark == "locomo" else None
        if source_benchmark == "locomo" and sample["category"] == 5:
            roster.append(dict(record, status="adversarial_no_answer_support_intervention"))
            continue
        if source_benchmark == "locomo" and not labels:
            roster.append(dict(record, status="no_evidence_annotation"))
            continue
        formatted = tok.apply_chat_template([{"role": "user", "content": sample["marked_prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(formatted.count(qa.BOUNDARY) == 1, "Invalid context/query marker")
        context, query = formatted.split(qa.BOUNDARY)
        context_key = ("hotpotqa_" + sample["id"]) if source_benchmark == "longbench" else sample["id"].split("_")[0]
        if context_key not in cache:
            encoded = tok(context, add_special_tokens=False, return_offsets_mapping=True)
            context_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            cache[context_key] = {"text": context, "ids": context_ids, "offsets": offsets,
                "chunks": [context_ids[i:i + CHUNK] for i in range(0, len(context_ids), CHUNK)]}
            contexts[context_key] = {"context_text": context, "context_ids": context_ids, "context_ids_sha256": digest(context_ids)}
            if source_benchmark == "locomo":
                ci = int(context_key[4:])
                cache[context_key]["support_spans"] = locomo_spans(context, locomo[ci]["conversation"])
        prepared = cache[context_key]
        require(context == prepared["text"], "Shared conversation context changed")
        context_ids, offsets, chunks = prepared["ids"], prepared["offsets"], prepared["chunks"]
        query_ids = tok.encode(query, add_special_tokens=False)
        bare_ids = tok.encode(sample.get("retrieval_question", sample["question"]), add_special_tokens=False)
        scores = qa.selectors.bm25_scores(chunks, bare_ids)
        ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        natural = sorted(ranked[:TOPK])
        natural_pack = make_pack(context_ids, query_ids, natural)
        if source_benchmark == "longbench":
            raw, mapping = hotpot[sample["id"]], mappings[sample["id"]]
            require(mapping["question"] == raw["input"] and mapping["context_chars"] == len(raw["context"]), "Stale Hotpot support mapping")
            shift = context.find(raw["context"])
            require(shift >= 0 and context.find(raw["context"], shift + 1) < 0, "Ambiguous Hotpot source context")
            support = coverage(mapping, offsets, shift, natural, CHUNK)
            required = minimum_union(support["facts"])
            require(required is not None, "Aligned Hotpot support has empty token span")
            support["scope"] = "all officially marked supporting sentences, exact or Unicode/whitespace aligned within the matched passage title"
        else:
            turns, headers, turn_sessions = prepared["support_spans"]
            missing = [label for label in labels if label not in turns or label not in turn_sessions]
            if missing:
                roster.append(dict(record, status="support_alignment_unknown", unresolved_labels=missing))
                continue
            required_set, facts = set(), []
            for label in labels:
                session = turn_sessions[label]
                turn_indices = list(span_tokens(offsets, *turns[label]))
                header_indices = list(span_tokens(offsets, *headers[session]))
                require(turn_indices and header_indices, "Empty mapped turn/header")
                turn_chunks = sorted({i // CHUNK for i in turn_indices})
                header_chunks = sorted({i // CHUNK for i in header_indices})
                required_set.update(turn_chunks + header_chunks)
                facts.append({"id": label, "session": session, "turn_chunks": turn_chunks,
                    "date_header_chunks": header_chunks, "turn_token_n": len(turn_indices),
                    "date_header_token_n": len(header_indices),
                    "natural_complete_turn_visible": set(turn_chunks).issubset(natural),
                    "natural_date_header_visible": set(header_chunks).issubset(natural)})
            required = sorted(required_set)
            support = {"facts": facts, "scope": "complete annotated dialogue turns plus the DATE/CONVERSATION header of each supporting session; annotations are not assumed exhaustive"}
        oracle = forced_indices(required, ranked, natural)
        record.update(required_chunks=required, natural_pack=natural_pack,
            natural_all_required_visible=set(required).issubset(natural))
        if oracle is None:
            roster.append(dict(record, status="support_exceeds_chunk_budget", required_chunk_n=len(required)))
            continue
        oracle_pack = make_pack(context_ids, query_ids, oracle)
        require(set(required).issubset(oracle) and len(oracle) == len(natural), "Oracle coverage/budget failure")
        # Check the optimized shared-context path against the original driver.
        if context_key not in verified_contexts:
            reference_ids, n_context, selection, pack = qa.tokenize_pack(tok, sample, CHUNK, "bm25", TOPK)
            require(reference_ids[0].tolist() == context_ids + query_ids and n_context == len(context_ids), "Official tokenization mismatch")
            require(selection == natural and pack == natural_pack, "Official natural pack mismatch")
            verified_contexts.add(context_key)
        task = ("hotpotqa" if source_benchmark == "longbench" else "locomo_" + sample["task"]) + "/oracle"
        fixed = {"schema_version": 1, "id": sample["id"], "index": sample["index"], "task": task,
            "source_benchmark": source_benchmark, "source_task": sample["task"], "context_key": context_key,
            "sample": {k: v for k, v in sample.items() if k != "marked_prompt"},
            "query_text": query, "query_ids": query_ids, "bare_question_ids": bare_ids,
            "input_ids_sha256": digest(context_ids + query_ids), "natural_pack": natural_pack, "oracle_pack": oracle_pack,
            "required_chunks": required, "support": support, "natural_all_required_visible": set(required).issubset(natural),
            "read_pack_token_delta": oracle_pack["read_pack_tokens"] - natural_pack["read_pack_tokens"]}
        fixed["fixture_row_sha256"] = digest(fixed)
        inputs.append(fixed)
        roster.append(dict(record, status="ready", oracle_pack=oracle_pack, task=task,
            fixture_row_sha256=fixed["fixture_row_sha256"], read_pack_token_delta=fixed["read_pack_token_delta"]))
        if len(inputs) % 100 == 0:
            print(json.dumps({"source_scanned": number, "eligible_inputs": len(inputs)}), flush=True)
    require(not torch.cuda.is_initialized(), "Preparation initialized CUDA")
    destination.mkdir(parents=True, exist_ok=True)
    qa.write_json(destination / "contexts.json", contexts)
    write_jsonl(destination / "inputs.jsonl", inputs)
    write_jsonl(destination / "selection_roster.jsonl", roster)
    summary = {task: {"eligible_n": len(rows), "natural_all_required_visible_n": sum(r["natural_all_required_visible"] for r in rows),
        "identical_packs_n": sum(r["natural_pack"] == r["oracle_pack"] for r in rows),
        "read_pack_token_delta_counts": dict(Counter(r["read_pack_token_delta"] for r in rows))}
        for task, rows in ((task, [r for r in inputs if r["task"] == task]) for task in sorted({r["task"] for r in inputs}))}
    manifest = {"schema_version": 1, "status": "ready", "created_at": qa.utc_now(), "model_at_preparation": str(model),
        "source_rows": len(roster), "eligible_inputs": len(inputs), "contexts": len(contexts),
        "selection_status_counts": dict(Counter(r["status"] for r in roster)), "tasks": summary,
        "config": {"j": 12, "selector": "bm25", "topk": TOPK, "chunk_size": CHUNK, "seed": SEED,
            "dtype": "bfloat16", "attn_impl": "sdpa", "adapter": "", "pack_variant": "oracle"},
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination / "contexts.json", destination / "inputs.jsonl", destination / "selection_roster.jsonl")},
        "cuda_initialized": False, "official_driver_context_crosschecks": len(verified_contexts),
        "natural_comparison": "reuse original main full outputs after exact ID/sample/config/pack and durable token+generation cache-key checks; incomplete sources remain waiting; no duplicate natural GPU work"}
    qa.write_json(destination / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return inputs, manifest


def build_plans(inputs, manifest, remote_root):
    root = remote_root.rstrip("/")
    workspace = root + "/workspace"
    scripts = workspace + "/exp/comem_v2_benchmarks_20260908"
    for mode in ("smoke", "full"):
        plan = {"schema_version": 1, "cwd": workspace, "workspace_root": workspace,
            "model": root + "/models/Qwen3-8B", "state_dir": root + "/outputs/queue_oracle_support/" + mode,
            "output_root": root + "/outputs/oracle_support/" + mode, "mode": mode, "arms": list(ARMS),
            "env": {"COMEM_REMOTE_QUEUE": "1", "PAPER_EVOLVE_ALLOW_8B": "1", "TOKENIZERS_PARALLELISM": "false"},
            "jobs": [], "scope": "annotated-support forced packs only; main natural controls reused separately; no pooled cross-task metric", "reused_results": [],
            "deferred_natural_reuse": [], "deferred_natural_reuse_n": {arm: 0 for arm in ARMS}}
        groups = defaultdict(list)
        for row in inputs:
            groups[row["task"]].append(row)
        if mode == "smoke":
            # Include one strict Hotpot case and one real multi-hop LoCoMo case.
            groups = {key: [next((r for r in rows if not r["natural_all_required_visible"]), rows[0])]
                for key, rows in groups.items() if key in ("hotpotqa/oracle", "locomo_category_1/oracle")}
        for arm in ARMS:
            for task, rows in sorted(groups.items()):
                # Equal packs are the same intervention input, so reuse the
                # already planned main prediction after the CPU pairing audit.
                # LoCoMo fix_none has no main natural run and must be generated.
                reused = [r for r in rows if r["natural_pack"] == r["oracle_pack"]
                    and (arm in ("fix_all", "j0") or r["source_benchmark"] == "longbench")]
                if reused:
                    plan["deferred_natural_reuse"].append({"arm": arm, "task": task,
                        "indices": [r["index"] for r in reused], "ids": [r["id"] for r in reused],
                        "expected_n": len(reused), "provenance": "derived_from_natural after complete source/config/token/pack validation; pending source stays pending"})
                    plan["deferred_natural_reuse_n"][arm] += len(reused)
                reused_indices = {r["index"] for r in reused}
                rows = [r for r in rows if r["index"] not in reused_indices]
                for start in range(0, len(rows), 100):
                    batch = rows[start:start + 100]
                    tag = task.replace("/", "_") + f"_part{start // 100:02d}"
                    relative = f"{arm}/{tag}"
                    output = plan["output_root"] + "/" + relative
                    argv = [scripts + "/run_oracle_support.py", "--inputs", scripts + "/data/oracle_support/inputs.jsonl",
                        "--expected-inputs-sha256", manifest["files"]["inputs.jsonl"], "--task", task,
                        "--indices", ",".join(str(r["index"]) for r in batch), "--arm", arm,
                        "--model", plan["model"], "--out", output]
                    if mode == "smoke":
                        argv.append("--smoke-only")
                    plan["jobs"].append({"id": f"oracle_support__{arm}__{tag}", "argv": argv, "depends_on": [],
                        "benchmark": "oracle_support", "arm": arm, "method": arm, "pack_variant": "oracle",
                        "output": output, "relative_output": relative, "layout": "official_qa", "expected_n": len(batch),
                        "cells": [{"key": task, "task": task, "length": None, "metric": "f1",
                            "indices": [r["index"] for r in batch], "expected_n": len(batch)}]})
        qa.write_json(HERE / f"oracle_support_{mode}_plan.json", plan)
        if mode == "full":
            manifest["full_plan_scope"] = {"gpu_jobs": len(plan["jobs"]),
                "gpu_predictions": sum(j["expected_n"] for j in plan["jobs"]),
                "deferred_natural_reuse_n": plan["deferred_natural_reuse_n"],
                "total_condition_predictions": len(inputs) * len(ARMS),
                "completion": "GPU job completion is separate from complete paired analysis; deferred natural controls may still be pending"}
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    p.add_argument("--out", type=Path, default=HERE / "data/oracle_support")
    p.add_argument("--remote-root", default="/data/liuhanzuo/comem_v2_20260908")
    p.add_argument("--plans-only", action="store_true", help="Filter existing immutable fixtures; no tokenization")
    args = p.parse_args()
    if args.plans_only:
        inputs = [json.loads(line) for line in (args.out / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    else:
        require(not (args.out / "manifest.json").exists(), "Prepared fixture already exists; choose a new output, do not replace submitted inputs")
        inputs, manifest = prepare(args.model, args.out)
    manifest = build_plans(inputs, manifest, args.remote_root)
    qa.write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest["full_plan_scope"], indent=2))


if __name__ == "__main__":
    main()
