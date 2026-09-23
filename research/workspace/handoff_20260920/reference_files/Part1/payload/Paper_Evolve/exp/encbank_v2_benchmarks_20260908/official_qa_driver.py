"""LongBench/LoCoMo official metrics with a controlled Encbank retrieval protocol.

Whole source context, explicit complete query, chronological LoCoMo sessions,
official instructions, Qwen chat template with thinking disabled, greedy decode.
This is an explicitly named Encbank adaptation, not an unchanged HF benchmark run.
GPU admission is owned by the external queue.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import re
import string
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for path in (ROOT / "Encbank", ROOT / "exp", HERE):
    sys.path.insert(0, str(path))

import numpy as np
import torch
from nltk.stem import PorterStemmer
from encbank import selectors
from eval._common import load_backbone
from reader_adapter import ARMS, GenerationCache, make_reader_factory
from run_driver import OutputLock, file_info, utc_now, write_json

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain",
                  4: "single_hop", 5: "adversarial"}
LB_TASKS = ("qasper", "hotpotqa", "narrativeqa", "2wikimqa", "musique", "multifieldqa_en")
BOUNDARY = "\n<ENCBANK_CONTEXT_QUERY_BOUNDARY_20260908>\n"


def compile_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError(f"Missing official metric functions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@lru_cache(maxsize=1)
def official_protocol():
    source = HERE / "protocol_sources"
    lb = compile_functions(source / "longbench/metrics.py",
        {"normalize_answer", "f1_score", "qa_f1_score"},
        {"re": re, "string": string, "Counter": Counter})
    lo = compile_functions(source / "locomo/task_eval/evaluation.py",
        {"normalize_answer", "f1_score", "f1"},
        {"regex": re, "string": string, "Counter": Counter, "ps": PorterStemmer(), "np": np})
    constants = {}
    for node in ast.parse((source / "locomo/task_eval/hf_llm_utils.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"QA_PROMPT", "CONV_START_PROMPT", "ANS_TOKENS_PER_QUES"}:
                    constants[target.id] = ast.literal_eval(node.value)
    prompts = json.loads((source / "longbench/config/dataset2prompt.json").read_text(encoding="utf-8"))
    maxgen = json.loads((source / "longbench/config/dataset2maxlen.json").read_text(encoding="utf-8"))
    return lb, lo, constants, prompts, maxgen


def longbench_samples(args):
    _, _, _, prompts, maxgen = official_protocol()
    for task in args.tasks:
        path = Path(args.data_dir) / (task + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No samples for {task}")
        seen = set()
        for index, row in enumerate(rows):
            if any(k not in row for k in ("context", "input", "answers")) or not row["answers"]:
                raise ValueError(f"Invalid LongBench {task} row {index}")
            source_id = str(row.get("_id", row.get("id", index)))
            if source_id in seen:
                raise ValueError(f"Duplicate LongBench source id {source_id}")
            seen.add(source_id)
            if args.max_samples > 0 and index >= args.max_samples:
                continue
            if index % args.num_shards != args.shard_index:
                continue
            marked = prompts[task].format(**dict(row, context=row["context"] + BOUNDARY))
            yield {"index": index, "id": source_id, "task": task,
                "question": row["input"], "answers": row["answers"],
                "marked_prompt": marked, "max_new_tokens": maxgen[task]}


def history_text(conv, constants):
    history = [constants["CONV_START_PROMPT"].format(conv["speaker_a"], conv["speaker_b"])]
    sessions = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
    for i in sessions:
        history.append("\nDATE: " + conv[f"session_{i}_date_time"] + "\nCONVERSATION:\n")
        for turn in conv[f"session_{i}"]:
            history.append(turn["speaker"] + ' said, "' + turn["text"] + '"\n')
            if "blip_caption" in turn:
                history.append(" and shared %s." % turn["blip_caption"])
            history.append("\n")
    return "".join(history)


def locomo_samples(args):
    _, _, constants, _, _ = official_protocol()
    data = json.loads(Path(args.locomo_data).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty official LoCoMo conversation list")
    rng = random.Random(args.seed)
    global_index, eligible_index = 0, 0
    for ci, conversation in enumerate(data):
        history = history_text(conversation["conversation"], constants)
        for qi, qa in enumerate(conversation["qa"]):
            index = global_index
            global_index += 1
            category = int(qa["category"])
            if category not in CATEGORY_NAMES:
                raise ValueError(f"Invalid LoCoMo category {category}")
            question, option_mapping = qa["question"], None
            if category == 2:
                question += " Use DATE of CONVERSATION to answer with an approximate date."
            if category == 5:
                distractor = qa.get("adversarial_answer", qa.get("answer"))
                if distractor is None:
                    raise ValueError(f"Missing category-5 distractor at {ci}/{qi}")
                options = ["No information available", str(distractor)]
                if rng.random() >= 0.5:
                    options.reverse()
                option_mapping = dict(zip(("a", "b"), options))
                question += " (a) {} (b) {}. Select the correct answer by writing (a) or (b).".format(*options)
                answers = ["No information available"]
            else:
                answers = [str(qa["answer"])]
            if category not in args.categories:
                continue
            eligible = eligible_index
            eligible_index += 1
            if args.max_samples > 0 and eligible >= args.max_samples:
                continue
            if eligible % args.num_shards != args.shard_index:
                continue
            yield {"index": index, "id": f"conv{ci}_qa{qi}", "task": f"category_{category}",
                "category": category, "category_name": CATEGORY_NAMES[category],
                "question": question, "retrieval_question": qa["question"], "answers": answers,
                "evidence": qa.get("evidence", []), "option_mapping": option_mapping,
                "marked_prompt": history + BOUNDARY + "\n\n" + constants["QA_PROMPT"].format(question),
                "max_new_tokens": constants["ANS_TOKENS_PER_QUES"]}


def score_prediction(pred, sample):
    lb, lo, _, _, _ = official_protocol()
    if not isinstance(pred, str) or pred == "[OOM]":
        raise ValueError("Missing/error generation cannot be scored")
    if "category" not in sample:
        return float(max(lb["qa_f1_score"](pred, str(a)) for a in sample["answers"])), pred
    # The HF single-question example evaluates its first nonblank output line.
    lines = [line.strip() for line in pred.replace('\\"', "'").splitlines() if line.strip()]
    decoded = lines[0].lower() if lines else ""
    if sample["category"] == 5:
        # Explicit single options only; empty/invalid/ambiguous output never falls
        # through to option b as in the original HF example.
        choices = set(re.findall(r"\(([ab])\)|\b([ab])\)", decoded))
        labels = {a or b for a, b in choices}
        bare = re.fullmatch(r"(?:answer\s*:\s*)?([ab])[.\s]*", decoded)
        if bare:
            labels.add(bare.group(1))
        if len(labels) == 1:
            decoded = sample["option_mapping"][labels.pop()].lower()
        elif labels:
            return 0.0, decoded
        return float("no information available" in decoded or "not mentioned" in decoded), decoded
    decoded = decoded.replace("(a)", "").replace("(b)", "").replace("a)", "").replace("b)", "").replace("answer:", "").strip()
    answer = sample["answers"][0]
    if sample["category"] == 3:
        answer = answer.split(";")[0].strip()
    scorer = lo["f1"] if sample["category"] == 1 else lo["f1_score"]
    return float(scorer(decoded, answer)), decoded


def tokenize_pack(tok, sample, chunk_size, selector, topk):
    marked = sample["marked_prompt"]
    if marked.count(BOUNDARY) != 1:
        raise ValueError("Ambiguous context/query marker")
    formatted = tok.apply_chat_template([{"role": "user", "content": marked}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    context, query = formatted.split(BOUNDARY)
    context_ids = tok.encode(context, add_special_tokens=False)
    query_ids = tok.encode(query, add_special_tokens=False)
    if not query_ids:
        raise ValueError("Empty query after chat template")
    ids = torch.tensor([context_ids + query_ids], dtype=torch.long)
    chunks = list(ids[0, :len(context_ids)].split(chunk_size)) if context_ids else []
    q_ids = tok.encode(sample.get("retrieval_question", sample["question"]), add_special_tokens=False)
    selected = selectors.select_context_chunk_indices(selector, chunks, q_ids, topk, None)
    return ids, len(context_ids), list(selected), {
        "input_tokens": len(context_ids) + len(query_ids), "context_tokens": len(context_ids),
        "query_tokens": len(query_ids), "context_chunks": len(chunks),
        "selected_indices": list(selected),
        "read_pack_tokens": 1 + len(query_ids) + sum(len(chunks[i]) for i in selected),
        "truncation": "none", "context_query_boundary": "explicit; independently tokenized; no padding"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--benchmark", choices=("longbench", "locomo"), required=True)
    p.add_argument("--arm", choices=ARMS, default="fix_all")
    p.add_argument("--model", required=True)
    p.add_argument("--j", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--adapter", default="", help="Flat lora.pt file or PEFT directory")
    p.add_argument("--selector", choices=("bm25", "recency"), default="bm25")
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--tasks", nargs="+", choices=LB_TASKS, default=list(LB_TASKS))
    p.add_argument("--categories", default="1,2,3,4,5")
    p.add_argument("--data-dir", default=str(HERE / "data/longbench"))
    p.add_argument("--locomo-data", default=str(HERE / "data/locomo/locomo10.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.categories = [int(c) for c in args.categories.replace(",", " ").split()]
    if not args.categories or any(c not in CATEGORY_NAMES for c in args.categories):
        raise ValueError("Categories must be selected from 1,2,3,4,5")
    if args.chunk_size < 1 or args.topk < 1 or args.max_samples == 0 or args.j < 0:
        raise ValueError("Require positive chunk/topk, nonzero sample limit and nonnegative split")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    samples = list(longbench_samples(args) if args.benchmark == "longbench" else locomo_samples(args))
    if not samples or len({(s["task"], s["id"]) for s in samples}) != len(samples):
        raise ValueError("Empty or duplicate selected sample set")
    if args.benchmark == "longbench" and set(args.tasks) != {s["task"] for s in samples}:
        raise ValueError("At least one requested task has no selected samples")
    files = [Path(__file__), HERE / "reader_adapter.py", ROOT / "exp/s15_ruler_lower.py",
        ROOT / "Encbank/encbank/model.py", ROOT / "Encbank/encbank/selectors.py"]
    if args.arm == "cacheblend16":
        files += [HERE / "cacheblend_contextual.py", ROOT / "Encbank/encbank/cacheblend.py"]
    files += list((HERE / "protocol_sources/longbench/config").glob("dataset2*.json"))
    files += [HERE / "protocol_sources/longbench/metrics.py", HERE / "protocol_sources/locomo/task_eval/evaluation.py",
        HERE / "protocol_sources/locomo/task_eval/hf_llm_utils.py"]
    files += ([Path(args.data_dir) / (task + ".jsonl") for task in args.tasks] if args.benchmark == "longbench"
              else [Path(args.locomo_data)])
    if args.adapter:
        adapter = Path(args.adapter)
        files += [adapter] if adapter.is_file() else [p for p in adapter.rglob("*") if p.is_file()]
        if not adapter.exists():
            raise FileNotFoundError(adapter)
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        if (Path(args.model) / name).is_file():
            files.append(Path(args.model) / name)
    config = {"schema_version": 1, "benchmark": args.benchmark, "arm": args.arm,
        "options": dict(vars(args), out=str(args.out)), "files": [file_info(f) for f in sorted(set(files))],
        "protocol": {"source_context": "whole source; no truncation", "query": "explicit complete query",
            "template": "tokenizer.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False)",
            "decoding": "greedy; native Encbank first-token EOS suppression; official per-task token caps",
            "retrieval": "same deterministic token-BM25/recency indices and ordering for every arm",
            "sink": "tokenizer BOS, falling back to tokenizer EOS, for both write/read",
            "locomo": "official HF short-answer prompts; chronological sessions and captions; stable seed option mapping; greedy adaptation",
            "metrics": "official functions compiled from recorded sources; LoCoMo robust explicit option decoding",
            "adapter_merge": "flat lora.pt uses fp32 accumulation; PEFT directory uses PEFT layers"},
        "expected": [{"index": s["index"], "id": s["id"], "task": s["task"]} for s in samples]}
    if args.arm == "cacheblend16":
        config["protocol"]["cacheblend"] = {"variant": "Qwen3 CacheBlend-style port",
            "selection": "layer-1 V summed squared contextual deviation; floor(.16 * context_tokens)",
            "bootstrap_full_layers": 2, "chunk_write_sink": False,
            "cost": "extra layer-1 attention/MLP work; not upstream performance"}
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with OutputLock(out):
        contract = out / "run_config.json"
        if contract.exists() and json.loads(contract.read_text(encoding="utf-8")) != config:
            raise ValueError("Output has different configuration/code/data; use a new --out")
        if not contract.exists():
            if any(p.name != "RUNNING.lock" for p in out.iterdir()):
                raise ValueError("Cannot adopt a nonempty output without a run configuration")
            write_json(contract, config)
        if (out / "COMPLETED.json").exists():
            print(f"Already complete: {out}")
            return 0
        metadata = {"started_at": utc_now(), "status": "running", "n": len(samples), "output_dir": str(out)}
        cache = GenerationCache(out / "generations.sqlite3")
        try:
            torch.manual_seed(args.seed)
            adapter_dir = args.adapter if args.adapter and Path(args.adapter).is_dir() else ""
            model, tok = load_backbone(args.model, args.dtype, args.attn_impl, args.device, adapter_dir)
            if args.adapter and Path(args.adapter).is_file():
                from s15_ruler_lower import apply_lora_pt
                apply_lora_pt(model, args.adapter, fp32_accum=True)
            def constructed(info):
                metadata["reader"] = info
                write_json(out / "metadata.json", metadata)
            reader = make_reader_factory(args.arm, cache, constructed, explicit_pack=True)(model, args.j, tokenizer=tok)
            values = defaultdict(list)
            with (out / "predictions.jsonl").open("w", encoding="utf-8") as output:
                for number, sample in enumerate(samples, 1):
                    ids, n_context, selected, pack = tokenize_pack(tok, sample, args.chunk_size, args.selector, args.topk)
                    if pack["read_pack_tokens"] + sample["max_new_tokens"] > model.config.max_position_embeddings:
                        raise ValueError("Read pack plus answer exceeds configured model positions; change the named protocol")
                    pred = reader.generate_from_ids(ids.to(args.device), context_token_count=n_context,
                        selected_indices=selected, chunk_size=args.chunk_size, max_new_tokens=sample["max_new_tokens"])
                    score, scored_pred = score_prediction(pred, sample)
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Nonfinite or out-of-range official score")
                    result = {k: v for k, v in sample.items() if k != "marked_prompt"}
                    result.update(pred=pred, scored_prediction=scored_pred, score=score, status="ok", pack=pack)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    values[sample["task"]].append(score)
                    print(f"[{args.benchmark}/{args.arm}] {number}/{len(samples)} {sample['id']} score={score:.4f}", flush=True)
            scores = {task: {"n": len(v), "score": 100 * sum(v) / len(v), "scale": "0..100",
                "metric": "accuracy" if task == "category_5" else "f1"} for task, v in values.items()}
            summary = {"benchmark": args.benchmark, "n": len(samples), "tasks": scores,
                "completion_scope": "requested shard/sample selection; not an unqualified full benchmark score"}
            if args.benchmark == "longbench":
                summary["requested_task_macro_f1"] = sum(v["score"] for v in scores.values()) / len(scores)
            else:
                answerable = [v for task, vs in values.items() if task != "category_5" for v in vs]
                if answerable:
                    summary["answerable_micro_f1"] = 100 * sum(answerable) / len(answerable)
                if "category_5" in scores:
                    summary["adversarial_accuracy"] = scores["category_5"]["score"]
            write_json(out / "scores.json", summary)
            metadata.update(status="completed", tasks=list(scores), scores=summary,
                files=[{"file": str(out / "predictions.jsonl"), "records": len(samples)}])
        except BaseException as exc:
            metadata.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            metadata.update(finished_at=utc_now(), reused_generations=cache.hits, new_generations=cache.misses)
            cache.close()
            write_json(out / "metadata.json", metadata)
        write_json(out / "COMPLETED.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
