"""InfiniteBench loop using the prepared official prompt and scoring helpers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from comem import CoMem
from eval import _cli
from eval._common import load_backbone, resolve_dense_retriever
from benchmark_pack import BOUNDARY, tokenize_explicit_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", "--model_path", dest="model_path", required=True)
    p.add_argument("--j", "--resume_j", dest="resume_j", type=_cli.j_type, default=12)
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--reuse_kv_blockdiag", action="store_true")
    p.add_argument("--baseline", default="none")
    p.add_argument("--adapter", "--lora_adapter", dest="lora_adapter", default="")
    p.add_argument("--selector", default="bm25", choices=_cli.SELECTOR_CHOICES)
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    p.add_argument("--retriever_path", default="")
    p.add_argument("--tasks", nargs="+", default=["longbook_qa_eng", "longbook_choice_eng"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--prompt_style", default="yarn-mistral")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--n", "--max_samples", dest="max_samples", type=int, default=-1)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--out", "--output_dir", dest="output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="sdpa")
    args = p.parse_args()
    _cli.normalize_args(args, task_attrs=("tasks",))
    from prepare_infinitebench import load_task, format_prompt, get_answer, score_prediction, MAX_NEW_TOKENS, TASKS
    if len(set(args.tasks)) != len(args.tasks) or any(task not in TASKS for task in args.tasks):
        raise ValueError("Select unique prepared InfiniteBench tasks")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model, tok = load_backbone(args.model_path, args.dtype, args.attn_impl, args.device, args.lora_adapter)
    reader = CoMem(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                   block_diagonal=args.reuse_kv_blockdiag, tokenizer=tok)
    retriever = resolve_dense_retriever(args.selector, args.retriever_path, args.device, args.dtype)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in args.tasks:
        data = load_task(task, args.data_dir)
        if not data:
            raise ValueError(f"No samples for {task}")
        if len({row["id"] for row in data}) != len(data):
            raise ValueError(f"Duplicate official source IDs for {task}")
        indices = list(range(len(data) if args.max_samples <= 0 else min(len(data), args.max_samples)))
        indices = indices[args.shard_index::args.num_shards]
        if not indices:
            raise ValueError(f"Empty shard for {task}")
        outfile = outdir / f"{task}_{args.shard_index}.jsonl"
        values = []
        with outfile.open("w", encoding="utf-8") as f:
            for index in indices:
                row = data[index]
                prompt = format_prompt(dict(row, context=row["context"] + BOUNDARY), task,
                                       prompt_style=args.prompt_style)
                question = row.get("input") or row.get("question") or ""
                ids, context_count, selected, pack = tokenize_explicit_prompt(tok, prompt,
                    str(question), args.chunk_size, args.selector, args.topk, retriever)
                maxgen = args.max_new_tokens or MAX_NEW_TOKENS[task]
                if pack["read_pack_tokens"] + maxgen > model.config.max_position_embeddings:
                    raise ValueError("Read pack plus answer exceeds configured model positions")
                pred = reader.generate_from_ids(ids.to(args.device), chunk_size=args.chunk_size,
                    context_token_count=context_count, selected_indices=selected, max_new_tokens=maxgen)
                score = float(score_prediction(pred, row, task))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid official InfiniteBench score")
                values.append(score)
                result = {"index": index, "id": row.get("id"), "task": task, "pred": pred,
                          "answers": get_answer(row, task), "score": score,
                          "input_tokens": int(ids.numel()), "truncation": "none",
                          "prompt_style": args.prompt_style, "pack": pack, "status": "ok"}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[InfiniteBench] {task} {len(values)}/{len(indices)} score={score:.4f}", flush=True)
        summary[task] = {"n": len(values), "score": sum(values) / len(values), "scale": "0..1"}
    (outdir / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
