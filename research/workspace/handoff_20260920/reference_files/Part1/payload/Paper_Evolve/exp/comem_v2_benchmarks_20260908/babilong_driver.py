"""Prepared official BABILong templates/accuracy with explicit complete queries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from comem import CoMem
from eval import _cli
from eval._common import load_backbone, resolve_dense_retriever
from benchmark_pack import BOUNDARY, tokenize_explicit_prompt
from prepare_babilong import TASKS, LENGTHS, load_task, format_prompt, score_prediction


def load_babilong_dataset(_dataset_name, length):
    return {task: load_task(task, length) for task in TASKS}


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model", "--model_path", dest="model_path", required=True)
    p.add_argument("--j", "--resume_j", dest="resume_j", type=_cli.j_type, default=12)
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--reuse_kv_blockdiag", action="store_true")
    p.add_argument("--baseline", default="none")
    p.add_argument("--adapter", "--lora_adapter", dest="lora_adapter", default="")
    p.add_argument("--selector", default="bm25", choices=_cli.SELECTOR_CHOICES)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--sink_tokens", default="bos", choices=["bos", "none"])
    p.add_argument("--retriever_path", default="")
    p.add_argument("--iter_rounds", type=int, default=0)
    p.add_argument("--iter_hop_topk", type=int, default=2)
    p.add_argument("--iter_score", default="meanpool")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--lengths", nargs="+", default=list(LENGTHS))
    p.add_argument("--dataset_name", default="RMT-team/babilong")
    p.add_argument("--n", "--limit", dest="limit", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--out", "--output_dir", dest="output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="sdpa")
    args = p.parse_args()
    _cli.normalize_args(args, task_attrs=("tasks",))
    if len(set(args.tasks)) != len(args.tasks) or any(t not in TASKS for t in args.tasks):
        raise ValueError("Select unique prepared BABILong tasks")
    if len(set(args.lengths)) != len(args.lengths) or any(l not in LENGTHS for l in args.lengths):
        raise ValueError("Select unique prepared BABILong lengths")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model, tok = load_backbone(args.model_path, args.dtype, args.attn_impl, args.device, args.lora_adapter)
    reader = CoMem(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                   block_diagonal=args.reuse_kv_blockdiag, tokenizer=tok)
    retriever = resolve_dense_retriever(args.selector, args.retriever_path, args.device, args.dtype)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    for length in args.lengths:
        data = load_babilong_dataset(args.dataset_name, length)
        for task in args.tasks:
            rows = data[task]
            n = len(rows) if args.limit <= 0 else min(len(rows), args.limit)
            indices = list(range(n))[args.shard_index::args.num_shards]
            if not indices:
                raise ValueError(f"Empty BABILong cell {task}/{length}")
            with (out / f"{task}_{length}_official_explicit{tag}.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "id", "task", "length", "target", "output", "question", "score", "status", "pack"])
                writer.writeheader()
                for index in indices:
                    row = rows[index]
                    # get_formatted_input strips the raw context. Put the marker
                    # after that strip so deleting it restores the official text.
                    marked = format_prompt(dict(row, input=row["input"].strip() + BOUNDARY), task)
                    ids, count, selected, pack = tokenize_explicit_prompt(tok, marked, row["question"],
                        args.chunk_size, args.selector, args.topk, retriever,
                        args.iter_rounds, args.iter_hop_topk, args.iter_score)
                    if pack["read_pack_tokens"] + args.max_new_tokens > model.config.max_position_embeddings:
                        raise ValueError("Read pack plus answer exceeds configured model positions")
                    pred = reader.generate_from_ids(ids.to(args.device), context_token_count=count,
                        selected_indices=selected, chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens)
                    score = score_prediction(pred, row, task)
                    if score not in (0.0, 1.0):
                        raise ValueError("Invalid official BABILong accuracy")
                    writer.writerow(dict(index=index, id=f"{task}/{length}/{index}", task=task, length=length,
                        target=row["target"], output=pred, question=row["question"], score=score,
                        status="ok", pack=json.dumps(pack)))
                    f.flush()
                    print(f"[BABILong] {task}/{length} index={index} score={score}", flush=True)


if __name__ == "__main__":
    main()
