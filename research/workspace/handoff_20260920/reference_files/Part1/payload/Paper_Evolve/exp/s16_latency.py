"""
S16 -- Per-query latency and measured memory of the repaired read, Qwen3-8B bf16, on the
paper's own RULER-style prompt (niah_multikey_1 at a chosen length), KV-cache decode path.

Arms are the same objects as in s15_ruler_lower.py: pub (published Encbank, no write sink),
pub_sink, fix_S (chunk K/V visible at --fix-layers), fix_all (all lower layers), j0 (RAG
reference, full recompute over the pack).

Phases, each bracketed by torch.cuda.synchronize(), median over --reps after one warmup:
  write    the selected top-k chunks written to depth j (+ lower K/V capture and rotation
           for the fix arms) plus the sink.  NOTE: EncbankLower.build_bottom writes chunks one
           at a time (unbatched); base Encbank.write_chunks batches equal-length chunks.  The
           difference is an implementation choice of this prototype, not of the method, and
           is reported as is.
  write_all  every context chunk of the prompt written once (what a store pays per document);
           base arms use the batched write_chunks, fix arms the unbatched capture.
  prefill  the query bottom band (write_prefill) + the top band over the pack (read_prefill).
  decode   --steps decode_step calls, reported per token.
Memory: torch.cuda.max_memory_allocated (reset before each arm's timed run) and the BYTES
of the cached objects for the read pack, measured from the tensors: h_j of the selected
chunks (bf16) and, for fix arms, the K/V entries of the chunk positions at the layers in S
(fix_S masks a full cache in this prototype; the measured figure counts only S layers, i.e.
what a deployment would store).  Per-token storage = bytes / selected tokens.

Everything here is one prompt (or a few), one GPU, one process; it measures this
implementation on this machine, not an optimised serving stack.
"""

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Encbank"))
sys.path.insert(0, str(ROOT / "exp"))

from encbank import Encbank                                   # noqa: E402
from encbank import selectors as _sel                       # noqa: E402
from eval import ruler as R                               # noqa: E402
from s15_ruler_lower import EncbankLower, build_arms        # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cache_chunk_bytes(cm):
    """Bytes of K+V for the chunk positions at layers in S (fix arms), else 0."""
    if not isinstance(cm, EncbankLower) or cm._bottom is None:
        return 0
    cache, M = cm._bottom["cache"], cm._bottom["M"]
    first = 1 if cm.sink_in_cache else 0
    total = 0
    for l in sorted(cm.lower_layers):
        layer = cache.layers[l] if hasattr(cache, "layers") else None
        if layer is not None and hasattr(layer, "keys"):
            k, v = layer.keys, layer.values
        else:  # older DynamicCache API
            k, v = cache.key_cache[l], cache.value_cache[l]
        total += k[:, :, first:M, :].numel() * k.element_size()
        total += v[:, :, first:M, :].numel() * v.element_size()
    return total


@torch.no_grad()
def run_arm(arm, name, bos_id, sel_chunks, query_ids, steps, context_chunks, time_all):
    """One timed pass of write -> prefill -> decode.  Returns a dict of seconds/bytes."""
    dev = arm.device
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    out = {}
    sync(); t0 = time.perf_counter()
    if isinstance(arm, EncbankLower):
        sink_hj, sel_hj = arm.build_bottom(bos_id, sel_chunks)
    else:
        sink_hj = arm.write_chunk([bos_id])
        sel_hj = arm.write_chunks(sel_chunks)
    sync(); t1 = time.perf_counter()
    out["write_s"] = t1 - t0
    out["hj_bytes"] = int(sum(h.numel() * h.element_size() for h in sel_hj))
    out["kv_bytes"] = int(cache_chunk_bytes(arm))
    out["sel_tokens"] = int(sum(h.shape[1] for h in sel_hj))

    q_hj, bottom_cache, q_pos = arm.write_prefill(query_ids)
    logits1, top_cache, pack_pos = arm.read_prefill(sink_hj, sel_hj, q_hj)
    sync(); t2 = time.perf_counter()
    out["prefill_s"] = t2 - t1
    out["pack_len"] = int(pack_pos)

    next_tok = int(logits1[0, -1].float().argmax().item())
    for _ in range(steps - 1):
        logits = arm.decode_step(next_tok, bottom_cache, top_cache, q_pos, pack_pos)
        q_pos += 1
        pack_pos += 1
        next_tok = int(logits[0, -1].float().argmax().item())
    sync(); t3 = time.perf_counter()
    out["decode_s_per_tok"] = (t3 - t2) / max(1, steps - 1)
    out["peak_gib"] = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else 0.0
    if isinstance(arm, EncbankLower):
        arm._bottom = None
    del bottom_cache, top_cache, sel_hj, q_hj, sink_hj

    if time_all:
        sync(); t4 = time.perf_counter()
        if isinstance(arm, EncbankLower):
            for ch in context_chunks:
                arm._capture_lower(ch)
        else:
            arm.write_chunks(context_chunks)
        sync(); t5 = time.perf_counter()
        out["write_all_s"] = t5 - t4
        out["n_context_chunks"] = len(context_chunks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--arms", default="pub,pub_sink,fix_S,fix_all,j0")
    ap.add_argument("--fix-layers", default="1,9,11")
    ap.add_argument("--task", default="niah_multikey_1")
    ap.add_argument("--length", default="32k")
    ap.add_argument("--n", type=int, default=2, help="prompts")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--essay", default=str(ROOT / "exp/data/pg19_essay.txt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    ap.add_argument("--cap-gb", type=float, default=30.0)
    ap.add_argument("--need-gb", type=float, default=22.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0)
    ap.add_argument("--cpu-smoke", action="store_true")
    args = ap.parse_args()

    names = [a for a in args.arms.split(",") if a]
    fix_layers = [int(x) for x in args.fix_layers.split(",") if x]
    if args.cpu_smoke:
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model)
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False)
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(cfg).float().eval()
        j, chunk, topk, steps = 2, 8, 3, 4
        text = ("The grass is green. The sky is blue. One of the special magic numbers for "
                "apple-tree is: 1234567. The sun is yellow. Here we go. There and back again. "
                "What is the special magic number for apple-tree mentioned in the text? The "
                "special magic number for apple-tree mentioned in the provided text is")
        prompts = [(text, tok.encode("What is the special magic number for apple-tree?",
                                     add_special_tokens=False))]
        out_path = Path("exp/results/s16_smoke.json")
        fix_layers = [0]
    else:
        from gpu_gate import acquire_gpu
        from eval._common import load_backbone
        acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb, idle_slack_gb=args.idle_slack_gb,
                    tag="s16_latency")
        R._ESSAY_PATH = args.essay
        model, tok = load_backbone(args.model, "bfloat16", "sdpa", "cuda:0", "")
        j, chunk, topk, steps = args.j, 512, args.topk, args.steps
        task = R._resolve_task(args.task)
        target = R._LENGTH_TOKENS[args.length]
        base_seed = args.seed + (hash((task, args.length)) % 100000)
        prompts = []
        for i in range(args.n):
            rng = random.Random(base_seed * 1000 + i)
            prompt, answers, gold = R._build_sample(task, target, tok, rng, None)
            prompts.append((prompt, tok.encode(R._bare_question(prompt), add_special_tokens=False)))
        out_path = Path(args.out or f"exp/results/s16_latency_{args.length}.json")
    arms = build_arms(model, tok, j, names, fix_layers)
    device = next(model.parameters()).device
    print(f"{args.model}: j={j} arms={names} fix_layers={fix_layers} task={args.task} "
          f"length={args.length} n={len(prompts)} reps={args.reps} steps={steps} "
          f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}", flush=True)

    rows = []
    for pi, (prompt, bare_q) in enumerate(prompts):
        ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        ids = ids.to(device)
        tokens = ids[0]
        chunks = list(tokens.split(chunk))
        context_chunks, query_chunk = chunks[:-1], chunks[-1]
        sel_idx = _sel.select_context_chunk_indices(
            "bm25", context_chunks, bare_q, topk, None, context_hj=None, query_hj=None,
            iter_rounds=0, iter_hop_topk=2, iter_score="meanpool", iter_conf_ratio=0.3,
            iter_max_chunks=64, dense_retriever=None, dense_tokenizer=tok)
        sel_chunks = [context_chunks[i] for i in sel_idx]
        bos_id = int(tokens[0].item()) if getattr(tok, "bos_token_id", None) is None else int(tok.bos_token_id)
        query_ids = query_chunk.tolist()
        for name in names:
            arm = arms[name]
            run_arm(arm, name, bos_id, sel_chunks, query_ids, steps, context_chunks, False)  # warmup
            reps = [run_arm(arm, name, bos_id, sel_chunks, query_ids, steps, context_chunks,
                            time_all=(r == 0)) for r in range(args.reps)]
            med = {k: st.median(x[k] for x in reps) for k in ("write_s", "prefill_s", "decode_s_per_tok", "peak_gib")}
            row = {"prompt": pi, "arm": name, "n_tokens": int(tokens.shape[0]),
                   "n_context_chunks": len(context_chunks), "sel_tokens": reps[0]["sel_tokens"],
                   "pack_len": reps[0]["pack_len"], "hj_bytes": reps[0]["hj_bytes"],
                   "kv_bytes": reps[0]["kv_bytes"],
                   "kb_per_token": (reps[0]["hj_bytes"] + reps[0]["kv_bytes"]) / 1024 / max(1, reps[0]["sel_tokens"]),
                   "write_all_s": reps[0].get("write_all_s"), **med,
                   "reps": reps}
            rows.append(row)
            print(f"  p{pi} {name:8s} write {med['write_s']*1e3:7.0f} ms  prefill {med['prefill_s']*1e3:7.0f} ms  "
                  f"decode {med['decode_s_per_tok']*1e3:6.1f} ms/tok  peak {med['peak_gib']:.2f} GiB  "
                  f"store {row['kb_per_token']:.1f} KB/tok  write_all {row['write_all_s']:.2f} s "
                  f"({len(context_chunks)} chunks)", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"args": vars(args), "j": j, "rows": rows}, indent=1))

    print("\n### medians over prompts")
    for name in names:
        rs = [r for r in rows if r["arm"] == name]
        m = lambda key: st.median(r[key] for r in rs)
        print(f"{name:8s} store {m('kb_per_token'):6.1f} KB/tok  write {m('write_s')*1e3:7.0f} ms  "
              f"write_all {m('write_all_s'):6.2f} s  prefill {m('prefill_s')*1e3:7.0f} ms  "
              f"decode {m('decode_s_per_tok')*1e3:6.1f} ms/tok  peak {m('peak_gib'):.2f} GiB")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
