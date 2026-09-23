"""
S12 -- The frac-vs-j curve on Qwen3-8B (L=36), same protocol as S7/S9 on 1.7B.

WHY
---
Every j-sweep so far (S2/S4/S7/S9) ran on Qwen3-1.7B and found the fraction of the
chunk's value that the cached h_j throws away rising monotonically with j, with no
plateau.  The paper's operating point is Qwen3-8B at j=12, and it reports that the
zero-shot *readable* depth deepens with scale (tab_depth: readout-crash 0.09L at 0.6B
-> 0.30L at 8B -> >0.42L at 32B).  So the 1.7B curve may simply be the wrong model:
the objection is that a small backbone commits to generation earlier, and the 8B
curve could have the knee the paper describes.  This script settles that with the
same metric on the same protocol, no training.

PROTOCOL (identical to S7 unless stated)
----------------------------------------
  document  n_doc chunks of c tokens; k of them retrieved, scattered evenly.
  ref       [sink; c_a; ...; c_k; query] forwarded from layer 0 -- the retrieval upper
            bound, i.e. the paper's matched j=0 replay and its distillation teacher.
  no_mem    [sink; query] -- the floor.
  full_doc  [sink; whole document; query] -- reported so the cost of retrieval itself
            is visible next to the cost of depth.
  A_off_j   deployed CoMem as published: every chunk written ALONE with NO prefix at
            chunk-local positions to h_j; query likewise; packed at fresh contiguous
            positions; layers [j, L) recomputed.  (COMem/comem/model.py write_chunk
            default, write_sink=False.)
  A_on_j    same, but a BOS is prepended during the write and dropped (write_sink=True).
  A_chunk_j / A_query_j   the S7 decomposition, write sink ON: chunks isolated with the
            query's h_j taken from ref (chunk staleness alone), and vice versa.

METRICS
-------
  frac   KL(p_ref || p_arm) / KL(p_ref || p_no_mem), full vocabulary, every query
         position, averaged over positions then samples.  0 = as good as having the
         chunks in context, 1 = contributed nothing.  Comparable to S2..S9.
  gap    ppl_arm / ppl_ref on the true query tokens -- the paper's multiplicative LM
         tax (tab_hy3_distill.tex), 1.0 exact.  Lets the 8B curve be read against the
         paper's Hy3 "fidelity smile" (1.64 at j=16 -> 1.50 at j~32-36 -> rising).
  top1   argmax agreement with ref on the query positions -- also the paper's metric.

j = 0 is included as an identity check (frac must be ~0; it is the paper's j=0 = RAG).
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, forward_from, kl


def pg19_text(path: str, n_chars: int, skip_ids=()) -> str:
    """Concatenate books of the local PG19 shard until n_chars is reached.

    ``skip_ids`` drops books by PG19 id.  The shard's first two books are the King James
    Bible (id 10) and a Shakespeare collection (id 100); Qwen3-8B has both memorised
    (retrieval-reference ppl 1.96 on the Bible vs 8.82 on wikitext), so a clean PG19 arm
    must skip them (pass --pg19-skip 10,100).
    """
    parts, total = [], 0
    skip = {int(x) for x in skip_ids}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["id"]) in skip:
                continue
            t = rec["text"]
            parts.append(t)
            total += len(t)
            if total >= n_chars:
                break
    return "\n".join(parts)[:n_chars]


def ppl_and_top1(logits, ref_logits, targets):
    """logits/ref_logits [1, nq, V] at the query positions; targets [nq-1] = q[1:]."""
    lp = F.log_softmax(logits[0, :-1].float(), -1)
    nll = -lp.gather(-1, targets[:, None]).squeeze(-1).mean()
    agree = (logits[0].argmax(-1) == ref_logits[0].argmax(-1)).float().mean()
    return float(nll), float(agree)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "pg19"])
    ap.add_argument("--pg19", default="/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl")
    ap.add_argument("--pg19-skip", default="", help="comma list of PG19 book ids to skip")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--layers", default="", help="comma list of j; default a 13-point grid")
    ap.add_argument("--decomp-layers", default="", help="j at which to run A_chunk/A_query")
    ap.add_argument("--out", default="")
    ap.add_argument("--forward-tol", type=float, default=1e-3,
                    help="max KL(lib||ours) accepted for the hand-rolled forward check")
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--need-gb", type=float, default=22.0)
    ap.add_argument("--idle-slack-gb", type=float, default=4.0,
                    help="desktop residency allowed while still counting the GPU as idle")
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb, idle_slack_gb=args.idle_slack_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    L = model.config.num_hidden_layers
    if args.layers:
        js = [int(x) for x in args.layers.split(",")]
    else:
        # relative grid so 1.7B (L=28) and 8B (L=36) land on the same j/L points
        rel = [0, .06, .11, .17, .25, .33, .42, .5, .58, .67, .75, .83, .92]
        js = sorted(set(int(round(r * L)) for r in rel))
    if args.decomp_layers:
        djs = [int(x) for x in args.decomp_layers.split(",")]
    else:
        djs = [j for j in js if j in (round(.17 * L), round(.33 * L), round(.5 * L), round(.67 * L))]
    tag = Path(args.model.rstrip("/\\")).name.lower()
    out = Path(args.out or f"exp/results/s12_jsweep_{tag}_{args.corpus}.json")

    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    c, nq, k = args.chunk, args.query, args.k
    doc_len = args.n_doc * c
    need_chars = (doc_len + nq) * args.n * 5
    if args.corpus == "wikitext":
        text = fetch_text("main", Path("exp/data"))
    else:
        skip = [x for x in args.pg19_skip.split(",") if x]
        text = pg19_text(args.pg19, need_chars, skip)
    ids_all = tok(text, return_tensors="pt").input_ids[0].to(dev)
    print(f"{args.model}: L={L}, corpus={args.corpus}, {len(ids_all)} tokens, js={js}, "
          f"decomp at {djs}", flush=True)

    # sanity: the hand-rolled forward must match the library forward
    probe = ids_all[: c + nq].unsqueeze(0)
    pos = torch.arange(probe.shape[1], device=dev).unsqueeze(0)
    ours = forward_from(model, embed_to(model, probe, pos, 0), pos, 0, final=nq)
    theirs = model(input_ids=probe, position_ids=pos, use_cache=False).logits[:, -nq:, :].float()
    d = kl(theirs, ours)
    print(f"manual-forward check: KL(lib || ours) = {d:.2e} nats", flush=True)
    assert d < args.forward_tol, "hand-rolled forward does not reproduce the library forward"

    def write(ids_1d, use_sink: bool):
        if use_sink:
            x = torch.cat([sink.unsqueeze(0), ids_1d], 1)
            return x, torch.arange(x.shape[1], device=dev).unsqueeze(0), 1
        return ids_1d, torch.arange(ids_1d.shape[1], device=dev).unsqueeze(0), 0

    rows = []
    for si in range(args.n):
        s = si * (doc_len + nq)
        if s + doc_len + nq > len(ids_all):
            print(f"corpus exhausted after {si} samples", flush=True)
            break
        doc = ids_all[s:s + doc_len]
        qi = ids_all[s + doc_len:s + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(args.n_doc)]
        picks = [round(i * (args.n_doc - 1) / max(k - 1, 1)) for i in range(k)]
        sel = [chunks[p] for p in sorted(set(picks))]
        kk = len(sel)
        targets = qi[0, 1:]

        pack = torch.cat([sink.unsqueeze(0)] + sel + [qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)
        ref_nll, _ = ppl_and_top1(ref, ref, targets)

        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        nm = forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq)
        base = kl(ref, nm)
        nm_nll, nm_top1 = ppl_and_top1(nm, ref, targets)

        full = torch.cat([sink.unsqueeze(0), doc.unsqueeze(0), qi], 1)
        fp = torch.arange(full.shape[1], device=dev).unsqueeze(0)
        fd = forward_from(model, embed_to(model, full, fp, 0), fp, 0, final=nq)
        fd_nll, fd_top1 = ppl_and_top1(fd, ref, targets)

        row = {"sample": si, "k": kk, "kl_no_mem": base, "ref_nll": ref_nll,
               "no_mem_frac": 1.0, "no_mem_gap": math.exp(nm_nll - ref_nll),
               "no_mem_top1": nm_top1,
               "full_doc_frac": kl(ref, fd) / base, "full_doc_gap": math.exp(fd_nll - ref_nll),
               "full_doc_top1": fd_top1}

        for j in js:
            h_ref = embed_to(model, pack, pp, j)
            h_s = h_ref[:, :1, :]
            hc_ref = h_ref[:, 1:1 + kk * c, :]
            hq_ref = h_ref[:, 1 + kk * c:, :]

            iso = {}
            for use_sink in (False, True):
                parts = []
                for ch in sel:
                    cid, cpos, drop = write(ch, use_sink)
                    parts.append(embed_to(model, cid, cpos, j)[:, drop:, :])
                qid, qpos, qdrop = write(qi, use_sink)
                iso[use_sink] = (torch.cat(parts, 1),
                                 embed_to(model, qid, qpos, j)[:, qdrop:, :])

            arms = [("A_off", iso[False][0], iso[False][1]),
                    ("A_on", iso[True][0], iso[True][1])]
            if j in djs:
                arms += [("A_chunk", iso[True][0], hq_ref), ("A_query", hc_ref, iso[True][1])]
            for name, hc, hq in arms:
                hp = torch.cat([h_s, hc, hq], 1)
                lg = forward_from(model, hp, pp, j, final=nq)
                nll, t1 = ppl_and_top1(lg, ref, targets)
                row[f"{name}_j{j}_frac"] = kl(ref, lg) / base
                row[f"{name}_j{j}_gap"] = math.exp(nll - ref_nll)
                row[f"{name}_j{j}_top1"] = t1
            for use_sink, nm_ in ((False, "off"), (True, "on")):
                hc = iso[use_sink][0]
                row[f"stale_{nm_}_j{j}"] = float(
                    ((hc - hc_ref).norm(dim=-1) / hc_ref.norm(dim=-1).clamp_min(1e-6)).mean())
        rows.append(row)
        jo = round(.33 * L)
        print(f"  sample {si}: base {base:.3f} nats  A_off@{jo} {row[f'A_off_j{jo}_frac']:.3f}"
              f"  A_on@{jo} {row[f'A_on_j{jo}_frac']:.3f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "L": L, "args": vars(args),
                                   "layers": js, "decomp_layers": djs, "rows": rows},
                                  indent=1))

    n = len(rows)

    def m(key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / max(1, len(vals))

    print(f"\n{args.model} L={L} corpus={args.corpus} n={n} k={rows[0]['k']}")
    print(f"KL(ref||no_mem) = {m('kl_no_mem'):.3f} nats; no_mem gap {m('no_mem_gap'):.3f} "
          f"top1 {m('no_mem_top1'):.3f}; full_doc frac {m('full_doc_frac'):.3f} gap "
          f"{m('full_doc_gap'):.3f}")
    print("\n   j   j/L |  A_off frac  gap   top1 |  A_on frac  gap   top1 |"
          " A_chunk  A_query | stale off/on")
    print("-----------+--------------------------+-------------------------+"
          "------------------+-------------")
    for j in js:
        if j in djs:
            dc = f"  {m(f'A_chunk_j{j}_frac'):.3f}   {m(f'A_query_j{j}_frac'):.3f} "
        else:
            dc = "     -        -   "
        print(f"{j:4d}  {j/L:.2f} |   {m(f'A_off_j{j}_frac'):.3f}   {m(f'A_off_j{j}_gap'):.3f}  "
              f"{m(f'A_off_j{j}_top1'):.3f} |  {m(f'A_on_j{j}_frac'):.3f}   "
              f"{m(f'A_on_j{j}_gap'):.3f}  {m(f'A_on_j{j}_top1'):.3f} |{dc}|  "
              f"{m(f'stale_off_j{j}'):.3f} / {m(f'stale_on_j{j}'):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
