"""
S5 -- For a fixed memory budget, does WHERE the cache is visible beat HOW MUCH is stored?

(Rewritten after S9.  The first version of this file used the gated memory branch from
S3, which needed a per-sample fitted gate and therefore only ever produced an upper
bound.  S6/S7/S9 established that a prefix KV cache needs no gate at all -- attention
over a cache is just attention -- so this now uses the gate-free formulation.  Nothing
is fitted, the numbers are deployable rather than oracle, and it runs far faster because
there is no inner optimisation.  The earlier version was never run, so no result depends
on it.)

WHY
---
Two runs showed placement mattering more than budget, both times as a side effect rather
than by design:
  * S3 (single chunk, gated): at a fixed 8 KB, layers [18,22] scored 0.668 while [14,26]
    scored 0.821 -- a 0.153 spread from placement, against 0.114 for doubling storage.
  * S9 (multi-chunk, gate-free): |S|=2 at the heuristic's [14,26] scored 0.790, WORSE
    than |S|=1 at [22] (0.759) despite twice the bytes.
Both used a spacing heuristic that happens to skip layer 22.  So the reported |S| curve
is a lower bound on what the KV family can do, and "how many layers" has been confounded
with "which layers" throughout.  Nobody in the literature separates them either:
Memorizing Transformers (arXiv:2203.08913) uses one layer and the decoder-only
Unlimiformer adaptation (arXiv:2410.01637) at most three, both choosing the COUNT for
indexing cost, never studying the PLACEMENT.

SELECTION MUST NOT HAPPEN ON THE EVALUATED SAMPLES
--------------------------------------------------
Greedily picking layers and then reporting that set's score on the same samples is the
"selected magnitude" error this project has already been burned by.  Layers are chosen by
greedy forward selection on a SELECTION split; every reported number comes from a
disjoint EVAL split; the heuristic sets are scored on that same eval split.

METRIC -- unchanged from S2/S3/S4/S6/S7/S8/S9
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem)
full-vocabulary KL in nats at every query position, averaged over positions then samples.
Multi-chunk (k=4), one sink always kept (S8: without it the cache is worse than no
memory at all).  Each memory layer costs 4*d_kv = 4 KB/token on Qwen3-1.7B.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, forward_from, kl
from s6_cacheblend_arm import chunk_kv_all_layers
from s9_frontier import forward_cache_subset


@torch.no_grad()
def build(model, tok, ids_all, sink, dev, c, nq, k, n_doc, idxs):
    """Precompute ref / floor / merged cache per sample; every layer set reuses them."""
    out = []
    doc_len = n_doc * c
    for si in idxs:
        st = si * (doc_len + nq)
        if st + doc_len + nq > len(ids_all):
            break
        doc = ids_all[st:st + doc_len]
        qi = ids_all[st + doc_len:st + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(n_doc)]
        picks = sorted({round(i * (n_doc - 1) / max(k - 1, 1)) for i in range(k)})
        sel = [chunks[p] for p in picks]

        pack = torch.cat([sink.unsqueeze(0)] + sel + [qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)
        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))

        caches, off = [], 1
        for ch in sel:
            cid = torch.cat([sink.unsqueeze(0), ch], 1)
            cpos = torch.arange(off - 1, off - 1 + cid.shape[1], device=dev).unsqueeze(0)
            cc = chunk_kv_all_layers(model, cid, cpos)
            caches.append({i: (v[0][:, :, 1:, :], v[1][:, :, 1:, :]) for i, v in cc.items()})
            off += c
        sk = chunk_kv_all_layers(model, sink.unsqueeze(0),
                                 torch.zeros(1, 1, dtype=torch.long, device=dev))
        merged = {i: (torch.cat([sk[i][0]] + [cv[i][0] for cv in caches], 2),
                      torch.cat([sk[i][1]] + [cv[i][1] for cv in caches], 2))
                  for i in caches[0]}
        # The merged cache is 28 layers x (K,V) x 8 heads x 2049 x 128 in bf16 = 235 MB
        # per sample.  Holding 18 of them pins 4.2 GB, which OOMs under a considerate
        # cap while somebody else is training.  Park them in host memory and page one in
        # at a time; the loops below are ordered so each sample is paged in ONCE per
        # greedy round, not once per candidate.
        out.append({"ref": ref, "base": base, "q": qi,
                    "qpos": torch.arange(off, off + nq, device=dev).unsqueeze(0),
                    "cache": {i: (a.cpu(), b.cpu()) for i, (a, b) in merged.items()},
                    "nq": nq})
        del merged, caches
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def score_many(model, samples, layer_sets, dev="cuda"):
    """Mean frac for EVERY layer set, paging each sample's cache in exactly once."""
    tot = [0.0] * len(layer_sets)
    for s in samples:
        gpu = {i: (a.to(dev, non_blocking=True), b.to(dev, non_blocking=True))
               for i, (a, b) in s["cache"].items()}
        for i, layers in enumerate(layer_sets):
            tot[i] += kl(s["ref"], forward_cache_subset(
                model, s["q"], s["qpos"], gpu, set(layers), s["nq"])) / s["base"]
        del gpu
        torch.cuda.empty_cache()
    return [t / len(samples) for t in tot]


def score(model, samples, layers):
    return score_many(model, samples, [layers])[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n-select", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--max-k", type=int, default=6)
    ap.add_argument("--out", default="exp/results/s5_placement.json")
    ap.add_argument("--cap-gb", type=float, default=8.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    L = model.config.num_hidden_layers
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // model.config.num_attention_heads)
    kb = 4 * model.config.num_key_value_heads * hd / 1024

    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)

    sel = build(model, tok, ids_all, sink, dev, args.chunk, args.query, args.k,
                args.n_doc, range(args.n_select))
    ev = build(model, tok, ids_all, sink, dev, args.chunk, args.query, args.k, args.n_doc,
               range(args.n_select, args.n_select + args.n_eval))
    print(f"{args.model}: L={L} | select {len(sel)} samples, eval {len(ev)} (disjoint)")

    chosen, hist = [], []
    for kk in range(1, args.max_k + 1):
        rem = [j for j in range(L) if j not in chosen]
        vals = score_many(model, sel, [chosen + [j] for j in rem])
        best_i = min(range(len(rem)), key=lambda i: vals[i])
        best_j, best_v = rem[best_i], vals[best_i]
        chosen = sorted(chosen + [best_j])
        hist.append({"k": kk, "layers": list(chosen), "sel_frac": best_v})
        print(f"  greedy |S|={kk}: +layer {best_j} -> {chosen}  (sel {best_v:.3f})", flush=True)

    def heur(s):
        return sorted({L // 2 + int(round(i * (L - 2 - L // 2) / max(s - 1, 1)))
                       for i in range(s)}) if s > 1 else [22]

    print("\n|S| | bytes/tok | greedy layers            | greedy | heuristic | gain")
    print("----+-----------+-------------------------+--------+-----------+------")
    rows = []
    for h in hist:
        hl = heur(h["k"])
        g, b = score_many(model, ev, [h["layers"], hl])
        rows.append({"k": h["k"], "greedy_layers": h["layers"], "heur_layers": hl,
                     "greedy_frac": g, "heur_frac": b})
        print(f"{h['k']:3d} |  {h['k']*kb:5.0f} KB  | {str(h['layers']):23s} | {g:.3f}  "
              f"|   {b:.3f}   | {b-g:+.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args),
                               "greedy": hist, "eval": rows}, indent=1))
    print("\nS9 reference (heuristic, multi-chunk): |S|=1 0.759, 2 0.790, 4 0.659, "
          "8 0.623, 14 0.505, 28 0.041")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
