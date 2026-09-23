"""
S10 -- Read compute in FLOPs, with the attention term included.

WHY
---
S9 reported read compute in "token-layers over the context" and gave the KV family a
flat ZERO, because it never recomputes context tokens.  That is the single simplification
in that table most favourable to the KV family and least favourable to Encbank: the query
still has to attend to every cached key at every layer where the cache is visible, and
that term is not zero.  This replaces the proxy with actual FLOPs.

It also adds the term S9 omitted entirely on the other side: the WRITE pass.  Encbank
writes a chunk through j layers; the KV family writes through all L.  That is Encbank's
real compute advantage and it belongs in the same ledger as its read disadvantage,
otherwise the comparison is rigged in the opposite direction.

MODEL (Qwen3-1.7B, read from config.json, not from memory)
    d = 2048, d_ff = 6144, L = 28, n_h = 16, n_kv = 8, head_dim = 128

COUNTING CONVENTION -- stated because every FLOP count depends on it
    * one multiply-accumulate = 2 FLOPs
    * linear layers only (no norms, no activations, no softmax); those are a few percent
    * attention counted as QK^T plus AV, with causal self-attention taken as half of the
      dense count and cross-attention against a cache taken as dense
    * the lm_head is excluded from both arms -- it is identical for both and applies to
      the query positions only
No GPU is used: this is arithmetic over the config.
"""

import argparse
import json
from pathlib import Path

CFG = dict(d=2048, d_ff=6144, L=28, n_h=16, n_kv=8, hd=128)


def linear_per_token_per_layer(c=CFG):
    """q, k, v, o projections plus a gated MLP."""
    d, d_ff, n_h, n_kv, hd = c["d"], c["d_ff"], c["n_h"], c["n_kv"], c["hd"]
    q = 2 * d * n_h * hd
    k = 2 * d * n_kv * hd
    v = 2 * d * n_kv * hd
    o = 2 * n_h * hd * d
    mlp = 3 * 2 * d * d_ff          # gate, up, down
    return q + k + v + o + mlp


def self_attn_flops(n_tok, c=CFG):
    """Causal self-attention over n_tok, one layer: QK^T + AV, halved for causality."""
    return 2 * (n_tok ** 2) * c["hd"] * c["n_h"]


def cross_attn_flops(n_q, n_kv_tok, c=CFG):
    """Dense attention of n_q queries against a cache of n_kv_tok keys, one layer."""
    return 2 * 2 * n_q * n_kv_tok * c["hd"] * c["n_h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=int, default=2081, help="tokens READ per query (k chunks)")
    ap.add_argument("--doc", type=int, default=6144, help="tokens WRITTEN per document")
    ap.add_argument("--ctx", type=int, default=2048, help="cached tokens attended at read")
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512, help="write is chunk-local")
    ap.add_argument("--js", default="2,6,10,14,18,22,26")
    ap.add_argument("--sizes", default="1,8,14,28")
    ap.add_argument("--out", default="exp/results/s10_flops.json")
    args = ap.parse_args()

    c = CFG
    L, d, hd, n_kv = c["L"], c["d"], c["hd"], c["n_kv"]
    lin = linear_per_token_per_layer()
    P, M, Q, D = args.pack, args.ctx, args.query, args.doc
    kb_layer = 4 * n_kv * hd / 1024
    kb_resid = 2 * d / 1024

    # frac measured in S9 (multi-chunk, k=4, n=12); carried here so the ledger is
    # accuracy-aware rather than a pure cost table
    FRAC_A = {2: 0.092, 6: 0.261, 10: 0.382, 14: 0.465, 18: 0.645, 22: 0.804, 26: 0.970}
    FRAC_KV = {1: 0.759, 2: 0.790, 4: 0.659, 8: 0.623, 14: 0.505, 28: 0.041}

    rows = []
    print(f"Qwen3-1.7B  linear {lin/1e6:.1f} MFLOP/token/layer")
    print(f"WRITE covers the whole document ({D} tok); READ covers only the retrieved "
          f"pack ({P} tok, cache {M}, query {Q}).")
    print(f"That asymmetry -- write once over D, read repeatedly over P -- is the whole "
          f"reason a depth split can pay for itself; D/P = {D/P:.2f}.")
    print(f"residual {kb_resid:.0f} KB/token, KV {kb_layer:.0f} KB/layer/token\n")
    print("arm            | bytes/tok | write GFLOP | read GFLOP | read x | frac")
    print("---------------+-----------+-------------+------------+--------+------")

    # The WRITE pass is chunk-local in BOTH families: each chunk is encoded on its own,
    # so its self-attention is quadratic in the CHUNK, not in the document.  Modelling it
    # as one D-token sequence makes the write term quadratic in D and inflates the
    # break-even by ~5x at D=128k, which is the wrong answer in Encbank's favour.
    n_chunks = max(1, D // args.chunk)
    write_attn = n_chunks * self_attn_flops(args.chunk)

    # arm A: write j layers over the document, read (L-j) layers over the pack
    jlist = [int(x) for x in args.js.split(",")]
    for j in jlist:
        w = j * (lin * D + write_attn)
        r = (L - j) * (lin * P + self_attn_flops(P))
        rows.append(dict(arm=f"A_j{j}", bytes_kb=kb_resid, write=w, read=r,
                         frac=FRAC_A.get(j)))
        print(f"A  Encbank j={j:<2d}  |  {kb_resid:5.0f} KB  |  {w/1e9:9.1f}  | "
              f"{r/1e9:9.1f}  |        | {FRAC_A.get(j)}")

    # KV family: write all L layers once, read = query through L layers + cache attention
    base_read = L * (lin * Q + self_attn_flops(Q))
    w_kv = L * (lin * D + write_attn)
    ref_read = None
    for s in [int(x) for x in args.sizes.split(",")]:
        r = base_read + s * cross_attn_flops(Q, M + 1)
        rows.append(dict(arm=f"KV_S{s}", bytes_kb=s * kb_layer, write=w_kv, read=r,
                         frac=FRAC_KV.get(s)))
        if s == L:
            ref_read = r
        print(f"KV |S|={s:<2d}      |  {s*kb_layer:5.0f} KB  |  {w_kv/1e9:9.1f}  | "
              f"{r/1e9:9.1f}  |        | {FRAC_KV.get(s)}")

    a2 = next(x for x in rows if x["arm"] == "A_j2")
    kv = next(x for x in rows if x["arm"] == f"KV_S{L}")
    print(f"\nread cost, Encbank j=2 vs full-depth KV: "
          f"{a2['read']/kv['read']:.0f}x  "
          f"({a2['read']/1e9:.0f} vs {kv['read']/1e9:.1f} GFLOP)")
    print("so the KV family's read is NOT free once the cache attention is counted, "
          "but including it moves the ratio from infinity to a finite number only.")

    # amortisation: Encbank writes cheaply, reads expensively. After how many reads of the
    # same document does the cheaper write stop paying for the more expensive read?
    print("\nbreak-even in TOTAL compute (write once, then R reads of the same document):")
    print("  R* = (write_KV - write_A) / (read_A - read_KV)")
    for j in jlist:
        a = next(x for x in rows if x["arm"] == f"A_j{j}")
        num = kv["write"] - a["write"]
        den = a["read"] - kv["read"]
        rstar = num / den if den > 0 else float("inf")
        print(f"  Encbank j={j:<2d}: R* = {rstar:6.2f} reads   "
              f"(write saving {num/1e9:.0f} GFLOP, read penalty {den/1e9:.0f} GFLOP/read)")
    print("  R* is the number of reads a document can absorb before Encbank's cheaper")
    print("  write is used up by its more expensive reads.  It scales with D/P, i.e.")
    print("  with how selective retrieval is: a long document read through a small")
    print("  pack is exactly where the depth split pays.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": c, "args": vars(args), "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
