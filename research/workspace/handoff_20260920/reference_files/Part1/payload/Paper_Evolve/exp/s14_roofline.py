"""
S14 -- Put memory traffic into the ledger, so the two families are compared on the
same ruler.

WHY THIS EXISTS
---------------
S10 priced both families in FLOPs only.  That is the last remaining systematic
simplification in the cost model, and it is not neutral: the two arms sit on OPPOSITE
sides of the roofline knee.

  * The KV family's read moves a full-depth cache (112 KB/token on Qwen3-1.7B) and then
    does almost no arithmetic -- 32 query tokens through 28 layers.  It is BANDWIDTH
    bound.  Pricing it in FLOPs charges it for the cheap resource and gives it the
    expensive one for free.
  * Encbank's read moves one residual (4 KB/token) and then recomputes L-j layers over the
    whole pack.  It is COMPUTE bound.  Pricing it in FLOPs charges it correctly.

So FLOPs-only understates the KV family's read cost and leaves Encbank's unchanged, which
biases the break-even against Encbank.  This script measures how much.

MODEL -- identical to S10, imported rather than restated so the two cannot drift.
Bytes are added on top:

    weight bytes / layer = 2 * params = `lin` (numerically, since lin FLOPs = 2*params)
    residual h_j         = 2d bytes/token          (4 KB on Qwen3-1.7B)
    KV, one layer        = 4 * n_kv * hd bytes/tok (4 KB on Qwen3-1.7B; 2:1 GQA here,
                           so one residual buys exactly ONE layer of KV)

    WRITE   A_j : j    weight-layers streamed + D tokens x 2d written
            KV  : L    weight-layers streamed + D tokens x L x KV written
    READ    A_j : L-j  weight-layers streamed + P tokens x 2d read
            KV  : L    weight-layers streamed + M tokens x |S| x KV read

TIME MODEL
    t = max(FLOPs / peak, bytes / bandwidth)
Compute and memory overlap on a GPU, so max() is the right composition, not sum().
This is a roofline bound, NOT a wall-clock prediction: it ignores kernel launch, softmax,
norms, occupancy, and cache reuse.  It is used only to compare two arms under one rule.

WHAT THIS CANNOT SHOW
    Nothing here is measured on hardware.  It cannot predict wall clock, and it cannot
    settle the paper's Q* (a measured quantity on their machine, not ours).  It settles
    exactly one question: does pricing memory traffic change the SIGN or the ORDER of
    magnitude of the S10 break-even?  No GPU is used.
"""

import argparse
import json
from pathlib import Path

from s10_flops import CFG, cross_attn_flops, linear_per_token_per_layer, self_attn_flops

# bf16 dense peak and HBM bandwidth.  5090 is the local card; the others are for the
# ridge-point column only, to show how hardware-dependent the conclusion is.
HW = {
    "rtx5090": (209.5e12, 1792e9),
    "h100": (494.7e12, 3350e9),
    "a100": (312.0e12, 2039e9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=int, default=6657)
    ap.add_argument("--doc", type=int, default=131072)
    ap.add_argument("--ctx", type=int, default=6144)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--js", default="2,9,12,26")
    ap.add_argument("--sizes", default="28")
    ap.add_argument("--hw", default="rtx5090", choices=sorted(HW))
    ap.add_argument("--out", default="exp/results/s14_roofline.json")
    args = ap.parse_args()

    c = CFG
    L, d, hd, n_kv = c["L"], c["d"], c["hd"], c["n_kv"]
    lin = linear_per_token_per_layer()
    P, M, Q, D = args.pack, args.ctx, args.query, args.doc
    peak, bw = HW[args.hw]
    ridge = peak / bw

    b_resid = 2 * d                      # bytes per token, one residual
    b_kv_layer = 4 * n_kv * hd           # bytes per token per layer, K and V
    w_layer = lin                        # bytes of weights per layer (= 2 * params)

    n_chunks = max(1, D // args.chunk)
    write_attn = n_chunks * self_attn_flops(args.chunk)

    def stage(fl, by):
        return dict(flops=fl, bytes=by, ai=fl / by,
                    t_flop=fl / peak, t_bw=by / bw, t=max(fl / peak, by / bw))

    rows = []
    for j in [int(x) for x in args.js.split(",")]:
        w = stage(j * (lin * D + write_attn), j * w_layer + D * b_resid)
        r = stage((L - j) * (lin * P + self_attn_flops(P)),
                  (L - j) * w_layer + P * b_resid)
        rows.append(dict(arm=f"A_j{j}", bytes_kb=b_resid / 1024, write=w, read=r))

    base_read = L * (lin * Q + self_attn_flops(Q))
    w_kv_fl = L * (lin * D + write_attn)
    for s in [int(x) for x in args.sizes.split(",")]:
        w = stage(w_kv_fl, L * w_layer + D * L * b_kv_layer)
        r = stage(base_read + s * cross_attn_flops(Q, M + 1),
                  L * w_layer + M * s * b_kv_layer)
        rows.append(dict(arm=f"KV_S{s}", bytes_kb=s * b_kv_layer / 1024, write=w, read=r))

    print(f"Qwen3-1.7B on {args.hw}: peak {peak/1e12:.0f} TFLOP/s, BW {bw/1e9:.0f} GB/s, "
          f"roofline ridge = {ridge:.0f} FLOP/byte")
    print(f"D={D} written, P={P} read per query, cache M={M}, query Q={Q}\n")
    print("arm      | KB/tok |         WRITE                  |          READ")
    print("         |        |  GB moved   AI     t (ms)      |  GB moved   AI      t (ms)  bound")
    print("---------+--------+--------------------------------+---------------------------------")
    for x in rows:
        w, r = x["write"], x["read"]
        bound = "BW " if r["t_bw"] > r["t_flop"] else "FLOP"
        print(f"{x['arm']:<8s} | {x['bytes_kb']:6.0f} | {w['bytes']/1e9:8.2f}  "
              f"{w['ai']:7.0f}  {1e3*w['t']:8.1f}      | {r['bytes']/1e9:8.3f}  "
              f"{r['ai']:8.0f}  {1e3*r['t']:8.2f}  {bound}")

    kv = next(x for x in rows if x["arm"] == f"KV_S{L}")
    print(f"\nread: KV_S{L} arithmetic intensity {kv['read']['ai']:.0f} FLOP/byte vs ridge "
          f"{ridge:.0f} -> {'BANDWIDTH' if kv['read']['ai'] < ridge else 'COMPUTE'} bound.")
    print("That is the bias S10 could not see: its read was priced on the resource it "
          "barely uses.\n")

    print("break-even R* = reads of one document before the KV family overtakes Encbank")
    print("  (higher R* = Encbank stays ahead longer)")
    print("   arm     | R* FLOPs-only | R* with bytes |  shift")
    print("  ---------+---------------+---------------+--------")
    out_rows = []
    for x in rows:
        if not x["arm"].startswith("A_"):
            continue
        f_num = kv["write"]["flops"] - x["write"]["flops"]
        f_den = x["read"]["flops"] - kv["read"]["flops"]
        t_num = kv["write"]["t"] - x["write"]["t"]
        t_den = x["read"]["t"] - kv["read"]["t"]
        rf = f_num / f_den if f_den > 0 else float("inf")
        rt = t_num / t_den if t_den > 0 else float("inf")
        sh = f"{100*(rt/rf - 1):+.1f}%" if rf not in (0, float("inf")) else "n/a"
        out_rows.append(dict(arm=x["arm"], r_flops=rf, r_bytes=rt))
        print(f"  {x['arm']:<8s} |    {rf:9.2f}  |    {rt:9.2f}  | {sh:>6s}")

    print("\nreading: the simplification was real -- the two arms are on opposite sides")
    print("of the roofline knee -- but correcting it moves R* by only a few percent, in")
    print("Encbank's favour.  It does not rescue the matched-bytes claim; the downgrade in")
    print("[[encbank-matched-bytes-selling-point]] stands on its own numbers.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(config=c, args=vars(args), hw=dict(peak=peak, bw=bw,
                 ridge=ridge), rows=rows, breakeven=out_rows), indent=1))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
