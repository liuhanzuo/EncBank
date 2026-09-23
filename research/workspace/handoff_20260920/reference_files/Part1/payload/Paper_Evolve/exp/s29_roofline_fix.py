"""S29 -- put the REPAIRED read F(S) into the S14 roofline ledger.

WHY. S14's roofline prices two arms on one ruler: CoMem's read (stream one residual per
token, then recompute L-j layers over the pack -- compute bound) and the KV family's read
(stream a full-depth cache, then do almost no arithmetic -- bandwidth bound). It showed they
sit on opposite sides of the roofline ridge, which is why a FLOPs-only ledger was biased.

The arm this paper actually recommends is in neither row. F(S) stores the depth-j residual
AND |S| layers of K/V, and it recomputes the upper band. So it should land BETWEEN the two,
and the open question the paper's cost section leaves ("the dial's system-level cost is left
open") is how far toward the bandwidth-bound end |S| pushes it. That is an arithmetic
question, not a measurement, and it is what this script answers.

THE MODEL, and it is S14's, unchanged. Imported rather than restated so the two cannot
drift: CFG, linear_per_token_per_layer, self_attn_flops, cross_attn_flops from s10_flops and
the HW table from s14_roofline. Bytes per token: residual 2d; one layer of K/V 4*n_kv*hd;
weights per layer = lin (numerically, since lin FLOPs = 2*params).

    F(S) WRITE   identical FLOPs to CoMem's write -- the tap is free, the keys and values
                 were already computed by the same forward -- and j weight-layers streamed,
                 D x 2d residual bytes written, PLUS D x |S| x KV bytes written.
    F(S) READ    the query's own lower band runs over Q tokens only, attending to a cache of
                 M+1 entries at the layers in S: j x lin x Q + |S| x cross_attn(Q, M+1).
                 Then the pack of P tokens goes through the upper L-j layers exactly as in
                 CoMem: (L-j) x (lin x P + self_attn(P)).
                 Bytes: ALL L weight-layers are streamed (j below, L-j above), the pack's
                 residuals P x 2d are read from the store, and the cache costs M x |S| x KV.

    t = max(FLOPs / peak, bytes / bandwidth), as in S14. Roofline bound, not wall clock:
    it ignores kernel launch, softmax, norms, occupancy and cache reuse.

A BOOKKEEPING ASYMMETRY, stated because it is easy to misread. S14's A_j read charges
(L-j) weight-layers over the pack and nothing else -- it does NOT charge the query's own
lower band, which CoMem must also run to produce h_j^q. The F(S) read here DOES charge it
(j x lin x Q). So F(S=0) is not "CoMem plus nothing"; it sits 0.179% above the A_j row for
that reason alone. The cost attributable to the cache is the F(S=0) -> F(S=j) difference,
+0.089%, and that is the number to quote for "what does |S| cost".

WHAT THIS CANNOT SHOW. Nothing here is measured. The measured numbers are S16
(exp/results/s16_latency_32k.json, one card, one prompt shape, medians of three) and they
are the ones the paper quotes; this is the analytic complement that says WHY the measured
decode penalty is small and where it would stop being small.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s10_flops import (CFG, cross_attn_flops, linear_per_token_per_layer,  # noqa: E402
                       self_attn_flops)
from s14_roofline import HW                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=int, default=6657)
    ap.add_argument("--doc", type=int, default=131072)
    ap.add_argument("--ctx", type=int, default=6144)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--j", type=int, default=9)
    ap.add_argument("--sizes", default="0,1,3,6,9", help="|S|, the cached lower layers")
    ap.add_argument("--hw", default="rtx5090", choices=sorted(HW))
    ap.add_argument("--out", default="exp/results/s29_roofline_fix.json")
    args = ap.parse_args()

    c = CFG
    L, d, hd, n_kv = c["L"], c["d"], c["hd"], c["n_kv"]
    lin = linear_per_token_per_layer()
    P, M, Q, D, j = args.pack, args.ctx, args.query, args.doc, args.j
    peak, bw = HW[args.hw]
    ridge = peak / bw

    b_resid = 2 * d
    b_kv_layer = 4 * n_kv * hd
    w_layer = lin
    n_chunks = max(1, D // args.chunk)
    write_attn = n_chunks * self_attn_flops(args.chunk)

    def stage(fl, by):
        return dict(flops=fl, bytes=by, ai=fl / by, t_flop=fl / peak,
                    t_bw=by / bw, t=max(fl / peak, by / bw))

    rows = []
    # the two S14 reference arms, recomputed here so every row is on one ruler
    rows.append(dict(
        arm=f"A_j{j} (CoMem)", kb=b_resid / 1024,
        write=stage(j * (lin * D + write_attn), j * w_layer + D * b_resid),
        read=stage((L - j) * (lin * P + self_attn_flops(P)),
                   (L - j) * w_layer + P * b_resid)))
    rows.append(dict(
        arm=f"KV_S{L} (full depth)", kb=L * b_kv_layer / 1024,
        write=stage(L * (lin * D + write_attn), L * w_layer + D * L * b_kv_layer),
        read=stage(L * (lin * Q + self_attn_flops(Q)) + L * cross_attn_flops(Q, M + 1),
                   L * w_layer + M * L * b_kv_layer)))
    # the repaired arm, one row per |S|
    for S in [int(x) for x in args.sizes.split(",")]:
        assert 0 <= S <= j, f"|S|={S} must lie in [0, j={j}]"
        rows.append(dict(
            arm=f"F(S={S}) at j={j}", kb=(b_resid + S * b_kv_layer) / 1024,
            write=stage(j * (lin * D + write_attn),
                        j * w_layer + D * b_resid + D * S * b_kv_layer),
            read=stage(j * lin * Q + S * cross_attn_flops(Q, M + 1)
                       + (L - j) * (lin * P + self_attn_flops(P)),
                       L * w_layer + P * b_resid + M * S * b_kv_layer)))

    print(f"Qwen3-1.7B ({L} layers, d={d}) on {args.hw}: peak {peak/1e12:.0f} TFLOP/s, "
          f"BW {bw/1e9:.0f} GB/s, roofline ridge = {ridge:.0f} FLOP/byte")
    print(f"D={D} written once, P={P} pack read per query, cache M={M}, query Q={Q}, j={j}\n")
    print(f"{'arm':22s} {'KB/tok':>7s} | {'read AI':>8s} {'bound':>6s} "
          f"{'read ms':>8s} | {'write AI':>8s} {'bound':>6s} {'write s':>8s}")
    for r in rows:
        rd, wr = r["read"], r["write"]
        print(f"{r['arm']:22s} {r['kb']:7.0f} | {rd['ai']:8.1f} "
              f"{'BW' if rd['t_bw'] > rd['t_flop'] else 'FLOP':>6s} {1000*rd['t']:8.2f} | "
              f"{wr['ai']:8.1f} {'BW' if wr['t_bw'] > wr['t_flop'] else 'FLOP':>6s} "
              f"{wr['t']:8.2f}")

    base = next(r for r in rows if r["arm"].startswith(f"A_j{j}"))
    print(f"\nread time relative to CoMem's own read ({1000*base['read']['t']:.2f} ms):")
    for r in rows:
        print(f"  {r['arm']:22s} {r['read']['t'] / base['read']['t']:6.3f}x   "
              f"AI {r['read']['ai']:7.1f} against ridge {ridge:.0f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"hw": args.hw, "peak": peak, "bw": bw, "ridge": ridge, "L": L, "j": j,
         "P": P, "M": M, "Q": Q, "D": D,
         "b_resid": b_resid, "b_kv_layer": b_kv_layer, "rows": rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
