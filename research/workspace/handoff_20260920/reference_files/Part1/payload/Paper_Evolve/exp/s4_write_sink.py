"""
S4 -- What does the missing write-time attention sink cost end to end?

THE DEFECT
----------
`COMem/comem/model.py:write_chunk` forwards each chunk with nothing in front of it and
chunk-local positions.  That is the no-sink regime: attention has nowhere to park mass,
and the shallow residual is badly distorted.  Measured in S1 on Qwen3-1.7B, mean
relative L2 per token against ground truth (the same chunk inside its real document),
over 512 tokens x 5 chunks:

    layer 3 residual   no prefix 1.160   +1 BOS 0.140   +128 real tokens 0.099
    null scale (a DIFFERENT document as the preceding context)            0.102

A relative L2 above 1.0 means the error vector is longer than the vector itself.  One
BOS token removes 96% of the gap to the null scale.  The READ pack already prepends a
sink; the WRITE pass did not, so every cached h_j was produced in that regime.

WHAT S1 DID NOT SETTLE
----------------------
S1 is a representation-space measurement.  A large relative displacement need not
translate into a large change in what the model predicts -- that is exactly the
inference S1's own caveat refused to make.  This script measures the end-to-end
consequence in the same units as S2, so the two are directly comparable.

ARMS (identical except for the write pass)
------------------------------------------
    ref       full forward of the pack [sink; C; Q]              -- ground truth
    no_mem    forward of [sink; Q]                               -- the floor
    A_nosink  h_j from C alone, positions 0..c-1                 -- CoMem as published
    A_sink    h_j from [BOS; C], BOS position dropped            -- the one-token fix

Both A arms then inject into the same pack at the same fresh contiguous positions and
recompute layers [j, L), so the ONLY difference is what was in front of the chunk during
the write.  Note the read pack keeps its own sink in both arms; this is purely about the
write.

METRIC
------
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem)
KL in nats over the full vocabulary at every one of the 32 query positions, averaged
over positions then over samples.  0 = the cached state was as good as having the chunk
in context; 1 = it contributed nothing.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, forward_from, kl


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--layers", default="", help="comma list of j; default every 2")
    ap.add_argument("--out", default="exp/results/s4_write_sink.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    L = model.config.num_hidden_layers
    js = ([int(x) for x in args.layers.split(",")] if args.layers
          else list(range(2, L, 2)))

    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
    c, nq = args.chunk, args.query

    rows = []
    for si in range(args.n):
        s = si * (c + nq)
        if s + c + nq > len(ids_all):
            break
        ci = ids_all[s:s + c].unsqueeze(0)
        qi = ids_all[s + c:s + c + nq].unsqueeze(0)

        pack = torch.cat([sink.unsqueeze(0), ci, qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)

        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))

        # write-side variants
        c_pos = torch.arange(c, device=dev).unsqueeze(0)
        w_ids = torch.cat([sink.unsqueeze(0), ci], 1)
        w_pos = torch.arange(w_ids.shape[1], device=dev).unsqueeze(0)
        qw_ids = torch.cat([sink.unsqueeze(0), qi], 1)
        qw_pos = torch.arange(qw_ids.shape[1], device=dev).unsqueeze(0)
        q_pos = torch.arange(nq, device=dev).unsqueeze(0)
        zero = torch.zeros(1, 1, dtype=torch.long, device=dev)

        row = {"sample": si, "kl_no_mem": base}
        for j in js:
            h_s = embed_to(model, sink.unsqueeze(0), zero, j)
            # the query is written the same way as the chunk in each arm, so the arms
            # differ in exactly one factor rather than two
            for name, hc, hq in (
                ("nosink", embed_to(model, ci, c_pos, j),
                           embed_to(model, qi, q_pos, j)),
                ("sink",   embed_to(model, w_ids, w_pos, j)[:, 1:, :],
                           embed_to(model, qw_ids, qw_pos, j)[:, 1:, :]),
            ):
                h_pack = torch.cat([h_s, hc, hq], 1)
                row[f"{name}_j{j}"] = kl(ref, forward_from(model, h_pack, pp, j, final=nq)) / base
        rows.append(row)
        if si % 6 == 0:
            print(f"  sample {si}: base {base:.3f} nats", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args),
                               "layers": js, "rows": rows}, indent=1))

    n = len(rows)
    mb = sum(r["kl_no_mem"] for r in rows) / n
    print(f"\nKL(ref || no_mem) = {mb:.3f} nats over {n} samples")
    print("\n  j | no sink (published) | +1 BOS sink |  delta")
    print("----+---------------------+-------------+--------")
    for j in js:
        a = sum(r[f"nosink_j{j}"] for r in rows) / n
        b = sum(r[f"sink_j{j}"] for r in rows) / n
        print(f"{j:3d} |        {a:.3f}        |    {b:.3f}    | {b - a:+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
