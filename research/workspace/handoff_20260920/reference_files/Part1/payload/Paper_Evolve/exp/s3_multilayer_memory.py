"""
S3 -- Does arm B escape the 0.86 floor with more memory layers and a per-head gate?

WHAT S2 ESTABLISHED
-------------------
Arm B (store K,V at ONE layer, never recompute the context, blend the memory branch
into that layer with a single global scalar gate) never got below
    frac = KL(p_ref || p_B) / KL(p_ref || p_no_mem) = 0.86
and the gate sweep was MONOTONE with its optimum pinned at the boundary g = 1.0.
g = 0 reproduces no_mem exactly (frac 1.000, an identity check), so the memory is not
harmful -- it is starved.  Even routing 100% of one layer's attention output through
the memory admits only 16.4% of the chunk's value.  That is a bandwidth limit, not a
tuning failure, which is what makes this experiment worth running.

WHAT THIS ADDS
--------------
The published configuration (Memorizing Transformers, arXiv:2203.08913) is not what S2
implemented.  Two things were missing, and both are added here:

  * MULTIPLE memory layers, |S| in {1, 2, 4, 8}.
  * A PER-HEAD gate rather than one global scalar.

The gate is not trained here; it is fitted per sample by Adam directly against the
objective, i.e. an ORACLE gate.  That is deliberate.  A fitted-on-the-evaluated-sample
gate cannot be deployed, but it upper-bounds what any trained gate could achieve at this
operating point, which is exactly the object a feasibility question needs.  If arm B
cannot clear the floor even with an oracle per-head gate, no training schedule will
rescue it; if it does clear the floor, the size of the gap tells us what training is
worth.

BYTES -- READ THIS BEFORE COMPARING TO ARM A
--------------------------------------------
On Qwen3-1.7B (d=2048, d_kv=1024, 2:1 GQA, bf16) one layer of K,V is 4*d_kv = 4096
B/token and one residual is 2*d = 4096 B/token, so |S| = 1 is matched against arm A and
**|S| = k costs k times arm A's storage**.  Arm A's cost is 4 KB/token at every j.  So
the honest axis is frac against bytes, not frac against |S|, and any |S| > 1 result is
arm B spending more storage than arm A to try to catch it.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, stored_kv


def _attn(layer, h, pos_emb, mem=None, gate=None):
    """Qwen3 attention with an optional non-positional memory branch and per-head gate.

    `gate` is [1, n_heads, 1, 1] in [0,1]; the memory branch uses the UN-rotated query
    against un-rotated stored keys, the standard choice for a kNN memory layer.
    """
    a = layer.self_attn
    B, T, _ = h.shape
    shape = (B, T, -1, a.head_dim)
    q_raw = a.q_norm(a.q_proj(h).view(shape)).transpose(1, 2)
    k_raw = a.k_norm(a.k_proj(h).view(shape)).transpose(1, 2)
    v = a.v_proj(h).view(shape).transpose(1, 2)
    cos, sin = pos_emb
    q, k = apply_rotary_pos_emb(q_raw, k_raw, cos, sin)
    n_rep = a.config.num_attention_heads // a.config.num_key_value_heads
    out = F.scaled_dot_product_attention(
        q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), is_causal=True, scale=a.scaling
    )
    if mem is not None:
        km, vm = mem
        out_mem = F.scaled_dot_product_attention(
            q_raw, repeat_kv(km, n_rep), repeat_kv(vm, n_rep),
            is_causal=False, scale=a.scaling,
        )
        out = (1.0 - gate) * out + gate * out_mem
    return a.o_proj(out.transpose(1, 2).reshape(B, T, -1))


def forward_mem(model, ids, position_ids, mems: dict, gates: dict, n_score: int):
    """Full forward of `ids` with memory spliced in at the layers named in `mems`."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    for i, layer in enumerate(model.model.layers):
        r = h
        x = layer.input_layernorm(h)
        x = _attn(layer, x, pos_emb, mem=mems.get(i), gate=gates.get(i))
        h = r + x
        r = h
        h = r + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float()


def kl_vec(ref_logits, q_logits):
    lp = F.log_softmax(ref_logits[0].float(), -1)
    lq = F.log_softmax(q_logits[0].float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sizes", default="1,2,4,8", help="|S| values")
    ap.add_argument("--steps", type=int, default=40, help="Adam steps for the oracle gate")
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--out", default="exp/results/s3_multilayer.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    model.requires_grad_(False)
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    sizes = [int(x) for x in args.sizes.split(",")]

    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
    c, nq = args.chunk, args.query

    # Memory layers: spread over the DEEP HALF.  S2 found j=22 admits far more than
    # j=12 (frac 0.836 vs 0.978 at g=1), so the shallow half is not where the signal is.
    def pick(k):
        return sorted({L // 2 + int(round(i * (L - 2 - L // 2) / max(k - 1, 1)))
                       for i in range(k)}) if k > 1 else [22]

    rows = []
    for si in range(args.n):
        s = si * (c + nq)
        if s + c + nq > len(ids_all):
            break
        ci = ids_all[s:s + c].unsqueeze(0)
        qi = ids_all[s + c:s + c + nq].unsqueeze(0)

        pack = torch.cat([sink.unsqueeze(0), ci, qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        w = torch.cat([sink.unsqueeze(0), ci], 1)
        wp = torch.arange(w.shape[1], device=dev).unsqueeze(0)

        with torch.no_grad():
            ref = forward_mem(model, pack, pp, {}, {}, nq)
            base = float(kl_vec(ref, forward_mem(model, short, sp, {}, {}, nq)))

        row = {"sample": si, "kl_no_mem": base}
        for k in sizes:
            layers = pick(k)
            with torch.no_grad():
                mems = {}
                for j in layers:
                    km, vm = stored_kv(model, w, wp, j)
                    mems[j] = (km[:, :, 1:, :], vm[:, :, 1:, :])  # drop the write sink
            theta = {j: torch.zeros(1, H, 1, 1, device=dev, dtype=torch.float32,
                                    requires_grad=True) for j in layers}
            opt = torch.optim.Adam(list(theta.values()), lr=args.lr)
            best = float("inf")
            for _ in range(args.steps):
                gates = {j: torch.sigmoid(t).to(torch.bfloat16) for j, t in theta.items()}
                loss = kl_vec(ref, forward_mem(model, short, sp, mems, gates, nq))
                opt.zero_grad(); loss.backward(); opt.step()
                best = min(best, float(loss))
            row[f"S{k}"] = best / base
            row[f"S{k}_layers"] = layers
            row[f"S{k}_gate_mean"] = float(
                torch.cat([torch.sigmoid(t).flatten() for t in theta.values()]).mean())
        rows.append(row)
        print(f"  sample {si}: base {base:.3f} nats | " +
              " ".join(f"|S|={k} {row[f'S{k}']:.3f}" for k in sizes), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args), "rows": rows}, indent=1))

    n = len(rows)
    print(f"\narm B with an ORACLE per-head gate, {n} samples")
    print("|S| | bytes/token | x arm A |  frac  | mean gate | layers")
    print("----+-------------+---------+--------+-----------+--------")
    for k in sizes:
        f = sum(r[f"S{k}"] for r in rows) / n
        g = sum(r[f"S{k}_gate_mean"] for r in rows) / n
        print(f"{k:3d} |   {4*k:5d} KB  |  {k:4d}x  | {f:.3f}  |   {g:.3f}   | "
              f"{rows[0][f'S{k}_layers']}")
    print("\nreference, arm A at 4096 B/token: j=2 0.03, j=12 0.37, j=22 0.80, j=26 0.95")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
