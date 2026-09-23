"""
S2 -- Functional cost of stored state: KL on the next-token distribution.

WHY THIS AND NOT S1
-------------------
S1 measured how far a stored vector *moves* when its context is removed (relative L2).
That is a representation-space quantity and it does not settle the design question,
because a given displacement in K can change the attention distribution by an arbitrary
amount depending on geometry.  It also repeats C2's own weakness from the other side:
C2 scored with a label-level metric that collapsed ~60% of raw generation differences
onto the same label, so its "relocation is null" result is explicitly a statement about
scorer resolution.  A KL between full next-token distributions collapses nothing.

THE MATCHED-BYTES CONTRAST
--------------------------
On Qwen3-1.7B (d=2048, d_kv=1024, GQA 2:1, bf16):
    residual h_l          = 2*d      = 4096 B/token
    K,V at ONE layer      = 4*d_kv   = 4096 B/token
so "store one residual" and "store one layer of KV" cost exactly the same.  (On a 4:1
GQA model such as the 8B, one residual buys two layers of KV instead; the identity is
d/(2*d_kv) layers.)  That makes the two families directly comparable at fixed storage:

  arm A (CoMem)   store h_j of the chunk, injected into a fresh pack, recompute [j,L).
                  The chunk keeps computing, and it does so in a pack it never saw.
  arm B (Memo-T)  store K_j,V_j of the chunk; the query runs all L layers alone and at
                  layer j additionally attends to that stored memory.  The chunk never
                  computes again, and is visible at exactly one layer.

ARMS
----
  ref      full forward of the pack [sink; C; Q].  Ground truth.
  no_mem   forward of [sink; Q] only.  The floor: what the query knows without C.
  armA_j   as above.
  armB_j   as above, memory branch blended with weight g.

  There is no trained gate here, so armB is swept over g and we report the BEST g per
  sample.  That is deliberate: it upper-bounds what any trained gate could achieve at
  this operating point, which is the right object for a feasibility question.  armA has
  no free parameter, so the comparison is generous to armB.

METRIC
------
  KL(p_ref || p_arm) in nats at the final position of Q, plus the normalised form
      loss_fraction = KL(ref||arm) / KL(ref||no_mem)
  which reads directly as "the fraction of the memory's value that this storage scheme
  threw away": 0 means the stored state was as good as having the chunk in context,
  1 means it contributed nothing over not having the chunk at all.

Everything is training-free and inference-only.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from s1_binding_curve import fetch_text, wait_for_gpu


# --------------------------------------------------------------------------------------
# a hand-rolled forward, so we can start at an arbitrary layer and splice in a memory
# --------------------------------------------------------------------------------------


def _attn(layer, h, pos_emb, causal: bool, mem=None, gate: float = 0.0):
    """One Qwen3 attention block, optionally with a non-positional memory branch.

    `mem` is (K_mem, V_mem) with shape [1, n_kv, M, head_dim], already q/k-normed and
    deliberately NOT rotary-embedded.  The memory branch also uses the un-rotated query,
    matching the standard practice of dropping positional encoding in a kNN memory layer
    (Memorizing Transformers): the stored keys carry the positions they had in their
    source document, which are meaningless relative to the query's positions.
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
        q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), is_causal=causal, scale=a.scaling
    )

    if mem is not None and gate > 0.0:
        k_mem, v_mem = mem
        out_mem = F.scaled_dot_product_attention(
            q_raw, repeat_kv(k_mem, n_rep), repeat_kv(v_mem, n_rep),
            is_causal=False, scale=a.scaling,
        )
        out = (1.0 - gate) * out + gate * out_mem

    out = out.transpose(1, 2).reshape(B, T, -1)
    return a.o_proj(out)


def forward_from(model, h, position_ids, start: int, mem_layer=None, mem=None, gate=0.0,
                 causal: bool = True, final: bool = True, ckpt: bool = False):
    """Run layers [start, L) on `h`, returning logits at the last position.

    `h` is a residual stream, so `start > 0` is exactly the CoMem read: hand the upper
    band a state it did not compute itself.
    """
    pos_emb = model.model.rotary_emb(h, position_ids)

    def run(layer, h, use_mem):
        r = h
        x = layer.input_layernorm(h)
        x = _attn(layer, x, pos_emb, causal, mem=mem if use_mem else None, gate=gate)
        h = r + x
        return h + layer.mlp(layer.post_attention_layernorm(h))

    for i in range(start, len(model.model.layers)):
        layer = model.model.layers[i]
        if ckpt and torch.is_grad_enabled():
            # recompute activations in the backward pass; without this, backprop through
            # the resumed band over a multi-thousand-token pack does not fit beside
            # somebody else's training job
            h = torch.utils.checkpoint.checkpoint(run, layer, h, i == mem_layer,
                                                  use_reentrant=False)
        else:
            h = run(layer, h, i == mem_layer)
    if not final:
        return h
    # logits over the LAST `final` positions.  We score every position of the query, not
    # just its end: the first query token has only the sink and the chunk behind it and
    # therefore depends on the chunk maximally, while by the last token the query's own
    # local context dominates and the chunk is nearly irrelevant.  Scoring only the end
    # would have measured almost nothing (0.07 nats) and called it a null.
    n = final if isinstance(final, int) else 1
    h = model.model.norm(h[:, -n:, :])
    return model.lm_head(h).float()


def embed_to(model, ids, position_ids, upto: int):
    """Embed and run layers [0, upto), returning h_upto. `upto=0` is just the embedding."""
    h = model.model.embed_tokens(ids)
    if upto == 0:
        return h
    pos_emb = model.model.rotary_emb(h, position_ids)
    for i in range(upto):
        layer = model.model.layers[i]
        r = h
        x = layer.input_layernorm(h)
        x = _attn(layer, x, pos_emb, causal=True)
        h = r + x
        r = h
        x = layer.post_attention_layernorm(h)
        h = r + layer.mlp(x)
    return h


@torch.no_grad()
def stored_kv(model, ids, position_ids, layer_idx: int):
    """K_j, V_j of a chunk encoded ALONE -- what arm B would have on disk."""
    h = embed_to(model, ids, position_ids, layer_idx)
    a = model.model.layers[layer_idx].self_attn
    x = model.model.layers[layer_idx].input_layernorm(h)
    B, T, _ = x.shape
    shape = (B, T, -1, a.head_dim)
    k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)  # pre-RoPE, as stored
    v = a.v_proj(x).view(shape).transpose(1, 2)
    return k, v


# --------------------------------------------------------------------------------------


def kl(p_ref_logits, q_logits, per_pos: bool = False):
    """KL(p_ref || q) in nats over the vocabulary, at every scored position.

    Returns the mean over positions, or the per-position vector when `per_pos`.
    """
    lp = F.log_softmax(p_ref_logits[0].float(), -1)
    lq = F.log_softmax(q_logits[0].float(), -1)
    d = (lp.exp() * (lp - lq)).sum(-1)
    return d.tolist() if per_pos else float(d.mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=64)
    ap.add_argument("--n", type=int, default=24, help="samples")
    ap.add_argument("--layers", default="", help="comma list of j; default = every 2")
    ap.add_argument("--gates", default="0.1,0.2,0.3,0.5,0.7,0.9")
    ap.add_argument("--out", default="exp/results/s2_functional_kl.json")
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
    gates = [float(x) for x in args.gates.split(",")]

    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    text = fetch_text("main", Path("exp/data"))
    ids_all = tok(text, return_tensors="pt").input_ids[0].to(dev)

    c, nq = args.chunk, args.query
    need = c + nq
    starts = [i * need for i in range(args.n) if (i + 1) * need <= len(ids_all)]
    print(f"{args.model}: L={L}, {len(starts)} samples, chunk={c} query={nq}")

    # ---- sanity: the hand-rolled forward must match the library's ----
    probe = ids_all[: c + nq].unsqueeze(0)
    pos = torch.arange(probe.shape[1], device=dev).unsqueeze(0)
    ours = forward_from(model, embed_to(model, probe, pos, 0), pos, 0, final=nq)
    theirs = model(input_ids=probe, position_ids=pos, use_cache=False).logits[:, -nq:, :].float()
    d = kl(theirs, ours)
    print(f"manual-forward check: KL(lib || ours) = {d:.2e} nats "
          f"(bf16 noise floor; anything above ~1e-4 is a bug)")
    assert d < 1e-3, "hand-rolled forward does not reproduce the library forward"

    rec = []
    for si, s in enumerate(starts):
        c_ids = ids_all[s : s + c].unsqueeze(0)
        q_ids = ids_all[s + c : s + c + nq].unsqueeze(0)

        pack = torch.cat([sink.unsqueeze(0), c_ids, q_ids], 1)
        pack_pos = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pack_pos, 0), pack_pos, 0, final=nq)

        short = torch.cat([sink.unsqueeze(0), q_ids], 1)
        short_pos = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        no_mem = forward_from(model, embed_to(model, short, short_pos, 0), short_pos, 0, final=nq)
        base = kl(ref, no_mem)

        # write side: chunk alone with a sink, chunk-local positions (S1 showed the sink
        # matters enormously; without it the write is in the degenerate no-sink regime)
        w_ids = torch.cat([sink.unsqueeze(0), c_ids], 1)
        w_pos = torch.arange(w_ids.shape[1], device=dev).unsqueeze(0)
        qw_ids = torch.cat([sink.unsqueeze(0), q_ids], 1)
        qw_pos = torch.arange(qw_ids.shape[1], device=dev).unsqueeze(0)

        row = {"sample": si, "kl_no_mem": base,
               "kl_no_mem_pos": kl(ref, no_mem, per_pos=True)}
        for j in js:
            # ---- arm A: inject h_j, recompute [j, L) over the pack ----
            h_c = embed_to(model, w_ids, w_pos, j)[:, 1:, :]      # drop the write sink
            h_q = embed_to(model, qw_ids, qw_pos, j)[:, 1:, :]
            h_s = embed_to(model, sink.unsqueeze(0),
                           torch.zeros(1, 1, dtype=torch.long, device=dev), j)
            h_pack = torch.cat([h_s, h_c, h_q], 1)
            row[f"A_j{j}"] = kl(ref, forward_from(model, h_pack, pack_pos, j, final=nq))

            # ---- arm B: query alone, memory attended at layer j only ----
            mem = stored_kv(model, w_ids, w_pos, j)
            mem = (mem[0][:, :, 1:, :], mem[1][:, :, 1:, :])      # drop the write sink
            best = min(
                kl(ref, forward_from(model, embed_to(model, short, short_pos, 0),
                                     short_pos, 0, mem_layer=j, mem=mem, gate=g, final=nq))
                for g in gates
            )
            row[f"B_j{j}"] = best
        rec.append(row)
        if si % 4 == 0:
            print(f"  sample {si}: KL(ref||no_mem) = {base:.3f} nats")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args),
                               "layers": js, "gates": gates, "rows": rec}, indent=1))

    n = len(rec)
    mean_base = sum(r["kl_no_mem"] for r in rec) / n
    print(f"\nKL(ref || no_mem) = {mean_base:.3f} nats  <- the whole value of the chunk")
    print("\n  j |  arm A KL   frac |  arm B KL   frac   (frac = KL/KL_no_mem, 0=perfect 1=useless)")
    print("----+------------------+------------------")
    for j in js:
        a = sum(r[f"A_j{j}"] for r in rec) / n
        b = sum(r[f"B_j{j}"] for r in rec) / n
        print(f"{j:3d} |   {a:6.3f}  {a/mean_base:5.2f} |   {b:6.3f}  {b/mean_base:5.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
