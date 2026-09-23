"""
S6 -- Put the CacheBlend cache object on the same axes as arm A and arm B.

WHY THIS ARM
------------
Arms A and B so far were mine.  `COMem/comem/cacheblend.py` (450 lines of code, wired
into eval/_cli.py) implements the published member of the store-KV family: CacheBlend
(Yao et al., EuroSys'25, arXiv:2405.16444) caches the FULL per-layer K/V of every chunk,
concatenates the retrieved caches at query time, repairs positions, and selectively
recomputes a small fraction of tokens.  Its own docstring gives the storage as 144
KiB/token on Qwen3-8B against CoMem's 8 KiB -- arithmetic that matches the independent
derivation used throughout these experiments.

Arm C here is that cache object without the selective-recompute repair, which makes it
the clean upper end of the storage axis:

    arm A (CoMem)      one depth-j residual        4 KB/token   recompute [j,L)
    arm B (Memo-T)     K,V at |S| chosen layers    4|S| KB      no recompute, gated
    arm C (CacheBlend) K,V at ALL L layers       112 KB/token   no recompute, no gate

ARM C NEEDS NO GATE, WHICH MATTERS
----------------------------------
Arm B had to blend a memory branch into one layer, so it needed a gate, and with no
training available that gate had to be fitted per sample -- an oracle, i.e. an upper
bound rather than a deployable number.  Arm C has no such problem: a full-depth prefix
KV cache is just attention.  Nothing is fitted, nothing is trained, and the number it
produces is directly deployable.  So arm C is the fair, unhandicapped representative of
"store KV instead of a residual", and if the matched-bytes claim survives against it,
it survives against the strongest form of the objection.

POSITIONS
---------
The chunk is encoded at the positions it will occupy in the read pack, so the cached K
already carries the right rotation and no repair step is needed.  That is the favourable
choice for arm C.

METRIC -- identical to S2/S3/S4 so all four arms are directly comparable
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem)
full-vocabulary KL in nats at every query position, averaged over positions then samples.
0 = as good as having the chunk in context, 1 = contributed nothing.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, forward_from, kl


@torch.no_grad()
def chunk_kv_all_layers(model, ids, position_ids):
    """Post-RoPE K,V at every layer for a chunk encoded ALONE at the given positions.

    This is the CacheBlend cache object: full depth, produced without the query and
    without any neighbouring chunk, exactly as a chunk-KV cache would be built offline.
    """
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    cache = {}
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        r = h
        x = layer.input_layernorm(h)
        B, T, _ = x.shape
        shape = (B, T, -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
        v = a.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cache[i] = (k, v)
        n_rep = a.config.num_attention_heads // a.config.num_key_value_heads
        o = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                                           is_causal=True, scale=a.scaling)
        h = r + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return cache


@torch.no_grad()
def forward_with_cache(model, ids, position_ids, cache, n_score):
    """Run `ids` with a prefix KV cache in front of it at every layer."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    T = ids.shape[1]
    M = cache[0][0].shape[2]
    # every query token sees the whole cache, plus itself causally
    mask = torch.zeros(T, M + T, dtype=torch.bool, device=ids.device)
    mask[:, :M] = True
    mask[:, M:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        r = h
        x = layer.input_layernorm(h)
        B, _, _ = x.shape
        shape = (B, T, -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
        v = a.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ck, cv = cache[i]
        k = torch.cat([ck, k], dim=2)
        v = torch.cat([cv, v], dim=2)
        n_rep = a.config.num_attention_heads // a.config.num_key_value_heads
        o = F.scaled_dot_product_attention(
            q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
            attn_mask=mask.view(1, 1, T, M + T), scale=a.scaling)
        h = r + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="exp/results/s6_cacheblend.json")
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
    d_kv = model.config.num_key_value_heads * getattr(
        model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    kb_per_layer = 4 * d_kv / 1024
    print(f"{args.model}: L={L}, KV {kb_per_layer:.0f} KB/layer/token, "
          f"full depth {L*kb_per_layer:.0f} KB/token; residual "
          f"{2*model.config.hidden_size/1024:.0f} KB/token")

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

        # cache the chunk (with a write sink) at the positions it occupies in the pack
        w_ids = torch.cat([sink.unsqueeze(0), ci], 1)
        w_pos = torch.arange(w_ids.shape[1], device=dev).unsqueeze(0)
        cache = chunk_kv_all_layers(model, w_ids, w_pos)
        q_pos = torch.arange(w_ids.shape[1], w_ids.shape[1] + nq, device=dev).unsqueeze(0)
        armC = kl(ref, forward_with_cache(model, qi, q_pos, cache, nq)) / base

        rows.append({"sample": si, "kl_no_mem": base, "C": armC})
        if si % 6 == 0:
            print(f"  sample {si}: base {base:.3f} nats, arm C {armC:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args), "rows": rows}, indent=1))

    n = len(rows)
    mb = sum(r["kl_no_mem"] for r in rows) / n
    cf = sum(r["C"] for r in rows) / n
    print(f"\nKL(ref || no_mem) = {mb:.3f} nats over {n} samples\n")
    print("arm                        | bytes/token | gate      | frac")
    print("---------------------------+-------------+-----------+------")
    print(f"C  CacheBlend, full depth  | {L*kb_per_layer:6.0f} KB   | none      | {cf:.3f}")
    print("B  1 KV layer  (S3)        |      4 KB   | oracle    | 0.782")
    print("B  8 KV layers (S3)        |     32 KB   | oracle    | 0.313")
    print("A  CoMem j=12  (S2)        |      4 KB   | none      | 0.372")
    print("A  CoMem j=2   (S2)        |      4 KB   | none      | 0.029")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
