"""
S9 -- One coherent frontier: accuracy against BOTH stored bytes and read compute.

WHY THIS SUPERSEDES THE EARLIER ARM B
-------------------------------------
S3's arm B spliced a memory branch into |S| layers and blended it with a gate, which had
to be fitted per sample because nothing was trained -- an oracle, so an upper bound
rather than a deployable number.  S6/S7 then showed that a plain prefix KV cache needs no
gate at all: attention over a cache is just attention.  So the whole "store KV" family is
ONE knob -- at how many layers is the cache present -- and it needs no fitted parameter:

    |S| = 1   cache visible at one layer          4 KB/token
    |S| = 8   cache visible at eight layers      32 KB/token
    |S| = L   cache visible everywhere          112 KB/token   (= CacheBlend's object)

At a layer with no cache the query simply attends to itself, exactly as in arm B, minus
the gate.  This is both more faithful to the published methods and free of the oracle
caveat, so it replaces arm B rather than sitting alongside it.

THE TWO AXES
------------
Arm A and the KV family sit at opposite corners, and neither dominates:

  arm A (Encbank)   4 KB/token always, but recomputes (L-j) layers over the WHOLE pack
  KV family       4|S| KB/token, but recomputes NOTHING over the context

Read compute is reported in token-layers, the only unit in which the two are comparable:
    arm A     (L - j) * pack_tokens          + attention
    KV @|S|   L * query_tokens               + |S| * query_tokens * cache_tokens of attn
With a 2081-token pack and a 32-token query these differ by more than an order of
magnitude, so quoting "matched bytes" alone hides most of the trade.  Every row below
carries both numbers.

METRIC -- unchanged from S2/S3/S4/S6/S7/S8
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem)
full-vocabulary KL in nats at every query position, averaged over positions then samples.
Multi-chunk throughout (S7 showed the single-chunk setting makes chunk staleness
identically zero), and one sink is always kept (S8: without it the cache is worse than
no memory at all, by up to 4235x).
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
from s6_cacheblend_arm import chunk_kv_all_layers


@torch.no_grad()
def forward_cache_subset(model, ids, position_ids, cache, layers, n_score):
    """Run `ids` with the prefix cache visible ONLY at the layers in `layers`."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    T = ids.shape[1]
    M = cache[0][0].shape[2]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
    mask_c = torch.cat([torch.ones(T, M, dtype=torch.bool, device=ids.device), causal], 1)
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        r = h
        x = layer.input_layernorm(h)
        B = x.shape[0]
        shape = (B, T, -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
        v = a.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if i in layers:
            ck, cv = cache[i]
            k, v = torch.cat([ck, k], 2), torch.cat([cv, v], 2)
            am = mask_c.view(1, 1, T, M + T)
        else:
            am = causal.view(1, 1, T, T)
        n_rep = a.config.num_attention_heads // a.config.num_key_value_heads
        o = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                                           attn_mask=am, scale=a.scaling)
        h = r + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--js", default="2,6,10,14,18,22,26")
    ap.add_argument("--sizes", default="1,2,4,8,14,28")
    ap.add_argument("--out", default="exp/results/s9_frontier.json")
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
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // model.config.num_attention_heads)
    kb_layer = 4 * model.config.num_key_value_heads * hd / 1024
    kb_resid = 2 * model.config.hidden_size / 1024

    bos = tok.bos_token_id or tok.eos_token_id
    sink = torch.tensor([bos], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
    c, nq, k = args.chunk, args.query, args.k
    doc_len = args.n_doc * c
    js = [int(x) for x in args.js.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]

    # KV-family layer sets: spread over the deep half, which S2 showed transmits most
    def pick(s):
        if s >= L:
            return set(range(L))
        return {L // 2 + int(round(i * (L - 2 - L // 2) / max(s - 1, 1))) for i in range(s)} \
            if s > 1 else {22}

    rows = []
    for si in range(args.n):
        st = si * (doc_len + nq)
        if st + doc_len + nq > len(ids_all):
            break
        doc = ids_all[st:st + doc_len]
        qi = ids_all[st + doc_len:st + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(args.n_doc)]
        picks = sorted({round(i * (args.n_doc - 1) / max(k - 1, 1)) for i in range(k)})
        sel = [chunks[p] for p in picks]
        kk = len(sel)

        pack = torch.cat([sink.unsqueeze(0)] + sel + [qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)
        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))
        row = {"sample": si, "kl_no_mem": base, "pack_tokens": int(pack.shape[1])}

        # ---- arm A: Encbank, residual at depth j, recompute [j, L) ----
        for j in js:
            parts = []
            off = 1
            for ch in sel:
                cid = torch.cat([sink.unsqueeze(0), ch], 1)
                cpos = torch.arange(off - 1, off - 1 + cid.shape[1], device=dev).unsqueeze(0)
                parts.append(embed_to(model, cid, cpos, j)[:, 1:, :])
                off += c
            qid = torch.cat([sink.unsqueeze(0), qi], 1)
            qpos = torch.arange(qid.shape[1], device=dev).unsqueeze(0)
            hq = embed_to(model, qid, qpos, j)[:, 1:, :]
            h_s = embed_to(model, sink.unsqueeze(0),
                           torch.zeros(1, 1, dtype=torch.long, device=dev), j)
            hp = torch.cat([h_s] + parts + [hq], 1)
            row[f"A_j{j}"] = kl(ref, forward_from(model, hp, pp, j, final=nq)) / base

        # ---- KV family: prefix cache visible at |S| layers, no gate ----
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
        q_pos = torch.arange(off, off + nq, device=dev).unsqueeze(0)
        for s in sizes:
            row[f"KV_S{s}"] = kl(
                ref, forward_cache_subset(model, qi, q_pos, merged, pick(s), nq)) / base
        rows.append(row)
        if si % 4 == 0:
            print(f"  sample {si}: base {base:.3f} nats, pack {row['pack_tokens']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "args": vars(args), "rows": rows}, indent=1))

    n = len(rows)
    m = lambda key: sum(r[key] for r in rows) / n
    P = int(m("pack_tokens"))
    print(f"\nKL(ref || no_mem) = {m('kl_no_mem'):.3f} nats, {n} samples, k={k}, pack={P} tok")
    print(f"residual {kb_resid:.0f} KB/token; KV {kb_layer:.0f} KB/layer/token\n")
    print("arm            | bytes/tok | read compute (token-layers over context) | frac")
    print("---------------+-----------+------------------------------------------+------")
    for j in js:
        print(f"A  Encbank j={j:<2d}  |  {kb_resid:5.0f} KB  |  {(L-j)*P:>10d}  ({L-j} layers x pack)      "
              f"| {m(f'A_j{j}'):.3f}")
    for s in sizes:
        tag = " = CacheBlend" if s >= L else ""
        print(f"KV |S|={s:<2d}{tag:<7s}|  {s*kb_layer:5.0f} KB  |  {0:>10d}  ({s} attn layers only)    "
              f"| {m(f'KV_S{s}'):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
