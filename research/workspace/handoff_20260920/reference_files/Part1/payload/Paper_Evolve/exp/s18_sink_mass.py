"""
S18 -- Where does the attention mass GO when the sink is missing?  (queue item 6)

WHAT IS ALREADY ESTABLISHED, AND WHAT IS NOT
--------------------------------------------
S7/S8 measured the OUTCOME: merging k independently-cached chunks without a sink token at
the head of the merged cache costs frac 1.355 at k=4 (worse than having no memory at all),
and 0.041 with one BOS sink.  At k=1 the sinkless arm is 3.246 and the sinked arm 0.0006.
That is a large, reproducible effect, and the attention-sink reading (Xiao et al.'s
StreamingLLM: some heads emit near-no-op attention that has to land somewhere, and the
first position absorbs it) is CONSISTENT with it.  Consistent is not evidence.  Nothing so
far has looked at a single attention weight.

This script does.  It re-runs the same merged-cache read with and without the sink and
partitions every query token's attention distribution over three key regions:

    sink    the n_sink BOS tokens at the head of the merged cache (absent when n_sink=0)
    chunk   the k*chunk cached chunk tokens
    query   the query's own tokens, attended causally

Those three sum to 1 by construction, so "where the mass went" is a closed accounting.

PREDICTIONS THE ATTENTION-SINK READING MAKES (written before the run)
  P1  With a sink, a substantial share of mass sits on it, concentrated in some layers
      rather than spread evenly.  If m_sink is uniformly tiny, the mechanism is NOT
      absorption and the S8 effect needs a different explanation.
  P2  Removing it does not spread the freed mass evenly.  It should pile onto the FIRST
      few chunk tokens, which become the de facto sink.  Reported as m_chunk_first4.
  P3  The layers that gain the most mass are the layers where the sink held the most.
      Reported as the Pearson correlation between m_sink(n_sink=1) and
      delta m_chunk(n_sink=0 minus n_sink=1) across layers.
P1-P3 are falsifiable and I state the numbers either way.

METRIC DEFINITIONS
    m_region(layer) = attention probability summed over the key positions of that region,
    averaged first over the 16 attention heads, then over the 32 query positions, then
    over samples.  Unitless, in [0,1]; the three regions sum to 1.0 per layer.
    entropy(layer) = Shannon entropy in nats of the full attention distribution over all
    M+T keys, same averaging.  Higher = more diffuse.
    Reference for the deltas is always the n_sink=1 arm on the SAME sample and layer, so
    the comparison is paired.

WHAT THIS CANNOT SHOW
    Attention mass is not causal attribution: a head can place mass on a token and route
    little information through it (the value vector may be near-zero).  This measures
    where the softmax puts its probability, nothing more.  It also cannot show that the
    sink is the only repair, or that the same layers matter in another model, chunk size,
    k, or corpus.  Qwen3-1.7B only.
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
def forward_with_cache_attn(model, ids, position_ids, cache, n_score, bounds):
    """forward_with_cache, but with the softmax written out so the attention
    probabilities can be partitioned.  `bounds` = (n_sink, M): keys [0,n_sink) are sink,
    [n_sink,M) chunk, [M, M+T) query.  Returns (logits, per-layer stats)."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    T = ids.shape[1]
    M = cache[0][0].shape[2]
    ns, _ = bounds
    mask = torch.zeros(T, M + T, dtype=torch.bool, device=ids.device)
    mask[:, :M] = True
    mask[:, M:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))

    stats = []
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
        ck, cv = cache[i]
        k = torch.cat([ck, k], dim=2)
        v = torch.cat([cv, v], dim=2)
        n_rep = a.config.num_attention_heads // a.config.num_key_value_heads
        kr, vr = repeat_kv(k, n_rep), repeat_kv(v, n_rep)

        scores = (q.float() @ kr.float().transpose(-1, -2)) * a.scaling
        scores = scores.masked_fill(~mask.view(1, 1, T, M + T), float("-inf"))
        p = scores.softmax(-1)                       # [1, n_h, T, M+T]

        pm = p.mean(dim=(0, 1, 2))                   # mean over heads and query positions
        ent = float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1).mean())
        stats.append(dict(
            layer=i,
            m_sink=float(pm[:ns].sum()) if ns else 0.0,
            m_chunk=float(pm[ns:M].sum()),
            m_query=float(pm[M:].sum()),
            m_chunk_first4=float(pm[ns:ns + 4].sum()),
            entropy=ent))

        o = (p.to(vr.dtype) @ vr)
        h = r + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float(), stats


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx * dy else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--ks", default="1,4")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="exp/results/s18_sink_mass.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    ap.add_argument("--idle-slack-gb", type=float, default=5.0,
                    help="card counts as idle at or below this; owner set 5 GB on 2026-09-05")
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb, idle_slack_gb=args.idle_slack_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    L = model.config.num_hidden_layers
    bos = tok.bos_token_id or tok.eos_token_id
    sink1 = torch.tensor([bos], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")),
                  return_tensors="pt").input_ids[0].to(dev)
    c, nq = args.chunk, args.query
    doc_len = args.n_doc * c
    ks = [int(x) for x in args.ks.split(",")]

    acc = {k: {ns: [[] for _ in range(L)] for ns in (0, 1)} for k in ks}
    fracs = {k: {0: [], 1: []} for k in ks}
    nsamp = 0
    for si in range(args.n):
        s = si * (doc_len + nq)
        if s + doc_len + nq > len(ids_all):
            break
        doc = ids_all[s:s + doc_len]
        qi = ids_all[s + doc_len:s + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(args.n_doc)]
        nsamp += 1

        for k in ks:
            picks = sorted({round(i * (args.n_doc - 1) / max(k - 1, 1)) for i in range(k)})
            sel = [chunks[p] for p in picks]
            pack = torch.cat([sink1.unsqueeze(0)] + sel + [qi], 1)
            pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
            ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)
            short = torch.cat([sink1.unsqueeze(0), qi], 1)
            sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
            base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))

            caches, off = [], 1
            for ch in sel:
                cid = torch.cat([sink1.unsqueeze(0), ch], 1)
                cpos = torch.arange(off - 1, off - 1 + cid.shape[1], device=dev).unsqueeze(0)
                cc = chunk_kv_all_layers(model, cid, cpos)
                caches.append({i: (v[0][:, :, 1:, :], v[1][:, :, 1:, :]) for i, v in cc.items()})
                off += c
            q_pos = torch.arange(off, off + nq, device=dev).unsqueeze(0)

            for ns in (0, 1):
                parts = []
                if ns:
                    sids = torch.full((1, ns), bos, device=dev, dtype=torch.long)
                    spos = torch.arange(ns, device=dev).unsqueeze(0)
                    parts.append(chunk_kv_all_layers(model, sids, spos))
                parts.extend(caches)
                merged = {i: (torch.cat([p[i][0] for p in parts], 2),
                              torch.cat([p[i][1] for p in parts], 2)) for i in range(L)}
                M = merged[0][0].shape[2]
                lg, st = forward_with_cache_attn(model, qi, q_pos, merged, nq, (ns, M))
                fracs[k][ns].append(kl(ref, lg) / base)
                for i in range(L):
                    acc[k][ns][i].append(st[i])
                del merged
            del caches
            torch.cuda.empty_cache()
        print(f"  sample {si} done", flush=True)

    def m(k, ns, i, key):
        v = [d[key] for d in acc[k][ns][i]]
        return sum(v) / len(v)

    out = {}
    for k in ks:
        f0 = sum(fracs[k][0]) / len(fracs[k][0])
        f1 = sum(fracs[k][1]) / len(fracs[k][1])
        print(f"\n{'='*78}\nk={k}  frac: no sink {f0:.3f}   one sink {f1:.4f}   "
              f"(n={nsamp} samples, {L} layers)")
        print("     |  WITH SINK (ns=1)                  |  NO SINK (ns=0)          | moved")
        print("layer| m_sink m_chunk m_query  ent  1st4  | m_chunk m_query 1st4 ent | d_chunk")
        rows = []
        for i in range(L):
            s1 = {key: m(k, 1, i, key) for key in
                  ("m_sink", "m_chunk", "m_query", "entropy", "m_chunk_first4")}
            s0 = {key: m(k, 0, i, key) for key in
                  ("m_chunk", "m_query", "entropy", "m_chunk_first4")}
            d = s0["m_chunk"] - s1["m_chunk"]
            rows.append(dict(layer=i, ns1=s1, ns0=s0, d_chunk=d))
            if i % 3 == 0 or i == L - 1:
                print(f"{i:4d} | {s1['m_sink']:.3f}  {s1['m_chunk']:.3f}  {s1['m_query']:.3f} "
                      f"{s1['entropy']:5.2f} {s1['m_chunk_first4']:.3f} | "
                      f"{s0['m_chunk']:.3f}  {s0['m_query']:.3f} {s0['m_chunk_first4']:.3f} "
                      f"{s0['entropy']:5.2f} | {d:+.3f}")

        msink = [r["ns1"]["m_sink"] for r in rows]
        dchunk = [r["d_chunk"] for r in rows]
        top = sorted(rows, key=lambda r: -r["ns1"]["m_sink"])[:3]
        print(f"\n  P1  mean m_sink over layers {sum(msink)/L:.3f}, max {max(msink):.3f} "
              f"at layer {msink.index(max(msink))}; top-3 sink layers "
              f"{[r['layer'] for r in top]}")
        f4_1 = sum(r["ns1"]["m_chunk_first4"] for r in rows) / L
        f4_0 = sum(r["ns0"]["m_chunk_first4"] for r in rows) / L
        print(f"  P2  mass on the first 4 chunk tokens: {f4_1:.3f} with sink -> "
              f"{f4_0:.3f} without  ({f4_0-f4_1:+.3f})")
        e1 = sum(r["ns1"]["entropy"] for r in rows) / L
        e0 = sum(r["ns0"]["entropy"] for r in rows) / L
        print(f"      attention entropy {e1:.2f} -> {e0:.2f} nats ({e0-e1:+.2f})")
        print(f"  P3  corr(m_sink, delta m_chunk) over {L} layers = "
              f"{pearson(msink, dchunk):+.3f}")
        out[k] = dict(frac_no_sink=f0, frac_sink=f1, rows=rows,
                      p1_mean_msink=sum(msink)/L, p2_first4=[f4_1, f4_0],
                      p3_corr=pearson(msink, dchunk))

    print("\nreminder: attention mass is not causal attribution -- a head can place "
          "probability\non a token and route little through it.  This is where the softmax "
          "puts mass, only.")
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(model=args.model, args=vars(args), L=L,
                 n_samples=nsamp, k={str(a): b for a, b in out.items()}), indent=1))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
