"""
S13 -- Repair the query side: let the query's lower band [0, j) attend to the retrieved
chunks' cached K/V at a small set of layers S.  Qwen3-8B, inference only.

WHAT S12 ESTABLISHED (Qwen3-8B, k=4 of 12 chunks, 32-token query)
-----------------------------------------------------------------
Making the query side exact leaves frac 0.005-0.015 at every j to 0.67L (A_chunk); making
the chunk side exact leaves all of the loss (A_query ~ A_on).  The chunk's h_j is a
sufficient upper-band input at any depth; what CoMem loses is that the query's own h_j is
computed through layers [0, j) without ever seeing the retrieved chunks.

THE FIX -- arm F(S)
-------------------
The chunk's write pass through [0, j) already produces K_l, V_l at every lower layer as a
by-product.  Keep them for l in S (a deployment stores pre-RoPE K and rotates at read; here
the chunk is written at the positions it occupies in the pack, which is the same thing by
RoPE translation invariance).  At read time the query runs [0, j) with, at l in S, its keys
extended by the cached chunk K_l/V_l (query at pack positions, its own BOS at 0); the
resulting h_j goes into the pack [h_s; chunk h_j; query h_j] and [j, L) is recomputed
exactly as in CoMem.  Nothing is trained.

  storage    h_j = 2d bytes/token (8 KB on 8B) + 4*n_kv*hd bytes per layer in S (4 KB)
  read       upper band unchanged; plus the query's cross-attention to |S| cached layers
  F(empty)   the query at pack positions with no cache -- equals A_on up to the BOS
             geometry (A_on writes the query at local positions 0..nq)
  F(all)     S = [0, j): the family's ceiling, 8 + 4j KB/token
  A_chunk    query h_j taken from the reference pack: the floor (chunk staleness only)

LAYER SETS
----------
Greedy forward selection (|S| = 1, 2, 3) on a SELECTION split; every reported number is
from a disjoint EVAL split (S5's lesson).  Fixed heuristics on the eval split: {0}, {j-1},
three evenly spaced in [0, j), and all of [0, j).

REFERENCES ON THE SAME SAMPLES
------------------------------
  A_off / A_on   CoMem as published (no write sink) / with a write sink       8 KB
  C              full-depth isolated chunk KV + one sink, no recompute          4L KB
  KV(|S|)        S9's family: query runs all L layers alone, cache visible at
                 |S| evenly spaced layers, no recompute                           4|S| KB

METRIC -- as S12: frac = KL(p_ref||p_arm)/KL(p_ref||p_no_mem) over the 32 query
positions (0 = as good as the chunks in context, 1 = no memory); gap = ppl ratio on the
true query tokens; top1 = argmax agreement with ref.
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


# --------------------------------------------------------------------------------------
# forward pieces (hand-rolled, so a cache can be spliced in at chosen layers)
# --------------------------------------------------------------------------------------


def _qkv(layer, x, pos_emb):
    a = layer.self_attn
    B, T, _ = x.shape
    shape = (B, T, -1, a.head_dim)
    q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
    k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
    v = a.v_proj(x).view(shape).transpose(1, 2)
    cos, sin = pos_emb
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _block(layer, h, q, k, v, mask):
    a = layer.self_attn
    B, T = h.shape[0], h.shape[1]
    n_rep = a.config.num_attention_heads // a.config.num_key_value_heads
    if mask is None:
        o = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                                           is_causal=True, scale=a.scaling)
    else:
        o = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep),
                                           attn_mask=mask, scale=a.scaling)
    h = h + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
    return h + layer.mlp(layer.post_attention_layernorm(h))


@torch.no_grad()
def write_kv(model, ids, position_ids, upto, keep_h=()):
    """Run layers [0, upto) on `ids` alone.  Returns (h_upto, {l: (K_l, V_l)}, {i: h_i}).

    K is post-RoPE at `position_ids`; h_i is the input to layer i for i in keep_h.
    """
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    cache, hs = {}, {}
    for i in range(upto):
        if i in keep_h:
            hs[i] = h
        layer = model.model.layers[i]
        q, k, v = _qkv(layer, layer.input_layernorm(h), pos_emb)
        cache[i] = (k, v)
        h = _block(layer, h, q, k, v, None)
    if upto in keep_h:
        hs[upto] = h
    return h, cache, hs


@torch.no_grad()
def embed_to(model, ids, position_ids, upto):
    return write_kv(model, ids, position_ids, upto)[0]


@torch.no_grad()
def forward_from(model, h, position_ids, start, n_score):
    """Layers [start, L) over a residual `h`; logits at the last n_score positions."""
    pos_emb = model.model.rotary_emb(h, position_ids)
    for i in range(start, len(model.model.layers)):
        layer = model.model.layers[i]
        q, k, v = _qkv(layer, layer.input_layernorm(h), pos_emb)
        h = _block(layer, h, q, k, v, None)
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float()


@torch.no_grad()
def lower_with_cache(model, ids, position_ids, j, cache, S, bos_first):
    """Layers [0, j) on `ids` (= [BOS; query] when bos_first) with the chunk cache visible
    at layers in S.  The BOS attends only to itself, so it stays the canonical sink."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    T = ids.shape[1]
    dev = ids.device
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    mask_c = None
    if S:
        M = cache[S[0]][0].shape[2]
        mask_c = torch.zeros(T, M + T, dtype=torch.bool, device=dev)
        mask_c[:, M:] = causal
        mask_c[(1 if bos_first else 0):, :M] = True
        mask_c = mask_c.view(1, 1, T, M + T)
    for i in range(j):
        layer = model.model.layers[i]
        q, k, v = _qkv(layer, layer.input_layernorm(h), pos_emb)
        if i in S:
            ck, cv = cache[i]
            h = _block(layer, h, q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), mask_c)
        else:
            h = _block(layer, h, q, k, v, None)
    return h


@torch.no_grad()
def full_with_cache(model, ids, position_ids, cache, layers, n_score):
    """All L layers on `ids` alone, a prefix cache visible only at `layers` (S9/S6 arms)."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    T = ids.shape[1]
    dev = ids.device
    M = cache[0][0].shape[2]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    mask_c = torch.cat([torch.ones(T, M, dtype=torch.bool, device=dev), causal], 1)
    mask_c = mask_c.view(1, 1, T, M + T)
    for i, layer in enumerate(model.model.layers):
        q, k, v = _qkv(layer, layer.input_layernorm(h), pos_emb)
        if i in layers:
            ck, cv = cache[i]
            h = _block(layer, h, q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), mask_c)
        else:
            h = _block(layer, h, q, k, v, None)
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float()


def kl(p_ref_logits, q_logits):
    lp = F.log_softmax(p_ref_logits[0].float(), -1)
    lq = F.log_softmax(q_logits[0].float(), -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def score(logits, ref, base, targets, ref_nll):
    lp = F.log_softmax(logits[0, :-1].float(), -1)
    nll = float(-lp.gather(-1, targets[:, None]).squeeze(-1).mean())
    top1 = float((logits[0].argmax(-1) == ref[0].argmax(-1)).float().mean())
    return {"frac": kl(ref, logits) / base, "gap": math.exp(nll - ref_nll), "top1": top1}


def even_layers(j, m):
    if m >= j:
        return list(range(j))
    return sorted({round(i * (j - 1) / max(m - 1, 1)) for i in range(m)})


# --------------------------------------------------------------------------------------
# one sample: build every tensor the arms need
# --------------------------------------------------------------------------------------


@torch.no_grad()
def prepare(model, sink, doc_chunks, qi, js, upto):
    """Everything per sample.  `upto` = max depth whose caches are needed (L for arm C)."""
    dev = qi.device
    sel = doc_chunks
    kk, c, nq = len(sel), sel[0].shape[1], qi.shape[1]
    pack = torch.cat([sink] + sel + [qi], 1)
    pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
    ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, nq)
    targets = qi[0, 1:]
    ref_nll = float(-F.log_softmax(ref[0, :-1].float(), -1)
                    .gather(-1, targets[:, None]).squeeze(-1).mean())
    short = torch.cat([sink, qi], 1)
    sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
    base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, nq))

    # chunk writes: [BOS; chunk] with BOS at 0 and the chunk at its pack positions
    caches, hs_on = [], []
    for i, ch in enumerate(sel):
        off = 1 + i * c
        ids = torch.cat([sink, ch], 1)
        pos = torch.cat([torch.zeros(1, 1, dtype=torch.long, device=dev),
                         torch.arange(off, off + c, device=dev).unsqueeze(0)], 1)
        _, cache, hs = write_kv(model, ids, pos, upto, keep_h=set(js))
        caches.append({l: (v[0][:, :, 1:, :], v[1][:, :, 1:, :]) for l, v in cache.items()})
        hs_on.append({l: h[:, 1:, :] for l, h in hs.items()})
    # the sink alone at position 0 (its K/V and h are exact by causality)
    _, sink_cache, sink_hs = write_kv(model, sink, torch.zeros(1, 1, dtype=torch.long, device=dev),
                                      upto, keep_h=set(js))
    lower = {l: (torch.cat([cc[l][0] for cc in caches], 2),
                 torch.cat([cc[l][1] for cc in caches], 2)) for l in caches[0]}
    merged = {l: (torch.cat([sink_cache[l][0]] + [cc[l][0] for cc in caches], 2),
                  torch.cat([sink_cache[l][1]] + [cc[l][1] for cc in caches], 2))
              for l in caches[0]}
    q_off = 1 + kk * c
    q_pos = torch.arange(q_off, q_off + nq, device=dev).unsqueeze(0)
    qb_ids = torch.cat([sink, qi], 1)
    qb_pos = torch.cat([torch.zeros(1, 1, dtype=torch.long, device=dev), q_pos], 1)
    return dict(pack=pack, pp=pp, ref=ref, base=base, targets=targets, ref_nll=ref_nll,
                sel=sel, qi=qi, kk=kk, c=c, nq=nq, lower=lower, merged=merged,
                hs_on=hs_on, sink_hs=sink_hs, q_pos=q_pos, qb_ids=qb_ids, qb_pos=qb_pos)


@torch.no_grad()
def arm_F(model, P, j, S):
    """Query lower band with the chunk cache visible at S; then CoMem's upper recompute."""
    hq = lower_with_cache(model, P["qb_ids"], P["qb_pos"], j, P["lower"], sorted(S),
                          bos_first=True)[:, 1:, :]
    hc = torch.cat([hs[j] for hs in P["hs_on"]], 1)
    hp = torch.cat([P["sink_hs"][j], hc, hq], 1)
    return forward_from(model, hp, P["pp"], j, P["nq"])


@torch.no_grad()
def eval_arms(model, sink, P, j, layer_sets, with_refs):
    """All arms at depth j on one prepared sample.  Returns {arm: {frac, gap, top1}}."""
    dev = P["qi"].device
    c, nq, kk = P["c"], P["nq"], P["kk"]
    sc = lambda lg: score(lg, P["ref"], P["base"], P["targets"], P["ref_nll"])
    out = {}
    hc_on = torch.cat([hs[j] for hs in P["hs_on"]], 1)
    h_s = P["sink_hs"][j]
    # --- CoMem as published: no write sink, local positions ---
    loc = lambda n: torch.arange(n, device=dev).unsqueeze(0)
    hc_off = torch.cat([embed_to(model, ch, loc(c), j) for ch in P["sel"]], 1)
    hq_off = embed_to(model, P["qi"], loc(nq), j)
    out["A_off"] = sc(forward_from(model, torch.cat([h_s, hc_off, hq_off], 1), P["pp"], j, nq))
    # --- with a write sink, local positions (S12's A_on) ---
    qb_loc = torch.cat([sink, P["qi"]], 1)
    hq_on = embed_to(model, qb_loc, loc(nq + 1), j)[:, 1:, :]
    out["A_on"] = sc(forward_from(model, torch.cat([h_s, hc_on, hq_on], 1), P["pp"], j, nq))
    # --- floor: query h_j from the reference pack ---
    h_ref = embed_to(model, P["pack"], P["pp"], j)
    hq_ref = h_ref[:, 1 + kk * c:, :]
    out["A_chunk"] = sc(forward_from(model, torch.cat([h_s, hc_on, hq_ref], 1), P["pp"], j, nq))
    # --- the fix ---
    for name, S in layer_sets.items():
        out[name] = sc(arm_F(model, P, j, S))
    if with_refs:
        L = len(model.model.layers)
        out["C"] = sc(full_with_cache(model, P["qi"], P["q_pos"], P["merged"], set(range(L)), nq))
        for m in with_refs:
            out[f"KV{m}"] = sc(full_with_cache(model, P["qi"], P["q_pos"], P["merged"],
                                               set(even_layers(L, m)), nq))
    return out


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def samples_from(ids_all, c, n_doc, nq, k, n, start=0):
    """Consecutive windows of (n_doc chunks + query); k chunks scattered evenly."""
    doc_len = n_doc * c
    out = []
    for si in range(n):
        s = start + si * (doc_len + nq)
        if s + doc_len + nq > len(ids_all):
            break
        doc = ids_all[s:s + doc_len]
        qi = ids_all[s + doc_len:s + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(n_doc)]
        picks = sorted({round(i * (n_doc - 1) / max(k - 1, 1)) for i in range(k)})
        out.append(([chunks[p] for p in picks], qi))
    return out


def pg19_text(path, n_chars, skip):
    parts, total = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            t = json.loads(line)["text"]
            parts.append(t)
            total += len(t)
            if total >= n_chars:
                break
    return "\n".join(parts)[:n_chars]


def tiny_model():
    """A random 4-layer Qwen3 for the CPU smoke test; exercises every code path."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).float().eval()


# --------------------------------------------------------------------------------------


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "pg19"])
    ap.add_argument("--pg19", default="/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl")
    ap.add_argument("--pg19-skip", type=int, default=2, help="skip KJV (#10) and Shakespeare")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n-sel", type=int, default=12, help="selection-split samples (greedy)")
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--js", default="6,9,12,15,18,24")
    ap.add_argument("--greedy-max", type=int, default=3)
    ap.add_argument("--kv-sizes", default="2,4,14")
    ap.add_argument("--out", default="")
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--need-gb", type=float, default=22.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0,
                    help="GPU memory the desktop may hold while the card still counts as idle")
    ap.add_argument("--cpu-smoke", action="store_true")
    args = ap.parse_args()

    if args.cpu_smoke:
        dev = "cpu"
        model = tiny_model()
        L = 4
        js = [1, 2, 3]
        torch.manual_seed(1)
        ids_all = torch.randint(5, 512, (4000,))
        sink = torch.tensor([[3]])
        c, nq, k, n_doc = 8, 4, 2, 3
        n_sel, n_eval = 2, 3
        sel_ids, eval_ids = ids_all[:1500], ids_all[1500:]
        kv_sizes = [1, 2]
        out = Path("exp/results/s13_smoke.json")
    else:
        from gpu_gate import acquire_gpu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from s1_binding_curve import fetch_text
        acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb, idle_slack_gb=args.idle_slack_gb,
                    tag="s13_query_fix")
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
        L = model.config.num_hidden_layers
        js = [int(x) for x in args.js.split(",")]
        sink = torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device=dev)
        c, nq, k, n_doc = args.chunk, args.query, args.k, args.n_doc
        n_sel, n_eval = args.n_sel, args.n_eval
        kv_sizes = [int(x) for x in args.kv_sizes.split(",")]
        if args.corpus == "wikitext":
            sel_ids = tok(fetch_text("alt", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
            eval_ids = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
        else:
            need = (n_doc * c + nq) * (n_sel + n_eval) * 5
            ids = tok(pg19_text(args.pg19, need, args.pg19_skip), return_tensors="pt").input_ids[0].to(dev)
            cut = (n_doc * c + nq) * n_sel
            sel_ids, eval_ids = ids[:cut], ids[cut:]
        tag = Path(args.model.rstrip("/\\")).name.lower()
        out = Path(args.out or f"exp/results/s13_query_fix_{tag}_{args.corpus}.json")

    sel_samples = samples_from(sel_ids, c, n_doc, nq, k, n_sel)
    eval_samples = samples_from(eval_ids, c, n_doc, nq, k, n_eval)
    hd = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    kb_layer = 4 * model.config.num_key_value_heads * hd / 1024
    kb_resid = 2 * model.config.hidden_size / 1024
    print(f"{args.model if not args.cpu_smoke else 'tiny'}: L={L}, js={js}, "
          f"{len(sel_samples)} selection / {len(eval_samples)} eval samples, "
          f"resid {kb_resid:.0f} KB, KV {kb_layer:.0f} KB/layer/token", flush=True)

    if not args.cpu_smoke:
        # the hand-rolled forward must match the library forward
        probe = eval_ids[: c + nq].unsqueeze(0)
        pos = torch.arange(probe.shape[1], device=dev).unsqueeze(0)
        ours = forward_from(model, embed_to(model, probe, pos, 0), pos, 0, nq)
        theirs = model(input_ids=probe, position_ids=pos, use_cache=False).logits[:, -nq:, :].float()
        d = kl(theirs, ours)
        print(f"manual-forward check: KL(lib || ours) = {d:.2e} nats", flush=True)
        assert d < 3e-3, "hand-rolled forward does not reproduce the library forward"

    # ---------------- greedy layer selection on the selection split ----------------
    greedy = {}
    for j in js:
        S, trace = [], []
        for m in range(min(args.greedy_max, j)):
            cands = [l for l in range(j) if l not in S]
            scores = {l: [] for l in cands}
            for chunks, qi in sel_samples:
                P = prepare(model, sink, chunks, qi, [j], j)
                for l in cands:
                    scores[l].append(kl(P["ref"], arm_F(model, P, j, S + [l])) / P["base"])
            means = {l: st.mean(v) for l, v in scores.items()}
            best = min(means, key=means.get)
            S.append(best)
            trace.append({"size": m + 1, "layers": sorted(S), "sel_frac": means[best],
                          "cands": {int(l): round(v, 4) for l, v in means.items()}})
            print(f"  greedy j={j} |S|={m+1}: S={sorted(S)} sel_frac={means[best]:.3f}", flush=True)
        greedy[j] = trace

    # ---------------- eval split ----------------
    rows = []
    jm = js[len(js) // 2]
    for si, (chunks, qi) in enumerate(eval_samples):
        P = prepare(model, sink, chunks, qi, js, L)
        row = {"sample": si, "kl_no_mem": P["base"], "k": P["kk"]}
        for j in js:
            sets = {"F_empty": [], "F_first": [0], "F_last": [j - 1],
                    "F_even3": even_layers(j, 3), "F_all": list(range(j))}
            for tr in greedy[j]:
                sets[f"F_g{tr['size']}"] = tr["layers"]
            res = eval_arms(model, sink, P, j, sets, kv_sizes if j == js[0] else None)
            for arm, r in res.items():
                key = arm if arm == "C" or arm.startswith("KV") else f"{arm}_j{j}"
                for mname, v in r.items():
                    row[f"{key}_{mname}"] = v
        rows.append(row)
        gk = f"F_g{min(args.greedy_max, jm)}_j{jm}_frac"
        print(f"  eval {si}: base {P['base']:.3f}  j={jm}: A_on {row[f'A_on_j{jm}_frac']:.3f}  "
              f"F_greedy {row.get(gk, float('nan')):.3f}  F_all {row[f'F_all_j{jm}_frac']:.3f}  "
              f"A_chunk {row[f'A_chunk_j{jm}_frac']:.3f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "L": L, "args": vars(args), "js": js,
                                   "kb_resid": kb_resid, "kb_layer": kb_layer,
                                   "greedy": {str(j): v for j, v in greedy.items()},
                                   "rows": rows}, indent=1))

    # ---------------- summary ----------------
    n = len(rows)

    def m(key):
        vals = [r[key] for r in rows if key in r]
        return st.mean(vals) if vals else float("nan")

    print(f"\n{'tiny' if args.cpu_smoke else args.model} L={L} eval n={n} k={rows[0]['k']}  "
          f"KL(ref||no_mem)={m('kl_no_mem'):.3f}")
    print(f"references (storage KB/token): C {L*kb_layer:.0f} KB frac {m('C_frac'):.3f}; " +
          "  ".join(f"KV{s} {s*kb_layer:.0f} KB frac {m(f'KV{s}_frac'):.3f}" for s in kv_sizes))
    print("\n  j | A_off  A_on  | F_empty F_first F_last F_even3 | F_g1   F_g2   F_g3   | F_all  A_chunk | KB: greedy / F_all")
    print("----+--------------+--------------------------------+----------------------+----------------+-------------------")
    for j in js:
        g = {tr["size"]: tr["layers"] for tr in greedy[j]}
        gmax = g[max(g)] if g else []
        print(f"{j:3d} | {m(f'A_off_j{j}_frac'):.3f}  {m(f'A_on_j{j}_frac'):.3f} | "
              f"{m(f'F_empty_j{j}_frac'):.3f}   {m(f'F_first_j{j}_frac'):.3f}   {m(f'F_last_j{j}_frac'):.3f}  "
              f"{m(f'F_even3_j{j}_frac'):.3f}  | " +
              "  ".join(f"{m(f'F_g{s}_j{j}_frac'):.3f}" if s in g else "  -  " for s in (1, 2, 3)) +
              f" | {m(f'F_all_j{j}_frac'):.3f}  {m(f'A_chunk_j{j}_frac'):.3f}  | "
              f"{kb_resid + kb_layer*len(gmax):.0f} / {kb_resid + kb_layer*j:.0f}   greedy S: {gmax}")
    print("\n  j | gap: A_on  F_greedy  F_all | top1: A_on  F_greedy  F_all")
    for j in js:
        g = {tr["size"]: tr["layers"] for tr in greedy[j]}
        gs = max(g) if g else 0
        print(f"{j:3d} | {m(f'A_on_j{j}_gap'):.2f}  {m(f'F_g{gs}_j{j}_gap'):.2f}  {m(f'F_all_j{j}_gap'):.2f} | "
              f"{m(f'A_on_j{j}_top1'):.3f}  {m(f'F_g{gs}_j{j}_top1'):.3f}  {m(f'F_all_j{j}_top1'):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
