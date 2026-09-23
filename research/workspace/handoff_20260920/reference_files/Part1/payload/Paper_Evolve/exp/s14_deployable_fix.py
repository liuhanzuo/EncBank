"""
S14 -- Make the S13 repair deployable, and fix the S13 KV-family control.

WHY (two corrections from the S13 adversarial verification)
-------------------------------------------------------------
1. S13 wrote every chunk at the pack slot it later occupies ([BOS at 0; chunk at 1+i*c ..]).
   That is NOT RoPE-translation-equivalent to CoMem's write (write_sink=True writes
   [BOS at 0; chunk at 1..c]): the BOS-to-chunk distance differs.  So S13's A_on and A_chunk
   are not S12's arms (0.350 vs 0.289 at j=12), and the repair's cached K/V were produced
   by a slot-aware write that a once-per-chunk store cannot reproduce.
   Here every chunk is written ONCE at local positions, exactly as CoMem does; K is kept
   PRE-RoPE (after k_norm), and at read time it is rotated to the chunk's pack positions.
   V and h_j are position-free.  This is what a deployment would store.
2. S13's KV(|S|) references (query runs all L layers alone, cache visible at |S| layers)
   scored frac > 1 on 8B; the S9 construction on 1.7B scored 0.5-0.8.  The suspected cause
   is that the query had no sink of its own at cache-free layers, but that was not tested.
   Here the KV arms give the query its own BOS at position 0 (attending only to itself),
   with the chunk cache (no sink entry) visible at the chosen layers.

ARMS per (sample, j), all with the local write
    A_off            CoMem published (no sink, local)               8 KB/token
    A_on             CoMem with write sink, local (== S12's A_on)   8 KB
    A_chunk          query h_j from the reference pack (floor, == S12's A_chunk)
    FL_empty         query at pack positions, no cache (BOS geometry control)
    FL_1             S = {1}                                        12 KB
    FL_g3w / FL_g3p  the |S|=3 sets greedy-selected in S13 on wikitext / PG19
                     (evaluated on BOTH corpora here: the transfer test)  20 KB
    FL_all           S = [0, j)                                     8 + 4j KB
REFERENCE arms, once per sample (j-independent)
    C_bos            full-depth local-written KV (rotated), query with own BOS, cache at all
                     L layers, no recompute                          4L KB  (S13's C had a
                     merged sink entry and a sinkless query; both are reported by pairing
                     against S13's json on the same samples in the analysis)
    KVb2/KVb4/KVb14  same, cache visible at 2/4/14 evenly spaced layers only

Samples reproduce S13's EVAL split exactly (same corpus halves, same windows; S13's
selection windows are skipped), so every arm here can be paired per sample with S13's.
Metric as S12/S13: frac, gap, top1.  Peak GPU memory is printed at the end.
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from s13_query_fix import (_block, forward_from, lower_with_cache, kl, score, even_layers,
                           samples_from, pg19_text, tiny_model, embed_to)

# |S|=3 sets chosen by greedy forward selection in S13 (selection splits disjoint from eval)
G3_WIKI = {6: [0, 3, 4], 9: [1, 4, 8], 12: [1, 9, 11], 15: [0, 9, 13], 18: [0, 3, 14], 24: [0, 19, 22]}
G3_PG19 = {6: [1, 4, 5], 9: [1, 4, 5], 12: [1, 4, 5], 15: [0, 9, 12], 18: [0, 15, 17], 24: [1, 17, 22]}


def _qkv_pre(layer, x, pos_emb):
    """Like s13._qkv but also returns the PRE-RoPE (post-k_norm) key."""
    a = layer.self_attn
    B, T, _ = x.shape
    shape = (B, T, -1, a.head_dim)
    q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
    k_pre = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
    v = a.v_proj(x).view(shape).transpose(1, 2)
    cos, sin = pos_emb
    q, k = apply_rotary_pos_emb(q, k_pre, cos, sin)
    return q, k, v, k_pre


@torch.no_grad()
def write_local(model, ids, upto, keep_h=()):
    """CoMem's write: `ids` alone at LOCAL positions 0..T-1.  Returns
    ({l: (K_pre_l, V_l)} for l < upto, {i: h_i} for i in keep_h)."""
    dev = ids.device
    pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, pos)
    cache, hs = {}, {}
    for i in range(upto):
        if i in keep_h:
            hs[i] = h
        layer = model.model.layers[i]
        q, k, v, k_pre = _qkv_pre(layer, layer.input_layernorm(h), pos_emb)
        cache[i] = (k_pre, v)
        h = _block(layer, h, q, k, v, None)
    if upto in keep_h:
        hs[upto] = h
    return cache, hs


@torch.no_grad()
def rotate(model, k_pre, position_ids):
    """Apply RoPE at `position_ids` to a stored pre-RoPE key [1, n_kv, T, hd]."""
    dummy = torch.zeros(1, position_ids.shape[1], 1, device=k_pre.device, dtype=k_pre.dtype)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    return apply_rotary_pos_emb(k_pre, k_pre, cos, sin)[1]


@torch.no_grad()
def full_with_cache_bos(model, ids, position_ids, cache, layers, n_score):
    """All L layers on [BOS; query]; BOS attends only to itself; query attends to the
    chunk cache at `layers` plus BOS plus itself causally."""
    h = model.model.embed_tokens(ids)
    pos_emb = model.model.rotary_emb(h, position_ids)
    T = ids.shape[1]
    dev = ids.device
    M = cache[0][0].shape[2]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    mask_c = torch.zeros(T, M + T, dtype=torch.bool, device=dev)
    mask_c[:, M:] = causal
    mask_c[1:, :M] = True
    mask_c = mask_c.view(1, 1, T, M + T)
    for i, layer in enumerate(model.model.layers):
        q, k, v, _ = _qkv_pre(layer, layer.input_layernorm(h), pos_emb)
        if i in layers:
            ck, cv = cache[i]
            h = _block(layer, h, q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), mask_c)
        else:
            h = _block(layer, h, q, k, v, None)
    return model.lm_head(model.model.norm(h[:, -n_score:, :])).float()


@torch.no_grad()
def prepare(model, sink, sel, qi, js, L):
    dev = qi.device
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

    # CoMem's write: [BOS; chunk] at local positions, once per chunk; keep pre-RoPE K, V, h_j
    rot, hs_on = {l: ([], []) for l in range(L)}, []
    for i, ch in enumerate(sel):
        cache, hs = write_local(model, torch.cat([sink, ch], 1), L, keep_h=set(js))
        off = 1 + i * c
        ppos = torch.arange(off, off + c, device=dev).unsqueeze(0)
        for l, (k_pre, v) in cache.items():
            rot[l][0].append(rotate(model, k_pre[:, :, 1:, :], ppos))   # drop BOS entry
            rot[l][1].append(v[:, :, 1:, :])
        hs_on.append({l: h[:, 1:, :] for l, h in hs.items()})
    lower = {l: (torch.cat(ks, 2), torch.cat(vs, 2)) for l, (ks, vs) in rot.items()}
    _, sink_hs = write_local(model, sink, L, keep_h=set(js))
    q_off = 1 + kk * c
    q_pos = torch.arange(q_off, q_off + nq, device=dev).unsqueeze(0)
    qb_ids = torch.cat([sink, qi], 1)
    qb_pos = torch.cat([torch.zeros(1, 1, dtype=torch.long, device=dev), q_pos], 1)
    return dict(pack=pack, pp=pp, ref=ref, base=base, targets=targets, ref_nll=ref_nll,
                sel=sel, qi=qi, kk=kk, c=c, nq=nq, lower=lower, hs_on=hs_on,
                sink_hs=sink_hs, q_pos=q_pos, qb_ids=qb_ids, qb_pos=qb_pos)


@torch.no_grad()
def arm_FL(model, P, j, S):
    hq = lower_with_cache(model, P["qb_ids"], P["qb_pos"], j, P["lower"], sorted(S),
                          bos_first=True)[:, 1:, :]
    hc = torch.cat([hs[j] for hs in P["hs_on"]], 1)
    return forward_from(model, torch.cat([P["sink_hs"][j], hc, hq], 1), P["pp"], j, P["nq"])


@torch.no_grad()
def eval_arms(model, sink, P, j, sets, kv_sizes, L):
    dev = P["qi"].device
    c, nq, kk = P["c"], P["nq"], P["kk"]
    sc = lambda lg: score(lg, P["ref"], P["base"], P["targets"], P["ref_nll"])
    loc = lambda n: torch.arange(n, device=dev).unsqueeze(0)
    out = {}
    hc_on = torch.cat([hs[j] for hs in P["hs_on"]], 1)
    h_s = P["sink_hs"][j]
    hc_off = torch.cat([embed_to(model, ch, loc(c), j) for ch in P["sel"]], 1)
    hq_off = embed_to(model, P["qi"], loc(nq), j)
    out["A_off"] = sc(forward_from(model, torch.cat([h_s, hc_off, hq_off], 1), P["pp"], j, nq))
    hq_on = embed_to(model, torch.cat([sink, P["qi"]], 1), loc(nq + 1), j)[:, 1:, :]
    out["A_on"] = sc(forward_from(model, torch.cat([h_s, hc_on, hq_on], 1), P["pp"], j, nq))
    hq_ref = embed_to(model, P["pack"], P["pp"], j)[:, 1 + kk * c:, :]
    out["A_chunk"] = sc(forward_from(model, torch.cat([h_s, hc_on, hq_ref], 1), P["pp"], j, nq))
    for name, S in sets.items():
        out[name] = sc(arm_FL(model, P, j, S))
    if kv_sizes is not None:
        out["C_bos"] = sc(full_with_cache_bos(model, P["qb_ids"], P["qb_pos"], P["lower"],
                                              set(range(L)), nq))
        for m in kv_sizes:
            out[f"KVb{m}"] = sc(full_with_cache_bos(model, P["qb_ids"], P["qb_pos"], P["lower"],
                                                    set(even_layers(L, m)), nq))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "pg19"])
    ap.add_argument("--pg19", default="/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl")
    ap.add_argument("--pg19-skip", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n-sel", type=int, default=12, help="S13 selection windows to skip (pg19)")
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--js", default="6,9,12,15,18,24")
    ap.add_argument("--kv-sizes", default="2,4,14")
    ap.add_argument("--out", default="")
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--need-gb", type=float, default=22.0)
    ap.add_argument("--idle-slack-gb", type=float, default=8.0)
    ap.add_argument("--cpu-smoke", action="store_true")
    args = ap.parse_args()

    if args.cpu_smoke:
        dev = "cpu"
        model = tiny_model()
        L, js = 4, [1, 2, 3]
        torch.manual_seed(1)
        ids_all = torch.randint(5, 512, (4000,))
        sink = torch.tensor([[3]])
        c, nq, k, n_doc, n_eval = 8, 4, 2, 3, 3
        eval_ids = ids_all[1500:]
        kv_sizes = [1, 2]
        g3w = g3p = {1: [0], 2: [0, 1], 3: [0, 2]}
        out = Path("exp/results/s14_smoke.json")
    else:
        from gpu_gate import acquire_gpu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from s1_binding_curve import fetch_text
        acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb, idle_slack_gb=args.idle_slack_gb,
                    tag="s14_deployable_fix")
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
        L = model.config.num_hidden_layers
        js = [int(x) for x in args.js.split(",")]
        sink = torch.tensor([[tok.bos_token_id or tok.eos_token_id]], device=dev)
        c, nq, k, n_doc, n_eval = args.chunk, args.query, args.k, args.n_doc, args.n_eval
        kv_sizes = [int(x) for x in args.kv_sizes.split(",")]
        g3w, g3p = G3_WIKI, G3_PG19
        if args.corpus == "wikitext":
            eval_ids = tok(fetch_text("main", Path("exp/data")), return_tensors="pt").input_ids[0].to(dev)
        else:
            need = (n_doc * c + nq) * (args.n_sel + n_eval) * 5
            ids = tok(pg19_text(args.pg19, need, args.pg19_skip), return_tensors="pt").input_ids[0].to(dev)
            eval_ids = ids[(n_doc * c + nq) * args.n_sel:]     # == S13's eval split
        tag = Path(args.model.rstrip("/\\")).name.lower()
        out = Path(args.out or f"exp/results/s14_deployable_{tag}_{args.corpus}.json")

    eval_samples = samples_from(eval_ids, c, n_doc, nq, k, n_eval)
    hd = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    kb_layer = 4 * model.config.num_key_value_heads * hd / 1024
    kb_resid = 2 * model.config.hidden_size / 1024
    print(f"{'tiny' if args.cpu_smoke else args.model}: L={L}, js={js}, {len(eval_samples)} eval "
          f"samples, resid {kb_resid:.0f} KB, KV {kb_layer:.0f} KB/layer/token", flush=True)

    if not args.cpu_smoke:
        probe = eval_ids[: c + nq].unsqueeze(0)
        pos = torch.arange(probe.shape[1], device=dev).unsqueeze(0)
        ours = forward_from(model, embed_to(model, probe, pos, 0), pos, 0, nq)
        theirs = model(input_ids=probe, position_ids=pos, use_cache=False).logits[:, -nq:, :].float()
        d = kl(theirs, ours)
        print(f"manual-forward check: KL(lib || ours) = {d:.2e} nats", flush=True)
        assert d < 3e-3
        # rotation check: rotating a pre-RoPE key to positions p must equal computing the key
        # at positions p directly (translation invariance of the write is a separate matter)
        x = model.model.embed_tokens(probe)
        lyr = model.model.layers[0]
        pe = model.model.rotary_emb(x, pos)
        _, k_direct, _, k_pre = _qkv_pre(lyr, lyr.input_layernorm(x), pe)
        k_rot = rotate(model, k_pre, pos)
        print(f"rotation check: max|K_rot - K_direct| = {(k_rot - k_direct).abs().max().item():.2e}",
              flush=True)

    rows = []
    for si, (chunks, qi) in enumerate(eval_samples):
        P = prepare(model, sink, chunks, qi, js, L)
        row = {"sample": si, "k": P["kk"], "kl_no_mem": P["base"]}
        for j in js:
            sets = {"FL_empty": [], "FL_1": [1] if j > 1 else [0], "FL_g3w": g3w[j],
                    "FL_g3p": g3p[j], "FL_all": list(range(j))}
            res = eval_arms(model, sink, P, j, sets, kv_sizes if j == js[0] else None, L)
            for arm, r in res.items():
                key = arm if arm == "C_bos" or arm.startswith("KVb") else f"{arm}_j{j}"
                for mname, v in r.items():
                    row[f"{key}_{mname}"] = v
        rows.append(row)
        jm = js[len(js) // 2]
        print(f"  eval {si}: base {P['base']:.3f}  j={jm}: A_on {row[f'A_on_j{jm}_frac']:.3f}  "
              f"FL_1 {row[f'FL_1_j{jm}_frac']:.3f}  FL_g3w {row[f'FL_g3w_j{jm}_frac']:.3f}  "
              f"FL_all {row[f'FL_all_j{jm}_frac']:.3f}  A_chunk {row[f'A_chunk_j{jm}_frac']:.3f}  "
              f"C_bos {row['C_bos_frac']:.3f}  KVb14 {row['KVb14_frac'] if 'KVb14_frac' in row else float('nan'):.3f}",
              flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "L": L, "args": vars(args), "js": js,
                                   "kb_resid": kb_resid, "kb_layer": kb_layer,
                                   "g3_wiki": {str(j): v for j, v in g3w.items()},
                                   "g3_pg19": {str(j): v for j, v in g3p.items()},
                                   "rows": rows}, indent=1))

    n = len(rows)

    def m(key):
        vals = [r[key] for r in rows if key in r]
        return st.mean(vals) if vals else float("nan")

    print(f"\n{'tiny' if args.cpu_smoke else args.model} L={L} eval n={n}  KL(ref||no_mem)={m('kl_no_mem'):.3f}")
    print(f"refs: C_bos ({L*kb_layer:.0f} KB) frac {m('C_bos_frac'):.3f}; " +
          "  ".join(f"KVb{s} ({s*kb_layer:.0f} KB) {m(f'KVb{s}_frac'):.3f}" for s in kv_sizes))
    print("\n  j | A_off  A_on   A_chunk | FL_empty FL_1   FL_g3w FL_g3p | FL_all | gap A_on/FL_g3w/FL_all | top1 A_on/FL_all")
    for j in js:
        print(f"{j:3d} | {m(f'A_off_j{j}_frac'):.3f}  {m(f'A_on_j{j}_frac'):.3f}  {m(f'A_chunk_j{j}_frac'):.3f}  | "
              f"{m(f'FL_empty_j{j}_frac'):.3f}    {m(f'FL_1_j{j}_frac'):.3f}  {m(f'FL_g3w_j{j}_frac'):.3f}  "
              f"{m(f'FL_g3p_j{j}_frac'):.3f} | {m(f'FL_all_j{j}_frac'):.3f}  | "
              f"{m(f'A_on_j{j}_gap'):.2f} / {m(f'FL_g3w_j{j}_gap'):.2f} / {m(f'FL_all_j{j}_gap'):.2f} | "
              f"{m(f'A_on_j{j}_top1'):.3f} / {m(f'FL_all_j{j}_top1'):.3f}")
    if not args.cpu_smoke:
        print(f"\npeak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
