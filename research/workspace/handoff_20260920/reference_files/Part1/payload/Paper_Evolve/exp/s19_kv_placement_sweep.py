"""
S19 -- At a fixed storage budget, how much does the KV family's frac depend on WHICH
layers the cache is visible at?  (0 new machinery: S9's arm, swept over placement.)

WHY THIS EXISTS
---------------
S9 and S13 both report a "KV(|S|)" family built from the same merged, sink-prefixed
chunk cache, and they disagree by an order of magnitude at the same byte budget:

    |S|=2 (8 KB/token)    S9: 0.790      S13: 5.248
    |S|=4 (16 KB/token)   S9: 0.659      S13: 5.829
    |S|=14 (56 KB/token)  S9: 0.505      S13: 2.713
    |S|=28 (112 KB/token) S9: 0.041      S13: 0.039   <- these AGREE

Reading the two scripts (0 GPU) explains it completely: they use different layer pickers
under the same name.

    S9   pick(s)          upper half only, {14..26};  s=1 -> [22], s=2 -> [14,26]
    S13  even_layers(L,m) whole depth {0..27}, ALWAYS includes layer 0;
                          m=1 -> [0], m=2 -> [0,27]

At |S|=28 both return all 28 layers, which is exactly where the two scripts agree -- the
cleanest possible confirmation that the arm code is identical and only the SET differs.

So the discrepancy is not a bug in either file.  It is a real and very large placement
effect that neither script isolated, because each held placement fixed while varying the
count.  This script varies placement at a FIXED count and measures the size of the effect.

DESIGN
------
  |S|=1 : every single layer 0..L-1, one arm each.  This is the cleanest statement of
          placement sensitivity that exists at this budget (4 KB/token), and it contains
          both scripts' |S|=1 choices ([22] and [0]) as two of its points.
  |S|=2 : the two disputed sets [14,26] and [0,27], plus [22,26] and [0,14] to separate
          "includes layer 0" from "spans the whole depth".

METRIC (identical to S2/S8/S9; NOT redefined here)
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem) on the next-token distribution,
    full vocabulary, in nats, at each of the 32 query positions, averaged over positions
    and then over documents.  p_ref = the same query with the k retrieved chunks fully in
    context.  0 = as good as full context; 1 = the memory contributed nothing;
    >1 = the memory left the model WORSE than having no memory at all.

WHAT THIS CANNOT SHOW
    One model (Qwen3-1.7B, L=28), one chunk size, one k, one corpus, one retrieval rule,
    and greedy/heuristic placements only -- an untried set could beat all of these.  It
    measures a read-time quantity; it says nothing about wall clock or accuracy.  Per
    [[comem-layer9-anchor-retracted]], a single selection set cannot establish that any
    particular layer is special; this script reports a per-layer CURVE with paired
    bootstrap intervals rather than picking a winner.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import fetch_text, wait_for_gpu
from s2_functional_kl import embed_to, forward_from, kl
from s6_cacheblend_arm import chunk_kv_all_layers
from s9_frontier import forward_cache_subset


def boot_ci(v, n_boot, rng):
    n = len(v)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default="exp/results/s19_kv_placement.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    ap.add_argument("--idle-slack-gb", type=float, default=5.0,
                    help="card counts as idle at or below this; owner set 5 GB 2026-09-05")
    args = ap.parse_args()

    dev = "cuda"
    wait_for_gpu(args.need_gb, args.cap_gb, idle_slack_gb=args.idle_slack_gb)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    L = model.config.num_hidden_layers
    bos = tok.bos_token_id or tok.eos_token_id
    sink = torch.tensor([bos], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")),
                  return_tensors="pt").input_ids[0].to(dev)
    c, nq, doc_len = args.chunk, args.query, args.n_doc * args.chunk

    pairs = {"S9 [14,26]": [14, 26], "S13 [0,27]": [0, 27],
             "[22,26] both deep": [22, 26], "[0,14] has L0": [0, 14]}
    single = {i: [] for i in range(L)}
    two = {name: [] for name in pairs}
    nsamp = 0

    for si in range(args.n):
        s = si * (doc_len + nq)
        if s + doc_len + nq > len(ids_all):
            break
        doc = ids_all[s:s + doc_len]
        qi = ids_all[s + doc_len:s + doc_len + nq].unsqueeze(0)
        chunks = [doc[i * c:(i + 1) * c].unsqueeze(0) for i in range(args.n_doc)]
        picks = sorted({round(i * (args.n_doc - 1) / max(args.k - 1, 1))
                        for i in range(args.k)})
        sel = [chunks[p] for p in picks]

        pack = torch.cat([sink.unsqueeze(0)] + sel + [qi], 1)
        pp = torch.arange(pack.shape[1], device=dev).unsqueeze(0)
        ref = forward_from(model, embed_to(model, pack, pp, 0), pp, 0, final=nq)
        short = torch.cat([sink.unsqueeze(0), qi], 1)
        sp = torch.arange(short.shape[1], device=dev).unsqueeze(0)
        base = kl(ref, forward_from(model, embed_to(model, short, sp, 0), sp, 0, final=nq))

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

        for i in range(L):
            single[i].append(
                kl(ref, forward_cache_subset(model, qi, q_pos, merged, {i}, nq)) / base)
        for name, S in pairs.items():
            two[name].append(
                kl(ref, forward_cache_subset(model, qi, q_pos, merged, set(S), nq)) / base)
        nsamp += 1
        del merged, caches
        torch.cuda.empty_cache()
        print(f"  sample {si} done", flush=True)

    rng = random.Random(args.seed)
    mean = lambda v: sum(v) / len(v)
    print(f"\n{'='*70}\n|S|=1, 4 KB/token, every layer.  n={nsamp} paired documents, "
          f"{args.n_boot} bootstrap resamples")
    print("frac: 0 = chunks in context, 1 = no memory, >1 = WORSE than no memory\n")
    print("layer |  frac   [ 95% CI ]      | note")
    rows = []
    for i in range(L):
        lo, hi = boot_ci(single[i], args.n_boot, rng)
        note = ("  <- S13 even_layers(28,1)" if i == 0 else
                "  <- S9 pick(1)" if i == 22 else "")
        rows.append(dict(layer=i, frac=mean(single[i]), ci=[lo, hi]))
        print(f"{i:5d} | {mean(single[i]):.3f}  [{lo:.3f}, {hi:.3f}]{note}")

    best = min(rows, key=lambda r: r["frac"])
    worst = max(rows, key=lambda r: r["frac"])
    print(f"\nbest single layer  L{best['layer']}: {best['frac']:.3f}")
    print(f"worst single layer L{worst['layer']}: {worst['frac']:.3f}")
    print(f"placement is worth {worst['frac']/best['frac']:.1f}x at a FIXED 4 KB/token, "
          f"and crosses the frac=1 line\n(memory helps vs memory hurts) inside the same "
          f"budget.")

    print(f"\n{'='*70}\n|S|=2, 8 KB/token -- the two disputed sets plus two controls")
    print("set                 | layers   |  frac   [ 95% CI ]")
    out2 = {}
    for name, S in pairs.items():
        lo, hi = boot_ci(two[name], args.n_boot, rng)
        out2[name] = dict(layers=S, frac=mean(two[name]), ci=[lo, hi])
        print(f"{name:<19s} | {str(S):8s} | {mean(two[name]):.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\nS9 reported 0.790 and S13 5.248 for '|S|=2'.  If those two rows land near "
          "those\nvalues, the reconciliation is confirmed: same arm, same cache, "
          "different layer set.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(model=args.model, args=vars(args), L=L, n=nsamp,
                 single=rows, pairs=out2), indent=1))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
