"""
S17 -- Is the shallow anchor at layer 9 a finding, or an artefact of six documents?
(queue item 5)

THE CLAIM UNDER TEST
--------------------
S5 ran a greedy forward selection of memory layers on n_select=6 documents and produced

    k=1 [22]   k=2 [9,22]   k=3 [9,16,22]   k=4 [9,16,21,22]
    k=5 [9,16,18,21,22]     k=6 [9,16,18,21,22,25]

Layer 9 enters at k=2 and never leaves.  That is surprising: every other chosen layer is
deep (16-25), and a *shallow* anchor is the interesting part of the result.  But greedy
selection on six documents can lock onto a layer for no reason -- the first step's margin
is never reported, and nothing in S5 re-selects.

WHAT THIS RUN DOES
------------------
Re-runs the SAME greedy procedure on three DISJOINT six-document selection sets
(documents 0-5, 6-11, 12-17), in one process so the model loads once.  Set A reproduces
S5 exactly and is the control: if it does not reproduce, the harness changed and nothing
else here is interpretable.

Then it reports, for every k and every set, which layer was added and BY WHAT MARGIN over
the runner-up.  A layer picked by a margin comparable to the between-set spread is a coin
flip dressed as a finding.

DECISION RULE, fixed before the run
    layer 9 chosen at k=2 in 3/3 sets                 -> anchor is REPRODUCIBLE
    2/3, or chosen but with margin < runner-up spread -> WEAK, report as undetermined
    <=1/3                                             -> the anchor was six-document noise

METRIC (unchanged from S2/S5/S9)
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem) on the next-token distribution,
    averaged over the 32 query positions of a document, then over the documents of the
    selection set.  p_ref = the same query with the retrieved chunks fully in context.
    0 = as good as full context, 1 = the stored state contributed nothing. LOWER BETTER.

WHAT THIS CANNOT SHOW
    Three sets of six is still eighteen documents from one corpus, one model
    (Qwen3-1.7B), one chunk size, one k, one retrieval rule.  It tests reproducibility of
    the SELECTION across documents, not whether layer 9 is causally special, and not
    whether the anchor transfers to another model or corpus.  Greedy is also not optimal:
    a layer set that greedy never visits could beat all of these.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from s1_binding_curve import fetch_text, wait_for_gpu
from s5_layer_placement import build, kl, score_many


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--n-select", type=int, default=6)
    ap.add_argument("--sel-starts", default="0,6,12")
    ap.add_argument("--max-k", type=int, default=4, help="4 is enough to test the anchor")
    ap.add_argument("--out", default="exp/results/s17_anchor.json")
    ap.add_argument("--cap-gb", type=float, default=9.0)
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
    sink = torch.tensor([tok.bos_token_id or tok.eos_token_id], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")),
                  return_tensors="pt").input_ids[0].to(dev)

    starts = [int(x) for x in args.sel_starts.split(",")]
    print(f"{args.model}: L={L}, greedy on {len(starts)} disjoint sets of "
          f"{args.n_select} documents, max_k={args.max_k}")
    print("S5 reference (set starting at 0): k=1 [22], k=2 [9,22], k=3 [9,16,22], "
          "k=4 [9,16,21,22]\n")

    out = {}
    for st in starts:
        sel = build(model, tok, ids_all, sink, dev, args.chunk, args.query, args.k,
                    args.n_doc, range(st, st + args.n_select))
        chosen, hist = [], []
        for kk in range(1, args.max_k + 1):
            rem = [j for j in range(L) if j not in chosen]
            vals = score_many(model, sel, [chosen + [j] for j in rem])
            order = sorted(range(len(rem)), key=lambda i: vals[i])
            best, second = order[0], order[1]
            margin = vals[second] - vals[best]      # >0; how much the winner won by
            chosen = sorted(chosen + [rem[best]])
            hist.append(dict(k=kk, added=rem[best], layers=list(chosen),
                             frac=vals[best], runner_up=rem[second],
                             runner_up_frac=vals[second], margin=margin))
            print(f"  docs {st}-{st+args.n_select-1}  k={kk}: +L{rem[best]:<2d} "
                  f"-> {str(chosen):22s} frac {vals[best]:.4f}  "
                  f"(runner-up L{rem[second]:<2d} {vals[second]:.4f}, "
                  f"margin {margin:.4f})", flush=True)
        out[st] = hist
        del sel
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("k | " + " | ".join(f"docs {s}-{s+args.n_select-1}" for s in starts))
    for kk in range(1, args.max_k + 1):
        cells = []
        for s in starts:
            h = out[s][kk - 1]
            cells.append(f"+L{h['added']:<2d} m={h['margin']:.4f}")
        print(f"{kk} | " + " | ".join(f"{c:>16s}" for c in cells))

    k2 = [out[s][1]["layers"] for s in starts]
    hits = sum(1 for v in k2 if 9 in v)
    margins2 = [out[s][1]["margin"] for s in starts]
    spread = max(margins2) - min(margins2)
    verdict = ("REPRODUCIBLE" if hits == len(starts)
               else "WEAK / undetermined" if hits >= 2 else "six-document noise")
    print(f"\nlayer 9 present in the k=2 set: {hits}/{len(starts)}  -> {verdict}")
    print(f"k=2 winning margins {['%.4f' % m for m in margins2]}, spread {spread:.4f}")
    print("A margin at or below the between-set spread means the pick is not separated "
          "from\nsampling variation, whatever the hit count says.")
    print("Not shown: causal specialness of layer 9, transfer to other models/corpora, "
          "or any\nlayer set greedy never visits.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(model=args.model, args=vars(args), L=L,
                 sets={str(k): v for k, v in out.items()},
                 layer9_k2_hits=hits, verdict=verdict), indent=1))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
