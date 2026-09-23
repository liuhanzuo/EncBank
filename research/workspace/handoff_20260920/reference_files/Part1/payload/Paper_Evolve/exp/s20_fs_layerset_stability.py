"""
S20 -- Is the F(S) greedy layer set a finding, or is it noise? (queue item B1)

WHY
---
S13 selected the query-side cache layers S by greedy forward selection on ONE selection
split of 8 documents, and reported sets like [3,6,8] at j=9, with layer 8 recurring in 4 of
6 rows.  Per [[encbank-layer9-anchor-retracted]] a single selection split cannot establish
that a layer is special.

A 0-GPU pass over S13's own stored `cands` (the mean frac of every candidate layer at every
greedy step) already shows the picks are close to ties: at j>=6 every winning margin over
the runner-up is 0.0016-0.0153 frac, while S17 measured a between-selection-set spread of
0.0239 on a comparable greedy.  That is indirect -- it borrows a spread from a different arm
family.  This script measures the spread directly for F(S).

DESIGN
------
Fix j = 9 (the `model_registry` operating point, round(0.33*28)).  Run S13's identical
greedy on THREE DISJOINT selection sets of 8 documents each, in one process.  At each step
report the chosen layer, the runner-up, and the margin.

DECISION RULE, fixed before the run
    same layer chosen at |S|=1 in 3/3 sets AND min margin > between-set spread
        -> the layer is REPRODUCIBLE
    3/3 but margin <= spread, or 2/3                -> WEAK, report as undetermined
    <=1/3                                           -> the set was selection-split noise

METRIC (identical to S13/S19, not redefined here)
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem) on the next-token distribution,
    averaged over the 32 query positions of a document, then over the 8 documents of the
    selection set.  p_ref = the same query with the k=4 retrieved chunks fully in context.
    0 = as good as full context; 1 = the memory contributed nothing.  LOWER IS BETTER.

WHAT THIS CANNOT SHOW
    24 documents from one corpus, one model (Qwen3-1.7B), one chunk size, one k, one j.
    It tests whether the SELECTION reproduces across documents -- not whether any layer is
    causally special, and not whether a set greedy never visits would beat all of these.
"""

import argparse
import json
import statistics as st
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu_gate import acquire_gpu
from s1_binding_curve import fetch_text
from s13_query_fix import arm_F, kl, prepare, samples_from


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--query", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-doc", type=int, default=12)
    ap.add_argument("--j", type=int, default=9)
    ap.add_argument("--n-sel", type=int, default=8)
    ap.add_argument("--n-sets", type=int, default=3)
    ap.add_argument("--greedy-max", type=int, default=3)
    ap.add_argument("--out", default="exp/results/s20_fs_stability.json")
    ap.add_argument("--cap-gb", type=float, default=10.0)
    ap.add_argument("--need-gb", type=float, default=11.0)
    ap.add_argument("--idle-slack-gb", type=float, default=5.0,
                    help="card counts as idle at or below this; owner set 5 GB 2026-09-05")
    args = ap.parse_args()

    acquire_gpu(need_gb=args.need_gb, cap_gb=args.cap_gb,
                idle_slack_gb=args.idle_slack_gb, tag="s20_fs_stability")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    L = model.config.num_hidden_layers
    bos = tok.bos_token_id or tok.eos_token_id
    sink = torch.tensor([[bos]], device=dev)
    ids_all = tok(fetch_text("main", Path("exp/data")),
                  return_tensors="pt").input_ids[0].to(dev)
    j = args.j

    print(f"{args.model}: L={L}, j={j}, {args.n_sets} disjoint selection sets of "
          f"{args.n_sel} documents, greedy to |S|={args.greedy_max}")
    print(f"S13 reference (one set of 8): |S|=1 L8, |S|=2 [6,8], |S|=3 [3,6,8]\n")

    out = {}
    for si in range(args.n_sets):
        start = si * args.n_sel
        samples = samples_from(ids_all, args.chunk, args.n_doc, args.query,
                               args.k, args.n_sel, start=start)
        S, trace = [], []
        for m in range(min(args.greedy_max, j)):
            cands = [l for l in range(j) if l not in S]
            scores = {l: [] for l in cands}
            for chunks, qi in samples:
                P = prepare(model, sink, chunks, qi, [j], j)
                for l in cands:
                    scores[l].append(kl(P["ref"], arm_F(model, P, j, S + [l])) / P["base"])
                del P
            means = {l: st.mean(v) for l, v in scores.items()}
            srt = sorted(means.items(), key=lambda kv: kv[1])
            best, second = srt[0], srt[1]
            S.append(best[0])
            trace.append(dict(size=m + 1, layers=sorted(S), added=best[0],
                              frac=best[1], runner_up=second[0],
                              runner_up_frac=second[1], margin=second[1] - best[1]))
            print(f"  set {si} (docs {start}-{start+args.n_sel-1})  |S|={m+1}: "
                  f"+L{best[0]:<2d} -> {str(sorted(S)):12s} frac {best[1]:.4f}  "
                  f"(runner-up L{second[0]:<2d} {second[1]:.4f}, margin "
                  f"{second[1]-best[1]:.4f})", flush=True)
        out[si] = trace
        torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    print("|S| | " + " | ".join(f"set {s}" for s in range(args.n_sets)))
    for m in range(args.greedy_max):
        print(f"{m+1:3d} | " + " | ".join(
            f"+L{out[s][m]['added']:<2d} m={out[s][m]['margin']:.4f}"
            for s in range(args.n_sets)))

    first = [out[s][0]["added"] for s in range(args.n_sets)]
    margins = [out[s][0]["margin"] for s in range(args.n_sets)]
    spread = max(margins) - min(margins)
    hits = first.count(max(set(first), key=first.count))
    top = max(set(first), key=first.count)
    sep = min(margins) > spread
    verdict = ("REPRODUCIBLE" if hits == args.n_sets and sep
               else "WEAK / undetermined" if hits >= 2 else "selection-split noise")
    print(f"\n|S|=1 picks {first}; most common L{top} in {hits}/{args.n_sets} sets")
    print(f"margins {['%.4f' % v for v in margins]}, between-set spread {spread:.4f}")
    print(f"min margin {min(margins):.4f} {'>' if sep else '<='} spread -> "
          f"{'separated' if sep else 'NOT separated'} from set-to-set variation")
    print(f"VERDICT: {verdict}")
    print("\nS13's own stored margins (0 GPU, same script's `cands`): every greedy step at "
          "j>=6\nwon by 0.0016-0.0153 frac.  If the spread here is comparable, the F(S) "
          "layer sets\ncarry no information about WHICH layers matter, only how many.")
    print("Not shown: causal specialness of any layer; transfer to another model or corpus; "
          "sets greedy never visits.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(model=args.model, args=vars(args), L=L, j=j,
                 sets={str(k): v for k, v in out.items()},
                 first_picks=first, spread=spread, verdict=verdict), indent=1))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
