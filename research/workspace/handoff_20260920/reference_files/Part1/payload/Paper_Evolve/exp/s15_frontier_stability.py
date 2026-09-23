"""
S15 -- How much of the S9 frontier survives its own sample size?  (0 GPU)

TWO QUESTIONS, BOTH ANSWERABLE FROM FROZEN RECORDS
--------------------------------------------------
Q1 (queue item 5).  The greedy layer set in S5 was chosen on n_select=6 samples and the
    S9 frontier was measured on n=12.  Is the frontier's ordering a finding or is it
    sampling noise?  S9 stored PER-SAMPLE frac for all 13 arms on the same 12 documents,
    so the arms are PAIRED and a paired bootstrap over documents is the right test.
    No GPU is needed and no new decoding happens here.

Q2 (queue item 4).  S9's KV_S arms used the evenly-spaced HEURISTIC layer set.  S5 showed
    a greedy set beats it on held-out data by 0.078-0.126 frac, so S9 understated the KV
    family by that much.  Before spending GPU on a re-measurement, check offline WHERE
    that correction can actually land:

      * at k=1 the greedy and heuristic sets are the SAME layer ([22] in S5), so the
        correction there is exactly 0.000 -- and k=1 is the matched-byte operating point
        (one layer of KV = 4096 B/token = one residual under 2:1 GQA on Qwen3-1.7B);
      * at S=28 every layer is already stored, so there is no selection left to improve
        and the correction is 0 by construction.

    Those are the two ENDPOINTS of the frontier.  A correction that cannot move either
    endpoint cannot overturn the frontier's shape, only its interior.  This script bounds
    the interior movement using S5's own measured greedy-minus-heuristic deltas.

METRIC (unchanged from S2/S9, restated so this file stands alone)
    frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem), next-token distribution, averaged
    over the 32 query positions of one document, then reported per document.
    p_ref = the same query with the retrieved chunks fully in context.
    0 = indistinguishable from having the chunks in context; 1 = the stored state
    contributed nothing.  LOWER IS BETTER.

WHAT THIS CANNOT SHOW
    It cannot tell us the frac of any layer set that was never decoded -- the Q2 bound is
    an interval carried over from S5's k-indexed deltas, not a measurement of S9's arms.
    It cannot extend to other models, chunk sizes, k, or corpora.  A bootstrap over 12
    documents estimates document-to-document variance only; it says nothing about
    variance over reveal seeds, retrieval draws, or checkpoints, none of which were varied.
"""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

BYTES_KB = {"A": 4.0, "KV": 4.0}  # per stored layer; residual = 1 layer of KV here


def mean(v):
    return sum(v) / len(v)


def boot_ci(vals, n_boot, rng, lo=2.5, hi=97.5):
    n = len(vals)
    ms = sorted(mean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return ms[int(lo / 100 * n_boot)], ms[int(hi / 100 * n_boot)]


def sign_test_p(diffs):
    """Exact two-sided sign test on nonzero paired differences."""
    nz = [d for d in diffs if d != 0]
    n, k = len(nz), sum(1 for d in nz if d > 0)
    if n == 0:
        return 1.0
    c = [1]
    for _ in range(n):  # binomial coefficients
        c = [1] + [c[i] + c[i + 1] for i in range(len(c) - 1)] + [1]
    tot = float(sum(c))
    tail = sum(c[i] for i in range(n + 1) if min(i, n - i) <= min(k, n - k))
    return min(1.0, tail / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s9", default="exp/results/s9_frontier.json")
    ap.add_argument("--s5", default="exp/results/s5_placement.json")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default="exp/results/s15_frontier_stability.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    s9 = json.loads(Path(args.s9).read_text())
    rows = s9["rows"]
    arms = [k for k in rows[0] if k not in ("sample", "kl_no_mem", "pack_tokens")]
    col = {a: [r[a] for r in rows] for a in arms}
    n = len(rows)
    kb = {a: (4.0 if a.startswith("A_") else 4.0 * int(a.split("S")[1])) for a in arms}

    print(f"S9 frontier, n={n} paired documents, {len(arms)} arms, "
          f"{args.n_boot} bootstrap resamples (seed {args.seed})")
    print("frac: 0 = as good as full context, 1 = memory contributed nothing. LOWER BETTER\n")
    print("arm      | KB/tok |  mean   [ 95% CI ]      | width | rank  LOO rank range")
    print("---------+--------+-------------------------+-------+---------------------")

    # leave-one-out rank stability: drop each document, re-rank all arms by mean frac
    loo_ranks = {a: [] for a in arms}
    for drop in range(n):
        order = sorted(arms, key=lambda a: mean([col[a][i] for i in range(n) if i != drop]))
        for pos, a in enumerate(order):
            loo_ranks[a].append(pos + 1)
    full_rank = {a: i + 1 for i, a in enumerate(sorted(arms, key=lambda a: mean(col[a])))}

    summary = {}
    for a in sorted(arms, key=lambda a: mean(col[a])):
        lo, hi = boot_ci(col[a], args.n_boot, rng)
        r = loo_ranks[a]
        flag = "" if min(r) == max(r) else "  <- unstable"
        summary[a] = dict(kb=kb[a], mean=mean(col[a]), ci=[lo, hi],
                          rank=full_rank[a], loo=[min(r), max(r)])
        print(f"{a:<8s} | {kb[a]:6.0f} | {mean(col[a]):.3f}  [{lo:.3f}, {hi:.3f}] | "
              f"{hi-lo:.3f} | {full_rank[a]:>4d}   {min(r)}-{max(r)}{flag}")

    stable = sum(1 for a in arms if min(loo_ranks[a]) == max(loo_ranks[a]))
    print(f"\n{stable}/{len(arms)} arms keep an identical rank under every leave-one-out "
          f"refit; the ordering is {'STABLE' if stable == len(arms) else 'NOT fully stable'} "
          f"at n={n}.")

    # --- the matched-byte contrast, paired ---------------------------------------
    print("\nMATCHED BYTES (4 KB/token): store one residual h_j vs store one layer of KV")
    print("  paired per document; negative diff = the A arm is better")
    pairs = [("A_j2", "KV_S1"), ("A_j6", "KV_S1"), ("A_j10", "KV_S1"), ("A_j26", "KV_S1")]
    contrasts = {}
    for x, y in pairs:
        d = [col[x][i] - col[y][i] for i in range(n)]
        lo, hi = boot_ci(d, args.n_boot, rng)
        p = sign_test_p(d)
        w = sum(1 for v in d if v < 0)
        contrasts[f"{x}-{y}"] = dict(diff=mean(d), ci=[lo, hi], wins=w, n=n, sign_p=p)
        print(f"  {x:>6s} - {y:<6s}  {mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"wins {w}/{n}  sign p={p:.4g}")

    # --- Q2: where can the greedy correction land? --------------------------------
    s5 = json.loads(Path(args.s5).read_text())
    print("\nQ2: the S5 greedy-minus-heuristic correction, by selection size k")
    print("  (held-out frac; negative = greedy better = S9's KV arms were understated)")
    deltas = {}
    for r in s5["eval"]:
        d = r["greedy_frac"] - r["heur_frac"]
        same = r["greedy_layers"] == r["heur_layers"]
        deltas[r["k"]] = d
        note = "  SAME LAYER SET -> correction is exactly 0" if same else ""
        print(f"  k={r['k']}: greedy {r['greedy_frac']:.3f}  heur {r['heur_frac']:.3f}  "
              f"delta {d:+.3f}{note}")

    k1 = next(r for r in s5["eval"] if r["k"] == 1)
    print(f"\n  At k=1 -- the MATCHED-BYTE operating point (4 KB/token) -- greedy and "
          f"heuristic\n  both choose {k1['greedy_layers']}, so the correction is "
          f"{k1['greedy_frac']-k1['heur_frac']:+.3f}.")
    print("  At S=28 every layer is stored, so there is no selection to improve: 0 by "
          "construction.")
    print("  Both ENDPOINTS of the frontier are therefore untouched by queue item 4.")

    worst = min(deltas.values())
    a_best, kv1 = mean(col["A_j2"]), mean(col["KV_S1"])
    print(f"\n  Interior bound: applying the LARGEST measured correction ({worst:+.3f}) to "
          f"KV_S1\n  would give {kv1 + worst:.3f}, still {(kv1+worst)/a_best:.1f}x worse "
          f"than A_j2 ({a_best:.3f}) at the same 4 KB/token.")
    print("  So re-measuring the KV family with greedy sets cannot overturn the "
          "matched-byte\n  comparison; it can only move the interior of the frontier.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(args=vars(args), n=n, arms=summary,
                 contrasts=contrasts, s5_deltas=deltas), indent=1))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
