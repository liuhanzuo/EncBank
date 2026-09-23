"""
S13 analysis -- paired statistics for the query-side repair (exp/s13_query_fix.py output).

For every depth j: mean +- sem of frac per arm, paired differences against the deployed
read (A_on and F_empty) with percentile-bootstrap 95% CIs, the fraction of the query-side
loss removed, and the storage each arm costs.  Success criterion fixed before the run:
|S| <= 3 removes at least half of (F_empty - A_chunk) at j = 12; failure: under a quarter.
"""

import json
import math
import random
import statistics as st
import sys


def boot_ci(v, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    rows, js, L = d["rows"], d["js"], d["L"]
    kb_r, kb_l = d["kb_resid"], d["kb_layer"]
    greedy = {int(k): v for k, v in d["greedy"].items()}
    n = len(rows)
    col = lambda key: [r[key] for r in rows if key in r]
    print(f"{d['model']}  L={L}  eval n={n}  corpus={d['args'].get('corpus')}  "
          f"KL(ref||no_mem)={st.mean(col('kl_no_mem')):.3f}")

    print("\n### references on the same samples (storage KB/token)")
    print(f"  C (full-depth isolated KV + sink, no recompute) {L*kb_l:.0f} KB: "
          f"frac {st.mean(col('C_frac')):.3f} +- {sem(col('C_frac')):.3f}")
    for s in d["args"].get("kv_sizes", "").split(","):
        if s and col(f"KV{s}_frac"):
            print(f"  KV{s} ({int(s)*kb_l:.0f} KB): frac {st.mean(col(f'KV{s}_frac')):.3f} "
                  f"+- {sem(col(f'KV{s}_frac')):.3f}")

    arms = ["A_off", "A_on", "F_empty", "F_first", "F_last", "F_even3", "F_g1", "F_g2", "F_g3",
            "F_all", "A_chunk"]
    print("\n### mean frac +- sem per arm and depth")
    print("   j  " + "".join(f"{a:>10s}" for a in arms))
    for j in js:
        line = f"{j:4d}  "
        for a in arms:
            v = col(f"{a}_j{j}_frac")
            line += f"{st.mean(v):10.3f}" if v else f"{'-':>10s}"
        print(line)
    print("   sem")
    for j in js:
        line = f"{j:4d}  "
        for a in arms:
            v = col(f"{a}_j{j}_frac")
            line += f"{sem(v):10.3f}" if v else f"{'-':>10s}"
        print(line)

    print("\n### greedy layer sets (chosen on the selection split) and storage")
    for j in js:
        g = {tr["size"]: tr for tr in greedy.get(j, [])}
        for s, tr in sorted(g.items()):
            print(f"  j={j:2d} |S|={s}: S={tr['layers']}  sel_frac={tr['sel_frac']:.3f}  "
                  f"storage {kb_r + kb_l*s:.0f} KB/token")

    print("\n### paired: how much of the query-side loss (F_empty - A_chunk) each repair removes")
    print("  removed = (F_empty - F_x) / (F_empty - A_chunk); paired bootstrap 95% CI on the "
          "frac difference F_empty - F_x; sign = samples where F_x < F_empty")
    for j in js:
        base = [r[f"F_empty_j{j}_frac"] - r[f"A_chunk_j{j}_frac"] for r in rows]
        loss = st.mean(base)
        print(f"  j={j:2d}  query-side loss F_empty-A_chunk = {loss:.3f}   "
              f"(A_on - A_chunk = {st.mean([r[f'A_on_j{j}_frac'] - r[f'A_chunk_j{j}_frac'] for r in rows]):.3f})")
        for a in ["F_first", "F_last", "F_even3", "F_g1", "F_g2", "F_g3", "F_all"]:
            if not col(f"{a}_j{j}_frac"):
                continue
            diffs = [r[f"F_empty_j{j}_frac"] - r[f"{a}_j{j}_frac"] for r in rows]
            ci = boot_ci(diffs)
            better = sum(x > 0 for x in diffs)
            g = {tr["size"]: tr["layers"] for tr in greedy.get(j, [])}
            layers = (g.get(int(a[-1])) if a.startswith("F_g") else
                      {"F_first": [0], "F_last": [j - 1], "F_all": "all"}.get(a, "even3"))
            kb = kb_r + kb_l * (j if layers == "all" else 3 if layers == "even3" else len(layers))
            print(f"      {a:8s} S={str(layers):16s} {kb:4.0f} KB  frac {st.mean(col(f'{a}_j{j}_frac')):.3f}  "
                  f"removed {st.mean(diffs)/loss if loss else float('nan'):6.1%}  "
                  f"diff {st.mean(diffs):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {better}/{n}")

    print("\n### paper-metric view at each j: gap (ppl ratio) and top1 for A_on / best |S|<=3 / F_all / A_chunk")
    for j in js:
        g = {tr["size"]: tr for tr in greedy.get(j, [])}
        gs = max(g) if g else 0
        best = f"F_g{gs}" if gs else "F_last"
        print(f"  j={j:2d}  gap  A_on {st.mean(col(f'A_on_j{j}_gap')):.2f}  {best} "
              f"{st.mean(col(f'{best}_j{j}_gap')):.2f}  F_all {st.mean(col(f'F_all_j{j}_gap')):.2f}  "
              f"A_chunk {st.mean(col(f'A_chunk_j{j}_gap')):.2f}   |  top1  A_on "
              f"{st.mean(col(f'A_on_j{j}_top1')):.3f}  {best} {st.mean(col(f'{best}_j{j}_top1')):.3f}  "
              f"F_all {st.mean(col(f'F_all_j{j}_top1')):.3f}  A_chunk {st.mean(col(f'A_chunk_j{j}_top1')):.3f}")

    print("\n### verdict on the pre-registered criterion at j=12 (or the middle j)")
    jm = 12 if 12 in js else js[len(js) // 2]
    base = [r[f"F_empty_j{jm}_frac"] - r[f"A_chunk_j{jm}_frac"] for r in rows]
    g = {tr["size"]: tr for tr in greedy.get(jm, [])}
    cands = [a for a in ["F_g3", "F_g2", "F_g1", "F_even3", "F_last", "F_first"] if col(f"{a}_j{jm}_frac")]
    best = min(cands, key=lambda a: st.mean(col(f"{a}_j{jm}_frac")))
    diffs = [r[f"F_empty_j{jm}_frac"] - r[f"{best}_j{jm}_frac"] for r in rows]
    frac_removed = [x / b if b > 1e-6 else float("nan") for x, b in zip(diffs, base)]
    fr = [x for x in frac_removed if not math.isnan(x)]
    ci = boot_ci(fr) if len(fr) > 1 else (float("nan"), float("nan"))
    print(f"  j={jm}: best |S|<=3 arm = {best}; mean removed = {st.mean(diffs)/st.mean(base):.1%} "
          f"(per-sample median {st.median(fr):.1%}, bootstrap CI [{ci[0]:.1%},{ci[1]:.1%}])")
    r = st.mean(diffs) / st.mean(base)
    print("  -> " + ("SUCCESS (>= 50% removed)" if r >= 0.5 else
                     "FAILURE (< 25% removed)" if r < 0.25 else "INCONCLUSIVE (25-50%)"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exp/results/s13_smoke.json")
