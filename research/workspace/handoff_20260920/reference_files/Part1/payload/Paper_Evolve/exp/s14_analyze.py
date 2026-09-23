"""
S14 analysis -- the deployable (local-write, pre-RoPE K) repair, paired against S13's
slot-aware run on the SAME eval samples.

Usage: python exp/s14_analyze.py exp/results/s14_deployable_qwen3-8b_wikitext.json \
                                 exp/results/s13_query_fix_qwen3-8b_wikitext.json
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


def main(p14, p13=None):
    d = json.load(open(p14, encoding="utf-8"))
    rows, js, L = d["rows"], d["js"], d["L"]
    kb_r, kb_l = d["kb_resid"], d["kb_layer"]
    n = len(rows)
    col = lambda key: [r[key] for r in rows if key in r]
    corpus = d["args"].get("corpus")
    print(f"{d['model']}  L={L}  eval n={n}  corpus={corpus}  KL(ref||no_mem)={st.mean(col('kl_no_mem')):.3f}")

    print("\n### references with the query's own BOS (storage KB/token)")
    print(f"  C_bos  ({L*kb_l:.0f} KB): frac {st.mean(col('C_bos_frac')):.3f} +- {sem(col('C_bos_frac')):.3f}"
          f"  gap {st.mean(col('C_bos_gap')):.3f}  top1 {st.mean(col('C_bos_top1')):.3f}")
    for s in d["args"].get("kv_sizes", "").split(","):
        if s and col(f"KVb{s}_frac"):
            print(f"  KVb{s} ({int(s)*kb_l:.0f} KB): frac {st.mean(col(f'KVb{s}_frac')):.3f} +- {sem(col(f'KVb{s}_frac')):.3f}")

    arms = ["A_off", "A_on", "A_chunk", "FL_empty", "FL_1", "FL_g3w", "FL_g3p", "FL_all"]
    print("\n### mean frac per arm (deployable local write); storage: A 8, FL_1 12, FL_g3 20, FL_all 8+4j KB")
    print("   j  " + "".join(f"{a:>10s}" for a in arms))
    for j in js:
        print(f"{j:4d}  " + "".join(f"{st.mean(col(f'{a}_j{j}_frac')):10.3f}" for a in arms))
    print("   gap (ppl ratio): A_on / FL_1 / FL_g3w / FL_g3p / FL_all;  top1: A_on / FL_all")
    for j in js:
        print(f"{j:4d}  {st.mean(col(f'A_on_j{j}_gap')):.2f} / {st.mean(col(f'FL_1_j{j}_gap')):.2f} / "
              f"{st.mean(col(f'FL_g3w_j{j}_gap')):.2f} / {st.mean(col(f'FL_g3p_j{j}_gap')):.2f} / "
              f"{st.mean(col(f'FL_all_j{j}_gap')):.2f};   {st.mean(col(f'A_on_j{j}_top1')):.3f} / "
              f"{st.mean(col(f'FL_all_j{j}_top1')):.3f}")

    print("\n### removal of the query-side loss, relative to the DEPLOYED read A_on (local write):")
    print("  removed = (A_on - FL_x)/(A_on - A_chunk); paired bootstrap 95% CI on A_on - FL_x; sign = FL_x < A_on")
    own = "FL_g3w" if corpus == "wikitext" else "FL_g3p"
    other = "FL_g3p" if corpus == "wikitext" else "FL_g3w"
    for j in js:
        loss = st.mean([r[f"A_on_j{j}_frac"] - r[f"A_chunk_j{j}_frac"] for r in rows])
        print(f"  j={j:2d}  A_on - A_chunk = {loss:.3f}")
        for a, label in (("FL_1", "S={1} 12 KB"), (own, f"{own} own-corpus 3 layers 20 KB"),
                         (other, f"{other} CROSS-corpus 3 layers 20 KB"), ("FL_all", f"all {j} layers {kb_r + kb_l*j:.0f} KB")):
            diffs = [r[f"A_on_j{j}_frac"] - r[f"{a}_j{j}_frac"] for r in rows]
            ci = boot_ci(diffs)
            print(f"      {label:36s} frac {st.mean(col(f'{a}_j{j}_frac')):.3f}  removed {st.mean(diffs)/loss if loss else float('nan'):6.1%}  "
                  f"diff {st.mean(diffs):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {sum(x > 0 for x in diffs)}/{n}")

    print("\n### deployable FL_all vs the BOS-corrected full-depth KV reference C_bos, paired")
    for j in js:
        diffs = [r[f"FL_all_j{j}_frac"] - r["C_bos_frac"] for r in rows]
        ci = boot_ci(diffs)
        print(f"  j={j:2d} ({kb_r + kb_l*j:.0f} vs {L*kb_l:.0f} KB): {st.mean(col(f'FL_all_j{j}_frac')):.3f} vs "
              f"{st.mean(col('C_bos_frac')):.3f}  diff {st.mean(diffs):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  FL_all better {sum(x < 0 for x in diffs)}/{n}")

    print("\n### depth sensitivity of the deployable repair: FL_all(j_max) - FL_all(j_min), paired")
    diffs = [r[f"FL_all_j{js[-1]}_frac"] - r[f"FL_all_j{js[0]}_frac"] for r in rows]
    ci = boot_ci(diffs)
    print(f"  FL_all j={js[-1]} - j={js[0]}: {st.mean(diffs):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]; "
          f"A_on: {st.mean([r[f'A_on_j{js[-1]}_frac'] - r[f'A_on_j{js[0]}_frac'] for r in rows]):+.3f}")

    if p13:
        d13 = json.load(open(p13, encoding="utf-8"))
        r13 = {r["sample"]: r for r in d13["rows"]}
        common = [r["sample"] for r in rows if r["sample"] in r13]
        same = all(abs(r["kl_no_mem"] - r13[r["sample"]]["kl_no_mem"]) < 1e-6 for r in rows if r["sample"] in r13)
        print(f"\n### pairing with S13 (slot-aware write) on {len(common)} shared samples; identical kl_no_mem: {same}")
        for j in js:
            for a14, a13, label in (("FL_all", "F_all", "all-lower-layers repair: local write vs slot-aware"),
                                    ("A_chunk", "A_chunk", "floor: local vs slot-aware"),
                                    ("A_on", "A_on", "deployed read: local vs slot-aware"),
                                    ("FL_empty", "F_empty", "no-cache control")):
                diffs = [r[f"{a14}_j{j}_frac"] - r13[r["sample"]][f"{a13}_j{j}_frac"] for r in rows if r["sample"] in r13]
                if not diffs:
                    continue
                ci = boot_ci(diffs)
                print(f"  j={j:2d} {label:52s} {st.mean([r[f'{a14}_j{j}_frac'] for r in rows]):.3f} vs "
                      f"{st.mean([r13[s][f'{a13}_j{j}_frac'] for s in common]):.3f}  diff {st.mean(diffs):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        diffs = [r["C_bos_frac"] - r13[r["sample"]]["C_frac"] for r in rows if r["sample"] in r13]
        ci = boot_ci(diffs)
        print(f"  C_bos (query BOS, chunk cache) vs S13 C (merged sink entry, sinkless query): "
              f"{st.mean(col('C_bos_frac')):.3f} vs {st.mean([r13[s]['C_frac'] for s in common]):.3f}  diff {st.mean(diffs):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        for s in ("2", "4", "14"):
            if col(f"KVb{s}_frac") and f"KV{s}_frac" in r13[common[0]]:
                print(f"  KVb{s} vs S13 KV{s}: {st.mean(col(f'KVb{s}_frac')):.3f} vs {st.mean([r13[x][f'KV{s}_frac'] for x in common]):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
