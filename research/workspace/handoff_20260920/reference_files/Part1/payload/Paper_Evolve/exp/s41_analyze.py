"""Is the FULL lower band load-bearing at depth, or would a sparse band do?

The paper recommends caching the whole lower band and refuses to recommend any layer-selection
method, because E15b showed our own continuation-KL objective misranks sparse sets against the
task. But every sparse-set measurement in the paper is at j=12. The deep end is where a sparse
band would be worth most: at j=33 the full band stores 8+4*33 = 140 KB/token, barely under the
144 KB full-depth K/V reference, while six visible layers would store 8+4*6 = 32 KB -- less
than the operating point's 56 -- at the same 3/36 upper-band recompute.

So this is the cell the recommendation talks about and had never looked at.

THE ARM. fix_S over an EVENLY SPACED set {2,8,14,20,26,32}: stride 6 inside [0,33). Evenly
spaced on purpose -- it is a fixed, task- and corpus-independent rule, not a selection, which
is the only kind E15b permits recommending. Starting at 2 because S19 established layers 0
and 1 are poison pills; including them would confound "sparse" with a known-bad choice.

Contrasts, all paired within a cell on the 50 shared samples (deterministic in
task/length/seed/PYTHONHASHSEED=0), house bootstrap B=4000, random.Random(0) reseeded per
interval:
    fix_S - fix_all   does the sparse band cost anything against the full one?
    fix_S - pub       does it buy anything over the published read at all?
    fix_all - j0      the control: must reproduce S38's j=33 numbers exactly, since the
                      samples and the arm are identical.

WHAT THIS CANNOT SHOW. One set at one depth on one checkpoint. A negative result here says
that THIS evenly spaced set fails at j=33; it does not prove no sparse set can work, and per
E15b we have no trustworthy way to search for one. It also cannot separate "too few layers"
from "wrong layers" -- |S| and placement move together between fix_S and fix_all.
"""
import json, random, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
SET = [2, 8, 14, 20, 26, 32]
J = 33


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


d = json.loads((RES / "s41_ruler_8b_j33_sparse.json").read_text(encoding="utf-8"))
rows = d["rows"]
ARMS = ("pub", "fix_S", "fix_all", "j0")

print(f"Qwen3-8B, j={J} (0.92 L). fix_S = {SET}, evenly spaced, |S|=6.")
print(f"store: fix_S 8+4*6 = {8+4*6} KB/tok, fix_all 8+4*{J} = {8+4*J} KB/tok, "
      f"full-depth K/V reference 144 KB.")
print(f"upper band recomputed at read: {(36-J)}/36 for both.\n")
print(f"{'task':18s} {'len':>4s} n | " + " ".join(f"{a:>8s}" for a in ARMS))
store = {}
for task in ("niah_single_2", "niah_multikey_1"):
    for L in ("16k", "32k"):
        sel = [r for r in rows if r["task"] == task and r["length"] == L]
        if len(sel) < 50:
            print(f"{task:18s} {L:>4s} n={len(sel)} INCOMPLETE -- not quoted")
            continue
        m = {a: [r[f"{a}_recall"] for r in sel] for a in ARMS}
        store[(task, L)] = m
        print(f"{task:18s} {L:>4s} {len(sel)} | "
              + " ".join(f"{100*st.mean(m[a]):8.1f}" for a in ARMS))
print()

print("PAIRED CONTRASTS (difference of means, bootstrap 95% CI, better/worse/tie)\n")
for (task, L), m in store.items():
    print(f"  {task}/{L}")
    for a, b in (("fix_S", "fix_all"), ("fix_S", "pub"), ("fix_all", "j0")):
        diff = [x - y for x, y in zip(m[a], m[b])]
        lo, hi = boot(diff)
        excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"    {a:8s} - {b:8s} = {100*st.mean(diff):+6.1f} "
              f"[{100*lo:+6.1f},{100*hi:+6.1f}] "
              f"({sum(x>0 for x in diff)}/{sum(x<0 for x in diff)}/{sum(x==0 for x in diff)}) {excl}")
    ident = sum(1 for x, y in zip(m["fix_S"], m["pub"]) if x == y)
    print(f"    fix_S identical to pub on {ident} of {len(m['pub'])} samples")
    print()

print("=" * 74)
print("THE VERDICT INPUTS")
for (task, L), m in store.items():
    d1 = [x - y for x, y in zip(m["fix_S"], m["fix_all"])]
    d2 = [x - y for x, y in zip(m["fix_S"], m["pub"])]
    lo1, hi1 = boot(d1)
    lo2, hi2 = boot(d2)
    v1 = "loses to the full band" if hi1 < 0 else "not separable from the full band"
    v2 = "beats pub" if lo2 > 0 else "NOT separable from pub"
    print(f"  {task:18s} {L:>4s}: sparse {v1}; sparse {v2}")
