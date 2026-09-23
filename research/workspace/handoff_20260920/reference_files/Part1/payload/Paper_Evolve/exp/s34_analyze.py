"""Does the easy task have a depth ceiling at all, or does depth tolerance really track the task?

E57 established that at j=18 the single-needle cells are still at the j=0 replay bound while
the multi-key and chained cells are separably below it. That leaves one reading unexcluded:
maybe every task breaks and the easy one's break is simply deeper. S34 runs j=24 -- the
deepest point of the LM sweep grid, 0.67 L, 104 KB/token, nearly twice the operating point's
56 and within a factor 1.4 of the 144 KB full-depth KV reference.

Two outcomes, both worth having:
  - single still ties the bound at j=24: depth tolerance really is set by the task, and at
    104 KB the dial has no quality headroom left to buy on that task;
  - single finally falls away: every task has a ceiling and the easy one's is deeper, which is
    the more conservative reading and lets the paper keep one ordering statement.

Everything is the residual `fix_all - j0` paired within a (task, length, depth) cell on the 50
shared samples, house bootstrap B=4000, random.Random(0) reseeded per interval. j0 is measured
in every run, never borrowed.

SCALE CAVEAT, carried from E57 and checkable at COMem/eval/ruler.py:250: recall is
sum(1 for r in refs if r in pred)/len(refs), and variable_tracking has five refs per sample
against one for the NIAH tasks. vt recall is five-way partial credit and NIAH recall is
binary, so recall points are NOT the same unit across the two families. Within-cell residuals
are fine; no ordering across the families is licensed by this script.
"""
import json, random, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def rows(n):
    p = RES / n
    return None if not p.exists() else {
        (r["task"], r["length"], r["i"]): r
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


# task -> depth -> file (one file may hold both lengths)
GRID = {
    "niah_multikey_1": {6: "s32_ruler_8b_j6_mk.json",
                        12: {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json"},
                        15: "s33_ruler_8b_j15_mk.json",
                        18: "s32_ruler_8b_j18_mk.json",
                        24: "s34_ruler_8b_j24_mk.json",
                        30: "s36_ruler_8b_j30_mk.json"},
    "niah_single_2": {12: {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json"},
                      18: "s33_ruler_8b_j18_single.json",
                      24: "s34_ruler_8b_j24_single.json",
                      30: "s36_ruler_8b_j30_single.json"},
    "variable_tracking": {12: {"16k": "s15c_ruler_vt_16k.json", "32k": "s15c_ruler_vt_32k.json"},
                          18: "s33_ruler_8b_j18_vt.json"},
}
ARMS = ("pub", "pub_sink", "fix_all", "j0")

print("Qwen3-8B, n=50 per cell, four arms, paired within each cell.")
print("Repaired-arm storage = 8 + 4j KB/token: j=6->32, 12->56, 15->68, 18->80,\n      24->104, 30->128 (the full-depth KV reference is 144).\n")
out = {}
for task, byj in GRID.items():
    print(f"--- {task}")
    print(f"    {'j':>3s} {'KB':>4s} {'len':>4s} | {'pub':>6s} {'p_sink':>7s} {'fix_all':>8s} "
          f"{'j0':>6s} |  fix_all - j0")
    for j in sorted(byj):
        src = byj[j]
        for L in ("16k", "32k"):
            r = rows(src[L] if isinstance(src, dict) else src)
            if r is None:
                continue
            keys = sorted(k for k in r if k[0] == task and k[1] == L)
            if not keys:
                continue
            m = {a: [r[k][f"{a}_recall"] for k in keys] for a in ARMS}
            d = [x - y for x, y in zip(m["fix_all"], m["j0"])]
            lo, hi = boot(d)
            excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"
            print(f"    {j:3d} {8+4*j:4d} {L:>4s} | " +
                  " ".join(f"{100*st.mean(m[a]):{w}.1f}" for a, w in
                           (("pub", 6), ("pub_sink", 7), ("fix_all", 8), ("j0", 6))) +
                  f" |  {100*st.mean(d):+6.1f} [{100*lo:+6.1f},{100*hi:+6.1f}] "
                  f"({sum(x>0 for x in d)}/{sum(x<0 for x in d)}/{sum(x==0 for x in d)}) {excl}")
            out[(task, L, j)] = (100 * st.mean(d), 100 * lo, 100 * hi,
                                 100 * st.mean(m["fix_all"]), 100 * st.mean(m["j0"]))
    print()

print("=" * 80)
print("THE QUESTION: how deep does the easy task hold? (0.67 L = 104 KB, 0.83 L = 128 KB)\n")
for task in ("niah_single_2", "niah_multikey_1"):
    for L in ("16k", "32k"):
        v = out.get((task, L, 30))
        if v:
            verdict = "STILL AT THE BOUND" if not (v[1] > 0 or v[2] < 0) else "separably below"
            print(f"  {task:18s} {L:>4s}  j=30: fix_all {v[3]:5.1f} vs j0 {v[4]:5.1f}   "
                  f"resid {v[0]:+6.1f} [{v[1]:+.0f},{v[2]:+.0f}]  -> {verdict}")

print("\nThe repaired arm across every depth measured, per task/length:")
for task in GRID:
    for L in ("16k", "32k"):
        xs = [(j, out[(task, L, j)]) for j in sorted(GRID[task]) if (task, L, j) in out]
        if xs:
            print(f"  {task:18s} {L:>4s}: " +
                  "  ".join(f"j={j}:{v[3]:.1f}" for j, v in xs))

print("\nCells separably below the bound, by depth:")
for j in (6, 12, 15, 18, 24, 30):
    bad = sorted(f"{t}/{L}" for (t, L, jj), v in out.items()
                 if jj == j and (v[1] > 0 or v[2] < 0))
    tot = sum(1 for (t, L, jj) in out if jj == j)
    print(f"  j={j:2d}: {len(bad)} of {tot}" + ("  -- " + ", ".join(bad) if bad else ""))
