"""At j=18, does the repair fail on every task, or only on the hard ones?

E56 established, on niah_multikey_1 only, that the repaired arm's residual against the j=0
replay bound is indistinguishable from zero at j=6 and j=12 and separably negative at j=15
and j=18. Its own scope note says the obvious alternative was untested: the break might not
be a property of the DEPTH at all but of the TASK, with the easy cells surviving depths the
hard cells do not. S33 runs the other two tasks at j=18.

The discriminating comparison is the residual `fix_all - j0` at j=18, per task, against the
same quantity at j=12 on the same samples:
  - if all three tasks fall away at j=18, the break is a depth effect and E56 stands as a
    general statement about j;
  - if the single-needle cells hold at j=18 while multikey and vt collapse, the break is
    (at least partly) task difficulty, and E56 must be scoped to the harder tasks.

All contrasts paired within a (task, length, depth) cell on the 50 shared samples, house
bootstrap B=4000, random.Random(0) reseeded per interval. j0 is the bound and is measured in
every run rather than borrowed (exp/s33_j0_check.py verified it does not move with j on the
multikey cells: 50/50 identical at four depths).
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


# (task, length) -> {depth: file}
SRC = {
    ("niah_multikey_1", "16k"): {12: "s15_ruler_j12_16k.json", 18: "s32_ruler_8b_j18_mk.json"},
    ("niah_multikey_1", "32k"): {12: "s15_ruler_j12_32k.json", 18: "s32_ruler_8b_j18_mk.json"},
    ("niah_single_2", "16k"): {12: "s15_ruler_j12_16k.json", 18: "s33_ruler_8b_j18_single.json"},
    ("niah_single_2", "32k"): {12: "s15_ruler_j12_32k.json", 18: "s33_ruler_8b_j18_single.json"},
    ("variable_tracking", "16k"): {12: "s15c_ruler_vt_16k.json", 18: "s33_ruler_8b_j18_vt.json"},
    ("variable_tracking", "32k"): {12: "s15c_ruler_vt_32k.json", 18: "s33_ruler_8b_j18_vt.json"},
}

print("Qwen3-8B, all four arms, n=50 per cell. Residual = fix_all - j0 (negative = below the")
print("full-replay bound). Storage: j=12 -> 56 KB/token, j=18 -> 80 KB/token.\n")
print(f"{'task':18s} {'len':>4s} {'j':>3s} | {'pub':>6s} {'p_sink':>7s} {'fix_all':>8s} "
      f"{'j0':>6s} |  fix_all - j0")
resid = {}
for (task, L), byj in SRC.items():
    for j in (12, 18):
        r = rows(byj[j])
        if r is None:
            print(f"  {task}/{L} j={j}: missing")
            continue
        keys = sorted(k for k in r if k[0] == task and k[1] == L)
        if not keys:
            continue
        m = {a: [r[k][f"{a}_recall"] for k in keys]
             for a in ("pub", "pub_sink", "fix_all", "j0")}
        d = [x - y for x, y in zip(m["fix_all"], m["j0"])]
        lo, hi = boot(d)
        excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"{task:18s} {L:>4s} {j:3d} | " +
              " ".join(f"{100*st.mean(m[a]):{w}.1f}" for a, w in
                       (("pub", 6), ("pub_sink", 7), ("fix_all", 8), ("j0", 6))) +
              f" |  {100*st.mean(d):+6.1f} [{100*lo:+6.1f},{100*hi:+6.1f}] "
              f"({sum(x>0 for x in d)}/{sum(x<0 for x in d)}/{sum(x==0 for x in d)}) {excl}")
        resid[(task, L, j)] = (100 * st.mean(d), 100 * lo, 100 * hi,
                               100 * st.mean(m["fix_all"]), 100 * st.mean(m["j0"]))
    print()

print("=" * 78)
print("THE DISCRIMINATING READ: how much does the residual move from j=12 to j=18?\n")
print(f"{'task':18s} {'len':>4s} | {'resid j=12':>22s} | {'resid j=18':>22s} | change")
for (task, L) in SRC:
    a, b = resid.get((task, L, 12)), resid.get((task, L, 18))
    if not a or not b:
        continue
    fa = f"{a[0]:+.1f} [{a[1]:+.0f},{a[2]:+.0f}]"
    fb = f"{b[0]:+.1f} [{b[1]:+.0f},{b[2]:+.0f}]"
    print(f"{task:18s} {L:>4s} | {fa:>22s} | {fb:>22s} | {b[0]-a[0]:+6.1f} pts")

print("\nRepaired-arm recall at j=18 against its own bound, per task:")
for (task, L) in SRC:
    b = resid.get((task, L, 18))
    if b:
        print(f"  {task:18s} {L:>4s}: fix_all {b[3]:5.1f} against j0 {b[4]:5.1f}")

print("\nVerdict inputs -- cells whose j=18 residual EXCLUDES zero:")
bad = [(t, L) for (t, L, j), v in resid.items() if j == 18 and (v[1] > 0 or v[2] < 0)]
ok = [(t, L) for (t, L, j), v in resid.items() if j == 18 and not (v[1] > 0 or v[2] < 0)]
print(f"  separably below the bound at j=18 ({len(bad)}): " + ", ".join(f"{t}/{L}" for t, L in sorted(bad)))
print(f"  not separable at j=18       ({len(ok)}): " + ", ".join(f"{t}/{L}" for t, L in sorted(ok)))
