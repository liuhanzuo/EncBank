"""Does "depth tolerance is set by the task, not by j" replicate on a second checkpoint?

On Qwen3-8B (E55-E60) the single-needle cells sit at the j=0 replay bound at every split
tested ON THEM, 0.33-0.83 L -- j=6 = 0.17 L was a multikey-only run and j=30 = 0.83 L is
the deepest measured -- while the multi-key cells are separably below it from 0.42 L on. That is
a strong claim resting on one checkpoint. S35 runs Qwen3-1.7B (L=28) at the SAME depth
fractions:

    8B j=12 = 0.33 L   ->  1.7B j=9  = 0.32 L   (the 1.7B operating point, from S19)
    8B j=18 = 0.50 L   ->  1.7B j=14 = 0.50 L
    8B j=24 = 0.67 L   ->  1.7B j=19 = 0.68 L

Matched by fraction and not by index, the convention S27 used for the LM protocol, because the
two checkpoints have different L.

Only the two NIAH tasks: within that family the retrieval, the haystack and the scoring are
identical and only the needle count differs, so single-vs-multikey is the one clean contrast
for a difficulty claim. variable_tracking is excluded because its recall is five-way partial
credit against NIAH's binary (COMem/eval/ruler.py:250 divides by len(refs)), so it is not on a
common scale.

Residual = fix_all - j0, paired within a cell on the 50 shared samples, house bootstrap
B=4000, random.Random(0) reseeded per interval. j0 is measured in every run, never borrowed.

WHAT THIS CANNOT SHOW. The 1.7B starts weaker: at its operating point the multi-key repaired
arm is already 58.0 against a bound of 98.0 (E21), so that arm has less room to fall and a
null there is weak evidence. The informative cells are the single-needle ones. Cross-checkpoint
differences are two cell means with no interval and are never paired.
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


L17, L8 = 28, 36
GRID17 = {9: "s19_ruler_17b_niah.json",
          14: "s35_ruler_17b_j14_niah.json",
          19: "s35_ruler_17b_j19_niah.json"}
# the 8B column, for the side-by-side (cell means only, never differenced across checkpoints)
GRID8 = {12: {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json"},
         18: {"niah_multikey_1": "s32_ruler_8b_j18_mk.json",
              "niah_single_2": "s33_ruler_8b_j18_single.json"},
         24: {"niah_multikey_1": "s34_ruler_8b_j24_mk.json",
              "niah_single_2": "s34_ruler_8b_j24_single.json"}}
ARMS = ("pub", "pub_sink", "fix_all", "j0")
TASKS = ("niah_single_2", "niah_multikey_1")


def cell(r, task, L):
    if r is None:
        return None
    keys = sorted(k for k in r if k[0] == task and k[1] == L)
    if not keys:
        return None
    m = {a: [r[k][f"{a}_recall"] for k in keys] for a in ARMS if f"{a}_recall" in r[keys[0]]}
    if "fix_all" not in m or "j0" not in m:
        return None
    d = [x - y for x, y in zip(m["fix_all"], m["j0"])]
    lo, hi = boot(d)
    return (100 * st.mean(d), 100 * lo, 100 * hi, 100 * st.mean(m["fix_all"]),
            100 * st.mean(m["j0"]), sum(x > 0 for x in d), sum(x < 0 for x in d),
            sum(x == 0 for x in d), {a: 100 * st.mean(m[a]) for a in m})


print("Qwen3-1.7B (L=28), n=50 per cell, four arms, paired within each cell.")
print("Repaired-arm storage on the 1.7B = 4 + 4j KB/token: j=9 -> 40, j=14 -> 60, j=19 -> 80.\n")
r17 = {}
for task in TASKS:
    print(f"--- {task}")
    print(f"    {'j':>3s} {'frac':>5s} {'KB':>4s} {'len':>4s} | {'pub':>6s} {'p_sink':>7s} "
          f"{'fix_all':>8s} {'j0':>6s} |  fix_all - j0")
    for j in sorted(GRID17):
        r = rows(GRID17[j])
        for Ln in ("16k", "32k"):
            c = cell(r, task, Ln)
            if c is None:
                continue
            excl = "EXCLUDES 0" if (c[1] > 0 or c[2] < 0) else "spans 0"
            print(f"    {j:3d} {j/L17:5.2f} {4+4*j:4d} {Ln:>4s} | " +
                  " ".join(f"{c[8][a]:{w}.1f}" for a, w in
                           (("pub", 6), ("pub_sink", 7), ("fix_all", 8), ("j0", 6))) +
                  f" |  {c[0]:+6.1f} [{c[1]:+6.1f},{c[2]:+6.1f}] ({c[5]}/{c[6]}/{c[7]}) {excl}")
            r17[(task, Ln, j)] = c
    print()

print("=" * 82)
print("THE REPLICATION: 8B vs 1.7B at matched depth FRACTIONS.")
print("(cell means side by side; cross-checkpoint values are NOT paired and carry no interval)\n")
PAIRS = [(12, 9), (18, 14), (24, 19)]
for task in TASKS:
    print(f"  {task}")
    for Ln in ("16k", "32k"):
        line8, line17 = [], []
        for j8, j17 in PAIRS:
            src = GRID8[j8]
            f8 = src[Ln] if j8 == 12 else src[task]
            c8 = cell(rows(f8), task, Ln)
            c17 = r17.get((task, Ln, j17))
            line8.append(f"{j8/L8:.2f}L:{c8[0]:+.1f}{'*' if c8 and (c8[1]>0 or c8[2]<0) else ''}"
                         if c8 else f"{j8/L8:.2f}L:--")
            line17.append(f"{j17/L17:.2f}L:{c17[0]:+.1f}{'*' if c17 and (c17[1]>0 or c17[2]<0) else ''}"
                          if c17 else f"{j17/L17:.2f}L:--")
        print(f"    {Ln:>4s}  8B   residual: " + "   ".join(line8))
        print(f"    {Ln:>4s}  1.7B residual: " + "   ".join(line17))
    print("        (* = interval excludes zero)")
    print()

print("VERDICT INPUTS -- on the 1.7B, which cells are separably below their own bound?")
for j in sorted(GRID17):
    bad = sorted(f"{t}/{Ln}" for (t, Ln, jj), c in r17.items()
                 if jj == j and (c[1] > 0 or c[2] < 0))
    tot = sum(1 for (t, Ln, jj) in r17 if jj == j)
    print(f"  j={j:2d} ({j/L17:.2f} L): {len(bad)} of {tot}"
          + ("  -- " + ", ".join(bad) if bad else "  -- none"))

print("\nRepaired-arm recall across depth, 1.7B:")
for task in TASKS:
    for Ln in ("16k", "32k"):
        xs = [(j, r17[(task, Ln, j)][3]) for j in sorted(GRID17) if (task, Ln, j) in r17]
        if xs:
            print(f"  {task:18s} {Ln:>4s}: " + "  ".join(f"j={j}:{v:.1f}" for j, v in xs))
