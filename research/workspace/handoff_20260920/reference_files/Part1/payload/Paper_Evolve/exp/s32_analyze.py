"""Does the storage-versus-recompute dial hold at the TASK level, or only on the LM metric?

C3 says j becomes a storage-recompute dial at near-reference quality, and sec:exp-dial says
"at every setting we tested it buys more fidelity per byte". Every one of those settings is
the continuation-KL LM protocol: checked 2026-09-07, every 8B task-grid file is j=12. S32
adds j=6 and j=18 on niah_multikey_1 at 16k/32k, all four arms, so the task grid finally has
three depths.

Contrasts, all paired within a depth on the 50 shared samples of the cell (deterministic in
task/length/seed/PYTHONHASHSEED=0), house bootstrap B=4000, random.Random(0) reseeded per
interval:
    fix_all - j0   the residual against the full-replay bound -- the quantity that decides
                   whether the repair still works at that depth
    fix_all - pub  the size of the repair
    pub_sink - pub the write-time sink on its own

The j=12 column is read from the existing s15/s15b files, so the three depths are compared on
the same samples and the same driver, not against a remembered number.
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


SRC = {6: ("s32_ruler_8b_j6_mk.json", "s32_ruler_8b_j6_mk.json"),
       12: ("s15_ruler_j12_16k.json", "s15_ruler_j12_32k.json"),
       15: ("s33_ruler_8b_j15_mk.json", "s33_ruler_8b_j15_mk.json"),
       18: ("s32_ruler_8b_j18_mk.json", "s32_ruler_8b_j18_mk.json")}
DEPTHS = (6, 12, 15, 18)
ARMS = ("pub", "pub_sink", "fix_all", "j0")
TASK = "niah_multikey_1"

print("Qwen3-8B, niah_multikey_1, all four arms, n=50 per cell.")
print("storage of the repaired arm = 8 + 4j KB/token: j=6 -> 32, j=12 -> 56, j=18 -> 80\n")
print(f"{'depth':>6s} {'len':>4s} {'KB':>4s} | " + " ".join(f"{a:>9s}" for a in ARMS)
      + " |  fix_all - j0 (paired)")
store = {}
for j in DEPTHS:
    for L in ("16k", "32k"):
        r = rows(SRC[j][0 if L == "16k" else 1])
        if r is None:
            print(f"{j:6d} {L:>4s}: missing")
            continue
        keys = sorted(k for k in r if k[0] == TASK and k[1] == L)
        if not keys:
            print(f"{j:6d} {L:>4s}: no rows")
            continue
        m = {a: [r[k][f"{a}_recall"] for k in keys] for a in ARMS if f"{a}_recall" in r[keys[0]]}
        d = [x - y for x, y in zip(m["fix_all"], m["j0"])]
        lo, hi = boot(d)
        excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"{j:6d} {L:>4s} {8+4*j:4d} | "
              + " ".join(f"{100*st.mean(m[a]):9.1f}" for a in ARMS)
              + f" |  {100*st.mean(d):+6.1f} [{100*lo:+6.1f},{100*hi:+6.1f}] "
                f"({sum(x>0 for x in d)}/{sum(x<0 for x in d)}/{sum(x==0 for x in d)}) {excl}")
        store[(j, L)] = {a: st.mean(m[a]) for a in m}
        store[(j, L)]["resid"] = (st.mean(d), lo, hi)
    print()

print("The residual against the replay bound, by depth (this is what the dial claim needs "
      "to stay small):")
for L in ("16k", "32k"):
    xs = [(j, store[(j, L)]["resid"]) for j in DEPTHS if (j, L) in store]
    print(f"  {L}: " + "   ".join(f"j={j} {100*v[0]:+.1f} [{100*v[1]:+.0f},{100*v[2]:+.0f}]"
                                  for j, v in xs))

print("\nRepaired-arm recall by depth (the dial's own quality axis):")
for L in ("16k", "32k"):
    xs = [(j, store[(j, L)]["fix_all"]) for j in DEPTHS if (j, L) in store]
    print(f"  {L}: " + "   ".join(f"j={j} ({8+4*j} KB) {100*v:.1f}" for j, v in xs))

print("\nPublished arm by depth, for reference (the base method says the split is usable "
      "zero-shot only to j<=9):")
for L in ("16k", "32k"):
    xs = [(j, store[(j, L)]["pub"]) for j in DEPTHS if (j, L) in store]
    print(f"  {L}: " + "   ".join(f"j={j} {100*v:.1f}" for j, v in xs))
