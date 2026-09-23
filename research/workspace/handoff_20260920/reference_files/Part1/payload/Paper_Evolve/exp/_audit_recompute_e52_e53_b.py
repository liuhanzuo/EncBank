"""Part B: reverse-orientation bootstraps (fix_all - fix_none) and fragility of the counts."""
import json, random, statistics as st
from pathlib import Path

RES = Path("F:/Paper_Evolve/exp/results")
M = "niah_multikey_1"; S = "niah_single_2"; V = "variable_tracking"


def load(n):
    d = json.loads((RES / n).read_text(encoding="utf-8"))
    return {(r["task"], r["length"], r["i"]): r for r in d["rows"]}


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


SRC = {
    ("1.7B", M, "16k"): ("s30_ruler_17b_fixnone_niah.json", "s19_ruler_17b_niah.json"),
    ("1.7B", M, "32k"): ("s30_ruler_17b_fixnone_niah.json", "s19_ruler_17b_niah.json"),
    ("1.7B", S, "16k"): ("s30_ruler_17b_fixnone_niah.json", "s19_ruler_17b_niah.json"),
    ("1.7B", S, "32k"): ("s30_ruler_17b_fixnone_niah.json", "s19_ruler_17b_niah.json"),
    ("1.7B", V, "16k"): ("s30_ruler_17b_fixnone_vt.json", "s19_ruler_17b_vt.json"),
    ("1.7B", V, "32k"): ("s30_ruler_17b_fixnone_vt.json", "s19_ruler_17b_vt.json"),
    ("8B", M, "8k"): ("s31_ruler_8b_fixnone_8k.json", "s15c_ruler_j12_8k.json"),
    ("8B", S, "8k"): ("s31_ruler_8b_fixnone_8k.json", "s15c_ruler_j12_8k.json"),
    ("8B", M, "16k"): ("s15b_ruler_j12_16k.json", "s15_ruler_j12_16k.json"),
    ("8B", S, "16k"): ("s15b_ruler_j12_16k.json", "s15_ruler_j12_16k.json"),
    ("8B", M, "32k"): ("s15b_ruler_j12_32k.json", "s15_ruler_j12_32k.json"),
    ("8B", S, "32k"): ("s15b_ruler_j12_32k.json", "s15_ruler_j12_32k.json"),
    ("8B", M, "64k"): ("s31_ruler_8b_fixnone_64k.json", "s15c_ruler_j12_64k.json"),
    ("8B", S, "64k"): ("s31_ruler_8b_fixnone_64k.json", "s15c_ruler_j12_64k.json"),
    ("8B", V, "16k"): ("s31_ruler_8b_fixnone_vt.json", "s15c_ruler_vt_16k.json"),
    ("8B", V, "32k"): ("s31_ruler_8b_fixnone_vt.json", "s15c_ruler_vt_32k.json"),
}
cache = {}
def R(n):
    cache.setdefault(n, load(n)); return cache[n]

print("=== C2 CHECK: fix_all - fix_none, in BOTH orientations, every cell")
vals = {}
for (ck, task, L), (fnf, rff) in SRC.items():
    fn, rf = R(fnf), R(rff)
    keys = sorted(k for k in fn if k[0] == task and k[1] == L)
    c = [fn[k]["fix_none_recall"] for k in keys]
    fa = [rf[k]["fix_all_recall"] for k in keys]
    pub = [rf[k]["pub_recall"] for k in keys]
    dpos = [x - y for x, y in zip(fa, c)]          # fix_all - fix_none  (the claimed direction)
    dneg = [y - x for x, y in zip(fa, c)]          # fix_none - fix_all
    lo1, hi1 = boot(dpos); lo2, hi2 = boot(dneg)
    e1 = "EXCL0" if (lo1 > 0 or hi1 < 0) else "*** SPANS 0 ***"
    e2 = "EXCL0" if (lo2 > 0 or hi2 < 0) else "*** SPANS 0 ***"
    print(f"{ck:5s} {task:17s} {L:4s}  fix_all-fix_none = {100*st.mean(dpos):+6.2f} "
          f"[{100*lo1:+7.2f},{100*hi1:+7.2f}] {e1:16s} | reverse [{100*lo2:+7.2f},{100*hi2:+7.2f}] {e2}")
    vals[(ck, task, L)] = (c, fa, pub)

print("\n=== FRAGILITY of the cross-checkpoint lead counts (share = (fn-pub)/(fa-pub), cell means)")
sh = {k: 100 * (st.mean(c) - st.mean(p)) / (st.mean(f) - st.mean(p))
      for k, (c, f, p) in vals.items()}
both = sorted({(k[1], k[2]) for k in sh if k[0] == "8B"} & {(k[1], k[2]) for k in sh if k[0] == "1.7B"})
for task, L in both:
    a, b = sh[("8B", task, L)], sh[("1.7B", task, L)]
    # how many single-sample flips of the 1.7B fix_none column reverse the ordering?
    c17, f17, p17 = vals[("1.7B", task, L)]
    n = len(c17); den = st.mean(f17) - st.mean(p17); base = st.mean(c17) - st.mean(p17)
    step = 100 * (1.0 / n) / den   # share change from flipping one binary sample by 1.0
    gap = abs(a - b)
    print(f"  {task:17s} {L:4s}: 8B={a:6.2f}  1.7B={b:6.2f}  lead={'8B' if a>b else '1.7B':4s} "
          f"gap={gap:5.2f}pp; one 1.7B sample flip moves the 1.7B share by {step:5.2f}pp "
          f"-> flips needed to reverse: {gap/step:.2f}")

print("\n=== LENGTH-GRID MATCHING: 8B ranges restricted to the lengths the 1.7B has (16k/32k)")
for name, tasks in [("single", {S}), ("multikey+chained", {M, V})]:
    for ck in ("1.7B", "8B"):
        allv = [sh[k] for k in sh if k[0] == ck and k[1] in tasks]
        m16 = [sh[k] for k in sh if k[0] == ck and k[1] in tasks and k[2] in ("16k", "32k")]
        print(f"  {name:18s} {ck:5s}: all lengths [{min(allv):6.2f},{max(allv):6.2f}] (n={len(allv)})"
              f"   16k/32k only [{min(m16):6.2f},{max(m16):6.2f}] (n={len(m16)})")
print("\n=== per-task overlap on the MATCHED 16k/32k grid")
for task in (S, M, V):
    a = [sh[k] for k in sh if k[0] == "8B" and k[1] == task and k[2] in ("16k", "32k")]
    b = [sh[k] for k in sh if k[0] == "1.7B" and k[1] == task and k[2] in ("16k", "32k")]
    ov = min(max(a), max(b)) >= max(min(a), min(b))
    print(f"  {task:17s}: 8B[{min(a):6.2f},{max(a):6.2f}] 1.7B[{min(b):6.2f},{max(b):6.2f}] overlap={ov}")
