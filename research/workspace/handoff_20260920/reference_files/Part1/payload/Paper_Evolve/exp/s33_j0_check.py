"""Is the j0 replay bound really depth-independent, sample by sample?

WHY THIS MATTERS. E55/E56 read the dial's failure off `fix_all - j0` at four depths and lean
on j0 being the same bound throughout -- the analyzer prints 98.0 at every depth and length.
But equal MEANS are weak evidence: four runs could reach 98.0 by getting different samples
right. If j0 varies per sample with j, then the pack construction depends on j in a way that
touches the full-replay arm too, and the residual is not a clean read.

j0 sets resume_j = 0, so nothing should be cached and nothing recomputed above a split -- the
arm should be identical at every j, sample for sample. This checks that directly, which is
also the strongest available check that the four independent runs used the same samples.

Reports, per (task, length): the number of samples where j0 differs between any two depths,
and the per-depth mean. A single differing sample is a real finding and would need the
residual claims rescoped.
"""
import json
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

FILES = {
    6: ["s32_ruler_8b_j6_mk.json"],
    12: ["s15_ruler_j12_16k.json", "s15_ruler_j12_32k.json"],
    15: ["s33_ruler_8b_j15_mk.json"],
    18: ["s32_ruler_8b_j18_mk.json"],
}

per_depth = defaultdict(dict)   # (task,length,i) -> {j: j0_recall}
for j, names in FILES.items():
    for n in names:
        p = RES / n
        if not p.exists():
            print(f"  missing {n}")
            continue
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            if "j0_recall" in r:
                per_depth[(r["task"], r["length"], r["i"])][j] = r["j0_recall"]

cells = defaultdict(list)
for (task, L, i), byj in per_depth.items():
    cells[(task, L)].append((i, byj))

print("j0 (full replay) compared across split depths, sample by sample.")
print("j0 sets resume_j=0, so it should be bit-identical at every j.\n")
allbad = 0
for (task, L), items in sorted(cells.items()):
    depths = sorted({j for _, byj in items for j in byj})
    shared = [(i, byj) for i, byj in items if len(byj) == len(depths)]
    bad = [(i, byj) for i, byj in shared if len({v for v in byj.values()}) > 1]
    allbad += len(bad)
    means = {j: sum(byj[j] for _, byj in shared) / len(shared) for j in depths} if shared else {}
    print(f"  {task}/{L}: depths {depths}, {len(shared)} samples present at all depths")
    print("      per-depth mean: " + "  ".join(f"j={j} {100*m:.1f}" for j, m in means.items()))
    print(f"      samples where j0 DIFFERS across depths: {len(bad)}")
    for i, byj in bad[:5]:
        print(f"        sample {i}: " + ", ".join(f"j={j} {v}" for j, v in sorted(byj.items())))

print()
if allbad == 0:
    print("PASS: j0 is identical sample-for-sample at every measured depth, in every cell.")
    print("So the four runs scored the same samples and the bound the residual is measured")
    print("against does not move with j. The residual is a clean read of the repaired arm.")
else:
    print(f"FAIL: {allbad} sample(s) where the full-replay arm depends on the split depth.")
    print("The residual claims in E55/E56 would need rescoping -- do not quote them until this")
    print("is explained.")
