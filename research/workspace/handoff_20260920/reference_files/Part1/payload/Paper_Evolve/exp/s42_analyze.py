"""Does the sparse arm scale with |S| at j=33, or is it inert there?

S41 ran an evenly spaced |S|=6 set at j=33 and found `fix_S` bit-identical to `pub` on 50/50
samples in every cell -- the E54 signature. A four-link code audit (capture over [0,j), cache
over [0,j), a no-model unit probe of `_bottom_mask`, and per-layer mask application inside
`_bottom_forward`) found the path sound, which leaves two readings:
  (a) real -- a sparse band recovers nothing at that depth and every arm but `fix_all` is at
      the floor, where equality is uninformative;
  (b) artefact -- something downstream still drops the set at large j.

S42 is the empirical discriminator: identical depth and set-construction rule, but |S|=16 at
stride 2 instead of |S|=6 at stride 6. If the arm scales with |S| at all, 16 of 33 visible
layers should move it off `pub`. If it is again pinned to `pub`, the arm carries no signal at
this depth whatever the code says.

Storage: |S|=6 -> 8+4*6 = 32 KB/token, |S|=16 -> 72, `fix_all` (|S|=33) -> 140, full-depth
K/V reference 144. The upper band recomputed at read is 3/36 for all of them.

Paired within a cell on the 50 shared samples, house bootstrap B=4000, random.Random(0)
reseeded per interval. The two runs share samples, so |S|=6 and |S|=16 are comparable
sample-by-sample as well.
"""
import json, random, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))["rows"]


sparse = load("s41_ruler_8b_j33_sparse.json")   # |S|=6
dense = load("s42_ruler_8b_j33_dense.json")     # |S|=16
CELLS = [("niah_single_2", "16k"), ("niah_single_2", "32k"),
         ("niah_multikey_1", "16k"), ("niah_multikey_1", "32k")]

print("Qwen3-8B, j=33 (0.92 L). Evenly spaced sets, same rule, two densities.")
print("store: |S|=6 -> 32 KB/tok, |S|=16 -> 72, fix_all (|S|=33) -> 140; reference 144.\n")
print(f"{'task':17s} {'len':>4s} | {'pub':>7s} {'S=6':>7s} {'S=16':>7s} {'fix_all':>8s} {'j0':>7s}")
keep = {}
for task, L in CELLS:
    a = [r for r in sparse if r["task"] == task and r["length"] == L]
    b = [r for r in dense if r["task"] == task and r["length"] == L]
    if len(a) < 50 or len(b) < 50:
        print(f"{task:17s} {L:>4s} | INCOMPLETE (n={len(a)}/{len(b)}) -- not quoted")
        continue
    m = dict(pub=[r["pub_recall"] for r in a], s6=[r["fix_S_recall"] for r in a],
             s16=[r["fix_S_recall"] for r in b], fa=[r["fix_all_recall"] for r in a],
             j0=[r["j0_recall"] for r in a])
    keep[(task, L)] = m
    print(f"{task:17s} {L:>4s} | " + " ".join(f"{100*st.mean(m[k]):7.1f}"
                                              for k in ("pub", "s6", "s16", "fa", "j0")))
print()

print("DOES DENSITY MOVE IT? (paired, difference of means, 95% CI, better/worse/tie)\n")
for (task, L), m in keep.items():
    print(f"  {task}/{L}")
    for lab, a, b in (("S=16 - S=6", "s16", "s6"), ("S=16 - pub", "s16", "pub"),
                      ("S=16 - fix_all", "s16", "fa")):
        d = [x - y for x, y in zip(m[a], m[b])]
        lo, hi = boot(d)
        excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"    {lab:14s} = {100*st.mean(d):+6.1f} [{100*lo:+6.1f},{100*hi:+6.1f}] "
              f"({sum(x>0 for x in d)}/{sum(x<0 for x in d)}/{sum(x==0 for x in d)}) {excl}")
    ident6 = sum(1 for x, y in zip(m["s6"], m["pub"]) if x == y)
    ident16 = sum(1 for x, y in zip(m["s16"], m["pub"]) if x == y)
    print(f"    identical to pub: S=6 on {ident6}/50, S=16 on {ident16}/50")
    print()

print("=" * 72)
print("VERDICT INPUTS")
pinned = 0
for (task, L), m in keep.items():
    d = [x - y for x, y in zip(m["s16"], m["pub"])]
    lo, hi = boot(d)
    moved = (lo > 0 or hi < 0)
    pinned += (not moved)
    print(f"  {task:17s} {L:>4s}: S=16 vs pub -> "
          + ("MOVED off pub" if moved else "still pinned to pub"))
print()
if keep and pinned == len(keep):
    print("S=16 is pinned to pub in every cell. Tripling the visible layers changes nothing,")
    print("so the arm carries no signal at this depth. With the code audit finding the path")
    print("sound, the reading is that every arm but fix_all sits at the FLOOR here -- which")
    print("makes the sparse-vs-pub comparison uninformative, not a result about sparsity.")
else:
    print("S=16 moves off pub somewhere, so the arm does scale with |S| at this depth and")
    print("the |S|=6 result is a real statement about how many layers depth needs.")
