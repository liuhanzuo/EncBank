"""Recompute the numbers the abstract and C4 lead with, straight from the result rows.

Everything here is stated in the paper as a headline, so it should be checkable without
trusting any analysis script. Reads only the raw `rows` arrays.
  abstract : "multikey recall at 16k/32k rises from 2 to 92/98 on Qwen3-8B"
             "matching full-depth replay sample for sample on the needle tasks"
  C4       : "+90 [82,98] / +96 [90,100] paired, 0 of 50 samples worse"
             "matching j=0 replay on 50/50 samples of both needle tasks at 32k and 64k"
             "a control that keeps the sink entry and position shift but hides the chunks (16/12)"
"""
import json, random, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

def rows(name):
    p = RES / name
    if not p.exists():
        return None
    return {(r["task"], r["length"], r["i"]): r
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}

def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]

SRC = {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json",
       "8k": "s15c_ruler_j12_8k.json", "64k": "s15c_ruler_j12_64k.json"}
CTRL = {"16k": "s15b_ruler_j12_16k.json", "32k": "s15b_ruler_j12_32k.json"}

print("Qwen3-8B, j=12, zero-shot. Cell means (recall x100), n = number of shared samples.\n")
print(f"{'task':18s} {'len':4s} {'n':>3s}  {'pub':>6s} {'pub_sink':>9s} {'fix_all':>8s} {'j0':>6s}")
cells = {}
for L, f in SRC.items():
    r = rows(f)
    if r is None:
        print(f"  ({f} absent)"); continue
    for task in sorted({k[0] for k in r}):
        keys = sorted(k for k in r if k[0] == task and k[1] == L)
        if not keys: continue
        m = {}
        for a in ("pub", "pub_sink", "fix_all", "j0"):
            try: m[a] = 100 * st.mean(r[k][f"{a}_recall"] for k in keys)
            except KeyError: m[a] = float("nan")
        cells[(task, L)] = (r, keys, m)
        print(f"{task:18s} {L:4s} {len(keys):3d}  {m['pub']:6.1f} {m['pub_sink']:9.1f} "
              f"{m['fix_all']:8.1f} {m['j0']:6.1f}")

print("\nfix_all - pub, paired:")
for (task, L), (r, keys, m) in sorted(cells.items()):
    d = [r[k]["fix_all_recall"] - r[k]["pub_recall"] for k in keys]
    lo, hi = boot(d)
    print(f"  {task:18s} {L:4s} {100*st.mean(d):+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]  "
          f"worse on {sum(x<0 for x in d)} of {len(d)}")

print("\nfix_all - j0, paired  (\"ties replay sample for sample\" means 0 non-ties):")
for (task, L), (r, keys, m) in sorted(cells.items()):
    d = [r[k]["fix_all_recall"] - r[k]["j0_recall"] for k in keys]
    lo, hi = boot(d)
    ties = sum(x == 0 for x in d)
    tag = "TIES 50/50" if ties == len(d) else f"{ties}/{len(d)} tie"
    print(f"  {task:18s} {L:4s} {100*st.mean(d):+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]  {tag}")

print("\nthe no-visibility control (fix_none), multikey:")
for L, f in CTRL.items():
    r = rows(f)
    if r is None: continue
    keys = sorted(k for k in r if k[0] == "niah_multikey_1" and k[1] == L)
    if keys:
        print(f"  {L}: fix_none = {100*st.mean(r[k]['fix_none_recall'] for k in keys):.1f}")
