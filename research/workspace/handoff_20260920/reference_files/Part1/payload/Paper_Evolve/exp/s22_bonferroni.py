"""Family-wise correction for the S21/S22 composition block.

The block computes 6 cells x 7 contrasts = 42 paired intervals. The paper argues from
three of them and must say whether those three survive a correction over the whole family.

Convention: percentile bootstrap on the paired per-sample difference, as everywhere else,
but at the Bonferroni per-interval level 1 - alpha/m instead of 95%. B=40000 rather than
the usual 4000, because a two-sided 99.881% interval at B=4000 sits at order statistics 2
and 3997, which is far too coarse to read.
"""
import json, random, statistics as st, sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
TAG = sys.argv[1] if len(sys.argv) > 1 else "s22"
M = 42
ALPHA = 0.05
B = 40000

def rows(n):
    p = RES / n
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {(r["task"], r["length"], r["i"]): r for r in d["rows"]}

def ci(diffs, level, b=B, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(b))
    lo_q = (1 - level) / 2
    return ms[int(lo_q * b)], ms[int((1 - lo_q) * b)]

SRC = {("none","niah"):"s19_ruler_17b_niah.json", ("none","vt"):"s19_ruler_17b_vt.json",
       ("base","niah"):f"{TAG}_ruler_baselora_niah.json", ("base","vt"):f"{TAG}_ruler_baselora_vt.json",
       ("lower","niah"):f"{TAG}_ruler_lowerlora_niah.json", ("lower","vt"):f"{TAG}_ruler_lowerlora_vt.json"}
R = {k: rows(v) for k, v in SRC.items()}
CELLS = [("niah_multikey_1","16k","niah"),("niah_multikey_1","32k","niah"),
         ("niah_single_2","16k","niah"),("niah_single_2","32k","niah"),
         ("variable_tracking","16k","vt"),("variable_tracking","32k","vt")]

ARGUED = [
    ("none","fix_all","none","pub","the repair over pub (ARGUED, all 6 cells)", None),
    ("lower","fix_all","none","fix_all","composition: LoRA on top of the repair (ARGUED, vt16k)",
     ("variable_tracking","16k")),
    ("base","fix_all","none","fix_all","cross-path: base adapter on a seeing reader (ARGUED, vt)",
     ("variable_tracking",)),
]
lvl = 1 - ALPHA / M
print(f"{TAG}: family of m={M} intervals, Bonferroni per-interval level {100*lvl:.3f}%, B={B}\n")
for a_ad, a_arm, b_ad, b_arm, label, restrict in ARGUED:
    print(label)
    for task, length, grp in CELLS:
        if restrict and (task not in restrict or (len(restrict) > 1 and length not in restrict)):
            continue
        keys = sorted(k for k in R[("none", grp)] if k[0] == task and k[1] == length)
        try:
            A = [R[(a_ad, grp)][k][f"{a_arm}_recall"] for k in keys]
            Bv = [R[(b_ad, grp)][k][f"{b_arm}_recall"] for k in keys]
        except (KeyError, TypeError):
            print(f"    {task}/{length}: missing"); continue
        d = [x - y for x, y in zip(A, Bv)]
        lo95, hi95 = ci(d, 0.95)
        lo, hi = ci(d, lvl)
        verdict = "SURVIVES" if (lo > 0 or hi < 0) else "does NOT survive"
        print(f"    {task:18s}/{length}  {100*st.mean(d):+6.1f}  "
              f"95% [{100*lo95:+6.1f},{100*hi95:+6.1f}]  "
              f"corrected [{100*lo:+6.1f},{100*hi:+6.1f}]  {verdict}")
    print()

# the retracted one, for the record
print("RETRACTED contrast, for the record (zero-shot repair vs LoRA-trained pub):")
for task, length, grp in CELLS:
    keys = sorted(k for k in R[("none", grp)] if k[0] == task and k[1] == length)
    A = [R[("none", grp)][k]["fix_all_recall"] for k in keys]
    Bv = [R[("base", grp)][k]["pub_recall"] for k in keys]
    d = [x - y for x, y in zip(A, Bv)]
    lo95, hi95 = ci(d, 0.95); lo, hi = ci(d, lvl)
    v = "SURVIVES" if (lo > 0 or hi < 0) else "does NOT survive"
    print(f"    {task:18s}/{length}  {100*st.mean(d):+6.1f}  "
          f"95% [{100*lo95:+6.1f},{100*hi95:+6.1f}]  corrected [{100*lo:+6.1f},{100*hi:+6.1f}]  {v}")

print("\nAlso leaned on in sec:exp-lora -- matched minus mismatched adapter, both on fix_all:")
for task, length, grp in CELLS:
    keys = sorted(k for k in R[("none", grp)] if k[0] == task and k[1] == length)
    try:
        A = [R[("lower", grp)][k]["fix_all_recall"] for k in keys]
        Bv = [R[("base", grp)][k]["fix_all_recall"] for k in keys]
    except (KeyError, TypeError):
        continue
    d = [x - y for x, y in zip(A, Bv)]
    lo95, hi95 = ci(d, 0.95); lo, hi = ci(d, lvl)
    v = "SURVIVES" if (lo > 0 or hi < 0) else "does NOT survive"
    print(f"    {task:18s}/{length}  {100*st.mean(d):+6.1f}  "
          f"95% [{100*lo95:+6.1f},{100*hi95:+6.1f}]  corrected [{100*lo:+6.1f},{100*hi:+6.1f}]  {v}")
