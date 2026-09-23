"""Does the control's share of the recovery track the TASK or the MODEL SIZE?

s30 (E52) refuted the reading that the positions+sink control is a small-model artefact, and
the sentence that replaced it in 05_experiment is that the control's share of the
published-to-repaired gap "tracks the task rather than the model size". That sentence rested
on four cells, because the 8B had no fix_none on the chained task. s31 supplies it (and the
8k/64k needle lengths), so the claim can be checked on the full grid instead of asserted from
a corner of it.

The claim makes a sharp prediction: the 8B control should buy roughly nothing on
variable_tracking, as it does on the 1.7B (7.5% / -3.1%). If it buys a lot there, the
sentence is FALSE and comes out of the main text.

Paired contrasts are within a checkpoint, on the fifty shared samples of each cell
(deterministic in task/length/seed/PYTHONHASHSEED=0). The SHARE is a ratio of cell means; when
compared across checkpoints it is not paired and gets no interval, and is printed that way.
Bootstrap: percentile, B=4000, random.Random(0) reseeded per interval.
"""
import json, random, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def rows(n):
    if n is None:
        return None
    p = RES / n
    return None if not p.exists() else {
        (r["task"], r["length"], r["i"]): r
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


CELLS = [("niah_multikey_1", "8k"), ("niah_multikey_1", "16k"),
         ("niah_multikey_1", "32k"), ("niah_multikey_1", "64k"),
         ("niah_single_2", "8k"), ("niah_single_2", "16k"),
         ("niah_single_2", "32k"), ("niah_single_2", "64k"),
         ("variable_tracking", "16k"), ("variable_tracking", "32k")]

# where each checkpoint's fix_none lives, and where its reference arms live
FN = {
    "8B": {"8k": "s31_ruler_8b_fixnone_8k.json", "16k": "s15b_ruler_j12_16k.json",
           "32k": "s15b_ruler_j12_32k.json", "64k": "s31_ruler_8b_fixnone_64k.json",
           "vt": "s31_ruler_8b_fixnone_vt.json"},
    "1.7B": {"niah": "s30_ruler_17b_fixnone_niah.json", "vt": "s30_ruler_17b_fixnone_vt.json"},
}
REF = {
    "8B": {"8k": "s15c_ruler_j12_8k.json", "16k": "s15_ruler_j12_16k.json",
           "32k": "s15_ruler_j12_32k.json", "64k": "s15c_ruler_j12_64k.json",
           "vt16k": "s15c_ruler_vt_16k.json", "vt32k": "s15c_ruler_vt_32k.json"},
    "1.7B": {"niah": "s19_ruler_17b_niah.json", "vt": "s19_ruler_17b_vt.json"},
}
ARMS = ("pub", "pub_sink", "fix_all", "j0")
shares = {}


def cell(ck, task, L, fn, rf):
    if fn is None or rf is None:
        return
    keys = sorted(k for k in fn if k[0] == task and k[1] == L)
    if not keys or any(k not in rf for k in keys):
        return
    c = [fn[k]["fix_none_recall"] for k in keys]
    m = {a: [rf[k][f"{a}_recall"] for k in keys] for a in ARMS if f"{a}_recall" in rf[keys[0]]}
    line = f"  {task:18s} {L:4s} n={len(keys):2d}  fix_none={100*st.mean(c):6.1f}"
    for a in ARMS:
        if a in m:
            line += f"  {a}={100*st.mean(m[a]):6.1f}"
    print(line)
    for a in ("pub", "pub_sink", "fix_all"):
        if a not in m:
            continue
        d = [x - y for x, y in zip(c, m[a])]
        lo, hi = boot(d)
        print(f"      fix_none - {a:8s} = {100*st.mean(d):+6.1f} [{100*lo:+6.1f},{100*hi:+6.1f}]"
              f"  (+{sum(x>0 for x in d)}/-{sum(x<0 for x in d)}/={sum(x==0 for x in d)})"
              + ("  EXCLUDES 0" if (lo > 0 or hi < 0) else "  spans 0"))
    if "pub" in m and "fix_all" in m:
        den = st.mean(m["fix_all"]) - st.mean(m["pub"])
        if abs(den) > 1e-9:
            sh = 100 * (st.mean(c) - st.mean(m["pub"])) / den
            shares[(ck, task, L)] = sh
            print(f"      share of the pub->fix_all recovery (cell means): {sh:5.1f}%")


print("===== Qwen3-8B, j=12 of 36")
for task, L in CELLS:
    if task == "variable_tracking":
        cell("8B", task, L, rows(FN["8B"]["vt"]), rows(REF["8B"]["vt" + L]))
    else:
        cell("8B", task, L, rows(FN["8B"][L]), rows(REF["8B"][L]))

print("\n===== Qwen3-1.7B, j=9 of 28")
for task, L in CELLS:
    g = "vt" if task == "variable_tracking" else "niah"
    cell("1.7B", task, L, rows(FN["1.7B"][g]), rows(REF["1.7B"][g]))

print("\n===== THE CLAIM: does the share track the task or the model size?")
print("     (ratios of cell means; cross-checkpoint, so NOT paired and no intervals)")
for task in ("niah_single_2", "niah_multikey_1", "variable_tracking"):
    for ck in ("8B", "1.7B"):
        v = [f"{L}={shares[(ck,task,L)]:.1f}%" for _, t, L in
             [k for k in shares if k[0] == ck and k[1] == task]
             for _ in [0]] if any(k[0] == ck and k[1] == task for k in shares) else []
        got = sorted((k[2], shares[k]) for k in shares if k[0] == ck and k[1] == task)
        if got:
            print(f"  {task:18s} {ck:5s}: " + "  ".join(f"{L}={s:6.1f}%" for L, s in got))
    print()
