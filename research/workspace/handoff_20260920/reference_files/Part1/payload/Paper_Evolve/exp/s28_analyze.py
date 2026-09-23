"""Does "full-depth K/V without upper recompute is worse" replicate on a second checkpoint?

Table~\ref{tab:main}'s caption uses the cbos arm to make an architectural claim that
positions this paper against the KV-reuse family: caching the chunks' K/V at ALL layers and
recomputing nothing above j is worse than caching the depth-j residual plus the lower band
and recomputing the upper band. That was one checkpoint. s28 runs cbos on Qwen3-1.7B at j=9
on the same fifty samples per cell as s19 (deterministic in task/length/seed), so every
contrast below is paired.

Storage, from each model's config: on the 8B cbos is 36 x 4 = 144 KB/token against fix_all's
8 + 4x12 = 56. On the 1.7B the hidden is half as wide but the KV geometry is NOT (8 KV heads x 128, same as
the 8B, so 4 KB/layer either way), so cbos is 28 x 4 = 112 KB/token against fix_all's
4 + 4x9 = 40. The earlier "28 x 2 = 56 against 22" in this docstring was wrong: it halved the
KV along with the residual. s27 prints both terms from config.json.
Intervals: percentile bootstrap, B=4000, random.Random(0) reseeded per interval, on the
paired per-sample difference.
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

CELLS = [("niah_multikey_1", "16k"), ("niah_multikey_1", "32k"),
         ("niah_single_2", "16k"), ("niah_single_2", "32k"),
         ("variable_tracking", "16k"), ("variable_tracking", "32k")]

SRC = {
  "1.7B": {"ref_niah": "s19_ruler_17b_niah.json", "ref_vt": "s19_ruler_17b_vt.json",
           "cbos_niah": "s28_ruler_17b_cbos_niah.json", "cbos_vt": "s28_ruler_17b_cbos_vt.json"},
  "8B":   {"ref_niah": None, "ref_vt": None,
           "cbos_niah": "s15e_ruler_cbos_niah.json", "cbos_vt": "s15e_ruler_cbos_vt.json"},
}
REF8 = {"niah_multikey_1": {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json"},
        "niah_single_2":  {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json"},
        "variable_tracking": {"16k": "s15c_ruler_vt_16k.json", "32k": "s15c_ruler_vt_32k.json"}}

for ck in ("8B", "1.7B"):
    print(f"===== {ck}")
    for task, L in CELLS:
        grp = "vt" if task == "variable_tracking" else "niah"
        cb = rows(SRC[ck]["cbos_" + grp])
        rf = rows(SRC[ck]["ref_" + grp]) if ck == "1.7B" else rows(REF8[task][L])
        if cb is None or rf is None:
            print(f"  {task:18s} {L}: missing"); continue
        keys = sorted(k for k in cb if k[0] == task and k[1] == L)
        if not keys or any(k not in rf for k in keys):
            print(f"  {task:18s} {L}: not run"); continue
        c = [cb[k]["cbos_recall"] for k in keys]
        m = {a: [rf[k][f"{a}_recall"] for k in keys] for a in ("pub", "pub_sink", "fix_all", "j0")
             if f"{a}_recall" in rf[keys[0]]}
        line = f"  {task:18s} {L}  cbos={100*st.mean(c):6.1f}"
        for a in ("pub", "pub_sink", "fix_all", "j0"):
            if a in m:
                line += f"  {a}={100*st.mean(m[a]):6.1f}"
        print(line)
        if "fix_all" in m:
            d = [x - y for x, y in zip(c, m["fix_all"])]
            lo, hi = boot(d)
            print(f"      cbos - fix_all = {100*st.mean(d):+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]"
                  f"  (+{sum(x>0 for x in d)} / -{sum(x<0 for x in d)} / ={sum(x==0 for x in d)})"
                  + ("   EXCLUDES 0" if (lo > 0 or hi < 0) else "   spans 0"))
        if "pub_sink" in m:
            d = [x - y for x, y in zip(c, m["pub_sink"])]
            lo, hi = boot(d)
            print(f"      cbos - pub_sink = {100*st.mean(d):+5.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]"
                  + ("   EXCLUDES 0" if (lo > 0 or hi < 0) else "   spans 0"))
