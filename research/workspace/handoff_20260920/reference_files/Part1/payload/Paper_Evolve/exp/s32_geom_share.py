"""Recompute the control's share with the CORRECT single-factor reference.

WHY. The refutation round (2026-09-07) found that `fix_none` is built as
`EncbankLower(model, j, tok, lower_layers=[])` and `EncbankLower.__init__` defaults
`chunk_write_sink=True` (exp/s15_ruler_lower.py:72,77), so `build_bottom` prepends a BOS to
every chunk before capturing h_j and the lower K/V (:142-145). `fix_none` therefore writes
its chunks WITH a sink, exactly as `pub_sink` does.

Consequence: `fix_none - pub` is NOT the query-side geometry. It is
    (write-sink effect) + (query-side geometry),
and the share statistic I published in E52/E53 used `pub` as the reference, so it measured
the sum, not the geometry. The single-factor contrast that holds the write sink fixed is
`fix_none - pub_sink`, which is also the estimand E50 uses on the LM protocol (it references
A_on, the write-sink-ON arm). So the "protocol disagreement" conclusion compared two
different estimands.

This script prints both shares side by side so the correction is auditable:
    share_pub  = (fix_none - pub)      / (fix_all - pub)        <- what E52/E53 published
    share_geom = (fix_none - pub_sink) / (fix_all - pub_sink)   <- the single-factor version

Both are ratios of cell means. Cross-checkpoint they are not paired and carry no interval.
The paired interval that licenses a direction is `fix_none - pub_sink` itself, printed here
with the house bootstrap (B=4000, random.Random(0) reseeded per interval).
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


CELLS = [("niah_multikey_1", "8k"), ("niah_multikey_1", "16k"),
         ("niah_multikey_1", "32k"), ("niah_multikey_1", "64k"),
         ("niah_single_2", "8k"), ("niah_single_2", "16k"),
         ("niah_single_2", "32k"), ("niah_single_2", "64k"),
         ("variable_tracking", "16k"), ("variable_tracking", "32k")]

FN8 = {"8k": "s31_ruler_8b_fixnone_8k.json", "16k": "s15b_ruler_j12_16k.json",
       "32k": "s15b_ruler_j12_32k.json", "64k": "s31_ruler_8b_fixnone_64k.json",
       "vt": "s31_ruler_8b_fixnone_vt.json"}
RF8 = {"8k": "s15c_ruler_j12_8k.json", "16k": "s15_ruler_j12_16k.json",
       "32k": "s15_ruler_j12_32k.json", "64k": "s15c_ruler_j12_64k.json",
       "vt16k": "s15c_ruler_vt_16k.json", "vt32k": "s15c_ruler_vt_32k.json"}

out = {}
for ck in ("8B", "1.7B"):
    print(f"===== {ck}")
    print(f"  {'cell':26s} {'pub':>6s} {'p_sink':>7s} {'f_none':>7s} {'f_all':>6s} | "
          f"{'share_pub':>9s} {'share_geom':>10s} | fix_none - pub_sink (paired)")
    for task, L in CELLS:
        vt = task == "variable_tracking"
        if ck == "8B":
            fn = rows(FN8["vt"] if vt else FN8[L])
            rf = rows(RF8["vt" + L] if vt else RF8[L])
        else:
            fn = rows("s30_ruler_17b_fixnone_vt.json" if vt else "s30_ruler_17b_fixnone_niah.json")
            rf = rows("s19_ruler_17b_vt.json" if vt else "s19_ruler_17b_niah.json")
        if fn is None or rf is None:
            continue
        keys = sorted(k for k in fn if k[0] == task and k[1] == L)
        if not keys or any(k not in rf for k in keys):
            continue
        c = [fn[k]["fix_none_recall"] for k in keys]
        m = {a: [rf[k][f"{a}_recall"] for k in keys] for a in ("pub", "pub_sink", "fix_all")}
        mc, mp, ms_, ma = st.mean(c), st.mean(m["pub"]), st.mean(m["pub_sink"]), st.mean(m["fix_all"])
        s_pub = 100 * (mc - mp) / (ma - mp) if abs(ma - mp) > 1e-9 else float("nan")
        s_geo = 100 * (mc - ms_) / (ma - ms_) if abs(ma - ms_) > 1e-9 else float("nan")
        d = [x - y for x, y in zip(c, m["pub_sink"])]
        lo, hi = boot(d)
        excl = "EXCL" if (lo > 0 or hi < 0) else "spans0"
        print(f"  {task+'/'+L:26s} {100*mp:6.1f} {100*ms_:7.1f} {100*mc:7.1f} {100*ma:6.1f} | "
              f"{s_pub:8.1f}% {s_geo:9.1f}% | {100*st.mean(d):+6.1f} "
              f"[{100*lo:+6.1f},{100*hi:+6.1f}] {excl}")
        out.setdefault(ck, {})[(task, L)] = (s_pub, s_geo, 100 * st.mean(d), 100 * lo, 100 * hi)
    print()

print("===== the corrected picture: geometry share by task (single-factor, vs pub_sink)")
for task in ("niah_single_2", "niah_multikey_1", "variable_tracking"):
    for ck in ("8B", "1.7B"):
        g = sorted((L, v[1]) for (t, L), v in out.get(ck, {}).items() if t == task)
        if g:
            print(f"  {task:18s} {ck:5s}: " + "  ".join(f"{L}={s:7.1f}%" for L, s in g))
    print()

print("===== sign test: is the geometry positive or negative, by checkpoint?")
for ck in ("8B", "1.7B"):
    v = out.get(ck, {})
    pos = sum(1 for x in v.values() if x[1] > 0)
    print(f"  {ck:5s}: geometry share positive in {pos} of {len(v)} cells; "
          f"range {min(x[1] for x in v.values()):.1f}% to {max(x[1] for x in v.values()):.1f}%")
    e = [k for k, x in v.items() if x[3] > 0 or x[4] < 0]
    print(f"         paired fix_none - pub_sink excludes 0 in {len(e)} of {len(v)}: "
          + ", ".join(f"{t}/{L}({v[(t,L)][2]:+.1f})" for t, L in sorted(e)))
