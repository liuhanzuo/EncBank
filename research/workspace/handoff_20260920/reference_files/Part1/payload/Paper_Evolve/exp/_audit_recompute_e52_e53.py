"""INDEPENDENT recomputation of E52/E53 numbers. Read-only. Does not import s30/s31 analyzers."""
import json, random, statistics as st
from pathlib import Path

RES = Path("F:/Paper_Evolve/exp/results")


def load(n):
    d = json.loads((RES / n).read_text(encoding="utf-8"))
    return {(r["task"], r["length"], r["i"]): r for r in d["rows"]}


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


# ---- source map (checkpoint, task, length) -> (fix_none file, reference file)
M = "niah_multikey_1"; S = "niah_single_2"; V = "variable_tracking"
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
    if n not in cache:
        cache[n] = load(n)
    return cache[n]

shares = {}
means = {}
print("cell                                 n  fix_none    pub  pub_sink  fix_all  |  pairing")
for (ck, task, L), (fnf, rff) in SRC.items():
    fn, rf = R(fnf), R(rff)
    keys = sorted(k for k in fn if k[0] == task and k[1] == L)
    miss = [k for k in keys if k not in rf]
    # pairing evidence: same answers + same n_tokens
    ans_ok = all(fn[k]["answers"] == rf[k]["answers"] for k in keys if k in rf)
    tok_ok = all(fn[k]["n_tokens"] == rf[k]["n_tokens"] for k in keys if k in rf)
    c = [fn[k]["fix_none_recall"] for k in keys]
    arms = {a: [rf[k][a + "_recall"] for k in keys] for a in ("pub", "pub_sink", "fix_all")}
    means[(ck, task, L)] = dict(fix_none=st.mean(c), **{a: st.mean(v) for a, v in arms.items()})
    print(f"{ck:5s} {task:16s} {L:4s} n={len(keys):2d} "
          f"{100*st.mean(c):7.2f} {100*st.mean(arms['pub']):7.2f} "
          f"{100*st.mean(arms['pub_sink']):8.2f} {100*st.mean(arms['fix_all']):8.2f} "
          f"| miss={len(miss)} ans_match={ans_ok} ntok_match={tok_ok}")
    for a in ("pub", "pub_sink", "fix_all"):
        d = [x - y for x, y in zip(c, arms[a])]
        lo, hi = boot(d)
        print(f"      fn-{a:9s} {100*st.mean(d):+7.2f} [{100*lo:+7.2f},{100*hi:+7.2f}] "
              f"(+{sum(x>0 for x in d)}/-{sum(x<0 for x in d)}/={sum(x==0 for x in d)})"
              + ("  EXCL0" if (lo > 0 or hi < 0) else "  spans0"))
    den = st.mean(arms["fix_all"]) - st.mean(arms["pub"])
    sh = 100 * (st.mean(c) - st.mean(arms["pub"])) / den
    shares[(ck, task, L)] = sh
    print(f"      SHARE = {sh:8.4f}%   (num={100*(st.mean(c)-st.mean(arms['pub'])):.2f}, den={100*den:.2f})")

print("\n===== SHARE TABLE")
for task in (S, M, V):
    for ck in ("1.7B", "8B"):
        got = sorted(((k[2], v) for k, v in shares.items() if k[0] == ck and k[1] == task),
                     key=lambda t: int(t[0][:-1]))
        if got:
            print(f"  {task:16s} {ck:5s}: " + "  ".join(f"{L}={v:7.2f}%" for L, v in got)
                  + f"   -> range [{min(v for _, v in got):.2f}, {max(v for _, v in got):.2f}]")
    print()

print("===== CLAIMED RANGES CHECK")
def rng(ck, tasks):
    v = [shares[k] for k in shares if k[0] == ck and k[1] in tasks]
    return min(v), max(v), len(v)
print("  single 1.7B :", rng("1.7B", {S}),   " claim 54-75")
print("  single 8B   :", rng("8B", {S}),     " claim 79-90")
print("  mk+vt 1.7B  :", rng("1.7B", {M, V}), " claim -3-32")
print("  mk+vt 8B    :", rng("8B", {M, V}),   " claim 10-29")

print("\n===== 'neither uniformly above the other' / counting, cells where BOTH exist")
both = sorted({(k[1], k[2]) for k in shares if k[0] == "8B"} &
              {(k[1], k[2]) for k in shares if k[0] == "1.7B"})
n8 = 0
for task, L in both:
    a, b = shares[("8B", task, L)], shares[("1.7B", task, L)]
    lead = "8B" if a > b else "1.7B"
    n8 += a > b
    print(f"  {task:16s} {L:4s}: 8B={a:7.2f}%  1.7B={b:7.2f}%  -> {lead} leads")
print(f"  8B leads in {n8} of {len(both)} shared cells")

print("\n===== per-task 'same band' check (overlap of the two checkpoints' ranges)")
for task in (S, M, V):
    a = [shares[k] for k in shares if k[0] == "8B" and k[1] == task]
    b = [shares[k] for k in shares if k[0] == "1.7B" and k[1] == task]
    if a and b:
        print(f"  {task:16s}: 8B[{min(a):.1f},{max(a):.1f}]  1.7B[{min(b):.1f},{max(b):.1f}]  "
              f"overlap={'YES' if min(max(a), max(b)) >= max(min(a), min(b)) else 'NO'}")
