"""Does the no-visibility control behave the same way on a second checkpoint?

WHY THIS RUN EXISTS. The paper's mechanism story credits lower-band VISIBILITY with the
repair. The control that tests it is fix_none: the query gets its pack positions and the
sink entry, but every chunk entry is hidden. On Qwen3-8B (E8a) that control sits ABOVE the
published write and BELOW the write-sink baseline on multikey (-28 / -30), which is what
licenses "the sink and the positions do not explain the gain".

Then the LM protocol on the 1.7B (E50, s27) disagreed with that picture: the same control,
called FL_empty there, carries 13-44% (wikitext) and 17-58% (PG19) of the removable loss on
the 1.7B, where on the 8B it carries nothing and at shallow depths is negative. So the
smaller checkpoint gets a large minority of the repair from positions+sink alone -- on the
LM protocol. The task-level twin was never run on the 1.7B (E24b says so explicitly: "no
fix_none control and no fix_S arm on 1.7B", so E8a's decomposition is not replicated).
s30 runs it, so the two protocols can be compared on the same checkpoint.

Paired throughout: RULER samples are deterministic in (task, length, seed, PYTHONHASHSEED=0),
so this single-arm run lands on the same fifty samples per cell that s19 already scored for
pub, pub_sink, fix_all and j0.
Intervals: percentile bootstrap, B=4000, random.Random(0) reseeded per interval, on the
paired per-sample difference. Endpoints are approximate to +-2 recall points (see the
CLAIM_EVIDENCE preamble); no claim here rests on an endpoint's exact value, only on whether
the interval excludes 0.

Six cells here against the 8B's two: the 8B ran fix_none on the niah tasks at 16k/32k only,
so the 1.7B has the more complete control and the cross-checkpoint lines below are printed
only where the 8B actually has the cell. Cross-checkpoint differences are two cell means with
no interval and are never called paired.
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
             "fn_niah": "s30_ruler_17b_fixnone_niah.json",
             "fn_vt": "s30_ruler_17b_fixnone_vt.json"},
    "8B": {"fn_niah": "s15b_ruler_j12_16k.json", "fn_vt": None},
}
# the 8B's fix_none lives in the two s15b files, one per length; vt was never run there
REF8 = {"16k": "s15_ruler_j12_16k.json", "32k": "s15_ruler_j12_32k.json"}
FN8 = {"16k": "s15b_ruler_j12_16k.json", "32k": "s15b_ruler_j12_32k.json"}

ARMS = ("pub", "pub_sink", "fix_all", "j0")


def block(label, fn, rf, task, L):
    keys = sorted(k for k in fn if k[0] == task and k[1] == L)
    if not keys or any(k not in rf for k in keys):
        print(f"  {task:18s} {L}: not run on {label}")
        return
    c = [fn[k]["fix_none_recall"] for k in keys]
    m = {a: [rf[k][f"{a}_recall"] for k in keys] for a in ARMS if f"{a}_recall" in rf[keys[0]]}
    line = f"  {task:18s} {L}  n={len(keys):2d}  fix_none={100*st.mean(c):6.1f}"
    for a in ARMS:
        if a in m:
            line += f"  {a}={100*st.mean(m[a]):6.1f}"
    print(line)
    for a in ("pub", "pub_sink", "fix_all"):
        if a not in m:
            continue
        d = [x - y for x, y in zip(c, m[a])]
        lo, hi = boot(d)
        print(f"      fix_none - {a:8s} = {100*st.mean(d):+6.1f} "
              f"[{100*lo:+6.1f}, {100*hi:+6.1f}]"
              f"  (+{sum(x>0 for x in d)} / -{sum(x<0 for x in d)} / ={sum(x==0 for x in d)})"
              + ("   EXCLUDES 0" if (lo > 0 or hi < 0) else "   spans 0"))
    # the share of the published->repaired recovery that the control alone reaches
    if "pub" in m and "fix_all" in m:
        num = st.mean(c) - st.mean(m["pub"])
        den = st.mean(m["fix_all"]) - st.mean(m["pub"])
        if abs(den) > 1e-9:
            print(f"      control share of the pub->fix_all recovery (cell means, no interval): "
                  f"{100*num/den:5.1f}%")


print("===== 1.7B (j=9 of 28), s30 against s19")
fn_n, fn_v = rows(SRC["1.7B"]["fn_niah"]), rows(SRC["1.7B"]["fn_vt"])
rf_n, rf_v = rows(SRC["1.7B"]["ref_niah"]), rows(SRC["1.7B"]["ref_vt"])
for task, L in CELLS:
    vt = task == "variable_tracking"
    fn, rf = (fn_v, rf_v) if vt else (fn_n, rf_n)
    if fn is None or rf is None:
        print(f"  {task:18s} {L}: source missing")
        continue
    block("1.7B", fn, rf, task, L)

print("\n===== 8B (j=12 of 36), s15b against s15 -- the two cells that exist")
for task, L in CELLS:
    if task == "variable_tracking":
        continue
    fn, rf = rows(FN8[L]), rows(REF8[L])
    if fn is None or rf is None:
        print(f"  {task:18s} {L}: source missing")
        continue
    block("8B", fn, rf, task, L)
