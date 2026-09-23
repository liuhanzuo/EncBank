"""Does the MECHANISM replicate on a second checkpoint?

The paper's C1 (the zero-shot depth loss is entirely query-side) and C2 (lower-band
visibility removes most of it) rest on Qwen3-8B alone. S27 ran the same LM fidelity protocol
on Qwen3-1.7B at matched depth fractions. This compares them.

Metric: frac = KL(p_ref || p_arm) / KL(p_ref || p_no_mem) on 32-token continuations, mean
over query positions then over samples. 0 = as good as the chunks in context, 1 = no better
than no memory. Lower is better. Arms: A_off (published write, no sink), A_on (published +
BOS at write), A_chunk (chunks written alone, query taken from the reference forward -- so
the reader is exact and only the memory is degraded), FL_all (the repair: whole lower band
visible), C_bos / KVb_m (full-depth isolated-chunk K/V references).

Intervals are the paper's percentile bootstrap over samples: B=4000, random.Random(0)
reseeded per interval, on the paired per-sample difference where a difference is taken.
"""
import json, random, statistics as st, sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

def load(n):
    p = RES / n
    return None if not p.exists() else json.loads(p.read_text(encoding="utf-8"))

def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]

def col(d, key):
    return [r[key] for r in d["rows"] if key in r]

CK = [("Qwen3-8B",   "s14_deployable_qwen3-8b_{c}.json"),
      ("Qwen3-1.7B", "s27_deployable_qwen3-1.7b_{c}.json")]

for name, pat in CK:
    print(f"===== {name}")
    for c in ("wikitext", "pg19"):
        d = load(pat.format(c=c))
        if d is None:
            print(f"  {c}: absent"); continue
        js, L, n = d["js"], d["L"], len(d["rows"])
        print(f"  {c}: L={L}, n={n} samples, js={js} "
              f"(fractions {'/'.join(f'{j/L:.2f}' for j in js)})")
        for arm in ("A_off", "A_on", "A_chunk", "FL_all"):
            vals = []
            for j in js:
                v = col(d, f"{arm}_j{j}_frac")
                vals.append(100 * st.mean(v) if v else float("nan"))
            print(f"    {arm:8s} " + " ".join(f"{x:6.1f}" for x in vals))

print("\n===== the two claims, per checkpoint")
for name, pat in CK:
    for c in ("wikitext", "pg19"):
        d = load(pat.format(c=c))
        if d is None:
            continue
        js = d["js"]; L = d["L"]
        ch = [100 * st.mean(col(d, f"A_chunk_j{j}_frac")) for j in js]
        # C2: the repair removes what fraction of the removable loss, per sample
        rem = []
        for j in js:
            a = col(d, f"A_on_j{j}_frac"); f = col(d, f"FL_all_j{j}_frac")
            ck = col(d, f"A_chunk_j{j}_frac")
            per = [(A - F) / (A - C) for A, F, C in zip(a, f, ck) if (A - C) > 1e-6]
            rem.append(100 * st.mean(per) if per else float("nan"))
        worse = []
        for j in js:
            a = col(d, f"A_on_j{j}_frac"); f = col(d, f"FL_all_j{j}_frac")
            worse.append(sum(1 for A, F in zip(a, f) if F >= A))
        print(f"  {name:11s} {c:8s}  C1: max chunk-side frac over the grid = {max(ch):.2f}%"
              f"   |  C2: removed {min(rem):.1f}-{max(rem):.1f}% of the removable loss,"
              f" worse on {max(worse)} of {len(col(d, f'A_on_j{js[0]}_frac'))} samples at worst depth")
