"""S21 -- does the base paper's LoRA still buy anything once the read path is repaired?

Qwen3-1.7B, L=28, j=9 (= 0.32 L), RULER, n=50 per cell, the SAME fifty samples in every
cell (PYTHONHASHSEED=0, seed 42), so every contrast below is paired.

THE 2x2 THIS ASSEMBLES
                              no adapter                 + LoRA
    published read path       pub      (s19)             pub      (s21, base adapter)
    repaired read path        fix_all  (s19)             fix_all  (s21, lower adapter)

Each adapter is scored on the path it was distilled for -- both against the same j=0
teacher, both with Encbank/train/README.md's recipe (r32/a32, lr 1e-4, 1000 steps, top-64
bidirectional KL, lam 0.6, seven 512-token PG19 context chunks) -- so the two columns are
the same training budget spent on two different readers.  The off-diagonal cells (an
adapter applied to the path it was NOT trained for) are also reported, because "the base
adapter makes a seeing reader worse" and "it makes it better" are different papers.

Zero-shot references come from the earlier no-adapter run, not from a re-run:
    exp/results/s19_ruler_17b_niah.json, s19_ruler_17b_vt.json
which carry pub / pub_sink / fix_all / j0 on those same fifty samples.

Every interval is the paper's percentile bootstrap: B=4000, random.Random(0) reseeded per
interval, on the paired per-sample difference (exp/s14_analyze.py::boot_ci).
"""
import json
import random
import sys
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
B = 4000


def load(name):
    p = RES / name
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d, {(r["task"], r["length"], r["i"]): r for r in d["rows"]}


def boot_ci(diffs, b=B, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(b))
    return ms[int(0.025 * b)], ms[int(0.975 * b)]


def cell(rows, keys, arm):
    return [rows[k][f"{arm}_recall"] for k in keys]


def contrast(a, b, label):
    d = [x - y for x, y in zip(a, b)]
    lo, hi = boot_ci(d)
    return (f"{label:<52s} {100*st.mean(d):+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]  "
            f"(+{sum(x>0 for x in d)} / -{sum(x<0 for x in d)} / ={sum(x==0 for x in d)})")


# s21 = 1000 training steps (Encbank/train/README.md's documented example);
# s22 = the same recipe at 4000 steps, so the published path is also compared at the
# budget the base paper reports. Pass the tag as argv[1].
TAG = sys.argv[1] if len(sys.argv) > 1 else "s21"
SRC = {
    ("none", "niah"): "s19_ruler_17b_niah.json",
    ("none", "vt"): "s19_ruler_17b_vt.json",
    ("base", "niah"): f"{TAG}_ruler_baselora_niah.json",
    ("base", "vt"): f"{TAG}_ruler_baselora_vt.json",
    ("lower", "niah"): f"{TAG}_ruler_lowerlora_niah.json",
    ("lower", "vt"): f"{TAG}_ruler_lowerlora_vt.json",
}
print(f"=== {TAG}: adapters against the zero-shot s19 references ===\n")
CELLS = [("niah_multikey_1", "16k", "niah"), ("niah_multikey_1", "32k", "niah"),
         ("niah_single_2", "16k", "niah"), ("niah_single_2", "32k", "niah"),
         ("variable_tracking", "16k", "vt"), ("variable_tracking", "32k", "vt")]

loaded, missing = {}, []
for k, name in SRC.items():
    d = load(name)
    if d is None:
        missing.append(f"{k[0]}/{k[1]} -> {name}")
    else:
        loaded[k] = d
if missing:
    print(f"MISSING (run exp/results/{TAG}_chain.cmd first):")
    for m in missing:
        print("  " + m)

# provenance: every s21 file must name the adapter it merged, and the s19 files must not
for (ad, grp), (d, _) in sorted(loaded.items()):
    a = d["args"]
    print(f"[{ad:5s}/{grp:4s}] j={a['j']} n={a['n']} arms={d['arms']} "
          f"adapter_pt={a.get('adapter_pt', '')!r} lora={d.get('lora')}")
print()

for task, length, grp in CELLS:
    have = [ad for ad in ("none", "base", "lower") if (ad, grp) in loaded]
    if "none" not in have:
        continue
    keys = sorted(k for k in loaded[("none", grp)][1] if k[0] == task and k[1] == length)
    if not keys:
        continue
    # every file must score the SAME fifty samples, or nothing below is paired
    bad = [ad for ad in have
           if sorted(k for k in loaded[(ad, grp)][1] if k[0] == task and k[1] == length) != keys]
    if bad:
        print(f"### {task}/{length}: SAMPLE SETS DIFFER for {bad} -- contrasts skipped\n")
        continue

    R = {ad: loaded[(ad, grp)][1] for ad in have}
    get = lambda ad, arm: cell(R[ad], keys, arm)                        # noqa: E731
    print(f"### {task} / {length}   (n={len(keys)}, Qwen3-1.7B, j=9)")
    hdr = {"none": "no adapter", "base": "+LoRA(published path)", "lower": "+LoRA(repaired path)"}
    print(f"    {'arm':<10s}" + "".join(f"{hdr[ad]:>24s}" for ad in have))
    for arm in ("pub", "fix_all"):
        vals = []
        for ad in have:
            try:
                vals.append(f"{100*st.mean(get(ad, arm)):>24.1f}")
            except KeyError:
                vals.append(f"{'--':>24s}")
        print(f"    {arm:<10s}" + "".join(vals))
    if "j0" in loaded[("none", grp)][0]["arms"]:
        print(f"    {'j0 (bound)':<10s}{100*st.mean(get('none', 'j0')):>24.1f}")
    print()

    def safe(ad, arm):
        try:
            return get(ad, arm)
        except KeyError:
            return None

    pairs = [
        ("base", "pub", "none", "pub",
         "LoRA on the published path (the base paper's fix)"),
        ("none", "fix_all", "none", "pub",
         "the repair, zero-shot (this paper's fix)"),
        ("lower", "fix_all", "none", "fix_all",
         "LoRA ON TOP of the repair (the composition question)"),
        ("none", "fix_all", "base", "pub",
         "zero-shot repair vs LoRA-trained published path"),
        ("base", "fix_all", "none", "fix_all",
         "cross-path: base adapter applied to a seeing reader"),
        ("lower", "pub", "none", "pub",
         "cross-path: lower adapter applied to a blind reader"),
        # The comparison that tests "each adapter is the right adapter for its path".
        # It is not: on vt 32k the adapter trained for the BLIND reader wins here.
        ("lower", "fix_all", "base", "fix_all",
         "path-matched vs mismatched adapter, both on fix_all"),
    ]
    for a_ad, a_arm, b_ad, b_arm, label in pairs:
        A, Bv = safe(a_ad, a_arm), safe(b_ad, b_arm)
        if A is None or Bv is None:
            print(f"    {label:<52s} (not run)")
        else:
            print("    " + contrast(A, Bv, label))
    print()
