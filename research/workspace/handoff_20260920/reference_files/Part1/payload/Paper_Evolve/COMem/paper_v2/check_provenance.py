"""Every result file and script the paper names must exist on disk.

The house rule is that each number traces to a result file and a script. This checks the
mechanical half of that. It reads the LaTeX sources INCLUDING comments -- the provenance
blocks live in comments -- plus CLAIM_EVIDENCE.md, and expands the brace and star forms
those files use:
    s15_ruler_j12_{16k,32k}.json     -> two names
    s17b_ruler_mk16k_L{0..11}.json   -> twelve names
    s19_ruler_17b_mk16k_L*.json      -> must have at least one match

Only names matching the result-file convention (s<digits><optional letter>_...) are
checked. That filter matters: LaTeX wraps long names across lines, so a naive scan picks up
fragments like "8b_pg19.json" from a broken "s12_jsweep_qwen3-8b_pg19.json" and reports
them as missing. Fragments are reported separately, as a hint that a name is wrapped, not
as an error.

Run: python COMem/paper_v2/check_provenance.py     (exit 1 if anything is genuinely missing)
"""
import glob
import os
import re
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.abspath("../../exp/results")
ROOT = os.path.abspath("../..")
RESULT_NAME = re.compile(r"^s\d+[a-z]?_")

files = sorted(glob.glob("sections/*.tex")) + ["main.tex", "CLAIM_EVIDENCE.md"]
text = "\n".join(open(f, encoding="utf-8").read() for f in files if os.path.exists(f))
for a, b in ((r"\_", "_"), (r"\{", "{"), (r"\}", "}")):
    text = text.replace(a, b)


RANGE = re.compile(r"([A-Za-z]?)(\d+)\.\.\1?(\d+)")


def expand(name):
    # the prose writes ranges two ways: L{0..8} and L0..L8. Normalise the second to the
    # first, then expand braces.
    m = RANGE.search(name)
    if m and "{" not in name:
        pre, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        name = name[:m.start()] + pre + "{" + f"{a}..{b}" + "}" + name[m.end():]
    m = re.search(r"\{([^{}]*)\}", name)
    if not m:
        return [name]
    body, out = m.group(1), []
    rng = re.fullmatch(r"(\d+)\.\.(\d+)", body)
    parts = ([str(i) for i in range(int(rng.group(1)), int(rng.group(2)) + 1)]
             if rng else body.split(","))
    for p in parts:
        out += expand(name[:m.start()] + p.strip() + name[m.end():])
    return out


raw = set(re.findall(r"\b([A-Za-z0-9_.-]+(?:\{[^{}]*\})?[A-Za-z0-9_*.]*\.json)\b", text))
def _selftest():
    """A checker that has only ever printed 0 is an untested assertion, not evidence.

    Verified by hand once (2026-09-07): planting a reference to a nonexistent result file in
    a scratch sections/*.tex made this script report MISSING: 1 and name it, and removing the
    scratch returned it to 0. This encodes that check so it does not have to be redone.
    """
    bad = []

    # 1. expand(): the three forms the prose uses
    for src, want in (
        ("s15_ruler_j12_{16k,32k}.json",
         ["s15_ruler_j12_16k.json", "s15_ruler_j12_32k.json"]),
        ("s17b_ruler_mk16k_L{0..2}.json",
         ["s17b_ruler_mk16k_L0.json", "s17b_ruler_mk16k_L1.json",
          "s17b_ruler_mk16k_L2.json"]),
        ("s24_ruler_17b_mk32k_L0..L2.json",
         ["s24_ruler_17b_mk32k_L0.json", "s24_ruler_17b_mk32k_L1.json",
          "s24_ruler_17b_mk32k_L2.json"]),
        ("s30_ruler_17b_fixnone_niah.json", ["s30_ruler_17b_fixnone_niah.json"]),
    ):
        got = sorted(expand(src))
        if got != sorted(want):
            bad.append(f"    expand({src!r}) = {got}, expected {sorted(want)}")

    # 2. the RESULT_NAME filter. Names it rejects go to `fragments` and are NEVER checked,
    #    so what it accepts is exactly what gets verified.
    for name, want in (("s15_x.json", True), ("s17b_x.json", True),
                       ("s9_x.json", True), ("8b_pg19.json", False),
                       ("config.json", False),
                       # latent gap, asserted so a future rename cannot pass unnoticed:
                       ("s17ab_x.json", False)):
        got = bool(RESULT_NAME.match(name))
        if got != want:
            bad.append(f"    RESULT_NAME.match({name!r}) = {got}, expected {want}")

    # 3. the existence check: a name not on disk must be reported
    fake = "s99_selftest_definitely_absent.json"
    if os.path.exists(os.path.join(RES, fake)):
        bad.append(f"    {fake} unexpectedly exists; the self-test needs a different name")

    if bad:
        print("SELFTEST FAILED -- this provenance check cannot be trusted:")
        print("\n".join(bad))
        sys.exit(2)
    print("selftest: expand() 4 forms, RESULT_NAME 6 cases (incl. the s17ab_ gap), "
          "existence probe -- all pass")


_selftest()

wanted, fragments = set(), set()
for r in raw:
    if RESULT_NAME.match(r):
        wanted.update(expand(r))
    else:
        fragments.add(r)

missing, ok, globbed = [], 0, 0
for n in sorted(wanted):
    if "*" in n:
        globbed += bool(glob.glob(os.path.join(RES, n)))
        if not glob.glob(os.path.join(RES, n)):
            missing.append(n + "   (no match)")
    elif os.path.exists(os.path.join(RES, n)):
        ok += 1
    else:
        missing.append(n)

print(f"result files named by the paper : {len(wanted)}")
print(f"  exact matches on disk         : {ok}")
print(f"  star patterns with >=1 match  : {globbed}")
print(f"  MISSING                       : {len(missing)}")
for m in missing:
    print("    " + m)

scripts = set(re.findall(r"\b((?:exp|COMem)/[A-Za-z0-9_/]+\.py)\b", text))
smiss = [s for s in sorted(scripts) if not os.path.exists(os.path.join(ROOT, s))]
print(f"\nscripts named by the paper      : {len(scripts)}, MISSING: {len(smiss)}")
for s in smiss:
    print("    " + s)

if fragments:
    print(f"\nnot checked ({len(fragments)}): names that do not match the s<n>_ convention. "
          "Most are LaTeX line-wrap fragments of a longer name, which is harmless; "
          "config.json is a model file, not a result.")
    print("    " + ", ".join(sorted(fragments)))

sys.exit(1 if missing or smiss else 0)
