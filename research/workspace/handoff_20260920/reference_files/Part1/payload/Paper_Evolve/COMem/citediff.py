"""Which of CoMem's citations did paper_v2 drop, and which of its own bib entries are dead?

Comment-stripped so that provenance notes in LaTeX comments do not count as citations.
"""
import glob
import re

CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")


def keys(path):
    return {m.group(1) for m in KEY.finditer(open(path, encoding="utf-8").read())}


def cited(files):
    out = set()
    for f in files:
        s = re.sub(r"(?m)^\s*%.*$", "", open(f, encoding="utf-8").read())
        for m in CITE.finditer(s):
            out |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    return out


v1b, v2b = keys("paper/qcmem.bib"), keys("paper_v2/qcmem.bib")
v1c = cited(glob.glob("paper/sections/*.tex") + ["paper/main.tex"])
v2c = cited(glob.glob("paper_v2/sections/*.tex") + ["paper_v2/main.tex"])

title = {}
for p in ("paper/qcmem.bib", "paper_v2/qcmem.bib"):
    s = open(p, encoding="utf-8").read()
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", s, re.S):
        t = re.search(r"title\s*=\s*[{\"](.+?)[}\"],?\s*\n", m.group(2), re.S)
        if t:
            title.setdefault(m.group(1), re.sub(r"\s+", " ", t.group(1)).strip("{} "))

print(f"CoMem cites {len(v1c)}, paper_v2 cites {len(v2c)}\n")
print("DROPPED -- CoMem cites these, paper_v2 does not:")
for k in sorted(v1c - v2c):
    print(f"  {k:20s} {title.get(k, '?')[:78]}")
print("\nDEAD -- in paper_v2's bib but never cited:")
for k in sorted(v2b - v2c):
    print(f"  {k:20s} {title.get(k, '?')[:78]}")
print("\nADDED -- paper_v2 cites these, CoMem does not:")
for k in sorted(v2c - v1c):
    print(f"  {k:20s} {title.get(k, '?')[:78]}")
