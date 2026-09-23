"""Resolve the label collision the motivation restructure created, and add the new bib entries.

Problems 1 and 2 moved from the appendix into Section 3, so sec:mot:visible and sec:mot:dial are
now defined twice. The main text keeps the short names (every \\ref to them wants the argument,
not the long form) and the appendix long forms are renamed app:mot:*. sec:mot:task never moved,
but it is renamed for consistency with the other two appendix blocks, and its four references are
updated with it.

Bib entries: ten copied verbatim from the survey's verifier agents (which returned BibTeX only
when they had opened the arXiv/ACL record), plus three whose metadata I confirmed myself on the
arXiv abs pages on 2026-09-07 -- Memory Inception 2605.06225, xRAG 2405.13792, SqueezeAttention
2404.04793. SqueezeAttention is filed in its arXiv form: the survey reported ICLR 2025 for it but
I did not confirm that on the proceedings page, so the venue is not asserted.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "sections/07_appendix.tex"

a = APP.read_text(encoding="utf-8")
for old, new in (("\\label{sec:mot:visible}", "\\label{app:mot:visible}"),
                 ("\\label{sec:mot:dial}", "\\label{app:mot:dial}"),
                 ("\\label{sec:mot:task}", "\\label{app:mot:task}")):
    if a.count(old) != 1:
        sys.exit(f"ABORT: {a.count(old)} defs of {old}")
    a = a.replace(old, new)
APP.write_text(a, encoding="utf-8")
print("07_appendix: 3 labels renamed to app:mot:*")

# sec:mot:task moved, so every reference to it has to move too.
n = 0
for p in sorted((ROOT / "sections").glob("*.tex")) + [ROOT / "main.tex"]:
    t = p.read_text(encoding="utf-8")
    if "ref{sec:mot:task}" not in t:
        continue
    k = t.count("ref{sec:mot:task}")
    p.write_text(t.replace("ref{sec:mot:task}", "ref{app:mot:task}"), encoding="utf-8")
    print(f"  {p.name}: {k} ref(s) -> app:mot:task")
    n += k
print(f"{n} references retargeted")

BIB = ROOT / "qcmem.bib"
b = BIB.read_text(encoding="utf-8")
new = (ROOT / ".new_entries.bib").read_text(encoding="utf-8")
added = re.findall(r"@\w+\s*\{\s*([^,\s]+)", new)
dupes = [k for k in added if re.search(r"@\w+\s*\{\s*" + re.escape(k) + r"\s*,", b)]
if dupes:
    sys.exit(f"ABORT: already in bib: {dupes}")
BIB.write_text(b.rstrip() + "\n\n" + new.strip() + "\n", encoding="utf-8")
print(f"qcmem.bib: +{len(added)} entries ({', '.join(added)})")
