"""Repair the three back-references that pointed into the deleted family list.

"the third line", "no line offers" and "the seven lines" all indexed into the enumeration of
five families that used to sit above the CoMem subsection. With that paragraph gone they point
at nothing, so each is rewritten to name the thing it meant. "the seven lines" keeps its count
because tab:priorart in the appendix really does lay out seven rows -- only the antecedent in
this section is gone, so it is renamed rather than renumbered.
"""
import sys
from pathlib import Path

P = Path(__file__).resolve().parent / "sections/02_related.tex"
s = P.read_text(encoding="utf-8")

EDITS = [
    ("CoMem's mechanism is therefore the third line's\nhidden-state recompute made retrievable and bounded",
     "CoMem's mechanism is therefore hidden-state caching with\nupper-layer recompute, made retrievable and bounded"),
    ("A third thing no line offers\nis a \\emph{depth budget}",
     "A third thing no prior line offers\nis a \\emph{depth budget}"),
    ("Table~\\ref{tab:priorart} in Appendix~\\ref{app:priorart} lays\nthe seven lines out on both axes.",
     "Table~\\ref{tab:priorart} in Appendix~\\ref{app:priorart} lays\nthe seven prior lines out on both axes."),
]
for old, new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"ABORT: {s.count(old)} matches for {old[:50]!r}")
    s = s.replace(old, new)
P.write_text(s, encoding="utf-8")
print("02_related: 3 dangling back-references repaired")
