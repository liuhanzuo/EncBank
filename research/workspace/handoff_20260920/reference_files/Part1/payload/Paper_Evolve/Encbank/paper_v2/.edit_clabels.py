"""Undo the C1 deletion; drop the C1/C2/C3/C4 labels instead.

The owner asked for the contribution list's "C1./C2./..." HEADERS to go, not for the
first contribution to be deleted -- I misread that last tick and cut the contribution.
This restores it, replaces the labelled enumerate with a plain itemize, and strips the
inline provenance parentheticals from the list items ("(\\S\\ref{sec:method},
\\S\\ref{sec:exp-scale}; Table~\\ref{tab:mechanism})"), which the owner also rejected.

With the numbers gone from the list, every "(Cn)" citation elsewhere would dangle, so
each one is rewritten to name the thing instead: the decomposition, the repair, the dial,
the task result. That also reverts last tick's renumbering, which is why the map below is
by MEANING at each site rather than by a global +1.

Every replacement is exact-match and asserted to occur exactly once; the script aborts
without writing if any count is wrong.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 1. the contribution list
INTRO = ROOT / "sections/01_introduction.tex"
s = INTRO.read_text(encoding="utf-8")

old_open = "\\begin{enumerate}[leftmargin=1.6em,itemsep=2pt,label=\\textbf{C\\arabic*.}]\n"
new_open = "\\begin{itemize}[leftmargin=1.2em,itemsep=2pt]\n"
c1 = (
    "\\item \\textbf{The zero-shot depth loss is monotone and entirely query-side.}\n"
    "The published write loses fidelity at every step of the depth grid, with no plateau\n"
    "on the grid. The decomposition then splits that loss\n"
    "cleanly: the chunk side contributes almost none of it at any depth we test, so the cached\n"
    "residual is a sufficient upper-band input and it is the reader that fails. This holds on\n"
    "\\emph{both} checkpoints.\n"
)

EDITS = [
    # restore the deleted contribution and unlabel the list
    (old_open, new_open + c1, "itemize + restore the diagnosis contribution"),
    ("\\end{enumerate}\n", "\\end{itemize}\n", "close itemize"),
    # strip the three inline provenance parentheticals the owner rejected
    ("the gap on their own (\\S\\ref{sec:method}, \\S\\ref{sec:exp-scale};"
     " Table~\\ref{tab:mechanism}).",
     "the gap on their own.", "provenance out of the repair item"),
    ("We\ntherefore recommend the arm that selects nothing (\\S\\ref{sec:exp-dial},"
     " \\S\\ref{sec:exp-ablation};\nTable~\\ref{tab:sparse}).",
     "We\ntherefore recommend the arm that selects nothing.",
     "provenance out of the dial item"),
    ("we state that here rather than in the limitations\n"
     "(\\S\\ref{sec:exp-main}, \\S\\ref{sec:exp-scale}; Tables~\\ref{tab:main}"
     " and~\\ref{tab:scale}).",
     "we state that here rather than in the limitations.",
     "provenance out of the task item"),
    # the deleted block is gone; the header comment still pointed at it
    ("with no interval and are never called paired (see the \"does not claim\" block).",
     "with no interval and are never called paired.", "stale comment pointer"),
]
for old, new, why in EDITS:
    if s.count(old) != 1:
        sys.exit(f"ABORT intro [{why}]: {s.count(old)} matches")
    s = s.replace(old, new)
INTRO.write_text(s, encoding="utf-8")
print("01_introduction: contribution restored, list unlabelled, 3 provenance refs stripped")

# ---------------------------------------------------------------- 2. name, don't number
BY_FILE = {
    "sections/02_related.tex": [
        ("the unmet need behind contribution C2\nis a depth setting",
         "the unmet need behind our dial contribution\nis a depth setting"),
    ],
    "sections/03_motivation.tex": [
        # this subsection continues Gap (i); the appendix owns (ii)--(iv)
        ("\\emph{Gap (ii); sets up C1.} A Encbank read pack",
         "\\emph{Gap (i), continued.} A Encbank read pack"),
        ("This is the requirement contribution C1 meets.",
         "This is the requirement our repair meets."),
    ],
    "sections/04_methodology.tex": [
        ("the retrieved memory (C1); \\S\\ref{subsec:accounting}",
         "the retrieved memory; \\S\\ref{subsec:accounting}"),
        ("the sparse layer set $S$ (C2); \\S\\ref{subsec:instantiation}",
         "the sparse layer set $S$; \\S\\ref{subsec:instantiation}"),
        ("RULER table of \\S\\ref{sec:experiments} measures (C3).",
         "RULER table of \\S\\ref{sec:experiments} measures."),
        ("This subsection addresses \\S\\ref{sec:mot:visible} (C1): KV-reuse",
         "This subsection addresses \\S\\ref{sec:mot:visible}: KV-reuse"),
        ("This subsection addresses \\S\\ref{sec:mot:dial} (C2): in the base paper",
         "This subsection addresses \\S\\ref{sec:mot:dial}: in the base paper"),
        ("This subsection addresses \\S\\ref{sec:mot:task} (C3): the base paper's",
         "This subsection addresses \\S\\ref{sec:mot:task}: the base paper's"),
    ],
    "sections/05_experiment.tex": [
        ("table (\\S\\ref{sec:exp-main}) measures C3 on the base paper's own task driver",
         "table (\\S\\ref{sec:exp-main}) is the task result on the base paper's own driver"),
        ("the repair curve (\\S\\ref{sec:exp-dial}) measures C1 and C2, the removal of the",
         "the repair curve (\\S\\ref{sec:exp-dial}) measures the removal of the"),
        ("It answers C3 and, read against\n\\texttt{cbos}, part of C2.",
         "It is the task result, and read against\n\\texttt{cbos} it also bears on the dial."),
        ("Figure~\\ref{fig:mechanism}(a) measures C1 on the LM protocol",
         "Figure~\\ref{fig:mechanism}(a) measures the decomposition on the LM protocol"),
        ("Table~\\ref{tab:mechanism}(b) measure C1 and C2 with the",
         "Table~\\ref{tab:mechanism}(b) measure the repair and the dial with the"),
        ("We therefore restate C2:", "We therefore restate the dial claim:"),
    ],
    "sections/06_conclusion.tex": [
        ("plateau (\\S\\ref{sec:experiments}, C1), yet", "plateau (\\S\\ref{sec:experiments}), yet"),
        ("query-side term (C1).", "query-side term."),
        ("trained (\\S\\ref{sec:method}, \\S\\ref{sec:experiments}, C1). With the",
         "trained (\\S\\ref{sec:method}, \\S\\ref{sec:experiments}). With the"),
        ("deepest point (C2). On the base paper's own",
         "deepest point. On the base paper's own"),
        ("not the sole one (C3); on the chained", "not the sole one; on the chained"),
        ("direction of that result and not its completeness (C3, Table~\\ref{tab:scale}).",
         "direction of that result and not its completeness (Table~\\ref{tab:scale})."),
    ],
    "sections/07_appendix.tex": [
        ("dial statement of C3 is that", "dial statement is that"),
        ("\\emph{Gap (ii), contribution C2.} Each family", "\\emph{Gap (ii): the repair.} Each family"),
        ("\\emph{Gap (iii), contribution C3.} In the base paper", "\\emph{Gap (iii): the dial.} In the base paper"),
        ("\\emph{Gap (iv), contribution C4.} The base paper", "\\emph{Gap (iv): the task result.} The base paper"),
        ("the comparison that would falsify C4. The base paper states",
         "the comparison that would falsify the task claim. The base paper states"),
    ],
    # LaTeX comments: provenance notes that named contributions by number
    "sections/00_abstract.tex": [
        ("had been dropped here while C4 and sec:exp-main kept",
         "had been dropped here while the task contribution and sec:exp-main kept"),
    ],
    "sections/08_limitations.tex": [
        ("lives in the main text (01_introduction C4,",
         "lives in the main text (01_introduction, task contribution;"),
    ],
    "main.tex": [
        ("used in the abstract, the C4 contribution and the",
         "used in the abstract, the task contribution and the"),
    ],
}
for rel, edits in BY_FILE.items():
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    for old, new in edits:
        if t.count(old) != 1:
            sys.exit(f"ABORT {rel}: {t.count(old)} matches for {old[:60]!r}")
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print(f"{rel}: {len(edits)} C-number reference(s) named instead")

# the three remaining intro comments name sparse-set / scale provenance by C-number
c = INTRO.read_text(encoding="utf-8")
for old, new in (("% C2 sparse-set numbers:", "% Sparse-set numbers:"),
                 ("% C2 control contrast:", "% Control contrast:"),
                 ("% C3 second checkpoint", "% Second checkpoint"),
                 ("the two contrasts that scope C3:", "the two contrasts that scope it:")):
    if c.count(old) != 1:
        sys.exit(f"ABORT intro comment: {c.count(old)} for {old!r}")
    c = c.replace(old, new)
INTRO.write_text(c, encoding="utf-8")
print("01_introduction: 4 comment headers de-numbered")
