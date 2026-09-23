"""Delete the pre-Encbank material from Section 2, keeping the one thing in it that was load-bearing.

The owner asked for everything before the Encbank subsection to go. Two paragraphs are affected:

  (a) the storage-units paragraph ("a residual hidden is 8 KB per token, one layer of K+V is
      4 KB, full-depth KV is 144 KB"). Safe to drop: 04_methodology.tex eq:storage defines all
      three quantities properly, as store(j,S) = 2d + |S|*4 n_kv d_h, store_all(j) = 8+4j KB,
      store_full KV = 4L = 144 KB. Section 2 only used them informally.

  (b) the pointer paragraph naming the five families and sending them to app:method. Also
      dropped -- Section 2 is about to be rebuilt with real content, and sec:rw-table still
      reaches the appendix through Table~\\ref{tab:priorart}.

But (a) carried a FOOTNOTE that is a verification disclosure, not exposition: which prior-art
descriptions were checked against the base method's released re-implementations, which were
checked only against the papers, which arXiv records were checked and on what date, and which
were checked against no code at all. Deleting that silently would drop a provenance statement,
so it moves verbatim into the appendix, at the head of the five-family write-up it describes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REL = ROOT / "sections/02_related.tex"
s = REL.read_text(encoding="utf-8")

start = s.index("Every method below reuses some persistent object")
end = s.index("\\subsection{Encbank: the base method}")
cut = s[start:end]
if "\\footnote{Descriptions of CacheBlend" not in cut:
    sys.exit("ABORT: the footnote is not inside the block being cut -- bounds are wrong")

fs = cut.index("\\footnote{Descriptions of CacheBlend")
fe = cut.index("none of these was checked against released code in this work.}") + len(
    "none of these was checked against released code in this work.}")
foot = cut[fs + len("\\footnote{"):fe - 1]

s = s[:start] + s[end:]
REL.write_text(s, encoding="utf-8")
print(f"02_related: cut {len(cut.split())} words before the Encbank subsection "
      f"(units paragraph + footnote + appendix pointer)")

APP = ROOT / "sections/07_appendix.tex"
a = APP.read_text(encoding="utf-8")
anchor = "\\subsection{Token-axis KV compression}\n\\label{sec:rw-token}\n"
if a.count(anchor) != 1:
    sys.exit(f"ABORT appendix: {a.count(anchor)} matches for the anchor")

moved = (
    "\\subsection{What was checked against code, and what only against papers}\n"
    "\\label{sec:rw-provenance}\n"
    + foot + "\n\n"
)
a = a.replace(anchor, moved + anchor)
APP.write_text(a, encoding="utf-8")
print(f"07_appendix: verification disclosure moved in as sec:rw-provenance "
      f"({len(foot.split())} words), ahead of sec:rw-token")
