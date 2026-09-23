"""Delete the four-gaps opening of Section 3, and the numbering it anchored.

The owner cut the paragraph that enumerated "four things that cannot currently be done".
That paragraph is the ONLY place the Gap (i)/(ii)/(iii)/(iv) scheme is defined, so five
markers elsewhere -- two in this section, three in the appendix's full motivation -- would be
left indexing into nothing. Each marker is dropped rather than renumbered, because every
subsection it sat in already has a descriptive title that says the same thing:

  Gap (i)             -> "The split depth is a slope, not a knee"
  Gap (i), continued  -> "The loss belongs to the reader, not the memory"
  Gap (ii)            -> "No existing mechanism gives the reader's lower band a retrieved memory"
  Gap (iii)           -> "With the reader repaired, j can become a budget dial"
  Gap (iv)            -> "The operating point must be usable zero-shot"

The section is due for a restructure into two prior-work problems plus one CoMem problem once
the literature survey lands; this pass only removes what the deletion orphaned.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MOT = ROOT / "sections/03_motivation.tex"
s = MOT.read_text(encoding="utf-8")
start = s.index("Section~\\ref{sec:related} closed with four things")
end = s.index("\\subsection{The split depth is a slope, not a knee}")
cut = s[start:end]
s = s[:start] + s[end:]

for old, new in (("\\emph{Gap (i).} The base CoMem paper", "The base CoMem paper"),
                 ("\\emph{Gap (i), continued.} A CoMem read pack", "A CoMem read pack")):
    if s.count(old) != 1:
        sys.exit(f"ABORT motivation: {s.count(old)} matches for {old[:40]!r}")
    s = s.replace(old, new)
MOT.write_text(s, encoding="utf-8")
print(f"03_motivation: opening cut ({len(cut.split())} words), 2 gap markers dropped")

APP = ROOT / "sections/07_appendix.tex"
a = APP.read_text(encoding="utf-8")
for old, new in (("\\emph{Gap (ii): the repair.} Each family", "Each family"),
                 ("\\emph{Gap (iii): the dial.} In the base paper", "In the base paper"),
                 ("\\emph{Gap (iv): the task result.} The base paper", "The base paper")):
    if a.count(old) != 1:
        sys.exit(f"ABORT appendix: {a.count(old)} matches for {old[:40]!r}")
    a = a.replace(old, new)
APP.write_text(a, encoding="utf-8")
print("07_appendix: 3 gap markers dropped")

left = [f"{p.name}:{i}" for p in (MOT, APP)
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "Gap (" in ln]
print("residual 'Gap (' occurrences:", left or "none")
