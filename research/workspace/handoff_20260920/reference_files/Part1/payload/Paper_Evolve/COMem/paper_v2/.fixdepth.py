"""Apply the depth-series refutation findings (wf_c73a4388-07f, 20 confirmed of 35).

All verified by me against the files before applying.

A. STORAGE EDGE (findings 1/6/10/12, the last rated fatal). "j=30 ... is therefore the last
   depth at which this method stores less than the thing it replaces" is arithmetically FALSE.
   The paper's own law is 8+4j KB against 144 KB full-depth, so 8+4j < 144 for every j <= 33:
   j=31/32/33 store 132/136/140 KB and none was run. j=34 is the first tie at exactly 144.
   j=30 is the deepest depth I RAN, not the edge of the storage argument.

B. C3's RANGE (2/9/11/13). "spanning every split we tested (0.17--0.67 L) on single needles"
   is wrong at both ends. Enumerated from every 8B result JSON: niah_single_2 exists at
   j=12/18/24/30 only -> 0.33--0.83 L. j=6 (0.17 L) is a multikey-only run, and 0.67 L is
   stale four lines after the j=30 result is reported.

C. "BOTH AT 0.0" (3/7). True only at j=24 and j=30. At j=12 on single, pub=38.0 and
   pub_sink=84.0/96.0; at j=18, pub_sink=12.0/24.0. The sentence names the range "from the
   operating point to j=30", over which it is false.

D. THE GAP (4). "58--66 points" is not attained: single-vs-multikey at the deepest 1.7B
   fraction is -8.0 vs -66.0 (32k) = 58 and -4.0 vs -66.0 (16k) = 62. Range is 58--62; the 66
   was the multikey residual magnitude, not a gap.

F. NEEDLE COUNT (20). COMem/eval/ruler.py:213-214 builds niah_multikey_1 with num_k = 4, and
   niah_single_2 with num_k = 1. The appendix calls them "the one- and two-key tasks".
"""
import sys
from pathlib import Path

E = []

# ---- A + the "both at 0.0" over-generalisation (C) ------------------------------------
E.append(("sections/05_experiment.tex",
    "\\emph{The two tasks never converge, out to the edge of the storage argument itself.} We\n"
    "ran the split out to $j{=}30$ of 36 ($0.83L$), which stores $128$\\,KB/token against the\n"
    "$144$\\,KB full-depth reference and is therefore the last depth at which this method stores\n"
    "less than the thing it replaces.",

    "\\emph{The two tasks never converge at any depth we measured.} We\n"
    "ran the split out to $j{=}30$ of 36 ($0.83L$), which stores $128$\\,KB/token against the\n"
    "$144$\\,KB full-depth reference. That is the deepest split we ran and \\emph{not} the end of\n"
    "the priceable range: $8{+}4j$ stays below $144$\\,KB up to $j{=}33$ ($140$\\,KB), so\n"
    "$j{=}31$--$33$ ($0.86$--$0.92\\,L$) are inside the storage argument and untested.",
    "A: j=30 is the deepest run, not the storage edge -- 8+4j<144 up to j=33"))

E.append(("sections/05_experiment.tex",
    "times less arithmetic per query --- at no measurable cost in recall. At those depths the\n"
    "published arm and the write-sink arm are both at $0.0$, so all of that recall is the repair.",

    "times less arithmetic per query --- at no measurable cost in recall. At the two deepest of\n"
    "those splits, $j{=}24$ and $j{=}30$, the published arm and the write-sink arm are both at\n"
    "$0.0$, so there all of that recall is the repair; at $j{=}12$ and $j{=}18$ the write-sink\n"
    "arm is still scoring ($84.0$/$96.0$ and $12.0$/$24.0$), so the repair's share is smaller.",
    "C: 'both at 0.0' holds only at j=24 and j=30, not over the range the sentence names"))

# ---- B: C3's restatement ---------------------------------------------------------------
E.append(("sections/05_experiment.tex",
    "a usable range whose upper end is set by the task, spanning every split we tested\n"
    "($0.17$--$0.67\\,L$) on single needles and ending at the base method's own split on the\n"
    "multi-key task.} Three tasks, two lengths; $j{=}15$ was measured on the\n"
    "multi-key task only and $j{=}24$ on the two needle tasks only.",

    "a usable range whose upper end is set by the task, spanning every split we tested on single\n"
    "needles ($0.33$--$0.83\\,L$) and ending at the base method's own split on the multi-key\n"
    "task.} Three tasks, two lengths, and an unbalanced depth grid we state rather than smooth\n"
    "over: $j{=}6$ and $j{=}15$ were measured on the multi-key task only, $j{=}24$ and $j{=}30$\n"
    "on the two needle tasks only, and \\texttt{variable\\_tracking} only at $j{=}12$ and\n"
    "$j{=}18$.",
    "B: single needles were run at 0.33-0.83 L, never at 0.17 L; and the grid is unbalanced "
    "at four depths, not two"))

# ---- D: the gap range ------------------------------------------------------------------
E.append(("sections/05_experiment.tex",
    "\\emph{ordering}, which\nholds on both checkpoints and by $58$--$66$ points at the deepest\nfraction.",
    "\\emph{ordering}, which\nholds on both checkpoints and by $58$--$62$ points at the deepest\nfraction.",
    "D: the single-minus-multikey gap at the deepest 1.7B fraction is 58 (32k) and 62 (16k); "
    "66 was the multikey residual magnitude, not a gap"))

# ---- F: needle count -------------------------------------------------------------------
E.append(("sections/07_appendix.tex",
    "on the one- and two-key tasks and $+4.8$/$+3.6$ on the five-needle one",
    "on the one- and four-key tasks and $+4.8$/$+3.6$ on the five-needle one",
    "F: ruler.py:213-214 builds niah_multikey_1 with num_k=4, not 2"))

for path, old, new, why in E:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    if s.count(old) != 1:
        sys.exit(f"ABORT {path} ({why}): {s.count(old)} matches\n{old[:130]!r}")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print(f"  {why}")

# ---- the same stale range is baked into an analysis script's docstring ------------------
q = Path("../../exp/s35_analyze.py")
t = q.read_text(encoding="utf-8")
oldd = "the single-needle cells sit at the j=0 replay bound at every split tested, 0.17-0.67 L"
newd = "the single-needle cells sit at the j=0 replay bound at every split tested on them, 0.33-0.83 L"
if t.count(oldd) == 1:
    q.write_text(t.replace(oldd, newd), encoding="utf-8")
    print("  B: exp/s35_analyze.py docstring carried the same wrong range -- fixed")
else:
    print(f"  NOTE: s35_analyze docstring matched {t.count(oldd)}x")
print("done")
