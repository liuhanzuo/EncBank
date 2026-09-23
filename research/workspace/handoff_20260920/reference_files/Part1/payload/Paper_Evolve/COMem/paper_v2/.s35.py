"""S35: the task-dependence replicates on a second checkpoint, with one exception that scopes E58.

Qwen3-1.7B (L=28) at depth FRACTIONS matched to the 8B's (S27's convention, since L differs):
8B j=12/18/24 = 0.33/0.50/0.67 L  ->  1.7B j=9/14/19 = 0.32/0.50/0.68 L.
Residual fix_all - j0, paired, n=50/cell (exp/s35_analyze.py):

  niah_single_2 16k   -2.0 [-6,0] | -4.0 [-10,0] | -4.0 [-10,0]      all span 0
                32k   +0.0 [0,0]  | -4.0 [-10,0] | -8.0 [-16,-2]     LAST ONE EXCLUDES 0
  niah_multikey 16k  -40.0*       | -68.0*       | -66.0*            all exclude 0
                32k  -40.0*       | -66.0*       | -66.0*            all exclude 0

The 8B's single-needle 16k column is -2.0 / -4.0 / -4.0 -- identical to the 1.7B's. But the
1.7B's single 32k cell at 0.68 L is -8.0 [-16,-2], 4 of 50 samples worse, excluding zero. So
the easy task DOES have a ceiling on the smaller checkpoint, and E58's "no depth ceiling
anywhere in the range this paper prices" is an 8B statement.

The ordering is untouched: multi-key is separably below at every fraction on the 1.7B,
including its own operating point (-40.0 at 0.32 L, where the 8B was -6.0 spanning zero), and
by 58-66 points against single needles' 8.
"""
import sys
from pathlib import Path

p = Path("sections/05_experiment.tex")
s = p.read_text(encoding="utf-8")

old = (
    "Three tasks, two\n"
    "lengths, one checkpoint; $j{=}15$ was measured on the multi-key task only, $j{=}24$ on the\n"
    "two needle tasks only, and whether the same\n"
    "task-dependence holds on a second checkpoint is being measured at matched depth fractions\n"
    "and is not claimed here."
)
new = (
    "Three tasks, two\n"
    "lengths; $j{=}15$ was measured on the multi-key task only and $j{=}24$ on the two needle\n"
    "tasks only.\n"
    "\n"
    "\\emph{The task-dependence replicates on the second checkpoint, and it puts a boundary on the\n"
    "sentence above.} Qwen3-1.7B was run at depth \\emph{fractions} matched to the 8B's --- $j{=}9$,\n"
    "$14$ and $19$ of 28, i.e.\\ $0.32$, $0.50$ and $0.68\\,L$ against $0.33$, $0.50$ and\n"
    "$0.67\\,L$ --- on the two needle tasks, which are the pair that share retrieval, haystack and\n"
    "scoring. The multi-key cells are separably below their own replay bound at \\emph{every}\n"
    "fraction, including the smaller model's own operating point ($-40.0$ [$-54$, $-28$] at\n"
    "$0.32\\,L$, where the 8B was $-6.0$ and did not resolve), and by $66$--$68$ points at the two\n"
    "deeper fractions. The single-needle cells at 16k track the 8B almost exactly: $-2.0$, $-4.0$,\n"
    "$-4.0$ against the 8B's $-2.0$, $-4.0$, $-4.0$, every interval spanning zero. \\emph{But at 32k\n"
    "the smaller model's easy task finally breaks}: $-8.0$ [$-16$, $-2$] at $0.68\\,L$, four of\n"
    "fifty samples worse, the interval excluding zero. So ``the easy task has no depth ceiling''\n"
    "is a Qwen3-8B statement, not a general one; what generalises is the \\emph{ordering}, which\n"
    "holds on both checkpoints and by $58$--$66$ points at the deepest fraction. Two checkpoints,\n"
    "two needle tasks, two lengths; the chained task was not run at these depths on either\n"
    "checkpoint, and cross-checkpoint values are cell means that we never pair or difference."
)
if s.count(old) != 1:
    sys.exit(f"ABORT: {s.count(old)}")
p.write_text(s.replace(old, new), encoding="utf-8")
print("05_experiment: cross-checkpoint replication added, with the 1.7B exception that")
print("  scopes 'no depth ceiling' to the 8B")
