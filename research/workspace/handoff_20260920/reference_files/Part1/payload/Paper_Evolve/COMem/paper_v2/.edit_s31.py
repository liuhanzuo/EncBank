"""Fold s31's 8B fix_none cells in, and correct two things I got wrong last tick.

s31 ran the control on Qwen3-8B where it had never been run: variable_tracking at 16k/32k
(complete) and niah at 8k (complete). The 64k block is still running and is NOT used here.

TWO CORRECTIONS, both to text I wrote last tick:
 1. "under a tenth on the chained task, where the 8B has no such cell" is now FALSE. The 8B
    has the cells and they are 20.3% and 15.4% -- the leg of the claim s31 was launched to
    test is the leg that failed.
 2. "The share is larger on the bigger model in three of the four cells where both exist" is
    an ARITHMETIC ERROR of mine. It is 2 of 4 (the 8B leads on both single-needle cells, the
    1.7B on both multikey cells). With the two new chained cells it becomes 4 of 6.
"""
import sys
from pathlib import Path

EDITS = []

# ---------------------------------------------------------------- 05_experiment
EDITS.append(("sections/05_experiment.tex",
    "control recovers a substantial share of the published-to-repaired gap on \\emph{both}, and the\n"
    "share tracks the task rather than the model size: $54$--$75\\,\\%$ at 1.7B against $79$--$90\\,\\%$\n"
    "at 8B on single needles, $14$--$32\\,\\%$ against $10$--$16\\,\\%$ on multikey, and under a tenth\n"
    "on the chained task, where the 8B has no such cell (Appendix~\\ref{app:ablation}; these are\n"
    "ratios of cell means, cross-checkpoint and therefore neither paired nor given intervals).",

    "control recovers a substantial share of the published-to-repaired gap on \\emph{both}, and what\n"
    "the share tracks is the task, not the model size: on single needles it is $54$--$75\\,\\%$ at\n"
    "1.7B and $79$--$90\\,\\%$ at 8B, while on the multi-key and chained tasks together it falls to\n"
    "$-3$--$32\\,\\%$ at 1.7B and $10$--$29\\,\\%$ at 8B (Appendix~\\ref{app:ablation}; these are\n"
    "ratios of cell means, cross-checkpoint and therefore neither paired nor given intervals).\n"
    "The two checkpoints sit in the same band within each task and neither is uniformly above the\n"
    "other.",
    "s31 supplied the 8B chained cells (20.3%, 15.4%), which refutes 'under a tenth on the "
    "chained task'; the claim's shape survives but its third leg does not"))

# ---------------------------------------------------------------- 07_appendix
EDITS.append(("sections/07_appendix.tex",
    "the control reaches $75.0\\,\\%$/$54.1\\,\\%$ on 1.7B single needles against $79.3\\,\\%$/$89.7\\,\\%$\n"
    "on the 8B, and $32.1\\,\\%$/$13.8\\,\\%$ on 1.7B multikey against $15.6\\,\\%$/$10.4\\,\\%$ on the 8B.\n"
    "The share is larger on the \\emph{bigger} model in three of the four cells where both exist.",

    "the control reaches $75.0\\,\\%$/$54.1\\,\\%$ on 1.7B single needles against $79.3\\,\\%$/$89.7\\,\\%$\n"
    "on the 8B, $32.1\\,\\%$/$13.8\\,\\%$ on 1.7B multikey against $15.6\\,\\%$/$10.4\\,\\%$ on the 8B, and\n"
    "$7.5\\,\\%$/$-3.1\\,\\%$ on the 1.7B chained task against $20.3\\,\\%$/$15.4\\,\\%$ on the 8B (S31).\n"
    "The share is larger on the \\emph{bigger} model in four of the six cells where both exist ---\n"
    "\\emph{corrected}: an earlier version of this paragraph said three of four, which was an\n"
    "arithmetic slip; over the original four cells it was two, the 8B leading on both single-needle\n"
    "cells and the 1.7B on both multikey cells.",
    "adds the s31 chained cells and corrects my 'three of four' to the true 2 of 4 / 4 of 6"))

EDITS.append(("sections/07_appendix.tex",
    "the geometry is worth a lot on saturated single-needle cells, a little on multikey and",
    "the geometry is worth a lot on saturated single-needle cells and much less on multikey and",
    "'a little on multikey and nothing on the chained task' understated the 8B chained cells"))

for path, old, new, why in EDITS:
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    n = src.count(old)
    if n != 1:
        sys.exit(f"ABORT: pattern occurs {n} times in {path} (need 1):\n{old[:160]!r}")
    p.write_text(src.replace(old, new), encoding="utf-8")
    print(f"  {path}: {why}")

# the trailing clause of that appendix sentence, checked separately because it spans a line
p = Path("sections/07_appendix.tex")
s = p.read_text(encoding="utf-8")
old = "nothing on the chained task, on both checkpoints."
new = ("on the chained task, on both checkpoints -- though on the 8B the chained cells are "
       "$20.3\\,\\%$ and $15.4\\,\\%$, not the near-zero the 1.7B shows.")
if s.count(old) == 1:
    p.write_text(s.replace(old, new), encoding="utf-8")
    print("  sections/07_appendix.tex: chained-task summary now names the 8B values")
else:
    print(f"  NOTE: trailing clause matched {s.count(old)} times, left unchanged -- check by hand")
print("done")
