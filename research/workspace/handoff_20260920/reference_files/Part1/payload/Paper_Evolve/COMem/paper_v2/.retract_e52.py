"""RETRACTION. E52/E53's central conclusion was my own error, and the refutation round found it.

WHAT WAS CLAIMED (two ticks ago, and written into 05_experiment, 07_appendix and the
conclusion): that the positions+sink control's value is "a property of the protocol, not of
the model" -- that the LM protocol (E50) rated it worthless on the 8B while the task grid
rated it a large part of the recovery, and that the share therefore "tracks the task, not the
model size".

WHY IT WAS WRONG. `fix_none` is built as `CoMemLower(model, j, tok, lower_layers=[])`
(s15_ruler_lower.py:321-323) and `CoMemLower.__init__` defaults `chunk_write_sink=True`
(:72,77), so `build_bottom` prepends a BOS to every chunk (:142-145). `fix_none` therefore
writes its chunks WITH a sink, exactly as `pub_sink` does. So

    fix_none - pub      = (write-sink effect) + (query-side geometry)      <- what I used
    fix_none - pub_sink = (query-side geometry) alone                      <- the right one

and E50's LM-protocol share references A_on, the write-sink-ON arm. I compared a sink-OFF
task estimand against a sink-ON LM estimand and called the difference a protocol disagreement.

WHAT THE CORRECT CONTRAST SHOWS (exp/s32_geom_share.py, from the same JSONs):
    8B   geometry share positive in 2 of 10 cells, range -118.8% to +5.5%;
         fix_none - pub_sink excludes 0 in 4 cells, ALL NEGATIVE (-28 to -38, every
         multi-key cell).
    1.7B positive in 4 of 6, range -17.9% to +67.7%;
         excludes 0 in 3: single/16k +42.0 [+28,+56], vt 16k -6.8, vt 32k -6.0.

So the geometry HURTS the larger reader and HELPS the smaller one -- which is what E50 said
on the LM protocol. The two protocols AGREE. The "second protocol disagreement" does not
exist; E15b's sparse-set misranking remains the only one.
"""
import sys
from pathlib import Path

E = []

E.append(("sections/05_experiment.tex",
    "\\emph{That checkpoint contrast\nis a property of the protocol, not of the model.} Run as a task arm on both checkpoints, the\n"
    "control recovers a substantial share of the published-to-repaired gap on \\emph{both}, and what\n"
    "the share tracks is the task, not the model size: on single needles it is $54$--$75\\,\\%$ at\n"
    "1.7B and $79$--$90\\,\\%$ at 8B, while on the multi-key and chained tasks together it falls to\n"
    "$-3$--$32\\,\\%$ at 1.7B and $10$--$29\\,\\%$ at 8B (Appendix~\\ref{app:ablation}; these are\n"
    "ratios of cell means, cross-checkpoint and therefore neither paired nor given intervals).\n"
    "The two checkpoints sit in the same band within each task and neither is uniformly above the\n"
    "other. This is the second disagreement between our continuation-KL protocol and RULER --- the first\n"
    "is the sparse-set misranking of \\S\\ref{sec:exp-ablation} --- and both times we defer to the\n"
    "task.",

    "\\emph{The task grid agrees, once both are\nmeasured against the same baseline.} \\texttt{fix\\_none} writes its chunks with the same\n"
    "write-time sink that \\texttt{pub\\_sink} uses, so the contrast isolating the query-side\n"
    "geometry is \\texttt{fix\\_none} $-$ \\texttt{pub\\_sink}, not \\texttt{fix\\_none} $-$\n"
    "\\texttt{pub}; the LM protocol's version of this share references the write-sink-on arm for the\n"
    "same reason. Measured that way the geometry is \\emph{negative} on the 8B --- $-28$, $-30$,\n"
    "$-32$ and $-38$ on the four multi-key cells, every interval excluding zero --- and positive on\n"
    "the 1.7B needles, $+42.0$ [28, 56] on single at 16k. So the positions and the sink help the\n"
    "smaller reader and hurt the larger one, and both protocols say so\n"
    "(Appendix~\\ref{app:ablation}). Visibility therefore carries the whole repair on the 8B and\n"
    "most but not all of it on the 1.7B.",
    "RETRACTED: the protocol-disagreement claim. fix_none is a sink-ON arm, so the share had "
    "to be referenced to pub_sink; done that way the two protocols agree"))

E.append(("sections/06_conclusion.tex",
    "And our own\ncontinuation-KL fidelity metric disagreed with the task protocol twice: once when it ranked\n"
    "sparse layer sets in the order RULER reverses, and once when it rated the positions-and-sink\n"
    "control worthless on the larger checkpoint where the task grid rates it a large part of the\n"
    "recovery. Both times we deferred to the task, and neither disagreement is explained. A\n"
    "metric that decides which layers to cache, and that agrees with the task it is a proxy for,\n"
    "is the thing this paper most conspicuously does not have.",

    "And our own\ncontinuation-KL fidelity metric ranked sparse layer sets in the order RULER reverses, a\n"
    "disagreement we deferred to the task on and did not explain. A metric that decides which\n"
    "layers to cache, and that agrees with the task it is a proxy for, is the thing this paper\n"
    "most conspicuously does not have.",
    "the second disagreement was retracted, so the conclusion no longer claims two"))

for path, old, new, why in E:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    if s.count(old) != 1:
        sys.exit(f"ABORT: {path} pattern occurs {s.count(old)} times:\n{old[:150]!r}")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print(f"  {path}: {why}")
print("done")
