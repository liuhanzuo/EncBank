from pathlib import Path
import sys

p = Path("CLAIM_EVIDENCE.md")
s = p.read_text(encoding="utf-8")
old = "j=9 and j=24 remain unmeasured on the task. |"
new = ("j=9 and j=24 remain unmeasured on the task. **Internal check run 2026-09-07 "
       "(`exp/s33_j0_check.py`):** `j0` sets `resume_j=0`, so it should not depend on the split "
       "at all; compared **sample by sample** across j=6/12/15/18 it is identical on **50 of 50 "
       "samples in both multi-key cells**, 0 samples differing. That is stronger than the equal "
       "means the analyzer prints -- four runs could reach 98.0 by getting different samples "
       "right -- and it simultaneously confirms the four independent runs scored the same "
       "samples and that the bound the residual is measured against does not move with j. This "
       "is deliberately the check the E54 failure would have needed: verify the comparison is "
       "single-factor from the data, not from the arm's name. |")
if s.count(old) != 1:
    sys.exit(f"ABORT: {s.count(old)}")
p.write_text(s.replace(old, new), encoding="utf-8")
print("E56: j0 depth-independence check recorded")
