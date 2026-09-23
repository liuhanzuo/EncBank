"""Structure audit for paper_v2 section files: headings, labels and stray CRs."""
import re
import sys
from pathlib import Path

for name in sys.argv[1:]:
    p = Path(name)
    s = p.read_text(encoding="utf-8", newline="")
    print(f"=== {name}")
    for m in re.finditer(r"\\(sub)?section\*?\{([^}]*)\}", s):
        kind = "subsection" if m.group(1) else "section"
        print(f"   {kind:11s} {m.group(2)[:64]}")
    labels = re.findall(r"\\label\{((?:app|tab|sec)[^}]*)\}", s)
    print("   labels:", ", ".join(labels))
    lone_cr = len(re.findall(r"\r(?!\n)", s))
    print(f"   CRLF {s.count(chr(13)+chr(10))}  lone-CR {lone_cr}")
