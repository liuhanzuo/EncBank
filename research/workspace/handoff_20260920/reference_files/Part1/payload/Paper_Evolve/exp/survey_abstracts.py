"""Pull the abstract out of every unique manuscript in research_portfolio/.

38 paper.pdf files there are only 24 distinct documents -- the same frozen face is
presented under manuscripts/ (rule B0 snapshot), submitted/ (rule B archive) and
external_best/ (top external scores).  Dedupe by PDF sha256 first so each paper is
read once.
"""

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def strip_tex(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    for _ in range(6):  # unwrap nested one-arg macros a few times
        s = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc|mbox|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:sconf|finj|ainj|mcu|rinj)\{?\}?", lambda m: m.group(0)
               .replace("\\sconf", "s_conf").replace("\\finj", "fixed_inject")
               .replace("\\ainj", "anti_inject").replace("\\mcu", "mc_uniform")
               .replace("\\rinj", "rand_inject").replace("{}", ""), s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    s = s.replace("``", '"').replace("''", '"').replace("--", "-")
    return re.sub(r"\s+", " ", s).strip()


def abstract(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if m:
        return strip_tex(m.group(1))
    m = re.search(r"\\section\{Introduction\}(.*?)\\section", tex, re.S)
    return strip_tex(m.group(1))[:900] if m else "(no abstract)"


def title(tex: str) -> str:
    m = re.search(r"\\title\s*\{", tex)
    if not m:
        return "(no title)"
    i, d, buf = m.end(), 1, []
    while i < len(tex) and d:
        c = tex[i]
        d += (c == "{") - (c == "}")
        if d:
            buf.append(c)
        i += 1
    return strip_tex("".join(buf))


groups = defaultdict(list)
for pdf in sorted(ROOT.glob("*/*/paper.pdf")):
    groups[hashlib.sha256(pdf.read_bytes()).hexdigest()[:8]].append(pdf.parent)

for h, dirs in sorted(groups.items(), key=lambda kv: kv[1][0].name.lower()):
    tex = (dirs[0] / "paper.tex").read_text(encoding="utf-8", errors="ignore")
    where = ",".join(sorted({d.parent.name for d in dirs}))
    print(f"\n{'=' * 100}\n[{h}] {dirs[0].name}   <{where}>")
    print(f"TITLE: {title(tex)}")
    print(f"ABSTRACT: {abstract(tex)[:1400]}")
