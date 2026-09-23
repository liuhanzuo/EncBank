"""Condense the related-work survey's 176 KB result into something readable.

The verifiers were told to return long adversarial arguments, so the raw JSON is unusable at
the console. This prints one line per surviving citation and the full text of the two
Critique-phase agents, which are the parts that decide what Section 2 and Section 3 say.
"""
import json
import re
import textwrap
from pathlib import Path

OUT = Path(r"/srv/encbank/client\AppData\Local\Temp\claude\F--Paper-Evolve"
           r"\2216fe1d-3e3a-4dc2-91c5-d005db387464\tasks\wjr5cxo2l.output")
res = json.loads(OUT.read_text(encoding="utf-8", errors="replace"))["result"]


def first(s, n=220):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:n]


print(f"SURVIVED: {len(res['surviving'])}   KILLED: {len(res['killed'])}\n")
print("=" * 100)
print("SURVIVING CITATIONS")
print("=" * 100)
by_axis = {}
for c in res["surviving"]:
    by_axis.setdefault(c["axis"], []).append(c)
for axis, cs in by_axis.items():
    print(f"\n### {axis}")
    for c in cs:
        t = first(c["title"], 95)
        ax = first(c["arxiv"], 40)
        print(f"  - {t}")
        print(f"      arXiv: {ax}")
        print(f"      mech : {first(c['mechanism'], 260)}")

print("\n" + "=" * 100)
print("KILLED (proposed, then rejected)")
print("=" * 100)
for k in res["killed"]:
    tag = "DOES NOT EXIST" if not k["exists"] else "irrelevant"
    print(f"  [{tag:14s}] {first(k['title'], 80)}")
    print(f"      {first(k['why'], 200)}")

for key in ("missing", "plan"):
    print("\n" + "=" * 100)
    print(key.upper())
    print("=" * 100)
    print(textwrap.fill(re.sub(r"\n{3,}", "\n\n", res[key]), 98,
                        replace_whitespace=False, drop_whitespace=False))
