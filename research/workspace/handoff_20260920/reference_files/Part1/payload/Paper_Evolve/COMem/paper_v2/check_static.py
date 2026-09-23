import re, glob, collections, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob('sections/*.tex')) + ['main.tex'] if os.path.exists('main.tex') else sorted(glob.glob('sections/*.tex'))
refs = collections.Counter(); labels = collections.Counter(); cites = set()
for f in files:
    t = open(f, encoding='utf8').read()
    t = re.sub(r'(?<!\\)%.*', '', t)
    for m in re.finditer(r'\\(?:ref|eqref|autoref)\{([^}]*)\}', t): refs[m.group(1)] += 1
    for m in re.finditer(r'\\label\{([^}]*)\}', t): labels[m.group(1)] += 1
    for m in re.finditer(r'\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}', t):
        for k in m.group(1).split(','): cites.add(k.strip())
    for m in re.finditer(r'\\input\{([^}]*)\}', t): print(f, 'inputs', m.group(1))
    b = collections.Counter(re.findall(r'\\begin\{([^}]*)\}', t)); e = collections.Counter(re.findall(r'\\end\{([^}]*)\}', t))
    if b != e: print('ENV MISMATCH', f, b, e)
print('MISSING LABELS', [r for r in refs if r not in labels])
print('DUP LABELS', [l for l, c in labels.items() if c > 1])
print('UNREFERENCED LABELS', [l for l in labels if l not in refs])
bib = open('qcmem.bib', encoding='utf8').read()
keys = set(re.findall(r'@\w+\{([^,]*),', bib))
print('MISSING CITES', [c for c in cites if c not in keys])
print('CITED', sorted(cites))
