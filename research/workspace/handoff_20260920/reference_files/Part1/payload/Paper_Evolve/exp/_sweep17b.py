import json, os, glob, random, statistics as st
R = r'F:\Paper_Evolve\exp\results'


def boot_ci(v, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


print('layer  fix_S   arms present')
base = {}
for L in range(9):
    fn = os.path.join(R, 's19_ruler_17b_mk16k_L%d.json' % L)
    if not os.path.exists(fn):
        print(L, 'MISSING'); continue
    d = json.load(open(fn, encoding='utf-8'))
    rows = [r for r in d['rows'] if r['task'] == 'niah_multikey_1' and r['length'] == '16k']
    arms = [k[:-7] for k in rows[0] if k.endswith('_recall')]
    m = {a: 100.0 * st.mean(r[a + '_recall'] for r in rows) for a in arms}
    base[L] = (rows, m)
    print('%-6d %-7.1f %s  fix_layers=%s n=%d' % (
        L, m.get('fix_S', float('nan')), {k: round(v, 1) for k, v in m.items()},
        d.get('fix_layers'), len(rows)))

print()
print('paired fix_S - pub_sink  (the control the 8B sweep used):')
for L in sorted(base):
    rows, m = base[L]
    if 'fix_S' not in m or 'pub_sink' not in m:
        print(L, 'arms missing:', sorted(m)); continue
    d = [100.0 * (r['fix_S_recall'] - r['pub_sink_recall']) for r in rows]
    b = sum(1 for x in d if x > 0); w = sum(1 for x in d if x < 0)
    if b == 0 and w == 0:
        print('  l=%d  %+6.1f (all ties)' % (L, st.mean(d)))
    else:
        lo, hi = boot_ci(d)
        print('  l=%d  %+6.1f [%.1f, %.1f]  (%d better / %d worse)' % (L, st.mean(d), lo, hi, b, w))
