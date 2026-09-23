import json, os, random, statistics as st
R = r'F:\Paper_Evolve\exp\results'


def boot_ci(v, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


main = json.load(open(os.path.join(R, 's19_ruler_17b_niah.json'), encoding='utf-8'))
mrows = [r for r in main['rows']
         if r['task'] == 'niah_multikey_1' and r['length'] == '16k']
mrows.sort(key=lambda r: r['i'])

print('sample alignment check (answers + n_tokens must match per index):')
ok_all = True
for L in range(9):
    d = json.load(open(os.path.join(R, 's19_ruler_17b_mk16k_L%d.json' % L), encoding='utf-8'))
    rows = sorted([r for r in d['rows']
                   if r['task'] == 'niah_multikey_1' and r['length'] == '16k'],
                  key=lambda r: r['i'])
    ok = (len(rows) == len(mrows)
          and all(a['answers'] == b['answers'] and a['n_tokens'] == b['n_tokens']
                  for a, b in zip(rows, mrows)))
    ok_all &= ok
    print('  L%d aligned=%s' % (L, ok))
print('ALL ALIGNED:', ok_all)
if not ok_all:
    raise SystemExit

print()
print('1.7B single-layer profile, niah_multikey_1 @16k, n=50, j=9')
print('reference arms on the SAME 50 samples: pub=%.1f pub_sink=%.1f fix_all=%.1f j0=%.1f'
      % tuple(100.0 * st.mean(r[a + '_recall'] for r in mrows)
              for a in ['pub', 'pub_sink', 'fix_all', 'j0']))
print()
print('layer  fix_S   vs pub_sink (paired)              vs fix_all (paired)')
for L in range(9):
    d = json.load(open(os.path.join(R, 's19_ruler_17b_mk16k_L%d.json' % L), encoding='utf-8'))
    rows = sorted([r for r in d['rows']
                   if r['task'] == 'niah_multikey_1' and r['length'] == '16k'],
                  key=lambda r: r['i'])
    s = [100.0 * r['fix_S_recall'] for r in rows]
    out = ['%-6d %-7.1f' % (L, st.mean(s))]
    for ref in ['pub_sink', 'fix_all']:
        v = [100.0 * (rows[i]['fix_S_recall'] - mrows[i][ref + '_recall'])
             for i in range(len(rows))]
        b = sum(1 for x in v if x > 0); w = sum(1 for x in v if x < 0)
        lo, hi = boot_ci(v)
        out.append('%+6.1f [%+.0f,%+.0f] (%d/%d)' % (st.mean(v), lo, hi, b, w))
    print('  '.join(out))
