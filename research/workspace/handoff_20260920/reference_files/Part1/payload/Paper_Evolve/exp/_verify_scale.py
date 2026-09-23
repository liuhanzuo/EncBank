import json, random, statistics as st, os

R = r'F:\Paper_Evolve\exp\results'


def boot_ci(v, B=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def load(fn):
    return json.load(open(os.path.join(R, fn), encoding='utf-8'))


def cells(files):
    """map (task,length) -> list of rows"""
    out = {}
    for fn in files:
        d = load(fn)
        for r in d['rows']:
            out.setdefault((r['task'], r['length']), []).append(r)
    return out


def report(name, files):
    print('=' * 70)
    print(name)
    c = cells(files)
    order = [('niah_multikey_1', '16k'), ('niah_multikey_1', '32k'),
             ('niah_single_2', '16k'), ('niah_single_2', '32k'),
             ('variable_tracking', '16k'), ('variable_tracking', '32k')]
    for k in order:
        if k not in c:
            continue
        rows = c[k]
        m = {}
        for arm in ['pub', 'pub_sink', 'fix_all', 'j0']:
            key = arm + '_recall'
            if key not in rows[0]:
                m[arm] = None
                continue
            m[arm] = 100.0 * st.mean(r[key] for r in rows)
        print('%-22s n=%d  pub=%s pub_sink=%s fix_all=%s j0=%s' % (
            '%s/%s' % k, len(rows),
            *[('%.1f' % m[a] if m[a] is not None else 'NA')
              for a in ['pub', 'pub_sink', 'fix_all', 'j0']]))
        for a, b in [('fix_all', 'pub'), ('j0', 'fix_all'), ('pub_sink', 'pub')]:
            ka, kb = a + '_recall', b + '_recall'
            if ka not in rows[0] or kb not in rows[0]:
                continue
            d = [100.0 * (r[ka] - r[kb]) for r in rows]
            better = sum(1 for x in d if x > 0)
            worse = sum(1 for x in d if x < 0)
            mean = st.mean(d)
            if better == 0 and worse == 0:
                print('      %-18s %+6.1f  (all %d ties)' % (a + '-' + b, mean, len(d)))
            else:
                lo, hi = boot_ci(d)
                print('      %-18s %+6.1f  [%.1f, %.1f]  (%d better / %d worse)'
                      % (a + '-' + b, mean, lo, hi, better, worse))


report('Qwen3-1.7B  j=9  (s19)',
       ['s19_ruler_17b_niah.json', 's19_ruler_17b_vt.json'])
report('Qwen3-8B  j=12  (s15/s15c)',
       ['s15_ruler_j12_16k.json', 's15_ruler_j12_32k.json',
        's15c_ruler_vt_16k.json', 's15c_ruler_vt_32k.json'])
