"""Task-specific depth curves, raw cell CSV and paired changes from anchor."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--partial', action='store_true', help='Explicitly label incomplete curves')
a = p.parse_args()
plan = json.loads((ROOT / 'plan.json').read_text())
data = ROOT / 'collected' / 'results'
target = ROOT / 'analysis'
target.mkdir(exist_ok=True)
tasks = ['niah_single_2', 'niah_multikey_1', 'longeval', 'qasper']
titles = ['RULER single-key', 'RULER multi-key', 'LongEval', 'Qasper']
arms = ['cache_lora', 'cache_without_lora', 'replay_shared_lora', 'replay_base']
labels = {'cache_lora': 'MidCache', 'cache_without_lora': 'MidCache (without LoRA)',
          'replay_shared_lora': 'Replay (same adapter)', 'replay_base': 'Replay (base)'}
styles = {'cache_lora': ('#176B9B', '-', 'o'), 'cache_without_lora': ('#C16B23', '-', 's'),
          'replay_shared_lora': ('#777777', '--', '^'), 'replay_base': ('#222222', ':', None)}
present, missing, cell_rows, task_rows, paired_rows = {}, [], [], [], []
rng = np.random.default_rng(20260915)


def interval(values, strata):
    """Bootstrap within length, preserving the fixed equal-length mixture."""
    groups = [values[np.array(strata) == key] for key in sorted(set(strata))]
    draws = sum(g[rng.integers(0, len(g), (10000, len(g)))].sum(1) for g in groups) / len(values)
    return np.quantile(draws, [.025, .975]).tolist()


for name, info in plan['models'].items():
    present[name] = {}
    for j in info['depths']:
        run = data / name / f'j{j:02}'
        if not (run / 'verified_summary.json').exists():
            missing.append({'model': name, 'j': j})
            continue
        verified = json.loads((run / 'verified_summary.json').read_text())
        assert verified['verified'] and verified['samples'] == 120 and verified['predictions'] == 480
        if j != info['anchor_j']:
            assert verified['depth_pairing_verified'] and verified['base_replay_reuse_verified']
        protocol = json.loads((run / 'protocol.json').read_text())
        assert protocol['j'] == j
        nparams = protocol.get('trainable_parameters', info['anchor_trainable_parameters'] if j == info['anchor_j'] else None)
        assert nparams is not None
        for key, value in verified['cells'].items():
            task, length, arm = key.split(':')
            cell_rows.append(dict(model=name, j=j, L=info['L'], j_over_L=j/info['L'], task=task,
                                  length=length, arm=arm, n=value['n'], score=value['mean'], trainable_parameters=nparams))
        rows = [json.loads(line) for line in (run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        present[name][j] = {(r['id'], r['arm']): r for r in rows}
        for task in tasks:
            for arm in arms:
                selected = sorted((r for r in rows if r['task'] == task and r['arm'] == arm), key=lambda r: r['id'])
                values = np.array([r['score'] for r in selected]) * 100
                strata = [r['length'] for r in selected]
                lo, hi = interval(values, strata)
                task_rows.append(dict(model=name, j=j, j_over_L=j/info['L'], task=task, arm=arm, n=len(selected),
                                      score=float(values.mean()), bootstrap95_low=lo, bootstrap95_high=hi,
                                      trainable_parameters=nparams))
    anchor_rows = present[name].get(info['anchor_j'], {})
    for j, records in present[name].items():
        if j == info['anchor_j'] or not anchor_rows:
            continue
        assert records.keys() == anchor_rows.keys()
        for task in tasks:
            for arm in arms:
                keys = sorted(k for k, r in records.items() if r['task'] == task and r['arm'] == arm)
                differences = np.array([records[k]['score'] - anchor_rows[k]['score'] for k in keys]) * 100
                strata = [records[k]['length'] for k in keys]
                lo, hi = interval(differences, strata)
                paired_rows.append(dict(model=name, j=j, anchor_j=info['anchor_j'], task=task, arm=arm, n=len(keys),
                                        difference=float(differences.mean()), paired95_low=lo, paired95_high=hi))

for filename, rows in [('cells.csv', cell_rows), ('tasks.csv', task_rows), ('paired_vs_anchor.csv', paired_rows)]:
    if rows:
        with (target / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
status = {'complete': not missing, 'missing': missing, 'limitations': plan['limitations']}
(target / 'coverage.json').write_text(json.dumps(status, indent=2) + '\n')
if missing and not a.partial:
    print(json.dumps({'curves_pending': True, 'missing': missing, 'csv': str(target)}))
    raise SystemExit(0)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 4, figsize=(14, 6.7), sharey=True)
for row_index, (name, info) in enumerate(plan['models'].items()):
    for col, task in enumerate(tasks):
        ax = axes[row_index, col]
        for arm in arms:
            items = sorted([r for r in task_rows if r['model'] == name and r['task'] == task and r['arm'] == arm], key=lambda r: r['j'])
            color, linestyle, marker = styles[arm]
            ax.plot([r['j_over_L'] for r in items], [r['score'] for r in items], color=color,
                    linestyle=linestyle, marker=marker, markersize=4, linewidth=1.5, label=labels[arm])
            if arm == 'cache_lora':
                ax.fill_between([r['j_over_L'] for r in items], [r['bootstrap95_low'] for r in items],
                                [r['bootstrap95_high'] for r in items], color=color, alpha=.12, linewidth=0)
        ax.set_xticks([j/info['L'] for j in info['depths']], [str(j) for j in info['depths']])
        ax.set_xlabel(f'Cache depth j (L={info["L"]}; spacing proportional to j/L)\n{name}', fontsize=8)
        ax.set_title(titles[col] + (' (F1)' if task == 'qasper' else ' (accuracy)'), fontsize=10)
        ax.set_ylim(-2, 102)
        ax.grid(axis='y', alpha=.2, linewidth=.6)
        if col == 0:
            ax.set_ylabel('Score (%)')
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.01))
caption = '200-step PG-19 adapters; one seed. Retrieval tasks average 8k/16k/32k (10 inputs each); Qasper: 30 inputs.\nShading: within-length bootstrap 95% CI for MidCache. Shared-adapter replay varies with j; base replay is reused.'
if missing:
    caption = 'PARTIAL: missing depths are listed in coverage.json.\n' + caption
fig.text(.5, .012, caption, ha='center', va='bottom', fontsize=8)
fig.tight_layout(rect=(0, .10 if missing else .07, 1, .95))
stem = 'depth_curves_partial' if missing else 'depth_curves'
fig.savefig(target / f'{stem}.pdf', bbox_inches='tight')
fig.savefig(target / f'{stem}.png', dpi=200, bbox_inches='tight')
print(json.dumps({'complete': not missing, 'figure': str(target / f'{stem}.pdf'), 'missing': missing}))
