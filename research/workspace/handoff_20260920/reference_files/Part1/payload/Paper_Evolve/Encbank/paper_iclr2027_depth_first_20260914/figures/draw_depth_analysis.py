"""Plot the existing depth results without inventing intermediate measurements.

Values are read from the bundled appendix table, which remains the numeric source.
Reported RULER-gap intervals are not reused as absolute-accuracy error bars.
"""
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'sections' / 'tab_multidepth.tex'
rows = []
for line in source.read_text(encoding='utf-8').splitlines():
    if re.match(r'^\d+\s*&', line):
        cells = [cell.strip() for cell in line.split('&')]
        rows.append(dict(j=int(cells[0]), params_m=float(cells[1]),
                         ruler=float(cells[2]), locomo=float(cells[4]),
                         read_ms=float(cells[5])))
assert [row['j'] for row in rows] == [6, 9, 12, 18]
(HERE / 'depth_analysis_data.json').write_text(json.dumps({
    'source': 'sections/tab_multidepth.tex',
    'protocol': 'Separately distilled suffix adapter at each split; H20 selected-pack prefill.',
    'data': rows}, indent=2) + '\n', encoding='utf-8')

fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [f for f in ['Arial', 'Helvetica', 'DejaVu Sans'] if f in fonts],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white',
    'mathtext.fontset': 'dejavusans',
})
fig, axes = plt.subplots(1, 3, figsize=(6.65, 2.03))
specs = [
    ('ruler', '(a) RULER', 'Score (%)', (48, 108), [50, 75, 100], '#0F4D92', 'o'),
    ('locomo', '(b) LoCoMo', 'Judge score', (25, 44), [25, 30, 35, 40], '#356C49', 's'),
    ('read_ms', '(c) Read time', 'Milliseconds', (445, 930), [500, 700, 900], '#4D4D4D', 'D'),
]
for ax, (key, title, ylabel, ylim, yticks, color, marker) in zip(axes, specs):
    x, y = [r['j'] for r in rows], [r[key] for r in rows]
    ax.axvspan(11.3, 12.7, color='#EAF1F8', zorder=0)
    ax.plot(x, y, color=color, marker=marker, markersize=4.1,
            linewidth=1.6, markeredgewidth=.6, zorder=3)
    ax.set(xlim=(4.8, 19.3), ylim=ylim, yticks=yticks,
           xticks=x, xlabel=r'Cache depth $j$', ylabel=ylabel)
    ax.set_title(title, loc='left', pad=7, fontweight='bold')
    ax.grid(axis='y', color='#E4E4E4', linewidth=.5, zorder=0)
    ax.tick_params(length=3, width=.7)
    for tick in ax.get_xticklabels():
        if tick.get_text() == '12':
            tick.set_color('#0F4D92')
            tick.set_fontweight('bold')
    for row in rows:
        # Endpoints and the chosen split carry exact values; all four are plotted.
        if row['j'] not in (6, 12, 18):
            continue
        text = f"{row[key]:.1f}" if key == 'read_ms' else f"{row[key]:.2f}"
        offset = (0, 6)
        if row['j'] == 6:
            offset = (6, 6)
        if row['j'] == 12:
            offset = (-2, -14)
        ax.annotate(text, (row['j'], row[key]), xytext=offset,
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=8, color=color)
fig.tight_layout(pad=.6, w_pad=1.1)
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(HERE / f'depth_analysis.{ext}', dpi=300,
                bbox_inches='tight', pad_inches=.025)
plt.close(fig)
print(HERE / 'depth_analysis.pdf')
