"""Plot preserved nuScenes reports; never reconstruct geometry from metrics."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures/nuscenes_scale'
reports = json.loads((OUT / 'evidence/all_summaries.json').read_text(encoding='utf-8-sig'))
variants = ['relative', 'optimized_original']
paths = [('depth', 'rel'), ('reconstruction_sequence_se3_mean', 'chamfer'), ('pose', 'ate', 'rmse')]
names = ['Depth estimation', 'Sparse 3D reconstruction', 'Ego pose']
units = ['Depth AbsRel', 'Chamfer distance (m)', 'ATE RMSE (m)']

def get(d, path):
    for k in path:
        d = d[k]
    d = float(d)
    if not np.isfinite(d):
        raise ValueError('Nonfinite metric')
    return d

rows = []
for report in reports:
    matched = []
    for seq in report['data']['sequence_results']:
        try:
            values = [[get(seq[v], path) for path in paths] for v in variants]
        except (KeyError, ValueError, TypeError):
            continue
        matched.append(values)
        for vi, v in enumerate(variants):
            rows.append(dict(model=report['data']['depth_model'], scene=seq['sequence_name'], variant=v,
                             depth_absrel=values[vi][0], sparse_chamfer_m=values[vi][1],
                             ate_rmse_m=values[vi][2], source=report['source']))
    report['matched_values'] = np.asarray(matched)

with (OUT / 'available_metrics.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

fig, axes = plt.subplots(1, 3, figsize=(17, 7))
fig.subplots_adjust(left=.14, right=.98, top=.80, bottom=.25, wspace=.30)
fig.suptitle('nuScenes | Scale recovery from preserved evaluation reports', fontsize=19, y=.96)
fig.text(.14, .885, 'Orange: relative    Teal: optimized_original    |    Lower is better', fontsize=11)
y = np.arange(len(reports))
for k, ax in enumerate(axes):
    for i, report in enumerate(reports):
        values = report['matched_values']
        if not len(values):
            continue
        before, after = values[:, :, k].mean(axis=0)
        ax.plot([before, after], [i, i], color='#cbd5e1', lw=3)
        ax.scatter(before, i, s=60, color='#d97706', zorder=3)
        ax.scatter(after, i, s=60, color='#078477', zorder=3)
        ax.annotate(f'{before:.3g} → {after:.3g}', (max(before, after), i),
                    xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_title(names[k], fontsize=14, pad=14)
    ax.set_xlabel(units[k] + ' ↓')
    ax.set_yticks(y)
    ax.set_yticklabels([r['data']['depth_model'].replace('_long', '') + f"  (n={len(r['matched_values'])})"
                        for r in reports] if k == 0 else [])
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    ax.set_xlim(left=0, right=ax.get_xlim()[1]*1.5)
    ax.spines[['top', 'right']].set_visible(False)
fig.text(.14, .065, 'Each point is an equal-weight mean over paired scenes valid for all three metrics within that model.\n'
         'n = scene count; Scal3R has only one reported scene and is not directly comparable to full runs.\n'
         'Reconstruction is a LiDAR-referenced sparse proxy. These are measured metrics, not predicted geometry visualizations.',
         fontsize=10, color='#475569', linespacing=1.6)
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / ('nuscenes_available_metrics.' + suffix), dpi=200, facecolor='white')
print('Saved figure and per-scene CSV:', OUT)
