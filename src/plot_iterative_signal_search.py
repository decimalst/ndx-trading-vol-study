"""Presentation-only figure of the frozen first-wave estimates."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/iterative_signal_search'


def main():
    metrics = json.loads((OUT / 'metrics.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7))
    colors = ['#83909f', '#127e9b']
    for ax, study, title in zip(axes, ['hf', 'index'],
                               ['SPX volatility: loss reduction (%)', 'QQQ daytime returns: loss reduction (%)'], strict=True):
        rows = [r for r in metrics['rows'] if r['study'] == study]
        labels = []
        for i, row in enumerate(rows):
            name = row['candidate'].replace('_', ' ')
            label = f"{name}, {row['horizon']} session" if study == 'hf' else f"{name} vs {row['control']}"
            labels.append(label)
            for phase, offset, color in zip(row['phases'], [-.12, .12], colors, strict=True):
                center = phase['improvement_pct']
                lo, hi = -100 * np.asarray(phase['ci95_envelope'])[::-1] / phase['control_loss']
                ax.errorbar(center, i + offset, xerr=[[center - lo], [hi - center]],
                            fmt='o', color=color, capsize=3, markersize=5,
                            label=phase['name'].capitalize() if i == 0 else None)
        ax.axvline(0, color='#394656', linewidth=1)
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontweight='bold', pad=14)
        ax.grid(axis='x', color='#dfe4e9', linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel('Worse  ←   change in forecast loss   →  Better', labelpad=12)
        ax.legend(loc='lower right', frameon=False)
    fig.suptitle('No new candidate passed the fixed improvement criteria',
                 x=.035, ha='left', fontsize=18, fontweight='bold')
    fig.text(.035, .875, 'Development and evaluation estimates with nominal 95% interval envelopes',
             fontsize=11, color='#445363')
    fig.text(.035, .025, 'Exploratory reused history • 12 new comparisons; 66 tracked cumulatively • All forecasts and inference independently reconstructed\n'
             'Intervals combine block bootstrap and HAC uncertainty. They are not simultaneous confidence intervals.',
             fontsize=9, color='#445363', linespacing=1.7)
    fig.subplots_adjust(left=.18, right=.98, top=.81, bottom=.18, wspace=.95)
    fig.savefig(OUT / 'comparison.png', dpi=180, facecolor='white')
    fig.savefig(OUT / 'comparison.pdf', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
