"""Presentation-only plot of the unchanged second-wave estimates."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/international_volatility'


def main():
    metrics = json.loads((OUT / 'metrics.json').read_text())
    fig, ax = plt.subplots(figsize=(10, 6.2))
    colors = ['#8794a1', '#167f9c']
    for i, row in enumerate(metrics['rows']):
        for phase, offset, color in zip(row['phases'], [-.13, .13], colors, strict=True):
            center = phase['improvement_pct']
            low, high = -100 * np.asarray(phase['ci95_envelope'])[::-1] / phase['control_loss']
            ax.errorbar(center, i + offset, xerr=[[center - low], [high - center]],
                        fmt='o', color=color, markersize=6, capsize=3,
                        label=phase['name'].capitalize() if i == 0 else None)
    labels = [f"{row['candidate'].capitalize()}, {row['horizon']} session" for row in metrics['rows']]
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='#354555', linewidth=1)
    ax.grid(axis='x', color='#e2e6e9', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Reduction in SPX volatility forecast loss (%) — positive is better', labelpad=12)
    ax.legend(loc='upper right', frameon=False)
    fig.suptitle('Overseas volatility: later gains did not repeat consistently',
                 x=.035, ha='left', fontsize=16, fontweight='bold')
    fig.text(.035, .88, 'Regional summaries and three training-only latent factors versus the strong SPX baseline', fontsize=10)
    fig.text(.035, .025, 'No candidate passed the fixed criteria. 6 new comparisons; 72 tracked cumulatively.\n'
             'Nominal 95% interval envelopes combine block-bootstrap and HAC uncertainty; they are not simultaneous.\n'
             'Reused history and archived publication-vintage uncertainty remain exploratory.',
             fontsize=8.8, color='#445363', linespacing=1.6)
    fig.subplots_adjust(left=.20, right=.97, top=.81, bottom=.20)
    fig.savefig(OUT / 'comparison.png', dpi=180, facecolor='white')
    fig.savefig(OUT / 'comparison.pdf', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
