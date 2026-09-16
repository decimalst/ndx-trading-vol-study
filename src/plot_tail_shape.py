"""Show every registered signed-tail comparison after independent verification."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'reports/tail_shape'
    verification = json.loads((folder/'verification.json').read_text())
    if verification['status'] != 'VERIFIED':
        raise ValueError('Independent empirical verification required')
    rows = json.loads((folder/'metrics.json').read_text())['rows']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True)
    for ax, row in zip(axes, rows, strict=True):
        for y, phase, color in zip([1, 0], row['phases'], ['#4a72a5', '#cc6c39'], strict=True):
            left, right = phase['ci95_envelope']
            ax.plot([left, right], [y, y], color=color, linewidth=2)
            ax.scatter(phase['delta'], y, color=color, s=38, zorder=3)
        threshold = .0005 if row['score'] == 'brier' else .005
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.axvline(-threshold, color='#999999', linestyle=':', linewidth=1)
        ax.set_title(f"{row['score'].upper()} vs {row['control'].replace('_', ' ')}", loc='left', fontsize=11)
        ax.set_yticks([1, 0], ['2016–2019', '2020–2025'])
        ax.set_ylim(-.65, 1.65)
        ax.grid(axis='x', alpha=.16)
        ax.set_xlabel('← Lower paired loss is better')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    fig.suptitle('Does SKEW improve downside-tail shape at the same predicted mean and variance?', x=.055, ha='left', fontsize=14)
    fig.text(.055, .035, 'Lines: envelope of nominal 95% block-bootstrap and HAC intervals. Dotted lines: fixed absolute effect gates.\n'
             'All three comparisons must pass in both periods, with stability and multiplicity checks. Historical archival experiment.', fontsize=9, color='#555555')
    fig.tight_layout(rect=(.025, .13, .99, .9), w_pad=2)
    for extension in ['png', 'pdf']:
        fig.savefig(folder/f'comparison_intervals.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
