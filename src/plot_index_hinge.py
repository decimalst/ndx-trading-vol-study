"""Render all four registered index contrasts after independent verification."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT/'reports/index_hinge'
    verification = json.loads((report/'verification.json').read_text())
    if verification['status'] != 'VERIFIED':
        raise ValueError('Independent verification required before rendering')
    rows = json.loads((report/'metrics.json').read_text())['rows']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharey=True)
    for ax, row in zip(axes.flat, rows, strict=True):
        for y, phase, color in zip([1, 0], row['phases'], ['#4a72a5', '#cc6c39'], strict=True):
            point = phase['gain_relative']*100
            lo, hi = phase['ci95_envelope']
            left, right = -hi/phase['control_loss']*100, -lo/phase['control_loss']*100
            ax.plot([left, right], [y, y], color=color, linewidth=2)
            ax.scatter(point, y, color=color, s=40, zorder=3)
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.axvline(.25, color='#999999', linestyle=':', linewidth=1)
        ax.set_title(f"{row['horizon']} sessions · hinge vs {row['control']}", loc='left', fontsize=11)
        ax.set_yticks([1, 0], ['2016–2019', '2020–2025'])
        ax.set_ylim(-.65, 1.65)
        ax.grid(axis='x', alpha=.16)
        ax.set_xlabel('Relative MSE improvement (%) →')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('Does the asymmetric implied–trailing volatility gap predict SPX returns?', x=.06, ha='left', fontsize=14)
    fig.text(.06, .025, 'Lines: envelope of nominal 95% block-bootstrap and HAC intervals. Dotted line: 0.25% effect gate.\n'
             'All offsets and both controls are required; correction and historical-vintage limits apply. Price returns exclude dividends.', fontsize=9, color='#555555')
    fig.tight_layout(rect=(.035, .09, .99, .94), h_pad=2, w_pad=2)
    for extension in ('png', 'pdf'):
        fig.savefig(report/f'comparison_intervals.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
