"""Presentation-only tick spacing for the verified, frozen wave8 figure.

Run from the repository root. The frozen plotting module validates the protocol
and verification before reading its metrics. This wrapper changes axis tick
spacing after the first rendered figure showed crowded labels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from matplotlib.ticker import MaxNLocator

from src import plot_calendar_variance as figure

original_subplots = figure.plt.subplots


def readable_subplots(*args, **kwargs):
    canvas, axes = original_subplots(*args, **kwargs)
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    return canvas, axes


figure.plt.subplots = readable_subplots
figure.main()
