"""Render all primary comparisons from a verified measurement-memory run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parents[1]/"reports/measurement_memory"


def main():
    metrics = json.loads((REPORT/"metrics.json").read_text())
    verified = json.loads((REPORT/"verification.json").read_text())
    if verified["status"] != "VERIFIED" or verified["protocol_sha256"] != metrics["protocol_sha256"]:
        raise ValueError("Matching independent verification required")
    groups = [([("width", "baseline"), ("quality", "baseline"), ("quality", "width")],
               ["Width vs market history", "Interaction vs market history", "Interaction vs width"], "Incremental information"),
              ([("width", "mean"), ("quality", "mean")],
               ["Width vs mean", "Interaction vs mean"], "Comparison with historical mean")]
    colors = {"development": "#23679a", "evaluation": "#b55e20"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))
    for ax, (pairs, labels, title) in zip(axes, groups, strict=True):
        ax.axvline(0, color="#555e68", linewidth=1)
        ax.axvline(-.005, color="#9299a3", linewidth=.8, linestyle=":")
        for position, (candidate, control) in enumerate(pairs):
            row, = [row for row in metrics["rows"] if row["candidate"] == candidate
                    and row["control"] == control and row["measure"] == "qmle"]
            for phase in row["phases"]:
                name = phase["name"]
                y = position + (-.12 if name == "development" else .12)
                ax.hlines(y, *phase["ci95_envelope"], color=colors[name], linewidth=2)
                ax.plot(phase["delta"], y, "o", color=colors[name], markersize=6,
                        label=name.capitalize() if position == 0 else None)
        ax.set_title(title, fontsize=12, loc="left", pad=16)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_ylim(len(labels)-.35, -.6)
        ax.set_xlabel("Absolute paired score difference (lower is better)", labelpad=10, fontsize=9)
        ax.grid(axis="x", color="#e6e9ed", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .12), ncol=2, frameon=False)
    fig.suptitle("Does reported measurement uncertainty help forecast volatility?", x=.055, ha="left", fontsize=16)
    fig.text(.055, .025, "Primary SPY trade-QMLE measurement; all five primary comparisons shown. Ten alternate-measurement comparisons are in the report.\n"
             "Intervals: envelope of 95% bootstrap and HAC intervals, not simultaneous family intervals. Dotted line: fixed −0.005 threshold.\n"
             "Retrospective source timing assumptions and historical reuse remain limitations.", fontsize=8.5, color="#505963")
    fig.subplots_adjust(left=.20, right=.97, top=.79, bottom=.29, wspace=.59)
    fig.savefig(REPORT/"primary_comparison_intervals.png", dpi=180, facecolor="white")
    fig.savefig(REPORT/"primary_comparison_intervals.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
