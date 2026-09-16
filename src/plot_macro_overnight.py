"""Render the verified fixed macro comparison family without selecting arms."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/macro_overnight"


def main():
    verified = json.loads((REPORT / "verification.json").read_text())
    metrics = json.loads((REPORT / "metrics.json").read_text())
    if verified["status"] != "VERIFIED" or verified["protocol_sha256"] != metrics["protocol_sha256"]:
        raise ValueError("Matching independent verification is required before plotting")
    candidates = ("cpi", "nfp", "fomc")
    labels = ("CPI plan", "Payroll plan", "Fed date control")
    colors = {"development": "#23679a", "evaluation": "#b55e20"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for ax, control, title in zip(axes, ("baseline", "mean"),
                                 ("Against market-history model", "Against historical mean"), strict=True):
        ax.axvline(0, color="#565c64", linewidth=1)
        ax.axvline(-.005, color="#9299a3", linewidth=.8, linestyle=":")
        for position, candidate in enumerate(candidates):
            row, = [item for item in metrics["rows"] if item["candidate"] == candidate and item["control"] == control]
            for phase in row["phases"]:
                name = phase["name"]
                y = position + (-.13 if name == "development" else .13)
                lower, upper = phase["ci95_envelope"]
                ax.hlines(y, lower, upper, color=colors[name], linewidth=2)
                ax.plot(phase["delta"], y, "o", color=colors[name], markersize=6,
                        label=name.capitalize() if position == 0 else None)
        ax.set_title(title, fontsize=12, loc="left", pad=16)
        ax.set_xlabel("Absolute paired score difference  (lower is better)", labelpad=10)
        ax.set_yticks(range(len(candidates)), labels)
        ax.set_ylim(2.55, -.55)
        ax.grid(axis="x", color="#e6e9ed", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Do advance release plans help predict overnight move size?", x=.09, ha="left", fontsize=15)
    fig.text(.09, .025, "Intervals: envelope of 95% block-bootstrap and HAC intervals; not simultaneous family intervals.\n"
             "Dotted line: fixed −0.005 effect threshold. Reused history and adjusted-return proxy; all six comparisons shown.",
             fontsize=8.5, color="#505963")
    fig.subplots_adjust(left=.16, right=.97, top=.80, bottom=.24, wspace=.23)
    fig.savefig(REPORT / "comparison_intervals.png", dpi=180, facecolor="white")
    fig.savefig(REPORT / "comparison_intervals.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
