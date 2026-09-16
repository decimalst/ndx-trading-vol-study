"""Plot both verified joint-risk contrasts without converting their intervals."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = (
    ("dynamic_correlation", "constant_correlation", "matrix_qlike"),
    ("dynamic_correlation", "constant_matrix", "matrix_qlike"),
)
PHASES = ("development", "evaluation")


def validated_rows(metrics, verification, protocol_hash):
    """Require current verification, both comparisons, and literal loss intervals."""
    if (
        verification.get("status") != "VERIFIED"
        or verification.get("protocol_sha256") != protocol_hash
        or metrics.get("protocol_sha256") != protocol_hash
    ):
        raise ValueError("Independent verification of this protocol required before rendering")
    rows = metrics.get("rows", [])
    keys = [(row["candidate"], row["control"], row["score"]) for row in rows]
    if len(keys) != len(CONTRASTS) or set(keys) != set(CONTRASTS):
        raise ValueError("Both registered joint-risk comparisons required")
    ordered = []
    for key in CONTRASTS:
        row = rows[keys.index(key)]
        phases = row.get("phases", [])
        names = [phase["name"] for phase in phases]
        if len(names) != len(PHASES) or set(names) != set(PHASES):
            raise ValueError("Both registered phases required for each comparison")
        arranged = [phases[names.index(name)] for name in PHASES]
        for phase in arranged:
            interval = phase["ci95_envelope"]
            if (
                len(interval) != 2
                or not all(
                    math.isfinite(x)
                    for x in [phase["delta"], phase["control_loss"], *interval]
                )
                or interval[0] > interval[1]
            ):
                raise ValueError("Finite ordered paired joint-score intervals required")
        ordered.append({**row, "phases": arranged})
    return ordered


def main():
    folder = ROOT / "reports/joint_risk"
    verification = json.loads((folder / "verification.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    protocol_hash = hashlib.sha256((ROOT / "joint_risk.yaml").read_bytes()).hexdigest()
    rows = validated_rows(metrics, verification, protocol_hash)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), sharey=True)
    for ax, row in zip(axes, rows, strict=True):
        for y, phase, color in zip([1, 0], row["phases"], ["#4a72a5", "#cc6c39"], strict=True):
            left, right = phase["ci95_envelope"]
            ax.plot([left, right], [y, y], color=color, linewidth=2)
            ax.scatter(phase["delta"], y, color=color, s=40, zorder=3)
            ax.text(
                0.03,
                y + 0.21,
                f"Joint-score difference: {phase['delta']:+.4f}",
                transform=ax.get_yaxis_transform(),
                color=color,
                fontsize=9,
            )
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.axvline(-0.005, color="#999999", linewidth=0.8, linestyle=":")
        control = (
            "constant correlation"
            if row["control"] == "constant_correlation"
            else "constant matrix"
        )
        ax.set_title(f"Dynamic vs {control}", loc="left", fontsize=11)
        ax.set_yticks([1, 0], ["2016–2019", "2020–2025"])
        ax.set_ylim(-0.6, 1.65)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", alpha=0.16)
        ax.set_xlabel("Candidate minus control score (negative is better)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    fig.suptitle(
        "Does changing dependence improve joint QQQ–SPX risk forecasts?",
        x=0.06,
        ha="left",
        fontsize=13,
    )
    fig.text(
        0.06,
        0.035,
        "Lines: nominal 95% block-bootstrap/HAC interval envelopes. Unhalved matrix score; no percentage gains.\n"
        "Both controls require ≥0.005 improvement in both periods, stability and multiplicity gates. Archival ETF/index daily returns.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.90), w_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"comparison_intervals.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
