"""Plot both independently verified direct cross-moment comparisons in native units."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = (
    ("dynamic_correlation", "constant_correlation", "product_mse"),
    ("dynamic_correlation", "constant_matrix", "product_mse"),
)
PHASES = ("development", "evaluation")
EFFECT_THRESHOLD = 1e-10


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validated_rows(metrics, verification, protocol_hash):
    """Admit only the complete current verified comparison/phase family."""
    if (
        verification.get("status") != "VERIFIED"
        or verification.get("protocol_sha256") != protocol_hash
        or metrics.get("protocol_sha256") != protocol_hash
        or metrics.get("status") == "UNEVALUABLE"
        or metrics.get("whole_wave_aborted") is True
    ):
        raise ValueError(
            "Current successful independent verification required before plotting"
        )
    if (
        metrics.get("hypothesis_count") != 2
        or metrics.get("cumulative_hypothesis_count") != 114
        or metrics.get("new_model_fits") != 0
        or metrics.get("new_forecasts") != 0
    ):
        raise ValueError("Exact registered family and no new fits or forecasts required")
    rows = metrics.get("rows", [])
    keys = [(row.get("candidate"), row.get("control"), row.get("score")) for row in rows]
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Both distinct registered product-MSE contrasts required")
    result = []
    for contrast in CONTRASTS:
        row = rows[keys.index(contrast)]
        if row.get("horizon") != 1 or any(
            not _finite(row.get(name)) or not 0 <= row[name] <= 1
            for name in ("p_holm_wave", "p_holm_cumulative")
        ):
            raise ValueError("Horizon one and finite corrected probabilities required")
        phases = row.get("phases", [])
        names = [phase.get("name") for phase in phases]
        if len(names) != 2 or set(names) != set(PHASES):
            raise ValueError("Both distinct registered phases required")
        ordered = [phases[names.index(name)] for name in PHASES]
        for phase in ordered:
            interval = phase.get("ci95_envelope", [])
            values = [
                phase.get("delta"),
                phase.get("candidate_loss"),
                phase.get("control_loss"),
                *interval,
            ]
            if (
                len(interval) != 2
                or not all(_finite(value) for value in values)
                or interval[0] > interval[1]
                or phase["candidate_loss"] < 0
                or phase["control_loss"] < 0
                or not _finite(phase.get("n"))
                or phase["n"] <= 0
                or int(phase["n"]) != phase["n"]
            ):
                raise ValueError(
                    "Finite native-unit differences, nonnegative MSE and ordered intervals required"
                )
        result.append({**row, "phases": ordered})
    return result


def render_verified(metrics, verification, protocol_hash, folder):
    """Render literal quartic-unit intervals only after the verification guard."""
    rows = validated_rows(metrics, verification, protocol_hash)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), sharey=True)
    try:
        for ax, row in zip(axes, rows, strict=True):
            for y, phase, color in zip(
                [1, 0], row["phases"], ["#4a72a5", "#cc6c39"], strict=True
            ):
                left, right = phase["ci95_envelope"]
                ax.plot([left, right], [y, y], color=color, linewidth=2)
                ax.scatter(phase["delta"], y, color=color, s=42, zorder=3)
                ax.text(
                    0.03,
                    y + 0.21,
                    f"MSE difference: {phase['delta']:+.2e}",
                    transform=ax.get_yaxis_transform(),
                    color=color,
                    fontsize=10,
                )
            ax.axvline(0.0, color="#555555", linewidth=0.8)
            ax.axvline(-EFFECT_THRESHOLD, color="#888888", linewidth=0.9, linestyle=":")
            control = (
                "constant correlation"
                if row["control"] == "constant_correlation"
                else "constant matrix"
            )
            ax.set_title(f"Dynamic vs {control}", loc="left", fontsize=11, y=1.13)
            ax.text(
                0.0,
                1.025,
                f"Wave Holm p: {row['p_holm_wave']:.3g}; cumulative: {row['p_holm_cumulative']:.3g}",
                transform=ax.transAxes,
                fontsize=8.5,
                color="#555555",
            )
            ax.set_yticks([1, 0], ["2016–2019", "2020–Oct 2025"])
            ax.set_ylim(-0.55, 1.7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlabel("Candidate − control MSE (decimal log return⁴)", fontsize=9)
            ax.grid(axis="x", alpha=0.16)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        fig.suptitle(
            "Do the issued forecasts predict signed cross moments more accurately?",
            x=0.06,
            ha="left",
            fontsize=13,
        )
        fig.text(
            0.06,
            0.035,
            "Lines: nominal 95% block-bootstrap/HAC interval envelopes. Negative is better; dotted reference: −1e−10.\n"
            "Both controls require the fixed effect, stability and multiplicity gates. Same frozen forecasts; no refitting or percentage gains.",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0.025, 0.15, 0.99, 0.88), w_pad=2.4)
        paths = {}
        for extension in ("png", "pdf"):
            path = folder / f"comparison_intervals.{extension}"
            fig.savefig(path, dpi=180, facecolor="white")
            paths[extension] = str(path)
        return paths
    finally:
        plt.close(fig)


def main(root=ROOT):
    root = Path(root)
    folder = root / "reports/cross_moment"
    if (folder / "failure.json").exists():
        raise ValueError("A canonical failure record blocks result plotting")
    metrics = json.loads((folder / "metrics.json").read_text())
    verification = json.loads((folder / "verification.json").read_text())
    protocol_hash = hashlib.sha256((root / "cross_moment.yaml").read_bytes()).hexdigest()
    return render_verified(metrics, verification, protocol_hash, folder)


if __name__ == "__main__":
    main()
