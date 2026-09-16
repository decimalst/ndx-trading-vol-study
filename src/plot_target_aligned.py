"""Plot the three independently verified target-aligned MSE comparisons."""

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
    ("aligned_dynamic", "aligned_constant", "product_mse"),
    ("aligned_dynamic", "constant_matrix", "product_mse"),
    ("aligned_dynamic", "dynamic_correlation", "product_mse"),
)
PHASES = ("development", "evaluation")
EFFECT_THRESHOLD = 1e-10
EXPECTED_COUNTS = {
    "hypothesis_count": 3,
    "cumulative_hypothesis_count": 117,
    "new_forecasts": 4924,
    "new_monthly_fits": 118,
    "new_scalar_fits": 236,
}
CONTROL_LABELS = {
    "aligned_constant": "Aligned constant correlation",
    "constant_matrix": "Original constant matrix",
    "dynamic_correlation": "Original QLIKE dynamic correlation",
}


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validated_rows(metrics, verification, protocol_hash):
    """Admit the complete current verified family before creating any artifacts."""
    if (
        verification.get("status") != "VERIFIED"
        or verification.get("protocol_sha256") != protocol_hash
        or metrics.get("protocol_sha256") != protocol_hash
        or metrics.get("status") == "UNEVALUABLE"
        or metrics.get("whole_wave_aborted") is True
    ):
        raise ValueError("Current successful independent verification required before plotting")
    if any(
        not _finite(metrics.get(name)) or metrics[name] != expected
        for name, expected in EXPECTED_COUNTS.items()
    ):
        raise ValueError("Exact registered family, new forecast and fit counts required")
    rows = metrics.get("rows", [])
    keys = [(row.get("candidate"), row.get("control"), row.get("score")) for row in rows]
    if len(keys) != len(CONTRASTS) or set(keys) != set(CONTRASTS):
        raise ValueError("All three distinct registered product-MSE contrasts required")
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
    """Render literal quartic-unit intervals after current independent verification."""
    rows = validated_rows(metrics, verification, protocol_hash)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.8), sharex=True, sharey=True)
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
            ax.set_title(CONTROL_LABELS[row["control"]], loc="left", fontsize=11, y=1.13)
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
            ax.set_xlabel("Candidate − control MSE\n(decimal log return⁴)", fontsize=9)
            ax.grid(axis="x", alpha=0.16)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        fig.suptitle(
            "Does fitting cross-moment MSE improve prediction? Aligned dynamic model versus each control",
            x=0.045,
            ha="left",
            fontsize=13,
        )
        fig.text(
            0.045,
            0.035,
            "Lines: nominal 95% block-bootstrap/HAC interval envelopes. Negative is better; dotted reference: −1e−10.\n"
            "All three comparisons require the fixed effect, stability and multiplicity gates. Shared means and marginal risks remain frozen.",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0.015, 0.15, 0.99, 0.88), w_pad=2.4)
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
    folder = root / "reports/target_aligned"
    if (folder / "failure.json").exists():
        raise ValueError("A canonical failure record blocks result plotting")
    metrics = json.loads((folder / "metrics.json").read_text())
    verification = json.loads((folder / "verification.json").read_text())
    protocol_hash = hashlib.sha256((root / "target_aligned.yaml").read_bytes()).hexdigest()
    return render_verified(metrics, verification, protocol_hash, folder)


if __name__ == "__main__":
    main()
