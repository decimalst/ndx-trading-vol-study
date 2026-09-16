"""Verified Brier comparisons for one issued-error calibration."""

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
    ("calibrated", "baseline", "brier"),
    ("calibrated", "recent_frequency", "brier"),
)
PHASES = ("development", "evaluation")
EFFECT_THRESHOLD = 0.0005
WAVE_ALPHA = 0.05 / (19 * 20)
CUMULATIVE_FAMILY = 131


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _positive_integer(value):
    return _finite(value) and value > 0 and int(value) == value


def validated_rows(metrics, verification, protocol_hash):
    """Require current verification and exactly two same-sample Brier contrasts."""
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
        or metrics.get("cumulative_hypothesis_count") != CUMULATIVE_FAMILY
    ):
        raise ValueError("Exact registered two-comparison and cumulative131 family required")
    proof = verification.get("forecast_reconstruction", {})
    count_names = (
        "new_forecasts",
        "reused_control_forecasts",
        "combined_forecasts",
        "common_scored_origins",
        "common_application_origins",
    )
    proof_fields = {
        "forecasts_verified": "combined_forecasts",
        "new_forecasts": "new_forecasts",
        "reused_control_forecasts": "reused_control_forecasts",
        "common_scored_origins": "common_scored_origins",
        "common_application_origins": "common_application_origins",
    }
    if (
        any(not _positive_integer(metrics.get(name)) for name in count_names)
        or metrics.get("new_monthly_fits") != 0
        or isinstance(metrics.get("new_monthly_fits"), bool)
        or proof.get("new_monthly_fits") != 0
        or isinstance(proof.get("new_monthly_fits"), bool)
        or metrics["new_forecasts"] != metrics["common_scored_origins"]
        or metrics["reused_control_forecasts"] != 2 * metrics["common_scored_origins"]
        or metrics["combined_forecasts"] != 3 * metrics["common_scored_origins"]
        or metrics["common_application_origins"] < metrics["common_scored_origins"]
        or any(
            not _positive_integer(proof.get(key)) or proof[key] != metrics[name]
            for key, name in proof_fields.items()
        )
    ):
        raise ValueError(
            "Complete three-model cohort, unchanged controls and verified application counts required"
        )
    rows = metrics.get("rows", [])
    keys = [(r.get("candidate"), r.get("control"), r.get("score")) for r in rows]
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Both distinct registered Brier comparisons required")
    result, cohort = [], None
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
            if (
                len(interval) != 2
                or not all(
                    _finite(v)
                    for v in (
                        phase.get("delta"),
                        phase.get("candidate_loss"),
                        phase.get("control_loss"),
                        *interval,
                    )
                )
                or not 0 <= phase["candidate_loss"] <= 1
                or not 0 <= phase["control_loss"] <= 1
                or not -1 <= phase["delta"] <= 1
                or interval[0] > interval[1]
                or not _positive_integer(phase.get("n"))
            ):
                raise ValueError(
                    "Finite Brier-loss differences and literal ordered intervals required"
                )
        sizes = [phase["n"] for phase in ordered]
        if sum(sizes) != metrics["common_scored_origins"] or (
            cohort is not None and sizes != cohort
        ):
            raise ValueError("Same complete phase cohorts required for both controls")
        cohort = sizes
        result.append({**row, "phases": ordered})
    return result


def render_verified(metrics, verification, protocol_hash, folder):
    """Render literal squared-probability-error intervals without percent rescaling."""
    rows = validated_rows(metrics, verification, protocol_hash)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharex=False, sharey=True)
    try:
        for ax, row in zip(axes, rows, strict=True):
            for y, phase, color in zip(
                [1, 0], row["phases"], ["#4a72a5", "#cc6c39"], strict=True
            ):
                ax.plot(phase["ci95_envelope"], [y, y], color=color, linewidth=2)
                ax.scatter(phase["delta"], y, color=color, s=42, zorder=3)
                ax.text(
                    0.03,
                    y + 0.21,
                    f"Loss difference: {phase['delta']:+.2e}",
                    transform=ax.get_yaxis_transform(),
                    color=color,
                    fontsize=10,
                )
            ax.axvline(0.0, color="#555555", linewidth=0.8)
            ax.axvline(-EFFECT_THRESHOLD, color="#888888", linewidth=0.9, linestyle=":")
            control = "full baseline" if row["control"] == "baseline" else "recent frequency"
            ax.set_title(f"Calibrated vs {control}", loc="left", fontsize=11, y=1.13)
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
            ax.set_xlabel(
                "Candidate − control Brier loss\n(squared probability error)",
                fontsize=9,
            )
            ax.grid(axis="x", alpha=0.16)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        fig.suptitle(
            "Can forecast-error memory improve SPX risk alerts?",
            x=0.06,
            ha="left",
            fontsize=13,
        )
        fig.text(
            0.06,
            0.035,
            "Lines: nominal 95% block-bootstrap/HAC interval envelopes. Negative is better; dotted reference: −0.0005.\n"
            f"Scales differ. Both controls require effect/stability gates, wave Holm2 p < {WAVE_ALPHA:.9g} and cumulative Holm131 p < 0.05.\n"
            "Archival daily OHLC risk proxy; event above twice the prior22-session mean; no trading-profit claim.",
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
    folder = root / "reports/issued_calibration"
    if (folder / "failure.json").exists():
        raise ValueError("A canonical failure record blocks result plotting")
    metrics_bytes = (folder / "metrics.json").read_bytes()
    verification = json.loads((folder / "verification.json").read_bytes())
    expected = verification.get("verified_output_hashes", {}).get(
        "reports/issued_calibration/metrics.json"
    )
    if expected != hashlib.sha256(metrics_bytes).hexdigest():
        raise ValueError("Metrics bytes differ from independently verified output snapshot")
    metrics = json.loads(metrics_bytes)
    protocol_hash = hashlib.sha256((root / "issued_calibration.yaml").read_bytes()).hexdigest()
    return render_verified(metrics, verification, protocol_hash, folder)


if __name__ == "__main__":
    main()
