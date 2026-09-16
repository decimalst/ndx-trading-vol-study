"""Plot only the exact event-cluster metrics admitted by the verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("baseline", "nuisance", "recent_frequency")
LABELS = ("Original forecast", "Count and recency control", "Recent event frequency")


def load_verified(root=ROOT):
    root = Path(root)
    if any(
        (root / f"reports/{name}/failure.json").exists()
        for name in ("event_cluster", "range_alert", "issued_calibration")
    ):
        raise ValueError("A failed study cannot supply a verified figure")
    report = root / "reports/event_cluster"
    proof = json.loads((report / "verification.json").read_bytes())
    payload = (report / "metrics.json").read_bytes()
    if (
        proof.get("status") != "VERIFIED"
        or proof.get("verified_output_hashes", {}).get("reports/event_cluster/metrics.json")
        != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError("Exact verified metric bytes required")
    result = json.loads(payload)
    if result.get("status") == "UNEVALUABLE" or result.get("whole_wave_aborted") is True:
        raise ValueError("Unevaluable trial has no forecast-comparison figure")
    return result


def draw(metrics):
    rows = metrics.get("rows", [])
    keys = [(r.get("candidate"), r.get("control"), r.get("score")) for r in rows]
    if len(keys) != 3 or set(keys) != {("cluster", c, "brier") for c in CONTROLS}:
        raise ValueError("All three candidate/control comparisons required")
    values = {}
    for row in rows:
        phases = row.get("phases", [])
        if len(phases) != 2 or {p["name"] for p in phases} != {"development", "evaluation"}:
            raise ValueError("Both phases must be visible")
        for phase in phases:
            a = np.asarray([phase["delta"], *phase["ci95_envelope"]], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all() or a[1] > a[2]:
                raise ValueError("Finite ordered uncertainty intervals required")
            values[row["control"], phase["name"]] = a * 10000
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.25, top=0.80)
    colors = {"development": "#65788C", "evaluation": "#007D79"}
    for phase, offset in (("development", 0.12), ("evaluation", -0.12)):
        for i, control in enumerate(CONTROLS):
            center, lo, hi = values[control, phase]
            y = i + offset
            ax.plot([lo, hi], [y, y], color=colors[phase], lw=2)
            ax.plot(
                center,
                y,
                "o",
                color=colors[phase],
                markersize=6,
                label=phase.title() if i == 0 else None,
            )
    ax.axvline(0, color="#36434B", lw=1)
    ax.axvline(-5, color="#A45521", lw=1, ls="--", label="Required gain: −5")
    ax.set_yticks(range(3), LABELS)
    ax.set_ylim(2.45, -0.45)
    ax.set_xlabel("Brier loss change × 10,000   •   lower is better")
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Does clustering of past risk events help?", x=0.03, ha="left", fontsize=16)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.40, -0.19), ncol=3, frameon=False, fontsize=9
    )
    fig.text(
        0.03,
        0.015,
        "Intervals are nominal envelopes across the fixed dependence checks.\n"
        "All three controls and all registered statistical gates are required; reused history is exploratory.",
        fontsize=8.5,
        color="#52616A",
    )
    return fig


def main():
    figure = draw(load_verified())
    report = ROOT / "reports/event_cluster"
    figure.savefig(report / "comparison_intervals.png", dpi=180)
    figure.savefig(report / "comparison_intervals.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
