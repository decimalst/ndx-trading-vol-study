"""Plot both registered calendar replication contrasts after verification."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = (("calendar", "baseline", "qlike"), ("calendar", "mean", "qlike"))
PHASES = ("development", "evaluation")


def validated_rows(metrics, verification, protocol_hash):
    """Reject missing comparisons, uncertain verification, or stale protocol links."""
    if (verification.get("status") != "VERIFIED"
            or verification.get("protocol_sha256") != protocol_hash
            or metrics.get("protocol_sha256") != protocol_hash):
        raise ValueError("Independent verification of this protocol required before rendering")
    rows = metrics.get("rows", [])
    keys = [(r["candidate"], r["control"], r["score"]) for r in rows]
    if len(keys) != len(CONTRASTS) or set(keys) != set(CONTRASTS):
        raise ValueError("Both registered calendar comparisons required")
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
            if (len(interval) != 2
                    or not all(math.isfinite(x) for x in [phase["delta"], *interval])
                    or interval[0] > interval[1]):
                raise ValueError("Finite ordered paired intervals required")
        ordered.append({**row, "phases": arranged})
    return ordered


def main():
    folder = ROOT / "reports/calendar_variance"
    verification = json.loads((folder / "verification.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    protocol_hash = hashlib.sha256((ROOT / "calendar_variance.yaml").read_bytes()).hexdigest()
    rows = validated_rows(metrics, verification, protocol_hash)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, row in zip(axes, rows, strict=True):
        for y, phase, color in zip([1, 0], row["phases"], ["#4a72a5", "#cc6c39"], strict=True):
            left, right = phase["ci95_envelope"]
            ax.plot([left, right], [y, y], color=color, linewidth=2)
            ax.scatter(phase["delta"], y, color=color, s=40, zorder=3)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.axvline(-.005, color="#999999", linestyle=":", linewidth=1)
        control = "market baseline" if row["control"] == "baseline" else "historical mean"
        ax.set_title(f"Calendar block vs {control}", loc="left", fontsize=11)
        ax.set_yticks([1, 0], ["2016–2019", "2020–2025"])
        ax.set_ylim(-.65, 1.65)
        ax.grid(axis="x", alpha=.16)
        ax.set_xlabel("Calendar minus control loss (negative is better)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    fig.suptitle("Do original announcement plans improve next-session SPX risk forecasts?",
                 x=.06, ha="left", fontsize=13)
    fig.text(.06, .035,
             "Lines: nominal 95% block-bootstrap/HAC interval envelopes. Dotted lines: fixed −0.005 effect gate.\n"
             "Both comparisons must pass both periods, stability and multiplicity gates. Archival daily OHLC proxy replication.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(.03, .14, .99, .90), w_pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"comparison_intervals.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
