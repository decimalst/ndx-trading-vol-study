"""Render the fixed saved wave27 estimates; no numerical model or inference run."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/joint_copula/predictive"
METRICS_SHA = "d1de728c8fb6909a4ce8ecd6edb055d8b3652693550de140dfca3ff688593442"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = REPORT / "metrics.json"
    if digest(source) != METRICS_SHA:
        raise ValueError("Original saved metrics changed")
    metrics = json.loads(source.read_bytes())
    if metrics["status"] != "COMPLETED" or metrics["leads"] != ["joint_copula"]:
        raise ValueError("Exact completed historical lead required")
    destinations = [REPORT / ("comparison_intervals." + ext) for ext in ("png", "pdf")]
    manifest = REPORT / "RENDER_MANIFEST.json"
    if any(path.exists() for path in [*destinations, manifest]):
        raise ValueError("Existing rendered evidence must be preserved")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    colors = ["#285979", "#008477"]
    rendered = []
    for ax, row, title in zip(axes, metrics["rows"], ("Against Gaussian dependence", "Against independence"), strict=True):
        for i, phase in enumerate(row["phases"]):
            y = 1 - i
            mean = phase["mean"]
            low, high = phase["ci95_envelope"]
            ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o", color=colors[i], markersize=8, capsize=5, linewidth=2.2)
            ax.annotate(f"{mean:+.4f}", (mean, y), xytext=(0, 17), textcoords="offset points", ha="center", color=colors[i], weight="bold")
            rendered.append({"control": row["control"], "phase": phase["name"], "mean": mean, "ci95_envelope": [low, high], "n": phase["n"]})
        ax.axvline(0, color="#7B8289", linewidth=1)
        ax.axvline(-.005, color="#B27425", linestyle="--", linewidth=1.3)
        ax.set_ylim(-.55, 1.55)
        ax.set_yticks([1, 0], ["2016–2019\nn = 1,005", "2020–Oct 2025\nn = 1,457"])
        ax.set_title(title, pad=18, weight="bold")
        ax.set_xlabel("Change in joint forecast loss (nats per pair)", labelpad=12)
        ax.grid(axis="x", alpha=.14)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlim(-.048, .004)
    axes[1].set_xlim(-.92, .04)
    fig.suptitle("Fixed t8 dependence improved joint index forecasts", x=.065, ha="left", y=.98, fontsize=17, weight="bold", color="#17384A")
    fig.text(.065, .89, "QQQ ETF and S&P 500 price index  •  Identical individual-index forecasts in every arm", fontsize=11, color="#4D5962")
    fig.text(.065, .12, "Negative values are improvements. Dashed line: required −0.005 mean improvement.", fontsize=10)
    fig.text(.065, .055, "Saved unadjusted 95% interval envelopes; qualification uses separate multiplicity corrections.\nExploratory reused history and current vendor snapshots. No trading-performance claim.", fontsize=9, color="#53616C")
    fig.subplots_adjust(left=.15, right=.98, top=.77, bottom=.28, wspace=.27)
    for path in destinations:
        fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    if digest(source) != METRICS_SHA:
        raise ValueError("Metrics changed during rendering")
    record = {"status": "RENDERED_FROM_SAVED_ESTIMATES_ONLY", "metrics_sha256": METRICS_SHA,
              "script_sha256": digest(Path(__file__)), "estimates": rendered,
              "outputs": {str(path.relative_to(ROOT)): digest(path) for path in destinations},
              "model_or_inference_rerun": False}
    with manifest.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
