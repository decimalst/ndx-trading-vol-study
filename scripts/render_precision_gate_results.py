"""Render saved verified wave26 means and intervals; no fitting or inference."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/precision_gate/predictive"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metrics = json.loads((REPORT / "metrics.json").read_bytes())
    verified = json.loads((REPORT / "verification.json").read_bytes())
    assert metrics["status"] == "COMPLETED" and verified["status"] == "VERIFIED"
    names = {
        "base": "Market baseline",
        "adaptive": "Adaptive model",
        "constant": "Fixed blend",
    }
    means, lowers, uppers, labels, colors, rows = [], [], [], [], [], []
    for row in metrics["rows"]:
        for phase in row["phases"]:
            m = phase["mean"]
            low, high = phase["ci95_envelope"]
            assert low <= m <= high
            means.append(m)
            lowers.append(m - low)
            uppers.append(high - m)
            period = "2016–2019" if phase["name"] == "development" else "2020–Oct 2025"
            labels.append(names[row["control"]] + " · " + period)
            colors.append("#3467a5" if phase["name"] == "development" else "#ba6328")
            rows.append(
                f"| {names[row['control']]} | {period} | {phase['n']:,} | {m:+.6f} | [{low:+.6f}, {high:+.6f}] |"
            )
    fig, ax = plt.subplots(figsize=(11, 5.7))
    fig.subplots_adjust(left=0.34, right=0.96, top=0.81, bottom=0.24)
    for i, (m, lo, hi, color) in enumerate(zip(means, lowers, uppers, colors)):
        ax.errorbar(m, i, xerr=[[lo], [hi]], fmt="o", color=color, capsize=4, lw=1.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", lw=1)
    ax.axvline(-0.005, color="#7e3153", lw=1, ls="--")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel(
        "QLIKE loss difference: contextual blend minus control (lower is better)", labelpad=10
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    passed = bool(metrics["leads"])
    fig.suptitle(
        "State-dependent model blending",
        x=0.035,
        y=0.965,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.035,
        0.895,
        "Historical exploratory lead; future confirmation required"
        if passed
        else "No qualifying improvement across all three controls",
        fontsize=12,
    )
    fig.text(
        0.035,
        0.08,
        "Bars: saved 95% HAC/block interval envelopes. Dashed line: required daily improvement (−0.005).\nReused historical windows and current data vintages; the later historical period is not untouched confirmation.",
        fontsize=9,
        color="#444444",
    )
    for suffix in ("png", "pdf"):
        path = REPORT / ("comparison_intervals." + suffix)
        if path.exists():
            raise ValueError("Preserve prior rendered artifact")
        fig.savefig(path, dpi=180)
    plt.close(fig)
    verdict = (
        "Exploratory historical lead; future confirmation remains required."
        if passed
        else "No new qualifying predictive improvement."
    )
    body = f"""# Model weights conditioned on current risk

**{verdict}** This fixed experiment compares a state-dependent precision blend against the market baseline, its adaptive counterpart, and a fitted constant blend. Both experts use the same twelve market predictors. The state is daily minus monthly log realized risk; the combination stays between the experts.

| Control | Period | Paired dates | QLIKE difference | 95% interval envelope |
| --- | --- | ---: | ---: | --- |
{chr(10).join(rows)}

Negative differences mean lower forecast loss. A lead requires an improvement of at least0.005 in both phases against all three controls, negative differences in every fixed later slice and calendar offset, all support floors, and both wave and cumulative multiple-testing corrections. Intervals shown here are unadjusted diagnostic envelopes; individual cells do not establish a lead.

![Saved model-blend estimates and uncertainty](comparison_intervals.png)

The common scored sample contains **{metrics["common_scored_origins"]:,} origins**, including **{metrics["common_scored_origins"] - metrics["fitted_gate_scored_origins"]:,} cold-start origins** and **{metrics["fitted_gate_scored_origins"]:,} origins with fitted gate weights**. Cold starts use baseline-valued gate forecasts and remain included. Expert forecasts were newly issued with monthly target-blind schedules. Each gate fit uses only previously issued forecast pairs whose outcomes matured by the preceding full-calendar session, within the fixed1260-session window. It never recalculates old forecasts using a newer expert.

All three new comparisons retain the146 inherited entries, giving **149 registered comparisons**. Previous unevaluable attempts remain counted. Verification independently reconstructs market inputs, targets, every expert fit, gate membership and optimum, scored and unscored applications, and the full-calendar paired inference. The independent gate optimizer differs from the producer; solver iteration counts are range checked rather than claimed to reproduce exactly.

## Qualifications

{metrics["evidence_limitation"]}

The exact six historical-replay regression tests are excluded from this run with their original files and prior-access correction preserved. The existing synthetic inference calibration is reused for the identical daily-mask procedure; it does not establish exact market interval coverage or adjusted-tail calibration. This model experiment introduces no Treasury auction quantities or release clocks. Historical nonqualification is not proof of equivalence or proof that all conditional combinations are useless. A historical lead would still require future confirmation and separate trading evaluation.

Evidence: [metrics](metrics.json), [independent reconstruction](verification.json), [freeze](freeze_record.json), [selected tests](TEST_GATE_RESULT.json), [registration journal](trial_ledger.jsonl), [terminal record](terminal.json), and [PDF figure](comparison_intervals.pdf).
"""
    with (REPORT / "SUMMARY.md").open("x") as stream:
        stream.write(body)
    manifest = {
        "status": "RENDERED_FROM_SAVED_VERIFIED_METRICS_ONLY",
        "metrics_sha256": sha(REPORT / "metrics.json"),
        "verification_sha256": sha(REPORT / "verification.json"),
        "renderer_sha256": sha(Path(__file__)),
        "fits_or_inference_rerun": False,
        "artifacts": {
            name: sha(REPORT / name)
            for name in ("SUMMARY.md", "comparison_intervals.png", "comparison_intervals.pdf")
        },
    }
    with (REPORT / "RENDER_MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
