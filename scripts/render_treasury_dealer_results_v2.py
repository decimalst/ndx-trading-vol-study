"""Render only saved, verified Treasury summaries; no forecast or inference run."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/treasury_dealer/predictive"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    terminal = json.loads((REPORT / "terminal.json").read_bytes())
    if terminal["status"] != "COMPLETED":
        raise ValueError("Completed verified Treasury run required")
    if sha(REPORT / "metrics.json") != terminal["metrics_sha256"]:
        raise ValueError("Saved metric hash mismatch")
    m = json.loads((REPORT / "metrics.json").read_bytes())
    v = json.loads((REPORT / "verification.json").read_bytes())
    if v["status"] != "VERIFIED" or m["leads"]:
        raise ValueError("This renderer describes the fixed nonqualifying result")
    for name in ("SUMMARY_V2.md", "comparison_intervals_v2.png", "comparison_intervals_v2.pdf", "RENDER_MANIFEST_V2.json"):
        if (REPORT / name).exists():
            raise ValueError("Refusing to overwrite a rendered research artifact")
    labels = {"matched": "Matched auction controls", "market": "Market baseline"}
    periods = {"development": "2016–2019", "evaluation": "2020–Oct 2025"}
    lines = [
        "# Treasury dealer allocation and five-session QQQ volatility", "",
        "**No new qualifying predictive signal.** The fixed experiment completed, passed sample support, and was independently reconstructed. Both comparisons failed the required improvement and stability gates.", "",
        "The candidate adds the newly available dealer-allocation surprise to the existing market predictors, delayed Treasury ETF context and matched auction controls. The matched comparison isolates the surprise's additional value; the market comparison evaluates the whole added block. Negative QLIKE differences mean lower forecast loss.", "",
        "| Control | Period | Daily dates | Daily loss difference | 95% interval envelope | Auction dates | Auction-date difference |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in m["rows"]:
        for phase in row["phases"]:
            d, a = phase["daily"], phase["active"]
            lo, hi = d["ci95_envelope"]
            lines.append(f"| {labels[row['control']]} | {periods[phase['name']]} | {d['n']:,} | {d['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {a['n']:,} | {a['mean']:+.6f} |")
    lines += [
        "", "The required daily improvement was at least 0.005 in **both** periods against **both** controls, with negative auction-date means, consistent fixed slices and offsets, and both multiple-testing corrections. Both whole-comparison adjusted p-values are 1.0. These probabilities apply to the complete comparisons, not individual table cells.", "",
        "Against matched auction controls, development deteriorated slightly and the evaluation change was nearly zero. The matched evaluation difference also turned positive in 2023–2025, both daily and on auction dates. Against the original market baseline, daily loss increased in both main periods. Several calendar offsets failed the consistency requirement. There is no basis to promote or tune this inspected specification.", "",
        "![Saved Treasury comparison estimates and uncertainty](comparison_intervals_v2.png)", "",
        f"Independent verification covered {m['common_scored_origins']:,} scored origins ({m['common_active_origins']:,} auction-activation dates), {v['forecasts']['forecasts_verified']:,} scored forecasts and {v['forecasts']['application_forecasts_verified']:,} total issued forecasts. It reconstructed {v['forecasts']['monthly_fits']} monthly refits and {v['forecasts']['model_fits']} individual model fits, all 1,138 source-event feature audits, every sample-support floor, eight phase/endpoint comparisons and 24 block-inference calculations. The complete calendar has {v['forecasts']['calendar_rows']:,} positions. All 3,252 selected repository tests passed before the freeze; six historical replay tests were explicitly quarantined.", "",
        "The two new comparisons preserve all 144 inherited entries, giving **146 registered comparisons**. Earlier unevaluable attempts remain in that family. This result is a completed nonqualifying test, not an insufficient-data result and not proof that every Treasury-related feature is useless or exactly equivalent.", "",
        "## Source and validation qualifications", "",
        "The ledger contains 1,135 known events, two events with unknown quantities, and one documented test auction excluded. January 27, 2020 Z52 uses only a qualified release-date assumption from cached Treasury-attributed text; the original PDF and historical delivery timestamp remain unavailable, and no quantities from that text are admitted. Rejecting the assumption leaves the previously documented unbounded strict-clock uncertainty. The saved result is conditional on this disclosed timing treatment.", "",
        "All numerical Treasury forecasting inputs stop at October 20, 2025. **The later historical period cannot be called untouched:** six old regression tests accessed or refitted it during earlier full-suite runs. That claim has been corrected in the [historical-access disclosure](../predictive_prefit/HISTORICAL_TEST_ACCESS_CORRECTION.md). Those tests were excluded by exact ID for this run, with original files preserved. Prior aggregate results also already exposed the later period. Current bounded decoding does not undo that access.", "",
        "Historical development and evaluation windows have been reused. Current archives do not certify every historical revision or intraday public release. The inherited vendor price field/vintage and implied-volatility history limitations remain. The fixed synthetic interval calibration passed its 90% diagnostic criterion but does not guarantee nominal market coverage or adjusted-tail calibration. This study establishes no causal auction effect, trading profit or untouched-future confirmation.", "",
        "## Next experiment", "",
        "Retain this result and return to the already designed model-state blend on admitted market data. That proposal combines the established baseline with a gradually adapting model and compares it with both individual models and a fitted constant blend. Its pure core is tested, but its historical issuance pipeline and prospective registration remain to be completed. The family and wave numbers must begin from the now-recorded 146 comparisons. A new source-engineering sweep is not required for that modeling question.", "",
        "Evidence: [metrics](metrics.json), [independent reconstruction](verification.json), [sample support](support.json), [freeze](freeze_record.json), [selected-test results](TEST_GATE_RESULT.json), [exact test selection](TEST_SELECTION.json), [terminal record](terminal.json), and [PDF figure](comparison_intervals_v2.pdf).",
    ]
    (REPORT / "SUMMARY_V2.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    colors = {"development": "#9a5b30", "evaluation": "#176878"}
    for ax, endpoint, title in zip(axes, ("daily", "active"), ("All scored dates", "Auction activation dates")):
        for pos, (row, phase) in enumerate((r,p) for r in m["rows"] for p in r["phases"]):
            stat = phase[endpoint]
            mean, (lo,hi) = stat["mean"], stat["ci95_envelope"]
            ax.errorbar(mean, pos, xerr=[[mean-lo],[hi-mean]], fmt="o", color=colors[phase["name"]], capsize=4, markersize=6, linewidth=2)
        ax.axvline(0,color="#505050",linewidth=1)
        if endpoint == "daily":
            ax.axvline(-0.005,color="#777777",linestyle="--",linewidth=1.2,label="Required daily gain")
            ax.legend(loc="upper right",fontsize=8,frameon=False)
        ax.set_yticks(range(4), [f"{labels[r['control']]}\n{periods[p['name']]}" for r in m["rows"] for p in r["phases"]],fontsize=9)
        ax.invert_yaxis()
        ax.set_ylim(3.6,-0.6)
        ax.set_title(title,loc="left",fontsize=12,fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel("Change in QLIKE loss  ← improvement",fontsize=9)
        ax.grid(axis="x",alpha=0.18)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0,pad=7)
    fig.suptitle("Treasury dealer allocation did not improve forecasts consistently",x=0.02,ha="left",fontsize=15,fontweight="bold")
    fig.text(0.02,0.055,"Points: saved paired means · Bars: nominal 95% interval envelopes · Both adjusted comparison p-values = 1.0",fontsize=9)
    fig.text(0.02,0.018,"Exploratory history; qualified Z52 release clock. The later historical period was previously accessed by regression tests.",fontsize=8,color="#555555")
    fig.tight_layout(rect=[0,0.10,1,0.94],w_pad=3.0)
    for suffix in ("png","pdf"):
        fig.savefig(REPORT/f"comparison_intervals_v2.{suffix}",dpi=180,facecolor="white")
    plt.close(fig)
    manifest = {"status":"RENDERED_FROM_SAVED_VERIFIED_METRICS_ONLY","metrics_sha256":sha(REPORT/"metrics.json"),
                "renderer_sha256":sha(Path(__file__)),"artifacts":{n:sha(REPORT/n) for n in ("SUMMARY_V2.md","comparison_intervals_v2.png","comparison_intervals_v2.pdf")},
                "fits_or_inference_rerun":False}
    with (REPORT/"RENDER_MANIFEST_V2.json").open("x") as stream:
        json.dump(manifest,stream,indent=2);stream.write("\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
