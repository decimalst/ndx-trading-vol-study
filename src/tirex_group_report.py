"""Render the group intervention results without changing saved measurements."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCALE = 1000.0
UNIT_LABEL = "10⁻³ log-variance units"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, field):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite reporting field: {field}")
    return result


def _fmt(value, digits=3):
    return f"{_number(value, 'display'):.{digits}f}"


def _pvalue(value):
    return f"{_number(value, 'pvalue'):.4g}"


def _label(group):
    return f"{group['family'].capitalize()} · L{group['layer']} · {group['size']} units"


def _crossings(primary):
    return sum(int(primary[key]) for key in (
        "base_crossing_count", "crossing_count", "control_crossing_count",
        "singleton_crossing_count"))


def _normalise_results(summary):
    results = summary["results"]
    if len(results) != 6:
        raise ValueError("the registered report requires all six group results")
    for result in results:
        group, primary = result["group"], result["primary"]
        if len(group["units"]) != group["size"]:
            raise ValueError("group size and unit list disagree")
        expected = (all(primary[key] for key in (
            "influence_pass", "advantage_pass", "singleton_pass")) and _crossings(primary) == 0)
        if bool(primary["qualified"]) != expected:
            raise ValueError("qualification disagrees with saved tests and crossing gate")
        for key in ("mean_influence_effect", "mean_abs_median_shift",
                    "mean_abs_control_median_shift", "singleton_advantage_mean",
                    "mean_abs_joint_minus_sum_median", "mean_pairwise_singleton_envelope"):
            _number(primary[key], key)
        interval = np.asarray(primary["influence_mean_ci95"], dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError("invalid influence interval")
    return results


def _singleton_envelope(primary):
    return primary["mean_pairwise_singleton_envelope"]


def _matching_summary(group):
    member_rms = np.array([item["rms_difference"] for item in group["members"]], dtype=float)
    if member_rms.shape != (group["size"],) or not np.isfinite(member_rms).all() or np.any(member_rms <= 0):
        raise ValueError("invalid member RMS differences in matching record")
    detail = group["control_matching"]
    ratios = []
    for control in detail:
        rms_ratio = np.array([item["rms_difference_ratio"] for item in control], dtype=float)
        if rms_ratio.shape != member_rms.shape or not np.isfinite(rms_ratio).all():
            raise ValueError("invalid control RMS ratios in matching record")
        ratios.append(float(np.sqrt(np.sum((member_rms * rms_ratio) ** 2) / np.sum(member_rms ** 2))))
    ranges = {}
    for field in ("sd_ratio", "norm_ratio", "rms_difference_ratio"):
        values = np.array([item[field] for control in detail for item in control], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("control matching ratios must be finite and positive")
        ranges[field] = [float(values.min()), float(values.max())]
    return {"input_l2_rms_ratios": ratios, "slot_ratio_ranges": ranges}


def _plot(results, floor, target):
    colors = {"group": "#235789", "controls": "#7D8793", "singleton": "#BB7926",
              "short": "#126E82", "long": "#A4508B"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), sharey=True)
    fig.subplots_adjust(left=.20, right=.98, top=.80, bottom=.31, wspace=.27)
    rows = np.arange(len(results))
    labels = [_label(item["group"]) for item in results]
    for index, result in enumerate(results):
        primary = result["primary"]
        mean = SCALE * primary["mean_influence_effect"]
        low, high = SCALE * np.asarray(primary["influence_mean_ci95"])
        # Draw endpoints directly; percentile intervals need not straddle means.
        axes[0].plot([low, high], [index, index], color=colors["group"], lw=2)
        axes[0].plot([low, low], [index-.065, index+.065], color=colors["group"], lw=1.3)
        axes[0].plot([high, high], [index-.065, index+.065], color=colors["group"], lw=1.3)
        axes[0].scatter([mean], [index], s=52,
                        marker="D" if primary["qualified"] else "o", color=colors["group"], zorder=4)
    axes[0].axvline(0, color="#ADADAD", lw=1)
    axes[0].axvline(SCALE * floor, color="#8B4B32", ls=":", lw=1.5, label="Per-pair effect floor")
    axes[0].set_title("A  Signed influence\nMean and descriptive 95% interval", loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), fontsize=9)
    for offset, key, label, color in (
        (-.22, "mean_abs_median_shift", "Selected group", colors["group"]),
        (0, "mean_abs_control_median_shift", "Matched groups", colors["controls"]),
        (.22, None, "Singleton envelope", colors["singleton"]),
    ):
        values = [SCALE * (item["primary"][key] if key else _singleton_envelope(item["primary"]))
                  for item in results]
        axes[1].barh(rows + offset, values, height=.19, color=color, label=label)
    axes[1].set_title("B  Absolute median movement\nGroup and comparison magnitudes", loc="left", fontsize=11)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), fontsize=9)
    for variant, offset in (("short", -.12), ("long", .12)):
        xs, ys = [], []
        for index, result in enumerate(results):
            description = result["descriptive_by_family_variant"].get(
                f"{result['group']['family']}/{variant}")
            if description is not None:
                xs.append(SCALE * description["mean_signed_shift"])
                ys.append(index + offset)
        axes[2].scatter(xs, ys, marker="o" if variant == "short" else "s",
                        s=45, color=colors[variant], label=f"{variant.capitalize()} motif")
    axes[2].axvline(0, color="#ADADAD", lw=1)
    axes[2].set_title("C  Changed motif lengths\nSigned mean; descriptive only", loc="left", fontsize=11)
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), fontsize=9)
    axes[0].set_yticks(rows, labels)
    axes[0].set_ylim(len(results)-.5, -.5)
    for axis in axes:
        axis.set_xlabel(UNIT_LABEL, labelpad=9)
        axis.xaxis.grid(True, color="#E8E8E8", lw=.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0, pad=12)
    qualified = sum(bool(item["primary"]["qualified"]) for item in results)
    fig.text(.04, .935, "TiRex: simultaneous interventions on small groups of native units",
             fontsize=19, weight="bold", ha="left")
    fig.text(.04, .884, f"{qualified}/6 groups meet all registered criteria · fresh synthetic validation · "
             "18 claims adjusted together", fontsize=12, color="#444444", ha="left")
    fig.text(.04, .04,
             "All axes use milli log-variance units: 1 = 0.001. Signed effects use directions fixed on discovery data.\n"
             "The dotted line is a per-pair floor; registered tests concern the probability of exceeding it, not the plotted mean.\n"
             "The singleton envelope takes the largest constituent-unit magnitude separately for each pair. "
             "Intervals and motif splits are descriptive.\n"
             "Qualification requires influence, advantage over matched groups and advantage over constituent singletons, "
             "plus ordered quantiles. No forecast-accuracy claim.",
             fontsize=9.5, color="#444444", linespacing=1.65, ha="left", va="bottom")
    fig.savefig(target, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _markdown(results, protocol, summary):
    by_id = {item["group"]["group_id"]: item["primary"] for item in results}
    recency = by_id["recency_k8"]
    persistence = by_id["persistence_k8"]
    recovery = by_id["recovery_k8"]
    control_ratio = persistence["mean_abs_control_median_shift"] / persistence["mean_abs_median_shift"]
    qualified = sum(bool(item["primary"]["qualified"]) for item in results)
    counts = {key: sum(bool(item["primary"][key]) for item in results)
              for key in ("influence_pass", "advantage_pass", "singleton_pass")}
    n_pairs = sorted({int(item["primary"]["n_pairs"]) for item in results})
    floor = protocol["inference"]["floor_log_variance"]
    lines = [
        "# TiRex native-unit group intervention pilot", "",
        f"**{qualified} of 6 registered groups met all three tests and the quantile-order gate.** "
        f"Influence passed for {counts['influence_pass']}/6 groups, advantage over matched groups for "
        f"{counts['advantage_pass']}/6, and advantage over the constituent-singleton envelope for "
        f"{counts['singleton_pass']}/6.", "",
        "This experiment measures how simultaneous replacements of native activations change a fixed "
        "TiRex model's forecasts on synthetic histories. It does not measure forecast accuracy, "
        "financial calibration, realized returns, or trading value.", "",
        "The swaps do change forecasts. However, none of the selected groups demonstrates "
        "the predeclared combination of consistent influence, superiority to matched groups, "
        "and superiority to its constituent units. This does not establish zero influence "
        "or reject distributed representation elsewhere in the model.", "",
        f"- The eight-unit recency group moves the median by "
        f"{recency['mean_abs_median_shift']:.6f} on average in absolute log-variance units; "
        f"the per-pair constituent-singleton envelope is already "
        f"{recency['mean_pairwise_singleton_envelope']:.6f}.",
        f"- The eight-unit persistence controls move forecasts about {control_ratio:.1f} times "
        "as much as the selected group. Approximate control matching is documented below.",
        f"- The eight-unit recovery group's held-out signed mean is "
        f"{recovery['mean_influence_effect']:.6f}, opposite to its discovery-frozen direction.", "",
        "Joint changes are generally close to the sum of the individual changes under these "
        "interventions. Additivity is compatible with distributed coding; it is not evidence "
        "that a representation must be localized.", "",
        "![Registered group comparisons](comparison_intervals.png)", "",
        "## Registered comparisons", "",
        "All effect sizes below are in **milli log-variance units**: a value of 1 means 0.001 "
        "in the native log-variance forecast. Signed influence uses each group's direction fixed "
        "on discovery data. The interval is a descriptive, pointwise 95% paired-bootstrap interval; "
        "it is not the registered test.", "",
        "| Family / layer / size | Signed mean [95% interval] | Influence p, Holm | "
        "Matched-group p, Holm | Singleton p, Holm | Crossings | Qualified |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        group, primary = result["group"], result["primary"]
        low, high = np.asarray(primary["influence_mean_ci95"]) * SCALE
        lines.append(f"| {_label(group)} | {_fmt(primary['mean_influence_effect'] * SCALE)} "
            f"[{_fmt(low)}, {_fmt(high)}] | {_pvalue(primary['influence_holm_p'])} | "
            f"{_pvalue(primary['advantage_holm_p'])} | {_pvalue(primary['singleton_holm_p'])} | "
            f"{_crossings(primary)} | {'Yes' if primary['qualified'] else 'No'} |")
    lines.extend([
        "",
        f"Each primary comparison uses {', '.join(map(str, n_pairs))} independent own-family pairs. "
        f"The strict numerical floor is {floor:g} log-variance units. Each of 18 one-sided exact "
        "binomial tests asks whether more than half of the generator's pairs exceed its floor; "
        "Holm correction covers all 18 claims. These are not tests of a population mean. "
        "A qualifying group must pass all three comparisons and have no own-family crossings "
        "in baseline, joint, matched-control, or constituent-singleton forecasts.", "",
        "## Native forecasts and the two swap directions", "",
        "The native contrast is the untouched donor median minus the untouched recipient "
        "median. Recipient and donor columns retain their raw signs; agreement uses the "
        "group's discovery-frozen predicted direction. All shifts and contrasts use milli "
        "log-variance units.", "",
        "| Group | Native donor−recipient contrast | Recipient median shift | "
        "Donor median shift | Both directions have expected sign |",
        "|---|---:|---:|---:|---:|",
    ])
    for result in results:
        group, primary = result["group"], result["primary"]
        values = [primary["mean_native_donor_recipient_contrast"],
                  primary["mean_recipient_median_shift"], primary["mean_donor_median_shift"]]
        lines.append(f"| {_label(group)} | " + " | ".join(_fmt(value * SCALE) for value in values)
            + f" | {_fmt(100 * primary['both_directions_expected_sign_fraction'], 1)}% |")
    lines.extend([
        "",
        "A partial-group replacement can disrupt compensating computations, especially if "
        "the native input-to-forecast contrast is small. Intervention sensitivity therefore "
        "does not establish that this group mediates the original contrast. No effect-to-contrast "
        "ratio is used, because a near-zero denominator can make that ratio misleading.", "",
        "## Effect magnitudes and nonadditivity", "",
        "| Family / layer / size | Group mean absolute shift | Matched groups | "
        "Per-pair singleton envelope | Mean absolute joint-minus-sum | q90−q10 width movement |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        group, primary = result["group"], result["primary"]
        values = [primary["mean_abs_median_shift"], primary["mean_abs_control_median_shift"],
                  _singleton_envelope(primary), primary["mean_abs_joint_minus_sum_median"],
                  primary["mean_abs_width_shift"]]
        lines.append(f"| {_label(group)} | " + " | ".join(_fmt(x * SCALE) for x in values) + " |")
    lines.extend([
        "",
        "The singleton envelope is the mean of the maximum constituent-unit magnitude for each "
        "pair, after averaging that unit's two swap directions; it is not the magnitude of one "
        "globally chosen best singleton. Matched groups average four disjoint matched control "
        "groups and both directions. Joint-minus-sum compares the joint median change against "
        "the sum of constituent changes, each measured from the same original baseline. "
        "Nonadditivity is descriptive and is neither required nor sufficient for qualification. "
        "An interval-width change measures a different output from a median log-variance change.", "",
        "## Approximate control matching", "",
        "Matching uses three discovery quantities, but it does not make every quantity equal. "
        "The table reports control-over-candidate ratios: aggregate RMS Euclidean displacement "
        "of the patched input, followed by the minimum and maximum slot-level matching ratios. "
        "A displacement ratio below 1 means that control moves its group by less on average. "
        "These are descriptive diagnostics, with no post-result exclusions or new pass criteria.", "",
        "| Group | Input displacement RMS ratios, C1 / C2 / C3 / C4 | "
        "Activation-SD slot range | Outgoing-norm slot range | Paired-difference RMS slot range |",
        "|---|---|---:|---:|---:|",
    ])
    for result in results:
        group = result["group"]
        match = _matching_summary(group)
        ratios = " / ".join(_fmt(value) for value in match["input_l2_rms_ratios"])
        ranges = ["–".join(_fmt(value) for value in match["slot_ratio_ranges"][field])
                  for field in ("sd_ratio", "norm_ratio", "rms_difference_ratio")]
        lines.append(f"| {_label(group)} | {ratios} | " + " | ".join(ranges) + " |")
    lines.extend([
        "",
        "For member RMS differences r and slot matching ratios c, the aggregate ratio is "
        "sqrt(sum((r × c)²) / sum(r²)). Some controls therefore apply smaller input displacements, "
        "and individual activation-scale matches can also differ materially. An advantage over "
        "these controls cannot isolate semantic specificity: perturbation size, covariance, "
        "and downstream sensitivity remain alternative explanations. Four-unit rows reuse "
        "prefixes of the eight-unit groups, so their matching slots are not independent "
        "additional matches. See the [independent matching diagnostic](control_match_review.json).", "",
        "## Synthetic counterexamples", "",
        "Every family and both motif lengths are reported below, including off-family responses. "
        "These descriptions add no significance claims. Activation contrast averages member "
        "donor-minus-recipient differences after applying each member's discovery-frozen sign "
        "and dividing by its pooled discovery activation SD. Activation consistency is the "
        "fraction of pairs with a positive mean standardized member contrast; it is not a "
        "probability of a correct forecast or proof that the group represents only the named concept.", "",
        "| Group | Evaluated family / motif | Pairs | Signed shift | Absolute shift | "
        "Both directions agree | Activation contrast | Activation consistency | Width movement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        for name, description in sorted(result["descriptive_by_family_variant"].items()):
            lines.append(
                f"| {_label(result['group'])} | {name} | {description['n_pairs']} | "
                f"{_fmt(description['mean_signed_shift'] * SCALE)} | "
                f"{_fmt(description['mean_abs_median_shift'] * SCALE)} | "
                f"{_fmt(100 * description['both_directions_expected_sign_fraction'], 1)}% | "
                f"{_fmt(description['activation_oriented_mean'])} | "
                f"{_fmt(100 * description['activation_consistency_fraction'], 1)}% | "
                f"{_fmt(description['mean_abs_width_shift'] * SCALE)} |")
    lines.extend([
        "",
        "## What the design establishes", "",
        "Candidate groups contain four or eight nested units from family layers fixed before "
        "this experiment using the prior single-unit pilot's results. Discovery data determine "
        "unit ranking, four disjoint matched control "
        "groups, and forecast directions; fresh validation data determine the reported results. "
        "Group members are replaced simultaneously from the same untouched paired donor state "
        "at context token 63. Layer, unit and token indices are zero-based. "
        "Tests check hook cleanup, no-op identity, normalization, batching "
        "and equivalence of a one-member group to the frozen scalar adapter.", "",
        "The synthetic pairs share their histogram and terminal value, but recent means, slopes, "
        "autocorrelation, and other temporal features can differ. Matching control activation "
        "scales, projection norms and donor-displacement magnitudes does not match covariance "
        "or all downstream influences. Natural donor coordinates can create an off-manifold "
        "hybrid state. Passing all criteria supports aggregate influence for this group under "
        "this intervention; it does not identify an exclusive concept, establish native-basis "
        "superiority over rotated representations, or characterize the whole model.", "",
        "## Research sharing", "",
        "This is a reproducible case study of hypothesis testing with an AI-assisted research "
        "harness: prewritten contracts, a pinned checkpoint, fresh synthetic splits, frozen "
        "selection rules, matched interventions and explicit negative results. The useful "
        "claim is that the experiment tests representation hypotheses through forecast "
        "interventions with controls. It makes no novelty claim, exclusive-concept claim, "
        "or proof of Goodfire's broader hypothesis.", "",
        "The model is credited to Podest and colleagues, "
        "[*TiRex-2: Generalizing TiRex to Multivariate Data and Streaming*]"
        "(https://arxiv.org/abs/2607.01204), as identified by the "
        "[official model documentation](https://nx-ai.github.io/tirex-2/introduction/); "
        "this study extends their released checkpoint with intervention experiments and "
        "does not replicate their forecasting benchmarks.", "",
        "This group study follows the "
        "[earlier single-unit intervention pilot](../tirex_intervention_pilot/REPORT.md) "
        "with fresh synthetic discovery and validation draws.", "",
        "[Goodfire's *Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) "
        "is an informal working-hypothesis essay motivating direct study of native mechanisms "
        "and attempts to challenge explanations.", "",
        "[Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models: "
        "Metrics and Methods*](https://arxiv.org/abs/2309.16042) shows that methodological choices "
        "can change activation-patching conclusions, motivating explicit metrics and controls here.", "",
        "## Verification and reproduction", "",
        "The tests-first implementation and separately implemented verifier were developed "
        "with AI assistance in this repository; this is internal verification, with no "
        "external replication or peer review claimed.", "",
        "Pre-execution checks passed 37 new synthetic contract tests and 19 prior-adapter "
        "regression tests, for 56 tests in total. Test success validates these software "
        "contracts; it does not by itself validate a scientific mechanism.", "",
        "The report reads `summary.json` and `protocol.json` without changing them. The run "
        "directory also contains its source/checkpoint receipts, selected groups and directions, "
        "raw forecasts, activation records, and instrument checks. Inspect those records when "
        "assessing the reported findings; the report receipt records the exact summary and "
        "protocol hashes used to create this figure and text.", "",
        "Evidence: [registered protocol](protocol.json), [full numerical results](summary.json), "
        "[frozen group selection](group_lock.json), [group instrumentation](group_instrumentation.json), "
        "[prewritten tests and receipts](tests_first/), and "
        "[independent verification](independent_verification.json).", "",
        "Run from the repository root in the project environment. A model rerun needs the "
        "pinned checkpoint and a new output directory. The prior pilot supplies context "
        "records for the disjointness check. The saved run's tests-first and execution "
        "receipts document this run, rather than being regenerated by these commands.", "",
        "```bash",
        "python -m unittest tests.test_tirex_group_pilot tests.test_tirex_group_adapter tests.test_tirex_group_run_helpers tests.test_tirex_intervention_adapter",
        "python -m src.tirex_group_run --protocol tirex_group_pilot.json --output reports/tirex_group_pilot/new_run --prior-run reports/tirex_intervention_pilot",
        "python scripts/verify_tirex_group_pilot.py --run-dir reports/tirex_group_pilot/new_run --source-root . --prior-run reports/tirex_intervention_pilot --output reports/tirex_group_pilot/new_run/independent_verification.json",
        "python -m src.tirex_group_report --run-dir reports/tirex_group_pilot/new_run",
        "```", "",
        f"Saved run duration: {_fmt(summary['elapsed_seconds'], 1)} seconds.", "",
    ])
    return "\n".join(lines)


def build_report(run_dir):
    """Write REPORT.md, a three-panel figure and a receipt from completed results."""
    run_dir = Path(run_dir).resolve()
    inputs = {name: _sha(run_dir / name) for name in ("summary.json", "protocol.json")}
    summary = json.loads((run_dir / "summary.json").read_text())
    protocol = json.loads((run_dir / "protocol.json").read_text())
    results = _normalise_results(summary)
    markdown = _markdown(results, protocol, summary)
    _plot(results, protocol["inference"]["floor_log_variance"], run_dir / "comparison_intervals.png")
    (run_dir / "REPORT.md").write_text(markdown)
    if any(_sha(run_dir / name) != value for name, value in inputs.items()):
        raise RuntimeError("report inputs changed during rendering")
    receipt = {"generated_utc": datetime.now(UTC).isoformat(), "input_sha256": inputs,
               "report_source_sha256": _sha(__file__),
               "output_sha256": {name: _sha(run_dir / name) for name in (
                   "REPORT.md", "comparison_intervals.png")},
               "qualification_count": sum(bool(item["primary"]["qualified"]) for item in results),
               "plot_scale": "native log-variance quantities multiplied by 1000",
               "raw_measurements_modified": False}
    (run_dir / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
