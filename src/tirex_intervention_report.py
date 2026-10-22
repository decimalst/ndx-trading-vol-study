"""Render saved TiRex intervention outputs without running or refitting the model."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _read(path):
    return json.loads(path.read_text())


def _number(value):
    return f"{float(value):.7g}"


def _interval(values):
    return f"[{_number(values[0])}, {_number(values[1])}]"


def _label(candidate):
    return f"L{candidate['layer']}/U{candidate['unit']}"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _magnitude_ci(values, seed):
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(5000, len(values)))
    return np.quantile(values[samples].mean(axis=1), [0.025, 0.975])


def _raw_magnitudes(run_dir, validation, results):
    """Compute per-pair candidate and matched-control magnitudes from saved rows."""
    rows = []
    baseline = validation["quantiles"].astype(np.float64)
    for index, result in enumerate(results):
        candidate = result["candidate"]
        layer, unit = candidate["layer"], candidate["unit"]
        mask = validation["family"] == candidate["family"]
        if int(mask.sum()) != result["primary"]["n_pairs"]:
            raise ValueError("Raw own-family pair count differs from summary")
        with np.load(run_dir / f"validation_swap_L{layer}_U{unit}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["pair_id"], validation["pair_id"]):
                raise ValueError("Candidate swap pair IDs differ from baseline")
            deltas = saved["quantiles"].astype(np.float64)[:, :, 1] - baseline[:, :, 1]
        control_deltas = []
        for control in candidate["controls"]:
            with np.load(run_dir / f"validation_swap_L{layer}_U{control}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["pair_id"], validation["pair_id"]):
                    raise ValueError("Control swap pair IDs differ from baseline")
                control_deltas.append(
                    saved["quantiles"].astype(np.float64)[:, :, 1] - baseline[:, :, 1]
                )
        candidate_magnitude = np.abs(deltas[mask]).mean(axis=1)
        control_magnitude = np.abs(np.stack(control_deltas)[:, mask]).mean(axis=(0, 2))
        if not np.isclose(candidate_magnitude.mean(), result["primary"]["mean_abs_median_shift"], atol=1e-12):
            raise ValueError("Raw candidate magnitude differs from summary")
        if not np.isclose(control_magnitude.mean(), result["primary"]["mean_abs_control_median_shift"], atol=1e-12):
            raise ValueError("Raw control magnitude differs from summary")
        seed = int(result["primary"]["seed"]) + 10000 + index
        rows.append({
            "candidate_mean": float(candidate_magnitude.mean()),
            "control_mean": float(control_magnitude.mean()),
            "candidate_ci": _magnitude_ci(candidate_magnitude, seed),
            "control_ci": _magnitude_ci(control_magnitude, seed),
        })
    return rows


def _activation_counterexamples(validation, results, families):
    """Challenge discovery activation labels on every held-out motif family."""
    output = []
    for result in results:
        candidate = result["candidate"]
        raw = validation[f"layer_{candidate['layer']}"].astype(np.float64)
        unit = candidate["unit"]
        values = candidate["orientation"] * (raw[:, 1, unit] - raw[:, 0, unit]) / candidate["activation_scale"]
        entries = {}
        for family in families:
            for variant in ("all", "short", "long"):
                mask = validation["family"] == family
                if variant != "all":
                    mask &= validation["variant"] == variant
                if not mask.any():
                    continue
                selected = values[mask]
                entries[f"{family}/{variant}"] = {
                    "n_pairs": int(mask.sum()),
                    "mean_oriented_standardized_activation_contrast": float(selected.mean()),
                    "positive_contrast_count": int(np.count_nonzero(selected > 0)),
                    "positive_contrast_fraction": float(np.mean(selected > 0)),
                }
        output.append({"layer": candidate["layer"], "unit": unit,
                       "selection_family": candidate["family"], "descriptive_by_family_variant": entries})
    return output


def _takeaways(results, activation_results):
    paragraphs = []
    for result, activation in zip(results, activation_results, strict=True):
        candidate, primary = result["candidate"], result["primary"]
        family = candidate["family"]
        contrasts = activation["descriptive_by_family_variant"]
        own = contrasts[f"{family}/all"]
        message = (f"- **{family.title()} {_label(candidate)}:** the activation contrast had its discovery sign "
                   f"in {own['positive_contrast_count']}/{own['n_pairs']} new pairs. Mean absolute forecast-median "
                   f"movement was {_number(primary['mean_abs_median_shift'])}, versus "
                   f"{_number(primary['mean_abs_control_median_shift'])} for matched controls (log-variance units).")
        short = contrasts.get(f"{family}/short")
        long = contrasts.get(f"{family}/long")
        if short and long:
            short_mean = short["mean_oriented_standardized_activation_contrast"]
            long_mean = long["mean_oriented_standardized_activation_contrast"]
            if short_mean * long_mean < 0:
                message += (" The mean oriented activation contrast changed sign between short "
                            f"({_number(short_mean)} discovery SD) and long ({_number(long_mean)}) motifs.")
        forecast = result["descriptive_by_family_variant"]
        short = forecast.get(f"{family}/short")
        long = forecast.get(f"{family}/long")
        if short and long:
            short_mean = short["frozen_direction_mean_shift"]
            long_mean = long["frozen_direction_mean_shift"]
            if short_mean * long_mean < 0:
                message += (" The mean signed forecast effect changed sign between short "
                            f"({_number(short_mean)}) and long ({_number(long_mean)}) motifs.")
        paragraphs.append(message)
    return paragraphs


def _draw_figure(run_dir, results, magnitudes, floor):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), sharey=True)
    positions = np.arange(len(results))
    labels = [f"{result['candidate']['family'].title()}\n{_label(result['candidate'])}" for result in results]
    blue, orange = "#0072B2", "#D55E00"
    for index, result in enumerate(results):
        primary = result["primary"]
        lo, hi = primary["influence_mean_ci95"]
        axes[0].hlines(index, lo, hi, color=blue, linewidth=2)
        axes[0].plot(primary["mean_influence_effect"], index, "o", color=blue, markersize=7)
        for key, offset, color, marker in (("candidate", -0.12, blue, "o"), ("control", 0.12, orange, "s")):
            lo, hi = magnitudes[index][key + "_ci"]
            axes[1].hlines(index + offset, lo, hi, color=color, linewidth=2)
            axes[1].plot(magnitudes[index][key + "_mean"], index + offset, marker,
                         color=color, markersize=6, label=key.title() if index == 0 else None)
    axes[0].axvline(0, color="#777777", linewidth=1)
    axes[0].axvline(floor, color="#777777", linestyle="--", linewidth=1,
                    label="Pairwise numerical floor")
    axes[0].set_title("Signed effect in the discovery-fixed direction", loc="left", pad=16)
    axes[0].set_xlabel("Mean signed median shift (10⁻³ log-variance units)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].set_title("Absolute movement: selected unit vs. controls", loc="left", pad=16)
    axes[1].set_xlabel("Mean absolute median shift (10⁻³ log-variance units)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.set_ylim(len(results) - 0.4, -0.6)
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:g}"))
        axis.xaxis.get_offset_text().set_visible(False)
    pair_counts = sorted({result["primary"]["n_pairs"] for result in results})
    pair_text = str(pair_counts[0]) if len(pair_counts) == 1 else "/".join(map(str, pair_counts))
    fig.suptitle("TiRex-2: swapping one native feed-forward unit", x=0.12, y=0.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(0.12, 0.895, f"Held-out synthetic counterexamples · {pair_text} own-family pairs per candidate · zero-based layer/unit IDs", fontsize=10)
    fig.text(0.12, 0.035, "Intervals: descriptive 95% paired bootstrap (5,000 draws). Significance uses six Holm-adjusted binomial tests.\n"
             "Signed effects and absolute magnitudes answer different questions. This is a model-influence test; forecast skill is untested.", fontsize=9)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.77, bottom=0.25, wspace=0.37)
    fig.savefig(run_dir / "comparison_intervals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _test_status(run_dir, tests_log):
    path = tests_log or run_dir / "tests_first" / "ALL_TESTS_GREEN.log"
    if not path.exists() and tests_log is None:
        path = run_dir.parent / "ALL_TESTS_GREEN.log"
    if not path.exists():
        return "The aggregate test log was not supplied to this renderer; consult the test receipts."
    text = path.read_text()
    counts = re.findall(r"Ran (\d+) tests? in", text)
    if not counts or not re.search(r"^OK\s*$", text, re.MULTILINE):
        return "The supplied aggregate test log does not establish a completed passing test suite."
    return f"**{int(counts[-1])} tests passed** in the saved aggregate test log. Test-first receipts preserve the initial pre-implementation failures."


def build_report(run_dir, tests_log=None):
    run_dir = Path(run_dir)
    protocol = _read(run_dir / "protocol.json")
    metadata = _read(run_dir / "model_metadata.json")
    instrumentation = _read(run_dir / "instrumentation.json")
    lock = _read(run_dir / "candidate_lock.json")
    summary = _read(run_dir / "summary.json")
    results = summary["results"]
    if [result["candidate"] for result in results] != lock["candidates"]:
        raise ValueError("Summary candidates differ from the pre-validation lock")
    if _sha(run_dir / "candidate_lock.json") != summary["candidate_lock_sha256"]:
        raise ValueError("Candidate-lock hash differs from completed summary")
    with np.load(run_dir / "validation.npz", allow_pickle=False) as raw:
        validation = {key: raw[key] for key in raw.files}
    magnitudes = _raw_magnitudes(run_dir, validation, results)
    activation_results = _activation_counterexamples(validation, results, protocol["families"])
    (run_dir / "activation_counterexamples.json").write_text(json.dumps(activation_results, indent=2, allow_nan=False) + "\n")
    floor = protocol["inference"]["floor_log_variance"]
    _draw_figure(run_dir, results, magnitudes, floor)
    influence_passes = sum(result["primary"]["influence_pass"] for result in results)
    advantage_passes = sum(result["primary"]["advantage_pass"] for result in results)
    if influence_passes:
        verdict = (f"**{influence_passes} of {len(results)} selected units passed the registered influence criterion** "
                   "on their own synthetic family.")
    else:
        verdict = ("**None of the selected units passed the registered influence criterion** "
                   "on its own synthetic family.")
    verdict += (f" {advantage_passes} of {len(results)} passed the matched-control advantage criterion. "
                "These are tests of local model influence under a specified synthetic generator; "
                "they do not establish a uniquely represented concept or better forecasts.")
    lines = [
        "# TiRex-2 native-unit activation swaps: synthetic pilot", "",
        f"Completed {summary['finished_utc']}. Status: **COMPLETED_SYNTHETIC_INTERVENTION_PILOT**.", "",
        verdict, "",
        *_takeaways(results, activation_results), "",
        "![Signed intervention effects and absolute movement against matched controls](comparison_intervals.png)", "",
        "## What was tested", "",
        f"The pinned `{metadata['checkpoint_id']}` checkpoint `{metadata['checkpoint_revision']}` "
        f"received {protocol['context_length']}-point synthetic log-variance histories and produced "
        f"{protocol['forecast_horizon']}-step q10/q50/q90 forecasts. No market prices, returns, future outcomes, "
        "or real-market prediction targets were used.", "",
        f"Discovery used {protocol['discovery_pairs_per_family']} independent pairs in each of three families; "
        f"validation used {protocol['validation_pairs_per_family']} new pairs per family. Each recipient/donor pair "
        "contains exactly the same multiset and final observation, with context mean −9 and population SD 0.6. "
        "The permutation changes temporal organization. Validation changes motif duration, amplitude and noise "
        "and includes both short and long versions, challenging explanations formed on discovery.", "",
        "| Family | Discovery contrast |", "| --- | --- |",
        "| Persistence | Alternating low/high values versus the same values grouped into runs. |",
        "| Recency | The same positive burst placed earlier versus later, followed by a common quiet tail. |",
        "| Recovery | A noisy rising ramp versus the same values in reverse order. |", "",
        f"Candidate selection searched layers {protocol['layers']} only, using absolute standardized donor-minus-recipient "
        "activation contrast. One unique unit per family was retained. Four control units in the same layer were "
        "matched by pooled activation SD and outgoing-weight norm. Unit identities, controls and the expected "
        "forecast-shift sign were fixed using discovery before any validation forecast. All layer and unit IDs are zero-based.", "",
        f"Each intervention replaced one gated feed-forward scalar at token {protocol['token_index']} "
        "with its paired donor value, in both directions. The hook is immediately before the feed-forward "
        "output projection. The model was not retrained.", "",
        "## Instrumentation and normalization", "",
        f"Overall instrumentation gate: **{'PASS' if instrumentation['passed'] else 'FAIL'}**. "
        "Forecasts from the capture hook are checked against the unhooked public forecast wrapper; same-value "
        "swaps, reversed batch order and singleton versus batched execution are also checked. Positive affine "
        "transformations of each entire input test invariance of normalized activations and corresponding "
        "equivariance of forecasts in the original units.", "",
        "| Check | Maximum forecast absolute error | Maximum activation relative error | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for check in instrumentation["checks"]:
        activation_errors = list(check["activation_relative_errors"].values())
        activation_text = _number(max(activation_errors)) if activation_errors else "not captured"
        lines.append(f"| {check['check']} | {_number(check['max_forecast_abs_error'])} | {activation_text} | {'PASS' if check['passed'] else 'FAIL'} |")
    lines += [
        "", f"The protocol tolerances were {_number(protocol['instrumentation']['max_forecast_abs_error'])} "
        "for forecast absolute error and "
        f"{_number(protocol['instrumentation']['max_activation_relative_error'])} for activation relative error. "
        f"The instrumentation baseline had {instrumentation['base_quantile_crossings']} quantile crossings. "
        "The normalization result limits any claim that a hidden scalar directly measures absolute volatility: "
        "global positive affine changes can leave normalized activations almost unchanged.", "",
        "## Held-out own-family intervention results", "",
        "The signed effect is the discovery-fixed sign multiplied by half the recipient median change minus "
        "the donor median change. The influence null says that at most half of generated pairs exceed the "
        f"strict +{_number(floor)} log-variance floor. Control advantage is the candidate's mean absolute "
        "median change across the two directions minus the corresponding average across four controls. "
        "Its null uses the same strict floor. Exact one-sided binomial p-values are adjusted together by Holm "
        "over all six claims. Bootstrap mean intervals are descriptive, pointwise 95% intervals from 5,000 "
        "pair-resampling draws; whether an interval includes zero is not the registered decision rule.", "",
        "| Candidate / family | Expected recipient sign | Signed mean [95% interval] | Influence successes | Holm p | Influence pass |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for result in results:
        c, p = result["candidate"], result["primary"]
        lines.append(f"| {_label(c)} / {c['family']} | {c['forecast_direction']:+d} | {_number(p['mean_influence_effect'])} {_interval(p['influence_mean_ci95'])} | {p['influence_successes']}/{p['n_pairs']} | {_number(p['influence_holm_p'])} | {str(p['influence_pass']).upper()} |")
    lines += ["", "| Candidate | Absolute median movement | Control movement | Advantage mean [95% interval] | Advantage successes | Holm p | Advantage pass |",
              "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for result in results:
        c, p = result["candidate"], result["primary"]
        lines.append(f"| {_label(c)} | {_number(p['mean_abs_median_shift'])} | {_number(p['mean_abs_control_median_shift'])} | {_number(p['mean_advantage_effect'])} {_interval(p['advantage_mean_ci95'])} | {p['advantage_successes']}/{p['n_pairs']} | {_number(p['advantage_holm_p'])} | {str(p['advantage_pass']).upper()} |")
    lines += ["", "All shifts above are in log-variance units. The 0.001 numerical floor corresponds to approximately "
              "0.1% multiplicative variance change, not a 0.1 percentage-point return, and is not a threshold of "
              "financial materiality. Absolute movement can be nonzero even when the signed effect cancels "
              "or runs against the discovery direction.", "",
              "| Candidate | Native donor-minus-recipient median contrast | Mean recipient median shift | Mean donor median shift | Both directions expected sign | Mean absolute interval-width change | Candidate/base/control crossings |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for result in results:
        c, p = result["candidate"], result["primary"]
        lines.append(f"| {_label(c)} | {_number(p['mean_native_donor_recipient_contrast'])} | {_number(p['mean_recipient_median_shift'])} | {_number(p['mean_donor_median_shift'])} | {p['both_directions_expected_sign_fraction']:.1%} | {_number(p['mean_abs_width_shift'])} | {p['crossing_count']}/{p['base_crossing_count']}/{p['control_crossing_count']} |")
    lines += ["", "Interval width means q90 minus q10 of the forecast log-variance distribution. A width change "
              "is a change in uncertainty about the model's predicted log variance, separately from a change "
              "in its median level. Crossings are counted without quantile rearrangement, and any own-family "
              "candidate, baseline or control crossing blocks a pass.", "",
              "## Counterexamples and off-family effects", "",
              "Every candidate is evaluated on every family. Short and long variants are shown separately. "
              "The signed values use the candidate's discovery-fixed forecast direction. These are descriptive "
              "checks, with no secondary significance declarations or selection of favorable variants.", "",
              "| Candidate | Test family | Relationship | All n / signed mean | Short n / signed mean | Long n / signed mean |",
              "| --- | --- | --- | --- | --- | --- |"]
    for result in results:
        c = result["candidate"]
        descriptions = result["descriptive_by_family_variant"]
        for family in protocol["families"]:
            cells = []
            for variant in ("all", "short", "long"):
                entry = descriptions.get(f"{family}/{variant}")
                cells.append(f"{entry['n_pairs']} / {_number(entry['frozen_direction_mean_shift'])}" if entry else "0 / not estimable")
            relationship = "own family" if family == c["family"] else "other family"
            lines.append(f"| {_label(c)} | {family} | {relationship} | {' | '.join(cells)} |")
    lines += ["", "The activation explanation is challenged separately from forecast influence. The next table "
              "uses donor-minus-recipient activation difference, oriented by its discovery sign and divided "
              "by the pooled discovery activation SD. Each cell is **mean standardized contrast; positive-sign "
              "count / pair count**. A stable positive contrast supports generalization of a response pattern; "
              "a large response on another family weakens any claim that the label is exclusive. These "
              "comparisons are descriptive and add no significance claims.", "",
              "| Candidate | Test family | All contrast; sign count | Short contrast; sign count | Long contrast; sign count |",
              "| --- | --- | --- | --- | --- |"]
    for candidate in activation_results:
        for family in protocol["families"]:
            cells = []
            for variant in ("all", "short", "long"):
                entry = candidate["descriptive_by_family_variant"].get(f"{family}/{variant}")
                cells.append(f"{_number(entry['mean_oriented_standardized_activation_contrast'])}; {entry['positive_contrast_count']}/{entry['n_pairs']}" if entry else "not estimable")
            lines.append(f"| {_label(candidate)} | {family} | {' | '.join(cells)} |")
    lines += ["", "Full activation-counterexample summaries are saved in "
              "[activation_counterexamples.json](activation_counterexamples.json).", "",
              "## Limits of the interpretation", "",
              "- These histories are synthetic log variance, not prices or returns. No realized future truth is "
              "available, so forecast accuracy, calibration, portfolio returns and trading alpha were not tested.",
              "- Within-pair global distributions and the last value are matched. Recent means, HAR-style "
              "volatility summaries, slopes and autocorrelation are not all matched, so a motif label does "
              "not uniquely identify the computation being represented.",
              "- Selection searched three preselected layers and one token. This does not survey every unit, "
              "exclude distributed mechanisms, or establish superiority of native units over rotated representations.",
              "- The transplanted scalar is observed in a natural forward pass, but the resulting hybrid "
              "state can be off manifold. An intervention effect establishes local model sensitivity under "
              "this operation, not concept exclusivity or a complete causal account of forecasting.",
              "- Validation p-values are conditional on the declared synthetic generator and its mixture "
              "of short and long motifs. They are not population claims about financial histories.",
              "- Small effects below the predeclared floor can be measurable. Failure of the criterion "
              "does not prove that a unit has exactly zero influence; a passing result does not by itself "
              "make its influence economically meaningful.", "",
              "## Verification and reproduction", "",
              _test_status(run_dir, tests_log), "",
              "See the [aggregate test log](tests_first/ALL_TESTS_GREEN.log) and "
              "[independent reconstruction receipt](independent_verification.json). "
              "The independent checker reconstructs saved summaries from raw interventions without rerunning TiRex.", "",
              f"The checkpoint weights SHA256 is `{metadata['weights_sha256']}`. "
              f"Execution used Python {metadata['python']}, TiRex-2 {metadata['versions']['tirex-2']}, "
              f"PyTorch {metadata['versions']['torch']}, {metadata['device']} and {metadata['threads']} threads. "
              "Compilation, flex attention and both test-time augmentation options were disabled. "
              "[Model metadata](model_metadata.json) includes source hashes and version details.", "",
              "The [protocol](protocol.json), [pre-execution hashes](pre_execution.json), "
              "[candidate lock](candidate_lock.json), [instrumentation](instrumentation.json), "
              "[summary](summary.json), and raw discovery/validation arrays are retained. "
              "The report reconstructs absolute movement from the saved validation swaps and checks "
              "pair identities, counts and candidate-lock consistency before plotting.", "",
              "To repeat from the repository environment with the pinned checkpoint available:", "",
              "```sh",
              "python -m src.tirex_intervention_run --protocol tirex_intervention_pilot.json --output reports/tirex_intervention_pilot/new_run",
              "python -m src.tirex_intervention_report --run-dir reports/tirex_intervention_pilot/new_run",
              "python scripts/verify_tirex_intervention_pilot.py --run-dir reports/tirex_intervention_pilot/new_run --source-root . --output reports/tirex_intervention_pilot/new_run/independent_verification.json",
              "```", "",
              "The run output must be a new directory. Existing run output is not overwritten. "
              "The report command reads completed outputs and does not load or run TiRex.", ""]
    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    receipt = {"report_generator_sha256": _sha(Path(__file__)),
               "inputs": {name: _sha(run_dir / name) for name in ("protocol.json", "model_metadata.json", "instrumentation.json", "candidate_lock.json", "summary.json", "validation.npz")},
               "outputs": {name: _sha(run_dir / name) for name in ("REPORT.md", "comparison_intervals.png", "activation_counterexamples.json")}}
    (run_dir / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tests-log", type=Path)
    args = parser.parse_args()
    print(build_report(args.run_dir, args.tests_log))


if __name__ == "__main__":
    main()
