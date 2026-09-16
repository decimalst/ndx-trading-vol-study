"""Render finalized, hash-bound commodity results without forecasts or inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports/commodity_implied/predictive"
ARTIFACTS = (
    "metrics.json",
    "freeze_record.json",
    "FULL_PRECHECK.json",
    "FULL_PRECHECK.log",
    "calibration.json",
    "calibration_verification.json",
    "verification.json",
)
OUTPUTS = ("SUMMARY.md", "comparison_intervals.png", "comparison_intervals.pdf")
CONTROL_LABELS = {
    "matched": "Matched commodity histories",
    "market": "Original market baseline",
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(payload):
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)

    result = json.loads(payload, parse_constant=reject)
    if type(result) is not dict:
        raise ValueError("Report object required")
    return result


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite literal report number required")
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative literal report count required")
    return value


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular report artifact required: " + path.name)
    return path.read_bytes()


def load_results(report):
    """Read only the fixed report whitelist; verify the terminal's byte bindings."""
    report = Path(report)
    if report.is_symlink() or not report.is_dir():
        raise ValueError("Existing regular report directory required")
    snapshots = {"terminal.json": _read(report / "terminal.json")}
    terminal = _json(snapshots["terminal.json"])
    bindings = terminal.get("report_artifact_hashes")
    if type(bindings) is not dict:
        raise ValueError("Terminal artifact hashes required")
    present = {
        name for name in ARTIFACTS if (report / name).exists() or (report / name).is_symlink()
    }
    if set(bindings) != present or not set(ARTIFACTS[:-1]) <= present:
        raise ValueError(
            "Exact available artifact bindings and completed prefit proofs required"
        )
    for name in ARTIFACTS:
        if name not in present:
            continue
        signature = bindings[name]
        if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            raise ValueError("Literal artifact SHA256 required")
        payload = _read(report / name)
        if _sha(payload) != signature:
            raise ValueError("Final report hash mismatch: " + name)
        snapshots[name] = payload
    decoded = {
        name: _json(value) for name, value in snapshots.items() if name.endswith(".json")
    }
    metrics = decoded["metrics.json"]
    if (
        terminal.get("metrics_sha256") != _sha(snapshots["metrics.json"])
        or terminal.get("status") != metrics.get("status")
        or terminal.get("leads") != metrics.get("leads")
        or terminal.get("cumulative_hypotheses") != 144
        or metrics.get("hypothesis_count") != 2
        or metrics.get("cumulative_hypothesis_count") != 144
        or type(metrics.get("inherited_rows")) is not list
        or len(metrics["inherited_rows"]) != 142
    ):
        raise ValueError("Terminal and complete144 comparison family disagree")
    for row in metrics["inherited_rows"]:
        if type(row) is not dict or not 0 <= _number(row.get("p_conservative")) <= 1:
            raise ValueError("Complete inherited probabilities required")
    rows = metrics.get("rows")
    if type(rows) is not list or len(rows) != 2:
        raise ValueError("Both fixed comparison rows required")
    for row, control in zip(rows, ("matched", "market")):
        if (
            row.get("study"),
            row.get("candidate"),
            row.get("control"),
            row.get("horizon"),
            row.get("score"),
        ) != ("commodity_implied", "candidate", control, 5, "qlike"):
            raise ValueError("Exact fixed comparison identity required")
        for field in ("p_conservative", "p_holm_wave", "p_holm_cumulative"):
            if not 0 <= _number(row.get(field)) <= 1:
                raise ValueError("Valid stored comparison probabilities required")
    precheck = decoded["FULL_PRECHECK.json"]
    if (
        precheck.get("exit_code") != 0
        or precheck.get("code_unchanged") is not True
        or _count(precheck.get("test_count")) == 0
        or precheck.get("log_sha256") != _sha(snapshots["FULL_PRECHECK.log"])
    ):
        raise ValueError("Successful hash-bound complete precheck required")
    matches = re.findall(
        r"(?m)^Ran ([0-9]+) tests in [0-9.]+s\n\nOK\n", snapshots["FULL_PRECHECK.log"].decode()
    )
    if matches != [str(precheck["test_count"])]:
        raise ValueError("Precheck count disagrees with its log")
    freeze = decoded["freeze_record.json"]
    if (
        freeze.get("status") != "FROZEN_BEFORE_COMMODITY_MARKET_COHORT_OR_FITS"
        or freeze.get("full_test_count") != precheck["test_count"]
        or freeze.get("cumulative_hypotheses_before_registration") != 142
    ):
        raise ValueError("Completed prospective freeze metadata required")
    calibration = decoded["calibration.json"]
    if (
        calibration.get("status") != "PASS"
        or decoded["calibration_verification.json"].get("status") != "VERIFIED"
        or not 0 <= _number(calibration.get("envelope_coverage")) <= 1
        or _count(calibration.get("replications")) == 0
    ):
        raise ValueError("Verified successful generated calibration required")
    status = metrics["status"]
    if status == "COMPLETED":
        proof = decoded.get("verification.json", {})
        if (
            proof.get("status") != "VERIFIED"
            or proof.get("protocol_sha256") != freeze.get("protocol_sha256")
            or proof.get("forecasts", {}).get("status") != "VERIFIED"
            or proof.get("scores", {}).get("status") != "VERIFIED"
        ):
            raise ValueError("Independent forecast and score verification required")
        counts = proof["forecasts"]
        for field in (
            "application_origins",
            "application_forecasts_verified",
            "scored_origins",
            "forecasts_verified",
            "monthly_fits",
            "model_fits",
            "coverage_origins",
            "calendar_rows",
        ):
            _count(counts.get(field))
        score_proof = proof["scores"]
        if (
            counts["scored_origins"] != _count(metrics.get("common_scored_origins"))
            or score_proof.get("common_scored_origins") != counts["scored_origins"]
            or score_proof.get("hypothesis_count") != 2
            or score_proof.get("cumulative_hypothesis_count") != 144
            or counts["forecasts_verified"] != 3 * counts["scored_origins"]
            or counts["application_forecasts_verified"] != 3 * counts["application_origins"]
            or counts["model_fits"] != 3 * counts["monthly_fits"]
        ):
            raise ValueError("Verified completed counts disagree")
        all_pass = all(row.get("verdict") == "COMPARISON_GATE_PASS" for row in rows)
        if metrics["leads"] != (["commodity_implied"] if all_pass else []):
            raise ValueError("Stored joint decision disagrees with row verdicts")
        for row in rows:
            if row.get("verdict") not in ("COMPARISON_GATE_PASS", "DOES_NOT_QUALIFY"):
                raise ValueError("Completed row verdict required")
            phases = row.get("phases")
            if type(phases) is not list or [phase.get("name") for phase in phases] != [
                "development",
                "evaluation",
            ]:
                raise ValueError("Both stored phase estimates required")
            for phase in phases:
                _count(phase.get("n"))
                delta = _number(phase.get("delta"))
                interval = phase.get("ci95_envelope")
                if (
                    type(interval) is not list
                    or len(interval) != 2
                    or not _number(interval[0]) <= delta <= _number(interval[1])
                ):
                    raise ValueError("Valid exact stored interval envelope required")
    elif status == "UNEVALUABLE":
        if metrics.get("whole_wave_aborted") is not True or metrics["leads"]:
            raise ValueError("Unevaluable wave must retain an explicit aborted state")
        for row in rows:
            if (
                row.get("status") not in ("INSUFFICIENT_DATA", "INVALID_RUN")
                or row.get("phases") != []
                or any(
                    row[field] != 1.0
                    for field in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
                )
            ):
                raise ValueError("Unevaluable rows must retain canonical p1 without estimates")
    else:
        raise ValueError("Only finalized completed or unevaluable outcomes can be rendered")
    return decoded, snapshots


def _summary(decoded):
    metrics, precheck, calibration = (
        decoded[name] for name in ("metrics.json", "FULL_PRECHECK.json", "calibration.json")
    )
    lines = ["# Oil/gold option-implied risk and five-session QQQ variance", ""]
    if metrics["status"] == "COMPLETED":
        passed = bool(metrics["leads"])
        lines += [
            "**Exploratory candidate: both fixed comparisons passed.**"
            if passed
            else "**No new qualifying predictive signal.** Both comparisons were completed and independently verified.",
            "",
            "The joint candidate adds delayed OVX/GVZ to the market baseline and matched oil/gold return-risk histories. Negative differences mean lower QLIKE forecast loss.",
            "",
            "| Control | Phase | Scored origins | Loss difference | 95% interval envelope | Wave Holm p | Cumulative Holm p |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
        for row in metrics["rows"]:
            for phase in row["phases"]:
                low, high = phase["ci95_envelope"]
                lines.append(
                    f"| {CONTROL_LABELS[row['control']]} | {phase['name'].capitalize()} | {phase['n']:,} | {phase['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] | {row['p_holm_wave']:.8f} | {row['p_holm_cumulative']:.8f} |"
                )
        counts = decoded["verification.json"]["forecasts"]
        lines += [
            "",
            "The required improvement is at least 0.005 in both phases against both controls, together with the fixed uncertainty, offset and stability gates. The figure uses the stored interval envelopes without new resampling.",
            "",
            f"Independent reconstruction checked {counts['forecasts_verified']:,} scored forecasts across {counts['scored_origins']:,} origins and {counts['application_forecasts_verified']:,} total application forecasts across {counts['application_origins']:,} origins. It verified {counts['monthly_fits']:,} monthly fits and {counts['model_fits']:,} individual model fits, retaining {counts['coverage_origins']:,} requested coverage origins on the {counts['calendar_rows']:,}-row reference calendar.",
            "",
        ]
        if passed:
            lines += [
                "A passing joint block does not identify which commodity index helps and does not establish a causal commodity effect. It requires confirmation on untouched future data before promotion.",
                "",
            ]
        else:
            lines += [
                "A completed nonqualifying result does not establish equivalence or rule out every possible commodity signal. No feature is promoted by this result.",
                "",
            ]
    else:
        statuses = sorted({row["status"] for row in metrics["rows"]})
        lines += [
            "**Unevaluable attempt: no valid completed comparison result.**",
            "",
            "Recorded disposition: "
            + ", ".join(statuses)
            + ". Both registered hypotheses remain p=1, and the whole wave is aborted. No effect estimates or confidence intervals are invented; partial artifacts are not reported as verified forecasting results.",
            "",
            "| Control | Disposition | Conservative p |",
            "| --- | --- | ---: |",
        ]
        for row in metrics["rows"]:
            lines.append(
                f"| {CONTROL_LABELS[row['control']]} | {row['status']} | 1.00000000 |"
            )
        lines += [""]
    lines += [
        f"All {precheck['test_count']:,} repository tests passed before the prospective freeze. Generated serial-null calibration achieved {100 * calibration['envelope_coverage']:.1f}% conservative interval-envelope coverage across {calibration['replications']:,} replications and was independently checked. This generated calibration is not a coverage guarantee for financial data.",
        "",
        "![Stored comparison estimates and uncertainty](comparison_intervals.png)",
        "",
        "The two registered comparisons preserve all 142 inherited entries, for **144 total**. OVX/GVZ describe 30-calendar-day USO/GLD option-implied risk; this differs from the five-session QQQ variance proxy. Current captures do not prove immutable historical vintages or exact file-publication clocks. The one-session lag does not repair later revisions.",
        "",
        "The inherited vendor ETF price series was intended to use adjusted close when available, with a raw-close fallback. Its exact captured field and historical vintage are not independently established; no split-adjustment proof or data repair is asserted. Cboe quotation-source and strike-selection changes remain part of the fixed history.",
        "",
        "Historical development/evaluation periods have been reused. No outcome here establishes untouched confirmation, a measured variance risk premium, execution profit or universal orthogonality. Numerical sources stop at 2025-10-20; market outcomes from 2025-11-03 onward remain protected.",
        "",
        "Evidence: [final metrics](metrics.json), [terminal hashes](terminal.json), [prospective freeze](freeze_record.json), [full checks](FULL_PRECHECK.json), [generated calibration](calibration.json), and [PDF figure](comparison_intervals.pdf).",
    ]
    if metrics["status"] == "COMPLETED":
        lines += [
            "Independent evidence: [forecast and score verification](verification.json)."
        ]
    return "\n".join(lines) + "\n"


def _figure(metrics):
    fig, ax = plt.subplots(figsize=(10.4, 5.3), layout="constrained")
    fig.set_facecolor("white")
    title = "Oil/gold option-implied risk: QQQ variance forecasts"
    if metrics["status"] == "UNEVALUABLE":
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=15, pad=22)
        ax.text(
            0.02,
            0.67,
            "UNEVALUABLE ATTEMPT",
            transform=ax.transAxes,
            fontsize=20,
            weight="bold",
            color="#6f4632",
        )
        ax.text(
            0.02,
            0.42,
            "No valid completed phase estimates or interval envelopes.\nBoth registered comparisons retained at p = 1.\n144 total comparisons remain in the research record.",
            transform=ax.transAxes,
            fontsize=13,
            linespacing=1.7,
        )
    else:
        labels, colors = [], ["#176b8f", "#ad602c"]
        values = []
        for phase_index, phase_name in enumerate(("development", "evaluation")):
            for row in metrics["rows"]:
                phase = row["phases"][phase_index]
                delta, (low, high) = phase["delta"], phase["ci95_envelope"]
                position = len(labels)
                ax.errorbar(
                    delta,
                    position,
                    xerr=[[delta - low], [high - delta]],
                    fmt="o",
                    color=colors[phase_index],
                    capsize=5,
                    markersize=7,
                    elinewidth=2,
                )
                labels.append(phase_name.capitalize() + "\n" + CONTROL_LABELS[row["control"]])
                values += [delta, low, high]
        ax.axvline(0, color="#717b82", linewidth=1, label="No change")
        ax.axvline(
            -0.005,
            color="#974245",
            linewidth=1.6,
            linestyle="--",
            label="Required improvement: −0.005",
        )
        ax.set_yticks(range(4), labels)
        ax.invert_yaxis()
        ax.set_ylim(3.65, -0.65)
        lo, hi = min(*values, -0.005, 0), max(*values, -0.005, 0)
        padding = max((hi - lo) * 0.12, 0.0005)
        ax.set_xlim(lo - padding, hi + padding)
        ax.set_xlabel("Candidate minus control QLIKE  ·  lower is better", labelpad=11)
        ax.set_title(
            title + "\nStored 95% interval envelopes; no new inference",
            loc="left",
            fontsize=14,
            pad=18,
        )
        ax.grid(axis="x", color="#e3e8eb", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False, fontsize=9
        )
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)
    payloads = {}
    for suffix in ("png", "pdf"):
        stream = io.BytesIO()
        fig.savefig(stream, format=suffix, dpi=180, bbox_inches="tight", facecolor="white")
        payloads[suffix] = stream.getvalue()
    plt.close(fig)
    return payloads


def render(report=DEFAULT_REPORT):
    report = Path(report)
    if any((report / name).exists() or (report / name).is_symlink() for name in OUTPUTS):
        raise ValueError("Refusing to overwrite an existing summary or comparison figure")
    decoded, snapshots = load_results(report)
    summary = _summary(decoded)
    figure = _figure(decoded["metrics.json"])
    # Confirm the same finalized bytes immediately before installing artifacts.
    for name, payload in snapshots.items():
        if _read(report / name) != payload:
            raise ValueError("Report artifact changed during rendering: " + name)
    payloads = {
        "SUMMARY.md": summary.encode(),
        "comparison_intervals.png": figure["png"],
        "comparison_intervals.pdf": figure["pdf"],
    }
    for name in OUTPUTS:
        with (report / name).open("xb") as stream:
            stream.write(payloads[name])
    return {name: _sha(payloads[name]) for name in OUTPUTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    print(json.dumps(render(arguments.report_dir), indent=2))


if __name__ == "__main__":
    main()
