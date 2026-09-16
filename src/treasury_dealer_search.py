"""One-shot prospective Treasury registration, execution and verification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from src.treasury_dealer_protocol import (
    CLOCK_CLASS,
    CLOCK_LIMITATION,
    EXECUTION,
    PROTOCOL_SHA256,
    load_protocol,
    pipeline_config,
    validate,
)
from src.treasury_dealer_score import (
    _prior,
    _probabilities,
    evaluate,
    failure_metrics,
    inherit_family,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump_x(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def verify_pins(root, pins):
    for name, expected in pins.items():
        if _sha(Path(root) / name) != expected:
            raise ValueError("Frozen artifact changed: " + name)


def _events(path, rows, mode):
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    with path.open(mode) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _qualified(result):
    result["source_clock_class"] = CLOCK_CLASS
    result["source_clock_limitation"] = CLOCK_LIMITATION
    return result


def _preserve_attempt(path):
    target = path.with_name("attempted_" + path.name)
    number = 1
    while target.exists():
        target = path.with_name(f"attempted_{number}_" + path.name)
        number += 1
    path.rename(target)


def _invalidate(report, signature, prior, error):
    published = (report / "metrics.json").exists()
    for name in ("metrics.json", "failure.json"):
        path = report / name
        if path.exists():
            _preserve_attempt(path)
    result = _qualified(failure_metrics(error, signature))
    result["inherited_rows"] = copy.deepcopy(prior)
    _dump_x(report / "failure.json", result)
    _dump_x(report / "metrics.json", result)
    _events(
        report / "trial_ledger.jsonl",
        [
            {"event": "invalidated" if published else "evaluated", **row}
            for row in result["rows"]
        ],
        "a",
    )
    return result


def execute_registered(report, signature, prior, work, final_check=None):
    """Journal the entire family before work; preserve registered failures at p1."""
    report = Path(report)
    _prior(prior)
    prior = copy.deepcopy(prior)
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        raise ValueError("Literal protocol hash required")
    report.mkdir(parents=True, exist_ok=True)
    if any(
        (report / name).exists()
        for name in ("trial_ledger.jsonl", "metrics.json", "failure.json")
    ):
        raise ValueError("Refusing to overwrite or restart a registered Treasury attempt")
    registrations = [
        {
            "event": "registered",
            "study": "treasury_dealer",
            "candidate": "candidate",
            "control": c,
            "horizon": 5,
            "score": "qlike",
            "protocol_sha256": signature,
        }
        for c in ("matched", "market")
    ]
    _events(
        report / "trial_ledger.jsonl",
        registrations + [{"event": "inherited", **row} for row in prior],
        "x",
    )
    try:
        result = work()
        if (
            result.get("status") != "COMPLETED"
            or type(result.get("hypothesis_count")) is not int
            or type(result.get("cumulative_hypothesis_count")) is not int
            or result.get("hypothesis_count") != 2
            or result.get("cumulative_hypothesis_count") != 146
            or [
                (r.get("candidate"), r.get("control"), r.get("horizon"), r.get("score"))
                for r in result.get("rows", [])
            ]
            != [("candidate", "matched", 5, "qlike"), ("candidate", "market", 5, "qlike")]
        ):
            raise ValueError("Complete registered two-comparison output family required")
        for row in result["rows"]:
            if row.get("study") != "treasury_dealer":
                raise ValueError("Exact registered study identity required")
            _probabilities(row)
        result = _qualified(result)
        result["inherited_rows"] = copy.deepcopy(prior)
        json.dumps(result, allow_nan=False)
        if final_check is not None:
            final_check()
        _dump_x(report / "metrics.json", result)
        _events(
            report / "trial_ledger.jsonl",
            [{"event": "evaluated", **row} for row in result["rows"]],
            "a",
        )
        if final_check is not None:
            final_check()
    except Exception as error:
        result = _invalidate(report, signature, prior, error)
    return result


def _paths(root):
    return Path(root) / EXECUTION["reports"], Path(root) / EXECUTION["data"]


def _full_test_count(text, returncode):
    summaries = re.findall(r"(?m)^Ran ([0-9]+) tests in [0-9.]+s\n\nOK\n", text)
    if returncode != 0 or len(summaries) != 1 or int(summaries[0]) <= 0:
        raise ValueError("One successful nonempty full-suite summary required")
    return int(summaries[0])


def _source_pins(root):
    """Hash-only admission of existing sources and frozen research inventories."""
    root = Path(root)
    pins = dict(EXECUTION["market_pins"])

    def merge(values):
        for name, signature in values.items():
            if name in pins and pins[name] != signature:
                raise ValueError("Conflicting immutable pins: " + name)
            pins[name] = signature

    for key in (
        "ledger",
        "source_terminal",
        "source_audit",
        "calibration",
        "prior",
        "prior_freeze",
    ):
        merge({EXECUTION[key]: EXECUTION[key + "_sha256"]})
    verify_pins(root, pins)
    terminal = json.loads((root / EXECUTION["source_terminal"]).read_bytes())
    audit = json.loads((root / EXECUTION["source_audit"]).read_bytes())
    calibration = json.loads((root / EXECUTION["calibration"]).read_bytes())
    if (
        terminal["status"] != "COMPLETED_QUALIFIED_SOURCE_PREPARATION_NO_PREDICTIVE_RUN"
        or audit["status"] != "VERIFIED_QUALIFIED_SAVED_SOURCE_LEDGER"
        or audit["discrepancies"]
        or audit["pin_verification"]["final_drift"]
        or calibration["status"] != "SYNTHETIC_CALIBRATION_PASS"
    ):
        raise ValueError("Completed source admission and fixed synthetic calibration required")
    merge(terminal["artifact_pins"])
    merge(audit["audited_input_hashes"])
    merge(calibration["pins"])
    old_terminal_path = (
        "reports/next_signal_review/TREASURY_HISTORY_RECONCILED_TERMINAL_AUDIT.json"
    )
    ledger_checkpoint_path = "reports/treasury_dealer/LEDGER_APPLICATION_CHECKPOINT.json"
    merge(
        {
            old_terminal_path: audit["old_terminal_sha256"],
            ledger_checkpoint_path: audit["checkpoint_sha256"],
        }
    )
    verify_pins(
        root,
        {
            old_terminal_path: pins[old_terminal_path],
            ledger_checkpoint_path: pins[ledger_checkpoint_path],
        },
    )
    merge(json.loads((root / old_terminal_path).read_bytes())["artifact_pins"])
    merge(json.loads((root / ledger_checkpoint_path).read_bytes())["new_pins"])
    prior_freeze = json.loads((root / EXECUTION["prior_freeze"]).read_bytes())
    for group in ("code", "inputs", "preserved", "prefit"):
        merge(prior_freeze[group])
    verify_pins(root, pins)
    prior = inherit_family(
        json.loads((root / EXECUTION["prior"]).read_bytes()), EXECUTION["prior_sha256"]
    )
    return pins, prior


def freeze(root=ROOT):
    """Complete all selected tests and freeze before empirical access."""
    root = Path(root)
    protocol = load_protocol()
    validate(protocol)
    report, out = _paths(root)
    report.mkdir(parents=True, exist_ok=True)
    if any(
        (report / name).exists()
        for name in ("freeze_record.json", "trial_ledger.jsonl", "FULL_PRECHECK.log")
    ):
        raise ValueError("Existing freeze, registration or full check must be preserved")
    if out.exists() and any(out.iterdir()):
        raise ValueError("No Treasury forecasting outputs may precede freeze")
    # Final independent prefit review is written only after all components are stable.
    review = root / "reports/treasury_dealer/predictive_prefit/INDEPENDENT_EXECUTION_REVIEW.md"
    if not review.is_file():
        raise ValueError("Final independent execution review required before freeze")
    inputs, _ = _source_pins(root)
    code = {
        str(p.relative_to(root)): _sha(p)
        for directory in ("src", "tests")
        for p in sorted((root / directory).rglob("*.py"))
    }
    print(
        "Running repository checks with six disclosed historical-replay quarantines",
        flush=True,
    )
    started, stamp = time.monotonic(), datetime.now(UTC).isoformat()
    with (report / "FULL_PRECHECK.log").open("x") as stream:
        checked = subprocess.run(
            [sys.executable, "-m", "src.treasury_dealer_test_gate", str(report)],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    verify_pins(root, code)
    text = (report / "FULL_PRECHECK.log").read_text()
    full = {
        "started_utc": stamp,
        "completed_utc": datetime.now(UTC).isoformat(),
        "exit_code": checked.returncode,
        "seconds": time.monotonic() - started,
        "log_sha256": _sha(report / "FULL_PRECHECK.log"),
        "code_unchanged": True,
    }
    try:
        full["test_count"] = _full_test_count(text, checked.returncode)
    except ValueError:
        full["test_count"] = 0
        _dump_x(report / "FULL_PRECHECK.json", full)
        raise
    _dump_x(report / "FULL_PRECHECK.json", full)
    gate = json.loads((report / "TEST_GATE_RESULT.json").read_bytes())
    selection = json.loads((report / "TEST_SELECTION.json").read_bytes())
    if (
        full["test_count"] < 2794
        or gate["status"] != "PASS"
        or gate["run_count"] != full["test_count"]
        or gate["executed_count"] != full["test_count"]
        or gate["quarantined_count"] != 6
        or selection["selected_count"] != full["test_count"]
    ):
        raise ValueError(
            "All selected repository tests must execute successfully; six quarantines disclosed"
        )
    verify_pins(root, inputs)
    preserved = {
        str(p.relative_to(root)): _sha(p)
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file() and report not in p.parents
    }
    prefit = {str(p.relative_to(root)): _sha(p) for p in report.iterdir() if p.is_file()}
    record = {
        "status": "FROZEN_BEFORE_TREASURY_MARKET_COHORT_OR_FITS",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": PROTOCOL_SHA256,
        "execution": EXECUTION,
        "code": code,
        "inputs": inputs,
        "preserved": preserved,
        "prefit": prefit,
        "selected_test_count": full["test_count"],
        "quarantined_test_count": 6,
        "cumulative_hypotheses_before_registration": 144,
        "previous_goal_turn": "PROGRESS_SOURCE_LEDGER_INDEPENDENT_AUDIT_AND_SAVED_CHECKPOINT",
        "environment": {
            "python": sys.version,
            "packages": {
                n: version(n) for n in ("numpy", "pandas", "scipy", "pyarrow", "PyYAML")
            },
            "threads": {
                n: os.environ.get(n)
                for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
            },
        },
    }
    _dump_x(report / "freeze_record.json", record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "tests": full["test_count"],
                "input_pins": len(inputs),
                "freeze_sha256": _sha(report / "freeze_record.json"),
            }
        ),
        flush=True,
    )
    return record


def _verify_freeze(root, protocol, frozen):
    from src.treasury_dealer_test_gate import QUARANTINED

    root = Path(root)
    validate(protocol)
    if (
        frozen["status"] != "FROZEN_BEFORE_TREASURY_MARKET_COHORT_OR_FITS"
        or frozen["protocol_sha256"] != PROTOCOL_SHA256
        or frozen["execution"] != EXECUTION
    ):
        raise ValueError("Exact completed prospective execution freeze required")
    current_code = {
        str(p.relative_to(root))
        for directory in ("src", "tests")
        for p in (root / directory).rglob("*.py")
    }
    if (
        "src/treasury_dealer_search.py" not in current_code
        or set(frozen["code"]) != current_code
    ):
        raise ValueError("Complete current source and test inventory required")
    report, _ = _paths(root)
    required_tests = {
        str((report / name).relative_to(root))
        for name in (
            "FULL_PRECHECK.json",
            "FULL_PRECHECK.log",
            "TEST_SELECTION.json",
            "TEST_GATE_RESULT.json",
        )
    }
    if not required_tests <= set(frozen["prefit"]):
        raise ValueError("Mandatory selected-test evidence missing from freeze")
    required_inputs = dict(EXECUTION["market_pins"])
    for name in (
        "ledger",
        "source_audit",
        "source_terminal",
        "calibration",
        "prior",
        "prior_freeze",
    ):
        required_inputs[EXECUTION[name]] = EXECUTION[name + "_sha256"]
    if any(frozen["inputs"].get(name) != digest for name, digest in required_inputs.items()):
        raise ValueError("Every mandatory admitted input pin required")
    for group in ("code", "inputs", "preserved", "prefit"):
        verify_pins(root, frozen[group])
    gate = json.loads((report / "TEST_GATE_RESULT.json").read_bytes())
    selection = json.loads((report / "TEST_SELECTION.json").read_bytes())
    full = json.loads((report / "FULL_PRECHECK.json").read_bytes())
    count = frozen.get("selected_test_count")
    if (
        type(count) is not int
        or count < 2794
        or frozen.get("quarantined_test_count") != 6
        or gate["status"] != "PASS"
        or gate["executed_count"] != count
        or gate["run_count"] != count
        or gate["skipped"]
        or gate["failure_count"]
        or gate["error_count"]
        or selection["selected_count"] != count
        or selection["discovered_count"] != count + 6
        or selection["quarantined_test_ids"] != sorted(QUARANTINED)
        or len(selection["selected_test_ids"]) != count
        or len(set(selection["selected_test_ids"])) != count
        or set(selection["selected_test_ids"]) & QUARANTINED
        or gate["selection_sha256"] != _sha(report / "TEST_SELECTION.json")
        or full["test_count"] != count
        or full["exit_code"] != 0
        or full["log_sha256"] != _sha(report / "FULL_PRECHECK.log")
    ):
        raise ValueError("Exact positive selected-test execution and six quarantines required")


def _json_native(value):
    """Lossless JSON date conversion for event audit output; never fill numbers."""
    import numpy as np
    import pandas as pd

    if isinstance(value, dict):
        return {k: _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(root=ROOT):
    from src.treasury_dealer_inputs import build_inputs, read_inputs
    from src.treasury_dealer_pipeline import InsufficientDataError, build_panel
    from src.treasury_dealer_support import audit_support
    from src.verify_treasury_dealer_forecasts import (
        verify_forecasts,
        verify_inputs_and_support,
    )
    from src.verify_treasury_dealer_scores import verify_scores

    root = Path(root)
    protocol = load_protocol()
    report, out = _paths(root)
    frozen = json.loads((report / "freeze_record.json").read_bytes())
    _verify_freeze(root, protocol, frozen)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite forecasting outputs")
    _, prior = _source_pins(root)

    def save_frame(name, frame):
        path = out / (name + ".parquet")
        if path.exists():
            raise ValueError("Refusing to overwrite saved research frame")
        frame.to_parquet(path)
        path.chmod(0o600)

    def work():
        print("Two comparisons registered; reading bounded admitted sources", flush=True)
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.chmod(0o700)
        _dump_x(
            report / "manifest.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "freeze_sha256": _sha(report / "freeze_record.json"),
                "protocol_sha256": PROTOCOL_SHA256,
                "source_clock_class": "QUALIFIED_REPORTED_CLOCK_ONLY_FOR_Z52",
            },
        )
        sources, ledger = read_inputs(
            root, EXECUTION["market_pins"], EXECUTION["ledger"], EXECUTION["ledger_sha256"]
        )
        inputs = build_inputs(sources, ledger["events"])
        features, targets = inputs["features"], inputs["targets"]
        save_frame("features", features)
        save_frame("targets", targets)
        _dump_x(out / "event_audit.json", _json_native(inputs["event_audit"]))
        (out / "event_audit.json").chmod(0o600)
        support = audit_support(
            features.index,
            features,
            targets,
            inputs["event_audit"],
            event_tenors={e["event_id"]: e["original_tenor_years"] for e in ledger["events"]},
        )
        _dump_x(report / "support.json", support)
        print("Independently reconstructing joined inputs and sample support", flush=True)
        input_check = verify_inputs_and_support(
            sources, ledger["events"], features, targets, inputs["event_audit"], support
        )
        _dump_x(report / "input_support_verification.json", input_check)
        if input_check["status"] != "VERIFIED":
            raise ValueError("Independent input/support reconstruction failed")
        if not support["passed"]:
            raise ValueError(
                "INSUFFICIENT_DATA: fixed Treasury sample support failed; see support.json"
            )
        print("Fitting fixed common32 monthly models", flush=True)
        try:
            produced = build_panel(features, targets, pipeline_config(protocol))
        except InsufficientDataError as error:
            save_frame("coverage", error.coverage)
            save_frame("schedules", error.schedules)
            raise
        for key in ("applications", "panel", "coverage", "schedules"):
            save_frame(key, produced[key])
        _dump_x(out / "fits.json", produced["fits"])
        (out / "fits.json").chmod(0o600)
        print("Independently reconstructing all fits and issued forecasts", flush=True)
        forecast_check = verify_forecasts(
            sources, ledger["events"], features, targets, produced, pipeline_config(protocol)
        )
        _dump_x(report / "forecast_verification.json", forecast_check)
        if forecast_check["status"] != "VERIFIED":
            raise ValueError("Independent forecast reconstruction failed")
        print("Scoring both comparisons and independently checking inference", flush=True)
        active = features.auction_count.gt(0)
        metrics = _qualified(
            evaluate(produced["panel"], features.index, active, prior, protocol, support)
        )
        score_check = verify_scores(
            produced["panel"], features.index, active, metrics, prior, protocol, support
        )
        _dump_x(report / "score_verification.json", score_check)
        if score_check["status"] != "VERIFIED":
            raise ValueError("Independent scoring verification failed")
        _verify_freeze(root, protocol, frozen)
        _dump_x(
            report / "verification.json",
            {
                "status": "VERIFIED",
                "inputs_and_support": input_check,
                "forecasts": forecast_check,
                "scores": score_check,
                "outputs": {
                    str(p.relative_to(root)): _sha(p) for p in out.iterdir() if p.is_file()
                },
                "protocol_sha256": PROTOCOL_SHA256,
            },
        )
        return metrics

    result = execute_registered(
        report,
        PROTOCOL_SHA256,
        prior,
        work,
        final_check=lambda: _verify_freeze(root, protocol, frozen),
    )

    def terminal_record():
        return _qualified(
            {
                "status": result["status"],
                "completed_utc": datetime.now(UTC).isoformat(),
                "metrics_sha256": _sha(report / "metrics.json"),
                "trial_ledger_sha256": _sha(report / "trial_ledger.jsonl"),
                "leads": result["leads"],
                "cumulative_hypotheses": 146,
                "output_hashes": {
                    str(p.relative_to(root)): _sha(p) for p in out.iterdir() if p.is_file()
                }
                if out.exists()
                else {},
                "report_artifact_hashes": {
                    p.name: _sha(p) for p in report.iterdir() if p.is_file()
                },
            }
        )

    try:
        _dump_x(report / "terminal.json", terminal_record())
    except Exception as error:
        if (report / "terminal.json").exists():
            _preserve_attempt(report / "terminal.json")
        result = _invalidate(report, PROTOCOL_SHA256, prior, error)
        _dump_x(report / "terminal.json", terminal_record())
    print(
        json.dumps({"status": result["status"], "leads": result["leads"], "comparisons": 146}),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif run()["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
