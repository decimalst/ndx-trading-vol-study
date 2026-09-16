"""One registered model-state gate experiment; no silent retry or fallback."""

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

from src.precision_gate_protocol import (
    EVIDENCE_CLASS,
    EVIDENCE_LIMITATION,
    EXECUTION,
    PROTOCOL_SHA256,
    load_protocol,
    pipeline_config,
    validate,
)
from src.treasury_dealer_search import (
    _dump_x,
    _events,
    _full_test_count,
    _preserve_attempt,
    _sha,
    verify_pins,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("base", "adaptive", "constant")
PIN_KEYS = (
    "prior",
    "prior_freeze",
    "prior_terminal",
    "prior_publication",
    "prior_audit",
    "calibration",
)


def _qualified(result):
    result["evidence_class"] = EVIDENCE_CLASS
    result["evidence_limitation"] = EVIDENCE_LIMITATION
    return result


def _save_bound(path, value, pins):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    _dump_x(path, value)
    pins[str(Path(path).resolve())] = hashlib.sha256(payload).hexdigest()
    verify_pins(ROOT, {str(Path(path).resolve()): pins[str(Path(path).resolve())]})


def _events_bound(path, rows, mode, pins):
    key = str(Path(path).resolve())
    previous = Path(path).read_bytes() if mode == "a" else b""
    if mode == "a" and hashlib.sha256(previous).hexdigest() != pins[key]:
        raise ValueError("Registered journal bytes changed before append")
    payload = "".join(
        json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows
    ).encode()
    _events(path, rows, mode)
    pins[key] = hashlib.sha256(previous + payload).hexdigest()
    verify_pins(ROOT, {key: pins[key]})


def _verify_bound(pins):
    verify_pins(ROOT, pins)


def _invalidate(report, signature, prior, error):
    from src.precision_gate_score import failure_metrics

    published = (report / "metrics.json").exists()
    for name in ("metrics.json", "failure.json"):
        if (report / name).exists():
            _preserve_attempt(report / name)
    result = _qualified(failure_metrics(error, signature, prior))
    result["inherited_rows"] = copy.deepcopy(prior)
    _dump_x(report / "failure.json", result)
    _dump_x(report / "metrics.json", result)
    _events(
        report / "trial_ledger.jsonl",
        [{"event": "invalidated" if published else "evaluated", **r} for r in result["rows"]],
        "a",
    )
    return result


def execute_registered(
    report, signature, prior, work, final_check=None, publication_pins=None
):
    from src.precision_gate_score import _prior, _probabilities

    report = Path(report)
    _prior(prior)
    prior = copy.deepcopy(prior)
    bound = {} if publication_pins is None else publication_pins
    if type(bound) is not dict or bound:
        raise ValueError("Fresh publication-pin mapping required")
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        raise ValueError("Literal protocol hash required")
    report.mkdir(parents=True, exist_ok=True)
    if any(
        (report / name).exists()
        for name in ("trial_ledger.jsonl", "metrics.json", "failure.json")
    ):
        raise ValueError("Refusing to overwrite or restart registered precision-gate attempt")
    _events_bound(
        report / "trial_ledger.jsonl",
        [
            {
                "event": "registered",
                "study": "precision_gate",
                "candidate": "contextual",
                "control": c,
                "horizon": 5,
                "score": "qlike",
                "protocol_sha256": signature,
            }
            for c in CONTROLS
        ]
        + [{"event": "inherited", **r} for r in prior],
        "x",
        bound,
    )
    try:
        result = work()
        if (
            result.get("status") != "COMPLETED"
            or type(result.get("hypothesis_count")) is not int
            or type(result.get("cumulative_hypothesis_count")) is not int
            or result["hypothesis_count"] != 3
            or result["cumulative_hypothesis_count"] != 149
            or [
                (
                    r.get("study"),
                    r.get("candidate"),
                    r.get("control"),
                    r.get("horizon"),
                    r.get("score"),
                )
                for r in result.get("rows", [])
            ]
            != [("precision_gate", "contextual", c, 5, "qlike") for c in CONTROLS]
        ):
            raise ValueError("Complete ordered registered three-comparison family required")
        for row in result["rows"]:
            _probabilities(row)
        result = _qualified(result)
        result["inherited_rows"] = copy.deepcopy(prior)
        json.dumps(result, allow_nan=False)
        _verify_bound(bound)
        if final_check is not None:
            final_check()
        _save_bound(report / "metrics.json", result, bound)
        _events_bound(
            report / "trial_ledger.jsonl",
            [{"event": "evaluated", **r} for r in result["rows"]],
            "a",
            bound,
        )
        _verify_bound(bound)
        if final_check is not None:
            final_check()
        _verify_bound(bound)
    except Exception as error:
        result = _invalidate(report, signature, prior, error)
    return result


def _paths(root):
    return Path(root) / EXECUTION["reports"], Path(root) / EXECUTION["data"]


def _source_pins(root):
    """Prior research is authenticated by bytes only, with no market decoding."""
    from src.precision_gate_score import inherit_family

    root = Path(root)
    pins = dict(EXECUTION["market_pins"])
    for key in PIN_KEYS:
        pins[EXECUTION[key]] = EXECUTION[key + "_sha256"]
    verify_pins(root, pins)
    documents = {key: json.loads((root / EXECUTION[key]).read_bytes()) for key in PIN_KEYS}
    if (
        documents["prior_terminal"]["status"] != "COMPLETED"
        or documents["prior_audit"]["status"] != "VERIFIED_QUALIFIED_SAVED_PUBLICATION"
        or documents["prior_audit"]["discrepancies"]
        or documents["prior_publication"]["status"] != "VERIFIED_QUALIFIED_REPORT_CHECKPOINT"
        or documents["calibration"]["status"] != "SYNTHETIC_CALIBRATION_PASS"
    ):
        raise ValueError(
            "Verified prior study and passed fixed inference calibration required"
        )

    def merge(values):
        for name, digest in values.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting frozen pin: " + name)
            pins[name] = digest

    for group in ("code", "inputs", "preserved", "prefit"):
        merge(documents["prior_freeze"][group])
    merge(documents["prior_publication"]["artifact_hashes"])
    merge(documents["prior_terminal"]["output_hashes"])
    parent = str(Path(EXECUTION["prior_terminal"]).parent)
    merge(
        {
            parent + "/" + name: digest
            for name, digest in documents["prior_terminal"]["report_artifact_hashes"].items()
        }
    )
    merge(documents["calibration"]["pins"])
    verify_pins(root, pins)
    return pins, inherit_family(documents["prior"], EXECUTION["prior_sha256"])


def _code(root):
    return {
        str(p.relative_to(root)): _sha(p)
        for directory in ("src", "tests")
        for p in sorted((root / directory).rglob("*.py"))
    }


def _test_evidence(root, frozen):
    from src.treasury_dealer_test_gate import QUARANTINED

    report, _ = _paths(root)
    required = {
        str((report / n).relative_to(root))
        for n in (
            "FULL_PRECHECK.json",
            "FULL_PRECHECK.log",
            "TEST_SELECTION.json",
            "TEST_GATE_RESULT.json",
        )
    }
    if not required <= set(frozen["prefit"]):
        raise ValueError("Mandatory selected-test receipts missing from freeze")
    gate = json.loads((report / "TEST_GATE_RESULT.json").read_bytes())
    selected = json.loads((report / "TEST_SELECTION.json").read_bytes())
    full = json.loads((report / "FULL_PRECHECK.json").read_bytes())
    count = frozen["selected_test_count"]
    if (
        type(count) is not int
        or count < 3252
        or frozen["quarantined_test_count"] != 6
        or gate["status"] != "PASS"
        or gate["executed_count"] != count
        or gate["run_count"] != count
        or gate["quarantined_count"] != 6
        or gate["skipped"]
        or gate["failure_count"]
        or gate["error_count"]
        or selected["selected_count"] != count
        or selected["discovered_count"] != count + 6
        or selected["quarantined_test_ids"] != sorted(QUARANTINED)
        or len(selected["selected_test_ids"]) != count
        or len(set(selected["selected_test_ids"])) != count
        or set(selected["selected_test_ids"]) & QUARANTINED
        or gate["selection_sha256"] != _sha(report / "TEST_SELECTION.json")
        or full["test_count"] != count
        or full["exit_code"] != 0
        or full["log_sha256"] != _sha(report / "FULL_PRECHECK.log")
    ):
        raise ValueError("Exact successful complete selected-test execution required")


def _verify_freeze(root, protocol, frozen):
    root = Path(root)
    validate(protocol)
    if (
        frozen["status"] != "FROZEN_BEFORE_PRECISION_GATE_MARKET_COHORT_OR_FITS"
        or frozen["protocol_sha256"] != PROTOCOL_SHA256
        or frozen["execution"] != EXECUTION
    ):
        raise ValueError("Exact prospective precision-gate execution freeze required")
    current = _code(root)
    if "src/precision_gate_search.py" not in current or set(frozen["code"]) != set(current):
        raise ValueError("Complete current source/test inventory required")
    required_inputs = {
        **EXECUTION["market_pins"],
        **{EXECUTION[k]: EXECUTION[k + "_sha256"] for k in PIN_KEYS},
    }
    if any(frozen["inputs"].get(n) != d for n, d in required_inputs.items()):
        raise ValueError("Every declared admitted input pin required")
    for group in ("code", "inputs", "preserved", "prefit"):
        verify_pins(root, frozen[group])
    _test_evidence(root, frozen)


def _check_frozen_snapshot(root, protocol, frozen, signature):
    report, _ = _paths(root)
    verify_pins(root, {str((report / "freeze_record.json").relative_to(root)): signature})
    _verify_freeze(root, protocol, frozen)


def _check_output_snapshot(root, pins):
    root = Path(root)
    _, out = _paths(root)
    required = {
        str((out / name).relative_to(root))
        for name in (
            "features.parquet",
            "targets.parquet",
            "applications.parquet",
            "panel.parquet",
            "coverage.parquet",
            "schedules.parquet",
            "fits.json",
        )
    }
    actual = {str(p.relative_to(root)) for p in out.iterdir() if p.is_file()}
    if set(pins) != required or actual != required:
        raise ValueError("Exact saved forecasting output inventory required")
    verify_pins(root, pins)


def freeze(root=ROOT):
    root = Path(root)
    protocol = load_protocol()
    validate(protocol)
    report, out = _paths(root)
    report.mkdir(parents=True, exist_ok=True)
    if any(
        (report / n).exists()
        for n in ("freeze_record.json", "trial_ledger.jsonl", "FULL_PRECHECK.log")
    ):
        raise ValueError(
            "Existing freeze, registration or selected-test run must be preserved"
        )
    if out.exists() and any(out.iterdir()):
        raise ValueError("Empirical outputs may not precede freeze")
    if not (root / "reports/precision_gate/prefit/INDEPENDENT_EXECUTION_REVIEW.md").is_file():
        raise ValueError("Completed independent prefit execution review required")
    inputs, _ = _source_pins(root)
    code = _code(root)
    print(
        "Running selected repository tests with six disclosed historical-replay exclusions",
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
    verify_pins(root, inputs)
    preserved = {
        str(p.relative_to(root)): _sha(p)
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file() and report not in p.parents
    }
    record = {
        "status": "FROZEN_BEFORE_PRECISION_GATE_MARKET_COHORT_OR_FITS",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": PROTOCOL_SHA256,
        "execution": EXECUTION,
        "code": code,
        "inputs": inputs,
        "preserved": preserved,
        "prefit": {str(p.relative_to(root)): _sha(p) for p in report.iterdir() if p.is_file()},
        "selected_test_count": full["test_count"],
        "quarantined_test_count": 6,
        "cumulative_hypotheses_before_registration": 146,
        "previous_goal_turn": "PROGRESS_TREASURY_COMPLETED_AND_PUBLICATION_CHECKPOINT_SAVED",
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
    _test_evidence(root, record)
    _dump_x(report / "freeze_record.json", record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "tests": full["test_count"],
                "input_pins": len(inputs),
            }
        ),
        flush=True,
    )
    return record


def run(root=ROOT):
    from src.claims_release_inputs import read_market_sources
    from src.claims_release_market import build_market_features, make_five_session_targets
    from src.precision_gate_pipeline import PipelineExecutionError, build_panel
    from src.precision_gate_score import evaluate
    from src.verify_precision_gate_forecasts import verify_forecasts
    from src.verify_precision_gate_scores import verify_scores

    root = Path(root)
    protocol = load_protocol()
    report, out = _paths(root)
    freeze_payload = (report / "freeze_record.json").read_bytes()
    freeze_signature = hashlib.sha256(freeze_payload).hexdigest()
    frozen = json.loads(freeze_payload)
    _check_frozen_snapshot(root, protocol, frozen, freeze_signature)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite issued forecasting outputs")
    _, prior = _source_pins(root)
    output_pins = {}
    publication_pins = {}

    def final_check():
        _check_frozen_snapshot(root, protocol, frozen, freeze_signature)
        _check_output_snapshot(root, output_pins)
        _verify_bound(publication_pins)

    def save_report(name, value):
        _save_bound(report / name, value, publication_pins)

    def save_frame(name, frame):
        path = out / (name + ".parquet")
        if path.exists():
            raise ValueError("Refusing to overwrite saved frame")
        frame.to_parquet(path)
        path.chmod(0o600)
        output_pins[str(path.relative_to(root))] = _sha(path)

    def save_produced(produced):
        for name in ("applications", "panel", "coverage", "schedules"):
            save_frame(name, produced[name])
        _dump_x(out / "fits.json", produced["fits"])
        (out / "fits.json").chmod(0o600)
        output_pins[str((out / "fits.json").relative_to(root))] = _sha(out / "fits.json")

    def work():
        print(
            "Three comparisons registered; decoding bounded admitted market inputs", flush=True
        )
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.chmod(0o700)
        save_report(
            "manifest.json",
            _qualified(
                {
                    "created_utc": datetime.now(UTC).isoformat(),
                    "freeze_sha256": freeze_signature,
                    "protocol_sha256": PROTOCOL_SHA256,
                }
            ),
        )
        sources = read_market_sources(root, EXECUTION["market_pins"])
        features = build_market_features(**sources)
        targets = make_five_session_targets(features.rv_total)
        save_frame("features", features)
        save_frame("targets", targets)
        print(
            "Issuing monthly expert and gate forecasts with mature historical memory",
            flush=True,
        )
        try:
            produced = build_panel(features, targets, pipeline_config(protocol))
        except PipelineExecutionError as error:
            save_produced(error.produced)
            raise
        save_produced(produced)
        _check_output_snapshot(root, output_pins)
        print(
            "Independently reconstructing features, all fits, memory and issued forecasts",
            flush=True,
        )
        forecast_check = verify_forecasts(
            sources, features, targets, produced, pipeline_config(protocol)
        )
        save_report("forecast_verification.json", forecast_check)
        if forecast_check["status"] != "VERIFIED":
            raise ValueError("Independent forecast reconstruction failed")
        print(
            "Scoring all three comparisons and independently reconstructing inference",
            flush=True,
        )
        metrics = _qualified(evaluate(produced["panel"], features.index, prior, protocol))
        score_check = verify_scores(
            produced["panel"], features.index, metrics, prior, protocol
        )
        save_report("score_verification.json", score_check)
        if score_check["status"] != "VERIFIED":
            raise ValueError("Independent score reconstruction failed")
        final_check()
        save_report(
            "verification.json",
            {
                "status": "VERIFIED",
                "forecasts": forecast_check,
                "scores": score_check,
                "protocol_sha256": PROTOCOL_SHA256,
                "outputs": dict(output_pins),
            },
        )
        return metrics

    result = execute_registered(
        report,
        PROTOCOL_SHA256,
        prior,
        work,
        final_check=final_check,
        publication_pins=publication_pins,
    )

    def terminal():
        return _qualified(
            {
                "status": result["status"],
                "completed_utc": datetime.now(UTC).isoformat(),
                "metrics_sha256": _sha(report / "metrics.json"),
                "trial_ledger_sha256": _sha(report / "trial_ledger.jsonl"),
                "leads": result["leads"],
                "cumulative_hypotheses": 149,
                "freeze_record_sha256": freeze_signature,
                "observed_freeze_record_sha256": _sha(report / "freeze_record.json")
                if (report / "freeze_record.json").is_file()
                else None,
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
        if result["status"] == "COMPLETED":
            final_check()
        save_report("terminal.json", terminal())
        if result["status"] == "COMPLETED":
            final_check()
    except Exception as error:
        if (report / "terminal.json").exists():
            _preserve_attempt(report / "terminal.json")
        result = _invalidate(report, PROTOCOL_SHA256, prior, error)
        save_report("terminal.json", terminal())
    print(
        json.dumps({"status": result["status"], "leads": result["leads"], "comparisons": 149}),
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
