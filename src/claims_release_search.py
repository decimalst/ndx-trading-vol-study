"""One-shot prospective claims registration, execution and verification."""

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

import yaml

from src.claims_release_protocol import validate
from src.claims_release_score import (
    _prior,
    calibrate,
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


def execute_registered(report, signature, prior, work):
    """Journal the entire family before work; preserve registered failures at p1."""
    report = Path(report)
    _prior(prior)
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        raise ValueError("Literal protocol hash required")
    report.mkdir(parents=True, exist_ok=True)
    if any(
        (report / name).exists()
        for name in ("trial_ledger.jsonl", "metrics.json", "failure.json")
    ):
        raise ValueError("Refusing to overwrite or restart a registered claims attempt")
    registrations = [
        {
            "event": "registered",
            "study": "claims_release",
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
            or result.get("hypothesis_count") != 2
            or result.get("cumulative_hypothesis_count") != 142
            or [
                (r.get("candidate"), r.get("control"), r.get("horizon"), r.get("score"))
                for r in result.get("rows", [])
            ]
            != [("candidate", "matched", 5, "qlike"), ("candidate", "market", 5, "qlike")]
        ):
            raise ValueError("Complete registered two-comparison output family required")
        _prior(result["rows"], 2)
        json.dumps(result, allow_nan=False)
    except Exception as error:
        result = failure_metrics(error, signature)
        _dump_x(report / "failure.json", result)
    result["inherited_rows"] = copy.deepcopy(prior)
    _dump_x(report / "metrics.json", result)
    _events(
        report / "trial_ledger.jsonl",
        [{"event": "evaluated", **row} for row in result["rows"]],
        "a",
    )
    return result


def _paths(root, protocol):
    return (root / protocol["outputs"]["reports"], root / protocol["outputs"]["data"])


def _source_pins(root, protocol):
    pins = dict(protocol["source_pins"])
    for section, path_key, hash_key in (
        ("ledger", "path", "sha256"),
        ("ledger", "admission", "admission_sha256"),
        ("prior", "path", "sha256"),
        ("prior", "publication", "publication_sha256"),
    ):
        pins[protocol[section][path_key]] = protocol[section][hash_key]
    verify_pins(root, pins)
    admission = json.loads((root / protocol["ledger"]["admission"]).read_bytes())
    if admission["status"] != "INDEPENDENTLY_VERIFIED_FIRST_REPORT_LEDGER_WITH_EXPLICIT_GAPS":
        raise ValueError("Independently verified source ledger required")
    pins.update(admission["input_and_component_sha256"])
    old = json.loads((root / protocol["prior"]["path"]).read_bytes())
    prior = inherit_family(old, protocol["prior"]["sha256"])
    for row in prior:
        pins[row["source"]] = row["source_sha256"]
    frozen = json.loads((root / "reports/peak_age/freeze_record.json").read_bytes())
    verify_pins(root, frozen["code"])
    publication = json.loads((root / protocol["prior"]["publication"]).read_bytes())
    verify_pins(
        root,
        {
            "reports/peak_age/" + key: value
            for key, value in publication["report_artifact_hashes"].items()
        },
    )
    verify_pins(root, pins)
    return pins, prior


def _full_test_count(text, returncode):
    summaries = re.findall(r"(?m)^Ran ([0-9]+) tests in [0-9.]+s\n\nOK\n", text)
    if returncode != 0 or len(summaries) != 1 or int(summaries[0]) <= 0:
        raise ValueError("One successful nonempty full-suite summary required")
    return int(summaries[0])


def freeze(root=ROOT):
    """Run generated calibration and full tests, then freeze before any new cohort."""
    from src.verify_claims_release_scores import verify_calibration

    root = Path(root)
    protocol = yaml.safe_load((root / "claims_release.yaml").read_bytes())
    validate(protocol)
    report, out = _paths(root, protocol)
    report.mkdir(parents=True, exist_ok=True)
    if (report / "freeze_record.json").exists() or (report / "trial_ledger.jsonl").exists():
        raise ValueError("Existing prospective freeze/registration must be preserved")
    if out.exists() and any(out.iterdir()):
        raise ValueError("No forecasting data may precede prospective freeze")
    inputs, _ = _source_pins(root, protocol)
    code = {
        str(p.relative_to(root)): _sha(p)
        for directory in ("src", "tests")
        for p in sorted((root / directory).rglob("*.py"))
    }
    print(
        "Running generated serial-null calibration and independent reconstruction", flush=True
    )
    calibration = calibrate(protocol)
    verification = verify_calibration(protocol, calibration)
    _dump_x(report / "calibration.json", calibration)
    _dump_x(report / "calibration_verification.json", verification)
    if calibration["status"] != "PASS" or verification["status"] != "VERIFIED":
        raise ValueError("Generated calibration failed before registration")
    print("Running full pre-registration repository tests", flush=True)
    started, stamp = time.monotonic(), datetime.now(UTC).isoformat()
    with (report / "FULL_PRECHECK.log").open("x") as stream:
        checked = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    verify_pins(root, code)
    text = (report / "FULL_PRECHECK.log").read_text()
    found = re.findall(r"(?m)^Ran ([0-9]+) tests in [0-9.]+s\n\nOK\n", text)
    full = {
        "started_utc": stamp,
        "completed_utc": datetime.now(UTC).isoformat(),
        "exit_code": checked.returncode,
        "seconds": elapsed,
        "test_count": int(found[0]) if len(found) == 1 else 0,
        "log_sha256": _sha(report / "FULL_PRECHECK.log"),
        "code_unchanged": True,
    }
    _dump_x(report / "FULL_PRECHECK.json", full)
    if _full_test_count(text, checked.returncode) < 2465:
        raise ValueError("Full repository checks failed before registration")
    verify_pins(root, inputs)
    preserved = {
        str(p.relative_to(root)): _sha(p)
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file() and p != root / "claims_release.yaml" and report not in p.parents
    }
    prefit = {str(p.relative_to(root)): _sha(p) for p in report.iterdir() if p.is_file()}
    record = {
        "status": "FROZEN_BEFORE_CLAIMS_MARKET_COHORT_OR_FITS",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha(root / "claims_release.yaml"),
        "code": code,
        "inputs": inputs,
        "preserved": preserved,
        "prefit": prefit,
        "full_test_count": full["test_count"],
        "cumulative_hypotheses_before_registration": 140,
        "environment": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "pyarrow", "PyYAML")
            },
            "thread_settings": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
            },
        },
    }
    _dump_x(report / "freeze_record.json", record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "tests": full["test_count"],
                "code_files": len(code),
                "freeze_sha256": _sha(report / "freeze_record.json"),
            }
        ),
        flush=True,
    )
    return record


def _verify_freeze(root, protocol, frozen):
    validate(protocol)
    if frozen["status"] != "FROZEN_BEFORE_CLAIMS_MARKET_COHORT_OR_FITS" or frozen[
        "protocol_sha256"
    ] != _sha(root / "claims_release.yaml"):
        raise ValueError("Exact completed prospective freeze required")
    for group in ("code", "inputs", "preserved", "prefit"):
        verify_pins(root, frozen[group])


def run(root=ROOT):
    from src.claims_release_features import build_claim_features
    from src.claims_release_inputs import read_market_sources
    from src.claims_release_market import build_market_features, make_five_session_targets
    from src.claims_release_pipeline import build_panel
    from src.verify_claims_release_forecasts import verify_forecasts
    from src.verify_claims_release_ledger import verify_ledger
    from src.verify_claims_release_scores import verify_scores

    root = Path(root)
    protocol = yaml.safe_load((root / "claims_release.yaml").read_bytes())
    report, out = _paths(root, protocol)
    frozen = json.loads((report / "freeze_record.json").read_bytes())
    _verify_freeze(root, protocol, frozen)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite forecasting outputs")
    _, prior = _source_pins(root, protocol)

    def work():
        print("Two comparisons registered; admitting bounded source snapshots", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        _dump_x(
            report / "manifest.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "freeze_sha256": _sha(report / "freeze_record.json"),
                "protocol_sha256": frozen["protocol_sha256"],
                "inputs": frozen["inputs"],
                "code": frozen["code"],
            },
        )
        ledger = json.loads((root / protocol["ledger"]["path"]).read_bytes())
        reports = json.loads(
            (root / "data/claims_release/full_source_reconciliation_formats.json").read_bytes()
        )["reports"]
        records = json.loads(
            (root / "data/claims_release/alfred_structural.json").read_bytes()
        )["records"]
        verify_ledger(ledger, reports, records)
        sources = read_market_sources(root, protocol["source_pins"])
        market = build_market_features(**sources)
        claims = build_claim_features(market.index, ledger["rows"])
        features = market.join(claims)
        targets = make_five_session_targets(market.rv_total)
        print("Fitting the fixed common-cohort monthly models", flush=True)
        produced = build_panel(features, targets, protocol["forecast"])
        for name, frame in {
            "features": features,
            "targets": targets,
            **{
                key: produced[key]
                for key in ("applications", "panel", "coverage", "schedules")
            },
        }.items():
            frame.to_parquet(out / (name + ".parquet"))
        _dump_x(out / "fits.json", produced["fits"])
        print(
            "Independently reconstructing all sources, clocks, fits and forecasts", flush=True
        )
        forecast_verification = verify_forecasts(
            **sources,
            ledger_rows=ledger["rows"],
            config=protocol["forecast"],
            produced=produced,
        )
        if forecast_verification["status"] != "VERIFIED":
            raise ValueError("Independent forecast verification failed")
        print(
            "Scoring both fixed comparisons and independently reconstructing inference",
            flush=True,
        )
        metrics = evaluate(produced["panel"], market.index, prior, protocol)
        score_verification = verify_scores(
            produced["panel"], market.index, metrics, prior, protocol
        )
        if score_verification["status"] != "VERIFIED":
            raise ValueError("Independent scoring verification failed")
        _verify_freeze(root, protocol, frozen)
        output_hashes = {
            str(p.relative_to(root)): _sha(p) for p in out.iterdir() if p.is_file()
        }
        _dump_x(
            report / "verification.json",
            {
                "status": "VERIFIED",
                "forecasts": forecast_verification,
                "scores": score_verification,
                "outputs": output_hashes,
                "protocol_sha256": frozen["protocol_sha256"],
            },
        )
        return metrics

    result = execute_registered(report, frozen["protocol_sha256"], prior, work)
    _verify_freeze(root, protocol, frozen)
    _dump_x(
        report / "terminal.json",
        {
            "status": result["status"],
            "completed_utc": datetime.now(UTC).isoformat(),
            "metrics_sha256": _sha(report / "metrics.json"),
            "trial_ledger_sha256": _sha(report / "trial_ledger.jsonl"),
            "leads": result["leads"],
            "cumulative_hypotheses": 142,
        },
    )
    print(
        json.dumps({"status": result["status"], "leads": result["leads"], "comparisons": 142}),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        result = run()
        if result["status"] != "COMPLETED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
