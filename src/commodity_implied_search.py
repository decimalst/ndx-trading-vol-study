"""One-shot prospective commodity registration, execution and verification."""

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

from src.commodity_implied_protocol import validate
from src.commodity_implied_score import (
    _prior,
    _probabilities,
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
    prior = copy.deepcopy(prior)
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        raise ValueError("Literal protocol hash required")
    report.mkdir(parents=True, exist_ok=True)
    if any(
        (report / name).exists()
        for name in ("trial_ledger.jsonl", "metrics.json", "failure.json")
    ):
        raise ValueError("Refusing to overwrite or restart a registered commodity attempt")
    registrations = [
        {
            "event": "registered",
            "study": "commodity_implied",
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
            or result.get("cumulative_hypothesis_count") != 144
            or [
                (r.get("candidate"), r.get("control"), r.get("horizon"), r.get("score"))
                for r in result.get("rows", [])
            ]
            != [("candidate", "matched", 5, "qlike"), ("candidate", "market", 5, "qlike")]
        ):
            raise ValueError("Complete registered two-comparison output family required")
        for row in result["rows"]:
            if row.get("study") != "commodity_implied":
                raise ValueError("Exact registered study identity required")
            _probabilities(row)
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
    """Authenticate admission and every inherited artifact without series decoding."""
    pins = dict(protocol["source_pins"])
    for section, path_key, hash_key in (
        ("commodity", "admission", "admission_sha256"),
        ("commodity", "checkpoint", "checkpoint_sha256"),
        ("prior", "path", "sha256"),
        ("prior", "publication", "publication_sha256"),
        ("prior", "freeze", "freeze_sha256"),
    ):
        pins[protocol[section][path_key]] = protocol[section][hash_key]
    for entry in protocol["commodity"]["histories"].values():
        for kind in ("raw", "parsed"):
            pins[entry[kind]] = entry[kind + "_sha256"]
    verify_pins(root, pins)
    admission = json.loads((root / protocol["commodity"]["admission"]).read_bytes())
    checkpoint = json.loads((root / protocol["commodity"]["checkpoint"]).read_bytes())
    if (
        admission["status"] != "BOUNDED_SOURCES_INDEPENDENTLY_VERIFIED_NO_FORECAST_ATTEMPT"
        or admission["source_pre_admission_sha256"]
        != protocol["commodity"]["checkpoint_sha256"]
        or admission["source_start"] != "2009-01-02"
        or admission["source_end"] != "2025-10-20"
        or admission["features_cohorts_fits_or_scores_computed"] is not False
        or admission["outside_scope_numerical_values_converted"] is not False
        or admission["predictive_family"] != 142
        or checkpoint["status"] != "FROZEN_SOURCE_CHECK_BEFORE_NUMERICAL_ADMISSION"
    ):
        raise ValueError("Exact independently completed bounded source admission required")
    for symbol, entry in protocol["commodity"]["histories"].items():
        proof = admission["sources"][symbol]
        if (
            proof["verification"]["status"] != "VERIFIED"
            or proof["verification"]["symbol"] != symbol
            or proof["source_sha256"] != entry["raw_sha256"]
            or proof["parsed_path"] != entry["parsed"]
            or proof["parsed_sha256"] != entry["parsed_sha256"]
        ):
            raise ValueError("Declared series differ from independently admitted bytes")
    for group in ("source_components", "evidence", "acquisition"):
        pins.update(checkpoint[group])
    old = json.loads((root / protocol["prior"]["path"]).read_bytes())
    prior = inherit_family(old, protocol["prior"]["sha256"])
    for row in prior:
        pins[row["source"]] = row["source_sha256"]
    frozen = json.loads((root / protocol["prior"]["freeze"]).read_bytes())
    for group in ("code", "inputs", "preserved", "prefit"):
        verify_pins(root, frozen[group])
        pins.update(frozen[group])
    publication = json.loads((root / protocol["prior"]["publication"]).read_bytes())
    parent = Path(protocol["prior"]["publication"]).parent
    pins.update(
        {
            str(parent / name): signature
            for name, signature in publication["report_artifact_hashes"].items()
        }
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
    from src.verify_commodity_implied_scores import verify_calibration

    root = Path(root)
    protocol = yaml.safe_load((root / "commodity_implied.yaml").read_bytes())
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
    if _full_test_count(text, checked.returncode) < 2644:
        raise ValueError("Full repository checks failed before registration")
    verify_pins(root, inputs)
    preserved = {
        str(p.relative_to(root)): _sha(p)
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file() and p != root / "commodity_implied.yaml" and report not in p.parents
    }
    prefit = {str(p.relative_to(root)): _sha(p) for p in report.iterdir() if p.is_file()}
    record = {
        "status": "FROZEN_BEFORE_COMMODITY_MARKET_COHORT_OR_FITS",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha(root / "commodity_implied.yaml"),
        "code": code,
        "inputs": inputs,
        "preserved": preserved,
        "prefit": prefit,
        "full_test_count": full["test_count"],
        "cumulative_hypotheses_before_registration": 142,
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
    if frozen["status"] != "FROZEN_BEFORE_COMMODITY_MARKET_COHORT_OR_FITS" or frozen[
        "protocol_sha256"
    ] != _sha(root / "commodity_implied.yaml"):
        raise ValueError("Exact completed prospective freeze required")
    for group in ("code", "inputs", "preserved", "prefit"):
        verify_pins(root, frozen[group])


def run(root=ROOT):
    from src.claims_release_inputs import read_market_sources
    from src.claims_release_market import build_market_features, make_five_session_targets
    from src.commodity_implied_features import build_commodity_features
    from src.commodity_implied_inputs import read_commodity_sources
    from src.commodity_implied_pipeline import InsufficientDataError, build_panel
    from src.verify_commodity_implied_forecasts import verify_forecasts
    from src.verify_commodity_implied_scores import verify_scores

    root = Path(root)
    protocol = yaml.safe_load((root / "commodity_implied.yaml").read_bytes())
    report, out = _paths(root, protocol)
    frozen = json.loads((report / "freeze_record.json").read_bytes())
    _verify_freeze(root, protocol, frozen)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite forecasting outputs")
    _, prior = _source_pins(root, protocol)

    def save_frame(name, frame):
        path = out / (name + ".parquet")
        if path.exists():
            raise ValueError("Refusing to overwrite a saved research frame")
        frame.to_parquet(path)
        path.chmod(0o600)

    def work():
        print("Two comparisons registered; admitting bounded source snapshots", flush=True)
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.chmod(0o700)
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
        sources = read_market_sources(root, protocol["source_pins"])
        commodity_iv = read_commodity_sources(root, protocol["commodity"]["histories"])
        market = build_market_features(**sources)
        commodity = build_commodity_features(market.index, sources["cross"], commodity_iv)
        features = market.join(commodity)
        targets = make_five_session_targets(market.rv_total)
        save_frame("features", features)
        save_frame("targets", targets)
        print("Fitting the fixed common22 monthly models", flush=True)
        try:
            produced = build_panel(features, targets, protocol["forecast"])
        except InsufficientDataError as error:
            save_frame("coverage", error.coverage)
            save_frame("schedules", error.schedules)
            raise
        for key in ("applications", "panel", "coverage", "schedules"):
            save_frame(key, produced[key])
        _dump_x(out / "fits.json", produced["fits"])
        (out / "fits.json").chmod(0o600)
        print(
            "Independently reconstructing all clocks, features, fits and forecasts", flush=True
        )
        forecast_verification = verify_forecasts(
            **sources,
            commodity_iv=commodity_iv,
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
            "cumulative_hypotheses": 144,
            "report_artifact_hashes": {
                name: _sha(report / name)
                for name in (
                    "metrics.json",
                    "freeze_record.json",
                    "FULL_PRECHECK.json",
                    "FULL_PRECHECK.log",
                    "calibration.json",
                    "calibration_verification.json",
                    "verification.json",
                )
                if (report / name).is_file()
            },
        },
    )
    print(
        json.dumps({"status": result["status"], "leads": result["leads"], "comparisons": 144}),
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
