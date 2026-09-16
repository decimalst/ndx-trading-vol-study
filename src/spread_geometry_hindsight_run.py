"""Authenticated additive execution; no frozen parent artifact is modified."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
import zipfile

import numpy as np
import pandas as pd
import yaml

from src.spread_geometry_hindsight import (
    DEFAULT_DISTANCE_BPS, DEFAULT_WIDTH_BPS, MODELS, POLICIES, STRUCTURES,
    _dates, _inputs, run_search,
)
from src.verify_spread_geometry_hindsight import verify_search

REPORT = Path("reports/spread_geometry_hindsight")
DATA = Path("data/model_memory_study/spread_geometry_hindsight")
PROTOCOL = "spread_geometry_hindsight.yaml"
TEST_MODULES = ["tests.test_spread_geometry_hindsight", "tests.test_spread_geometry_hindsight_run",
                "tests.test_verify_spread_geometry_hindsight", "tests.test_copula_spread_backtest"]
CODE = [PROTOCOL, "src/spread_geometry_hindsight.py", "src/spread_geometry_hindsight_run.py",
        "src/verify_spread_geometry_hindsight.py", "src/copula_spread_backtest.py",
        *[name.replace(".", "/") + ".py" for name in TEST_MODULES]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticate(root, pins):
    root = Path(root).resolve()
    for name, expected in pins.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Input pin outside repository")
        try:
            actual = digest(path)
        except (OSError, TypeError) as exc:
            raise ValueError(f"Unavailable pinned file: {name}") from exc
        if actual != expected:
            raise ValueError(f"Pinned bytes changed: {name}")


def extract_inputs(panel, positions):
    cols = ["origin", "target_end", "phase", "y_qqq", "y_spx"]
    required = {"origin", "target_end", "phase", "structure", "model", "sell", "distance", "width"}
    if (not isinstance(panel, pd.DataFrame) or not set(cols).issubset(panel.columns)
            or not panel.columns.is_unique or panel.empty or panel.origin.duplicated().any()
            or not isinstance(positions, pd.DataFrame) or not positions.columns.is_unique
            or not required.issubset(positions.columns)):
        raise ValueError("Complete source panel and saved positions required")
    returns = panel[cols].copy()
    anchor = positions[(positions.distance == .02) & (positions.width == .01)].copy()
    if (anchor.empty or len(anchor) != len(returns)*21 or
            set(anchor.model) != {"always_sell", *MODELS} or set(anchor.structure) != set(STRUCTURES)
            or anchor.duplicated(["origin", "structure", "model"]).any()
            or set(anchor.origin) != set(returns.origin) or anchor.sell.dtype != np.dtype(bool)):
        raise ValueError("Complete exact primary geometry anchor required")
    _dates(anchor.origin)
    _dates(anchor.target_end)
    clock = returns.set_index("origin")
    for name in ["target_end", "phase"]:
        if not np.array_equal(anchor[name].to_numpy(), anchor.origin.map(clock[name]).to_numpy()):
            raise ValueError("Saved selection clock differs from return panel")
    anchor["policy"] = anchor.model.map(lambda m: m if m == "always_sell" else "fixed_2pct_" + m)
    selections = anchor[["origin", "structure", "policy", "sell"]].copy()
    _inputs(returns, selections)
    return returns, selections


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_pins(root, protocol):
    source = protocol["source"]
    pins = {source[key]: source[key + "_sha256"]
            for key in ["parent_terminal", "parent_freeze", "returns", "positions"]}
    authenticate(root, pins)
    parent = json.loads((root / source["parent_terminal"]).read_text())
    if parent["status"] != "COMPLETED_VERIFIED_DESCRIPTIVE_PROBABILITY_REPLAY":
        raise ValueError("Completed authenticated parent required")
    summary = "data/model_memory_study/copula_spread_replay/summary.parquet"
    pins[summary] = parent["output_hashes"][summary]
    pins["reports/copula_shape/predictive/terminal.json"] = parent["original_failed_terminal_sha256"]
    old = json.loads((root / source["parent_freeze"]).read_text())
    pins["src/copula_spread_backtest.py"] = old["pins"]["src/copula_spread_backtest.py"]
    authenticate(root, pins)
    return pins


def prepare(root):
    report, data = root / REPORT, root / DATA
    report.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load((root / PROTOCOL).read_text())
    if (tuple(protocol["grid"]["distance_bps"]) != DEFAULT_DISTANCE_BPS or
            tuple(protocol["grid"]["width_bps"]) != DEFAULT_WIDTH_BPS or
            tuple(protocol["grid"]["policies"]) != POLICIES):
        raise ValueError("Declared exhaustive grid differs from implementation")
    if data.exists() or (report / "PREFIT_BEFORE.json").exists():
        raise ValueError("Preparation is exclusive; previous attempt retained")
    sources = _source_pins(root, protocol)
    before = {name: digest(root / name) for name in CODE}
    started = now()
    write_json(report / "PREFIT_BEFORE.json", {"created_utc": started, "pins": before})
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    selected = suite.countTestCases()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    with (report / "PREFIT.log").open("x") as log:
        log.write(stream.getvalue())
    after = {name: digest(root / name) for name in CODE}
    passed = result.wasSuccessful() and not result.skipped and selected == result.testsRun and before == after
    receipt = {"status": "PASS" if passed else "FAILED", "started_utc": started,
               "completed_utc": now(), "selected": selected, "run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
               "source_unchanged": before == after, "log_sha256": digest(report / "PREFIT.log"),
               "code_pins": after}
    write_json(report / "PREFIT.json", receipt)
    if not passed:
        raise ValueError("Pre-run tests failed; preparation preserved")
    authenticate(root, sources)
    data.mkdir(parents=True, mode=0o700)
    os.chmod(data, 0o700)
    pins = {**sources, **after}
    for name in ["PREFIT_BEFORE.json", "PREFIT.log", "PREFIT.json"]:
        relative = str(REPORT / name)
        pins[relative] = digest(root / relative)
    snapshot = data / "snapshot.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(pins):
            archive.write(root / name, arcname=name)
    os.chmod(snapshot, 0o600)
    with zipfile.ZipFile(snapshot) as archive:
        captured = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    if captured != pins:
        raise ValueError("Snapshot member bytes differ from registered inputs")
    freeze = {"created_utc": now(), "pins": pins,
              "snapshot": str(DATA / "snapshot.zip"), "snapshot_sha256": digest(snapshot)}
    write_json(report / "freeze.json", freeze)
    registration = {"created_utc": now(), "scope": protocol["scope"],
                    "freeze_sha256": digest(report / "freeze.json"), "new_formal_hypotheses": 0,
                    "unchanged_existing_formal_hypotheses": 164,
                    "descriptive_combinations_per_period": 13440, "periods": 3,
                    "test_gate": receipt["status"]}
    write_json(report / "registration.json", registration)
    print(json.dumps({"pretests": selected, "status": "FROZEN_READY", "snapshot_members": len(pins)}))


def _anchor_check(outputs, parent):
    account = outputs["accounts"]
    anchor = account[(account.distance_bps == 200) & (account.width_bps == 100)
                     & (account.phase != "pooled") & (account.scenario != "open_5bp")].copy()
    old = parent[(parent.distance == .02) & (parent.width == .01)].copy()
    old["policy"] = old.model.map(lambda m: m if m == "always_sell" else "fixed_2pct_" + m)
    old["scenario"] = old.credit_fraction.map({.05: "width_05", .10: "width_10", .20: "width_20"})
    keys = ["phase", "structure", "policy", "scenario"]
    joined = anchor.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(joined) != 126 or len(anchor) != 126 or len(old) != 126:
        raise ValueError("Incomplete original account anchor")
    for name in ["sessions", "sold_sessions", "ending_equity", "total_return", "max_drawdown"]:
        left, right = joined[name+"_new"], joined[name+"_old"]
        if name in ["sessions", "sold_sessions"]:
            same = np.array_equal(left, right)
        else:
            same = np.allclose(left, right, rtol=1e-10, atol=1e-11)
        if not same:
            raise ValueError(f"Original anchor mismatch: {name}")
    geometry = outputs["geometry"]
    geometry = geometry[(geometry.distance_bps == 200) & (geometry.width_bps == 100)
                        & (geometry.phase != "pooled")]
    old = old[old.scenario == "width_10"]
    paired = geometry.merge(old, on=["phase", "structure", "policy"], validate="one_to_one")
    if len(paired) != 42 or not np.allclose(paired.mean_debit, paired.sold_mean_debit, atol=1e-12, rtol=1e-10):
        raise ValueError("Original geometry liability anchor mismatch")
    for count, rate in [("any_breach_count", "sold_any_breach_rate"),
                        ("both_breach_count", "sold_both_breach_rate"),
                        ("any_full_count", "sold_any_full_loss_rate"),
                        ("both_full_count", "sold_both_full_loss_rate")]:
        expected = np.rint(paired[rate]*paired.sold_sessions_y).astype(int)
        if not np.array_equal(paired[count], expected):
            raise ValueError("Original geometry event anchor mismatch")
    return {"status": "VERIFIED", "account_rows": 126, "geometry_rows": 42,
            "scope": "Unchanged200bps/100bps cases reproduce parent accounts, liability and event counts."}


def run(root):
    report, data = root / REPORT, root / DATA
    freeze = json.loads((report / "freeze.json").read_text())
    registration = json.loads((report / "registration.json").read_text())
    if digest(report / "freeze.json") != registration["freeze_sha256"]:
        raise ValueError("Registration/freeze mismatch")
    anchors = {str(REPORT / name): digest(report / name) for name in ["freeze.json", "registration.json"]}
    authenticate(root, freeze["pins"])
    authenticate(root, {freeze["snapshot"]: freeze["snapshot_sha256"]})
    write_json(report / "started.json", {"started_utc": now(), "anchors": anchors,
                                        "interpretation": "Explicitly hindsight-selected, no new formal hypothesis."})
    output_pins = {}
    try:
        protocol = yaml.safe_load((root / PROTOCOL).read_text())
        snapshot_bytes = (root / freeze["snapshot"]).read_bytes()
        if hashlib.sha256(snapshot_bytes).hexdigest() != freeze["snapshot_sha256"]:
            raise ValueError("Snapshot changed before input capture")
        with zipfile.ZipFile(io.BytesIO(snapshot_bytes)) as archive:
            captured = {name: archive.read(name) for name in archive.namelist()}
        if {name: hashlib.sha256(value).hexdigest() for name, value in captured.items()} != freeze["pins"]:
            raise ValueError("Snapshot membership or content mismatch")
        panel = pd.read_parquet(io.BytesIO(captured[protocol["source"]["returns"]]))
        positions = pd.read_parquet(io.BytesIO(captured[protocol["source"]["positions"]]))
        returns, selections = extract_inputs(panel, positions)
        if len(returns) != 2190 or returns.phase.value_counts().to_dict() != {"evaluation": 1457, "development": 733}:
            raise ValueError("Frozen original cohort count mismatch")
        for name, frame in [("returns", returns), ("selections", selections)]:
            path = data / f"{name}.parquet"
            if path.exists():
                raise ValueError("No output overwrite")
            frame.to_parquet(path, index=False)
            os.chmod(path, 0o600)
            output_pins[str(path.relative_to(root))] = digest(path)
        outputs = run_search(returns, selections)
        for name, frame in outputs.items():
            path = data / f"{name}.parquet"
            if path.exists():
                raise ValueError("No output overwrite")
            frame.to_parquet(path, index=False)
            os.chmod(path, 0o600)
            output_pins[str(path.relative_to(root))] = digest(path)
        saved = {name: pd.read_parquet(data / f"{name}.parquet") for name in outputs}
        proof = verify_search(pd.read_parquet(data/"returns.parquet"), pd.read_parquet(data/"selections.parquet"),
                              saved, DEFAULT_DISTANCE_BPS, DEFAULT_WIDTH_BPS)
        parent = pd.read_parquet(io.BytesIO(captured["data/model_memory_study/copula_spread_replay/summary.parquet"]))
        proof["original_anchor"] = _anchor_check(saved, parent)
        write_json(report / "verification.json", proof)
        output_pins[str(REPORT / "verification.json")] = digest(report / "verification.json")
        for name, expected in [("geometry", 10080), ("accounts", 40320), ("winners", 252)]:
            if len(saved[name]) != expected:
                raise ValueError("Complete exhaustive grid count mismatch")
        authenticate(root, {**freeze["pins"], **anchors, **output_pins})
        terminal = {"status": "COMPLETED_VERIFIED_HINDSIGHT_SEARCH", "completed_utc": now(),
                    "new_formal_hypotheses": 0, "existing_formal_hypotheses": 164,
                    "descriptive_combinations_per_period": 13440,
                    "geometry_rows": 10080, "account_rows": 40320, "winner_rows": 252,
                    "anchors": anchors, "outputs": output_pins,
                    "scope": "Hindsight geometry selection under hypothetical premiums; no new breach forecasts, quotes, inference or promotion."}
        write_json(report / "terminal.json", terminal)
        print(json.dumps({"status": terminal["status"], "verification": proof}))
    except Exception as exc:
        failure = {"status": "FAILED_HINDSIGHT_ATTEMPT", "failed_utc": now(),
                   "error": repr(exc), "anchors": anchors, "partial_output_pins": output_pins}
        write_json(report / "failure.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    (prepare if args.action == "prepare" else run)(root)


if __name__ == "__main__":
    main()
