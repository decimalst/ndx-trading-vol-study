"""Probability-only descriptive replay; the original failed study stays intact.

The accounting engine's unused ES interface receives a universal payoff bound.
That legacy field is removed from outputs, not reported as an ES estimate.
"""

import argparse
import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.copula_calibration_run import (
    dump,
    read_bound_bytes,
    save_bound_bytes,
    sha,
    verify_pins,
)
from src.copula_spread_backtest import RISK_COLUMNS, run_backtest
from src.verify_copula_spreads import verify as verify_accounts

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("reports/copula_spread_replay")
DATA = Path("data/model_memory_study/copula_spread_replay")
FAILED = Path("reports/copula_shape/predictive")
OLD_DATA = Path("data/model_memory_study/copula_shape_wave29")
TERMINAL_SHA = "9f7a1ef5d19c16305f708aaa9bf8ce832c6de8d8e13ca73b9c4899a66e385a0c"
FREEZE_SHA = "737d9b68bcebf3578efc9ee2dba9e8c4141dfab6a174e1ce39ad110bee831b03"
REQUIRED = tuple(k for k in RISK_COLUMNS if k != "es97_5")


def _bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _engine_input(risks):
    if (
        not isinstance(risks, pd.DataFrame)
        or not risks.columns.is_unique
        or not set(REQUIRED) <= set(risks)
    ):
        raise ValueError("Complete probability and mean-liability forecast required")
    selected = risks.loc[:, list(REQUIRED)].copy()
    selected["es97_5"] = 1.0  # Universal support bound; unused by the selling rule.
    return selected.loc[:, list(RISK_COLUMNS)]


def independent_public_accounting(risks, realized, backtest, calendar):
    required = {"positions", "path", "summary", "outcomes"}
    if not required <= set(backtest):
        raise ValueError("Complete saved public accounting tables required")
    restored = {k: backtest[k].copy() for k in required}
    for frame in restored.values():
        if any("es97" in c or "var97" in c for c in frame.columns):
            raise ValueError("Legacy tail estimate leaked into probability-only output")
    positions = restored["positions"]
    if "max_terminal_debit" not in positions or not positions.max_terminal_debit.eq(1.0).all():
        raise ValueError("Explicit universal bounded-payoff limit required")
    positions.drop(columns="max_terminal_debit", inplace=True)
    positions["es97_5"] = np.where(positions.model.eq("always_sell"), np.nan, 1.0)
    restored["path"]["es97_5"] = np.where(
        restored["path"].model.eq("always_sell"), np.nan, 1.0
    )
    result = verify_accounts(_engine_input(risks), realized, restored, calendar)
    result.update(
        scope="Independent saved probability decisions and account reconstruction; universal payoff bound is not a tail estimate.",
        original_failed_result_preserved=True,
    )
    return result


def probability_backtest(risks, realized, calendar):
    internal = _engine_input(risks)
    result = run_backtest(internal, realized, calendar)
    for key, frame in result.items():
        dropped = [c for c in frame if "es97" in c or "var97" in c]
        result[key] = frame.drop(columns=dropped)
    result["positions"]["max_terminal_debit"] = 1.0
    result["verification"] = independent_public_accounting(risks, realized, result, calendar)
    return result


def source_pins(root=ROOT):
    root = Path(root)
    anchors = {
        str(FAILED / "terminal.json"): TERMINAL_SHA,
        str(FAILED / "freeze.json"): FREEZE_SHA,
    }
    verify_pins(root, anchors)
    failure = json.loads(read_bound_bytes(root, FAILED / "terminal.json", anchors))
    if (
        failure["status"] != "UNEVALUABLE"
        or len(failure["rows"]) != 6
        or any(r["p_conservative"] != 1 for r in failure["rows"])
    ):
        raise ValueError("Original six-comparison failure must remain intact")
    frozen = json.loads(read_bound_bytes(root, FAILED / "freeze.json", anchors))
    pins = {**frozen["pins"], **anchors}
    for name, expected in failure["expected_output_hashes"].items():
        if name in pins and pins[name] != expected:
            raise ValueError("Conflicting failed-attempt source pin")
        pins[name] = expected
    for path in (root / FAILED).iterdir():
        if path.is_file() and str(path.relative_to(root)) not in pins:
            pins[str(path.relative_to(root))] = sha(path)
    verify_pins(root, pins)
    return pins


def freeze(root=ROOT):
    root = Path(root)
    report = root / REPORT
    if (
        (report / "registration.json").exists()
        or (report / "freeze.json").exists()
        or (root / DATA).exists()
    ):
        raise ValueError("Refusing duplicate replay attempt")
    test = json.loads((report / "PREFIT.json").read_text())
    if (
        test["status"] != "PASS"
        or test["tests_run"] < 1
        or any(test[k] for k in ("failures", "errors", "skipped"))
    ):
        raise ValueError("Passing prewritten replay gate required")
    verify_pins(root, test["code_sha256"])
    pins = source_pins(root)
    for path in [
        root / "copula_spread_replay.yaml",
        root / "src/copula_spread_replay.py",
        root / "tests/test_copula_spread_replay.py",
    ] + list(report.glob("*")):
        if path.is_file():
            pins[str(path.relative_to(root))] = sha(path)
    dump(report / "freeze.json", {"created_utc": datetime.now(UTC).isoformat(), "pins": pins})
    dump(
        report / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "freeze_sha256": sha(report / "freeze.json"),
            "new_hypotheses": 0,
            "cumulative_hypotheses": 164,
            "original_failure": TERMINAL_SHA,
            "scope": "Descriptive accounting replay from unchanged saved probabilities; no refits, sampling or inference.",
        },
    )
    return {"status": "FROZEN_DESCRIPTIVE_REPLAY", "pins": len(pins)}


def safety_summary(risks, outcomes):
    keys = ["origin", "target_end", "phase", "structure", "distance", "width"]
    outcomes = outcomes[
        keys + ["portfolio_debit", "any_breach", "both_breach", "both_full_loss"]
    ]
    frame = risks.merge(outcomes, on=keys, validate="many_to_one")
    if len(frame) != len(risks):
        raise ValueError("Exact descriptive outcome cohort required")
    rows = []
    for labels, group in frame.groupby(["phase", "structure", "distance", "model"], sort=True):
        for selection, selected in (
            ("all", group),
            ("sell", group.loc[group.p_any_breach <= 0.10]),
            ("skip", group.loc[group.p_any_breach > 0.10]),
        ):
            row = dict(zip(("phase", "structure", "distance", "model"), labels, strict=True))
            row.update(
                selection=selection,
                sessions=len(selected),
                coverage=len(selected) / len(group),
            )
            if len(selected):
                for field in (
                    "p_any_breach",
                    "p_both_breach",
                    "p_both_full",
                    "mean_debit",
                    "portfolio_debit",
                    "any_breach",
                    "both_breach",
                    "both_full_loss",
                ):
                    row[field] = float(selected[field].mean())
                row.update(
                    brier_any=float(
                        np.mean(
                            (selected.p_any_breach - selected.any_breach.astype(float)) ** 2
                        )
                    ),
                    brier_both=float(
                        np.mean(
                            (selected.p_both_breach - selected.both_breach.astype(float)) ** 2
                        )
                    ),
                    payout_mse=float(
                        np.mean((selected.mean_debit - selected.portfolio_debit) ** 2)
                    ),
                    break_even_credit=float(selected.portfolio_debit.mean() + 0.02),
                    numerical_threshold_ambiguities=int(
                        selected.integration_gate_ambiguous.sum()
                    ),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def density_description(panel):
    rows = []
    for phase, part in panel.groupby("phase"):
        for asset in ("qqq", "spx"):
            for margin in ("original", "calibrated", "shape"):
                v = part[f"normal_{margin}_{asset}"].to_numpy(float)
                pit = part[f"pit_{margin}_{asset}"].to_numpy(float)
                rows.append(
                    {
                        "phase": phase,
                        "asset": asset,
                        "margin": margin,
                        "n": len(part),
                        "mean_loss": float(-part[f"marginal_{margin}_{asset}"].mean()),
                        "normal_mean": float(v.mean()),
                        "normal_variance": float(v.var()),
                        "normal_skew": float(np.mean((v - v.mean()) ** 3) / v.var() ** 1.5),
                        "pit_lower": float(np.mean(pit < 0.025)),
                        "pit_upper": float(np.mean(pit > 0.975)),
                    }
                )
    return rows


def run(root=ROOT):
    root = Path(root)
    report, data = root / REPORT, root / DATA
    if (report / "started.json").exists() or data.exists():
        raise ValueError("Refusing replay rerun")
    payloads = {
        name: (report / name).read_bytes() for name in ("freeze.json", "registration.json")
    }
    anchors = {
        str(REPORT / name): hashlib.sha256(v).hexdigest() for name, v in payloads.items()
    }
    frozen, registration = (
        json.loads(payloads[name]) for name in ("freeze.json", "registration.json")
    )
    if registration["freeze_sha256"] != anchors[str(REPORT / "freeze.json")]:
        raise ValueError("Replay registration mismatch")
    pins, bound = frozen["pins"], {}
    verify_pins(root, pins)
    verify_pins(root, anchors)
    data.mkdir(parents=True, mode=0o700)
    os.chmod(data, 0o700)

    def save_json(name, value):
        save_bound_bytes(root, REPORT / name, _bytes(value), bound)

    def save_frame(name, frame):
        save_bound_bytes(root, DATA / name, frame.to_parquet(index=False), bound)
        return pd.read_parquet(io.BytesIO(read_bound_bytes(root, DATA / name, bound)))

    def check():
        verify_pins(root, pins)
        verify_pins(root, anchors)
        verify_pins(root, bound)

    try:
        save_json(
            "started.json", {"started_utc": datetime.now(UTC).isoformat(), "anchors": anchors}
        )
        read = lambda name, **kw: pd.read_parquet(
            io.BytesIO(read_bound_bytes(root, name, pins)), **kw
        )
        risks = read(OLD_DATA / "spread_risks.parquet")
        panel = read(OLD_DATA / "panel.parquet")
        calendar = pd.DatetimeIndex(
            read(
                "data/model_memory_study/joint_copula_wave27/features.parquet", columns=[]
            ).index
        )
        actual = panel[["origin", "target_end", "y_qqq", "y_spx"]]
        print(
            "Replaying unchanged breach-probability rules and all fixed credit scenarios...",
            flush=True,
        )
        result = probability_backtest(risks, actual, calendar)
        for name in ("positions", "path", "summary", "outcomes"):
            result[name] = save_frame(name + ".parquet", result[name])
        print("Checking the exact saved account tables independently...", flush=True)
        proof = independent_public_accounting(risks, actual, result, calendar)
        save_json("accounting_verification.json", proof)
        save_frame("safety_summary.parquet", safety_summary(risks, result["outcomes"]))
        save_json(
            "density_descriptions.json",
            {
                "scope": "Descriptive verified saved rows only; original six p1 results unchanged.",
                "rows": density_description(panel),
            },
        )
        check()
        terminal = {
            "status": "COMPLETED_VERIFIED_DESCRIPTIVE_PROBABILITY_REPLAY",
            "completed_utc": datetime.now(UTC).isoformat(),
            "new_hypotheses": 0,
            "cumulative_hypotheses": 164,
            "leads": [],
            "original_failed_terminal_sha256": TERMINAL_SHA,
            "output_hashes": dict(bound),
            "anchors": anchors,
            "scope": "Expiration-liability safety and hypothetical-credit accounting only; no ES/VaR estimates, new model fits, sampling or inference.",
        }
        save_json("terminal.json", terminal)
        check()
        return terminal
    except Exception as error:
        path = report / "terminal.json"
        if path.exists():
            path.rename(report / "terminal.pre_invalidation.json")
        failure = {
            "status": "UNEVALUABLE_DESCRIPTIVE_REPLAY",
            "error": str(error),
            "new_hypotheses": 0,
            "cumulative_hypotheses": 164,
            "expected_output_hashes": bound,
            "leads": [],
        }
        dump(report / "failure.json", failure)
        dump(path, failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    arguments = parser.parse_args()
    result = freeze() if arguments.command == "freeze" else run()
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "pins", "new_hypotheses", "cumulative_hypotheses")
            },
            indent=2,
        )
    )
