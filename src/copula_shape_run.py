"""Exclusive frozen execution of shape calibration and bounded spread diagnostics."""

import argparse
import hashlib
import io
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.copula_calibration_run import (
    dump,
    read_bound_bytes,
    save_bound_bytes,
    sha,
    verify_pins,
)
from src.copula_shape import CONTRASTS, run_shape
from src.orthogonal_round2 import holm_adjust
from src.treasury_dealer_inference import masked_mean_inference

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("reports/copula_shape/predictive")
DATA = Path("data/model_memory_study/copula_shape_wave29")
OLD = Path("reports/copula_calibration/predictive")
OLD_DATA = Path("data/model_memory_study/copula_calibration_wave28")
CALENDAR_FILE = "data/model_memory_study/joint_copula_wave27/features.parquet"
OLD_TERMINAL = "483c26aac77924a64cc9d87fc017dd60c50dab9d7bdd8004c69343501b685587"


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def source_pins(root):
    root = Path(root)
    verify_pins(root, {str(OLD / "terminal.json"): OLD_TERMINAL})
    terminal = json.loads((root / OLD / "terminal.json").read_text())
    verify_pins(
        root,
        {
            str(OLD / "freeze.json"): terminal["freeze_sha256"],
            str(OLD / "metrics.json"): terminal["metrics_sha256"],
            str(OLD / "registration.json"): terminal["registration_sha256"],
        },
    )
    pins = json.loads((root / OLD / "freeze.json").read_text())["pins"]
    for mapping in (terminal["output_hashes"], terminal["report_artifact_hashes"]):
        for name, expected in mapping.items():
            if name in pins and pins[name] != expected:
                raise ValueError("Conflicting inherited pins")
            pins[name] = expected
    for file in (root / OLD).iterdir():
        if file.is_file():
            key = str(file.relative_to(root))
            if key not in pins:
                pins[key] = sha(file)
    verify_pins(root, pins)
    metrics = json.loads(read_bound_bytes(root, OLD / "metrics.json", pins))
    prior = metrics["inherited_rows"] + [
        {
            **r,
            "study": "copula_calibration_wave28",
            "source": str(OLD / "metrics.json"),
            "source_sha256": terminal["metrics_sha256"],
            "source_row_index": i,
        }
        for i, r in enumerate(metrics["rows"])
    ]
    if len(prior) != 158 or metrics["status"] != "COMPLETED":
        raise ValueError("Exactly158 authenticated predecessor comparisons required")
    return pins, prior


def freeze(root=ROOT):
    root = Path(root)
    report = root / REPORT
    if report.exists() or (root / DATA).exists():
        raise ValueError("Refusing to replace an existing registered attempt")
    prefit = json.loads((root / "reports/copula_shape/PREFIT.json").read_text())
    if (
        prefit["status"] != "PASS"
        or prefit["tests_run"] < 1
        or prefit["failures"]
        or prefit["errors"]
        or prefit["skipped"]
    ):
        raise ValueError("Passing nonempty prewritten test gate required")
    verify_pins(root, prefit["code_sha256"])
    pins, prior = source_pins(root)
    for path in (
        list((root / "src").glob("*.py"))
        + list((root / "tests").glob("*.py"))
        + [
            root / "copula_shape.yaml",
            root / ".gitignore",
            root / "Makefile",
            root / "docs/PROSPECTIVE_BENCHMARK.md",
        ]
    ):
        name = str(path.relative_to(root))
        digest = sha(path)
        if name in pins and pins[name] != digest:
            raise ValueError("Previously frozen source changed")
        pins[name] = digest
    for folder in (
        "reports/copula_shape",
        "reports/copula_spread_backtest",
        "reports/prospective_benchmark",
    ):
        for path in (root / folder).rglob("*"):
            if path.is_file():
                pins[str(path.relative_to(root))] = sha(path)
    report.mkdir(parents=True)
    frozen = {
        "created_utc": datetime.now(UTC).isoformat(),
        "pins": pins,
        "inherited_rows": prior,
        "protocol_sha256": pins["copula_shape.yaml"],
    }
    dump(report / "freeze.json", frozen)
    dump(
        report / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "freeze_sha256": sha(report / "freeze.json"),
            "contrasts": list(CONTRASTS),
            "inherited": 158,
            "new": 6,
            "cumulative": 164,
            "spread_scope": "All54 model/structure/distance risk cases and189 fixed conditional-premium policies, descriptive only.",
        },
    )
    return {"status": "FROZEN_REGISTERED", "pins": len(pins)}


def evaluate(panel, calendar, prior, protocol, inference=masked_mean_inference):
    if len(prior) != 158:
        raise ValueError("Exactly158 inherited rows required")
    indexed = panel.set_index("origin").sort_index()
    if not indexed.index.is_unique:
        raise ValueError("Unique paired origin required")
    for phase in ("development", "evaluation"):
        part = indexed.loc[indexed.phase.eq(phase)]
        if len(part) < protocol["support"]["phase_daily"] or any(
            part.offset.eq(k).sum() < protocol["support"]["offset_daily"] for k in range(5)
        ):
            raise ValueError("INSUFFICIENT_DATA: common phase/offset support")
    for start, end in protocol["forecast"]["stability"]:
        if len(indexed.loc[start:end]) < protocol["support"]["slice_daily"]:
            raise ValueError("INSUFFICIENT_DATA: stability support")
    rows = []
    inf = protocol["inference"]
    for contrast in CONTRASTS:
        phases = []
        for phase in ("development", "evaluation"):
            part = indexed.loc[indexed.phase.eq(phase)]
            start, end = protocol["forecast"][phase]
            full = calendar[(calendar >= start) & (calendar <= end)]
            result = inference(
                part[f"d_{contrast}"].reindex(full).to_numpy(float),
                full.isin(part.index),
                blocks=inf["blocks"],
                hac_lags=inf["hac_lags"],
                draws=inf["bootstrap_draws"],
                seed=inf["seed"] + inf["phase_codes"][phase] * 10000,
            )
            intervals = [result["hac"]["ci95"]] + [
                x["ci95"] for x in result["block_inference"].values()
            ]
            result.update(
                name=phase,
                ci95_envelope=[min(x[0] for x in intervals), max(x[1] for x in intervals)],
            )
            phases.append(result)
        rows.append(
            {
                "contrast": contrast,
                "phases": phases,
                "p_conservative": max(p["p_conservative"] for p in phases),
            }
        )
    wave = holm_adjust([r["p_conservative"] for r in rows])
    cumulative = holm_adjust([r["p_conservative"] for r in prior + rows])[-6:]
    for row, pw, pc in zip(rows, wave, cumulative, strict=True):
        row.update(
            p_holm_wave=float(pw),
            p_holm_cumulative=float(pc),
            adjusted_difference_detected=bool(
                row["phases"][0]["mean"] * row["phases"][1]["mean"] > 0
                and pw <= protocol["comparisons"]["wave_alpha"]
                and pc <= 0.05
            ),
        )
    diagnostics = {}
    for phase, part in panel.groupby("phase"):
        diagnostics[phase] = {}
        for asset in ("qqq", "spx"):
            diagnostics[phase][asset] = {}
            for margin in ("original", "calibrated", "shape"):
                v = part[f"normal_{margin}_{asset}"].to_numpy(float)
                pit = part[f"pit_{margin}_{asset}"].to_numpy(float)
                diagnostics[phase][asset][margin] = {
                    "mean_loss": float(-part[f"marginal_{margin}_{asset}"].mean()),
                    "normal_mean": float(v.mean()),
                    "normal_variance": float(v.var()),
                    "normal_skew": float(np.mean((v - v.mean()) ** 3) / v.var() ** 1.5),
                    "pit_lower": float(np.mean(pit < 0.025)),
                    "pit_upper": float(np.mean(pit > 0.975)),
                }
    return {
        "status": "COMPLETED",
        "rows": rows,
        "inherited_rows": prior,
        "new_hypotheses": 6,
        "cumulative_hypotheses": 164,
        "leads": [],
        "common_scored_origins": len(panel),
        "calibration_diagnostics": diagnostics,
    }


def verify_score_inference(panel, calendar, prior, protocol, metrics):
    from src.verify_copula_calibration_scores import _compare
    from src.verify_treasury_dealer_scores import _indexed_inference

    # Separate explicit-index resampling and HAC implementation; row densities
    # and contrast identities are independently covered by the forecast verifier.
    reconstructed = evaluate(panel, calendar, prior, protocol, inference=_indexed_inference)
    _compare(metrics, reconstructed)
    return {
        "status": "VERIFIED",
        "contrasts": 6,
        "scope": "Independent resampling/HAC primitive; shared declarative comparison assembly and Holm implementation. Row density and contrast identities independently reconstructed by forecast verifier.",
    }


def integration_check(risks, diagnostics):
    worst = {
        k: max(d["differences"][k] for d in diagnostics) for k in diagnostics[0]["differences"]
    }
    for key, value in worst.items():
        tolerance = 0.005 if key.startswith("p_") else 0.03
        if value > tolerance:
            raise ValueError(f"Fixed QMC consistency check failed: {key}={value}")
    if not np.isfinite(risks.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Nonfinite spread-risk forecast")
    return {
        "status": "TWO_SCRAMBLE_CONSISTENCY_CHECKED",
        "worst_differences": worst,
        "threshold_ambiguous_rows": int(risks.integration_gate_ambiguous.sum()),
        "limitation": "Two-scramble agreement is a numerical diagnostic, not a deterministic integration error bound.",
    }


def spread_safety_summary(risks, outcomes):
    keys = ["origin", "target_end", "phase", "structure", "distance", "width"]
    actual = outcomes[
        keys + ["portfolio_debit", "any_breach", "both_breach", "both_full_loss"]
    ]
    merged = risks.merge(actual, on=keys, validate="many_to_one")
    if len(merged) != len(risks):
        raise ValueError("Spread diagnostic outcome coverage mismatch")
    rows = []
    for labels, group in merged.groupby(
        ["phase", "structure", "distance", "model"], sort=True
    ):
        for selected, part in (
            ("all", group),
            ("sell", group.loc[group.p_any_breach <= 0.10]),
            ("skip", group.loc[group.p_any_breach > 0.10]),
        ):
            row = dict(zip(("phase", "structure", "distance", "model"), labels, strict=True))
            row.update(selection=selected, sessions=len(part), coverage=len(part) / len(group))
            if len(part):
                row.update(
                    predicted_any_breach=float(part.p_any_breach.mean()),
                    observed_any_breach=float(part.any_breach.mean()),
                    predicted_both_breach=float(part.p_both_breach.mean()),
                    observed_both_breach=float(part.both_breach.mean()),
                    predicted_both_full=float(part.p_both_full.mean()),
                    observed_both_full=float(part.both_full_loss.mean()),
                    predicted_mean_debit=float(part.mean_debit.mean()),
                    observed_mean_debit=float(part.portfolio_debit.mean()),
                    break_even_credit_with_scenario_fee=float(
                        part.portfolio_debit.mean() + 0.02
                    ),
                    brier_any=float(
                        np.mean((part.p_any_breach - part.any_breach.astype(float)) ** 2)
                    ),
                    brier_both=float(
                        np.mean((part.p_both_breach - part.both_breach.astype(float)) ** 2)
                    ),
                    payout_mean_squared_error=float(
                        np.mean((part.mean_debit - part.portfolio_debit) ** 2)
                    ),
                    var_exceedance_rate=float((part.portfolio_debit > part.var97_5).mean()),
                    mean_forecast_es=float(part.es97_5.mean()),
                    liability_exceeds_es_rate=float(
                        (part.portfolio_debit > part.es97_5).mean()
                    ),
                    numerical_threshold_ambiguities=int(part.integration_gate_ambiguous.sum()),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def retain_failure(root, error, bound, prior):
    report = Path(root) / REPORT
    retained = {}
    for name in ("metrics.json", "terminal.json"):
        path = report / name
        if path.exists():
            destination = path.with_name(path.stem + ".pre_invalidation.json")
            if destination.exists():
                raise ValueError("Refusing to overwrite retained failed publication")
            path.rename(destination)
            retained[str(destination.relative_to(root))] = sha(destination)
    failure = {
        "status": "UNEVALUABLE",
        "error": str(error),
        "completed_utc": datetime.now(UTC).isoformat(),
        "rows": [
            {"contrast": c, "p_conservative": 1.0, "adjusted_difference_detected": False}
            for c in CONTRASTS
        ],
        "inherited_rows": prior,
        "expected_output_hashes": dict(bound),
        "retained_unverified_artifacts": retained,
        "leads": [],
        "new_hypotheses": 6,
        "cumulative_hypotheses": 164,
    }
    dump(report / "failure.json", failure)
    dump(report / "terminal.json", failure)
    return failure


def commit_verified(root, metrics, bound, *, pins, anchors):
    root = Path(root)

    def check():
        verify_pins(root, pins)
        verify_pins(root, anchors)
        verify_pins(root, bound)

    try:
        check()
        save_bound_bytes(root, REPORT / "metrics.json", json_bytes(metrics), bound)
        check()
        terminal = {
            "status": "COMPLETED_VERIFIED_SHAPE_AND_SPREAD_DIAGNOSTIC",
            "completed_utc": datetime.now(UTC).isoformat(),
            "output_hashes": dict(bound),
            "anchors": anchors,
            "new_hypotheses": 6,
            "cumulative_hypotheses": 164,
            "leads": [],
            "limitation": "Historical reused data and hypothetical normalized spreads; conditional-premium scenarios are not executable option backtests. Spread accounting requires separate saved-output audit before economic interpretation.",
        }
        save_bound_bytes(root, REPORT / "terminal.json", json_bytes(terminal), bound)
        check()
        return terminal
    except Exception as error:
        retain_failure(root, error, bound, metrics["inherited_rows"])
        raise


def run(root=ROOT):
    from src.copula_spread_backtest import RISK_COLUMNS, run_backtest
    from src.copula_spread_forecasts import build_risks
    from src.verify_copula_shape import verify

    root = Path(root)
    report, data = root / REPORT, root / DATA
    if (report / "started.json").exists() or data.exists():
        raise ValueError("Refusing to rerun a registered attempt")
    anchor_payloads = {
        name: (report / name).read_bytes() for name in ("freeze.json", "registration.json")
    }
    anchors = {
        str(REPORT / name): hashlib.sha256(payload).hexdigest()
        for name, payload in anchor_payloads.items()
    }
    frozen = json.loads(anchor_payloads["freeze.json"])
    registration = json.loads(anchor_payloads["registration.json"])
    if registration["freeze_sha256"] != anchors[str(REPORT / "freeze.json")]:
        raise ValueError("Registration and freeze disagree")
    pins, bound = frozen["pins"], {}
    verify_pins(root, pins)
    verify_pins(root, anchors)
    protocol = yaml.safe_load(read_bound_bytes(root, "copula_shape.yaml", pins))
    save_bound_bytes(
        root,
        REPORT / "started.json",
        json_bytes({"started_utc": datetime.now(UTC).isoformat(), **anchors}),
        bound,
    )
    data.mkdir(parents=True, mode=0o700)
    os.chmod(data, 0o700)

    def save_json(name, value, *, public=False):
        relative = (REPORT if public else DATA) / name
        save_bound_bytes(root, relative, json_bytes(value), bound)

    def save_frame(name, frame):
        payload = frame.to_parquet(index=False)
        save_bound_bytes(root, DATA / name, payload, bound)
        return pd.read_parquet(io.BytesIO(read_bound_bytes(root, DATA / name, bound)))

    try:
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive_zip:
            for name in sorted(pins):
                if (
                    name.startswith(
                        (
                            "src/",
                            "tests/",
                            "docs/PROSPECTIVE_BENCHMARK",
                            "reports/copula_shape/",
                            "reports/copula_spread_backtest/",
                        )
                    )
                    or name == "copula_shape.yaml"
                    or name
                    in {
                        str(OLD_DATA / f)
                        for f in (
                            "archive.parquet",
                            "applications.parquet",
                            "panel.parquet",
                            "fits.json",
                        )
                    }
                ):
                    archive_zip.writestr(name, read_bound_bytes(root, name, pins))
        save_bound_bytes(root, DATA / "snapshot.zip", bundle.getvalue(), bound)
        read_old = lambda name: pd.read_parquet(
            io.BytesIO(read_bound_bytes(root, OLD_DATA / name, pins))
        )
        archive, baseline_app, baseline_panel = (
            read_old(f) for f in ("archive.parquet", "applications.parquet", "panel.parquet")
        )
        calendar = pd.DatetimeIndex(
            pd.read_parquet(
                io.BytesIO(read_bound_bytes(root, CALENDAR_FILE, pins)), columns=[]
            ).index
        )
        if calendar[-1] > pd.Timestamp("2025-10-20"):
            raise ValueError("Source ceiling exceeded")
        print("Fitting the frozen monthly shape correction...", flush=True)
        produced = run_shape(archive, baseline_app, baseline_panel)
        for name in ("applications", "panel"):
            produced[name] = save_frame(name + ".parquet", produced[name])
        save_json("fits.json", produced["fits"])
        produced["fits"] = json.loads(read_bound_bytes(root, DATA / "fits.json", bound))
        print("Independently verifying shape forecasts and fit certificates...", flush=True)
        proof = verify(archive, baseline_app, baseline_panel, produced, calendar=calendar)
        save_json("forecast_verification.json", proof, public=True)
        print("Evaluating six registered density comparisons...", flush=True)
        metrics = evaluate(produced["panel"], calendar, frozen["inherited_rows"], protocol)
        save_json(
            "score_verification.json",
            verify_score_inference(
                produced["panel"], calendar, frozen["inherited_rows"], protocol, metrics
            ),
            public=True,
        )
        print(
            "Generating bounded spread-risk forecasts with two fixed integrations...",
            flush=True,
        )
        risks, diagnostics = build_risks(
            produced["applications"], produced["panel"].origin, archive, calendar
        )
        risks = save_frame("spread_risks.parquet", risks)
        save_json("integration_diagnostics.json", diagnostics)
        save_json(
            "integration_verification.json", integration_check(risks, diagnostics), public=True
        )
        print(
            "Scoring all fixed spread filters and conditional-premium scenarios...", flush=True
        )
        actual = produced["panel"][["origin", "target_end", "y_qqq", "y_spx"]]
        backtest = run_backtest(risks.loc[:, list(RISK_COLUMNS)], actual, calendar)
        for name, value in backtest.items():
            if isinstance(value, pd.DataFrame):
                backtest[name] = save_frame("spread_" + name + ".parquet", value)
            else:
                save_json("spread_" + name + ".json", value, public=True)
        from src.verify_copula_spreads import verify as verify_spreads

        save_json(
            "spread_verification.json",
            verify_spreads(risks.loc[:, list(RISK_COLUMNS)], actual, backtest, calendar),
            public=True,
        )
        save_frame("spread_safety.parquet", spread_safety_summary(risks, backtest["outcomes"]))
        return commit_verified(root, metrics, bound, pins=pins, anchors=anchors)
    except Exception as error:
        if not (report / "failure.json").exists():
            retain_failure(root, error, bound, frozen["inherited_rows"])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    arguments = parser.parse_args()
    answer = freeze() if arguments.command == "freeze" else run()
    print(
        json.dumps(
            {
                k: v
                for k, v in answer.items()
                if k in ("status", "pins", "new_hypotheses", "cumulative_hypotheses")
            },
            indent=2,
        )
    )
