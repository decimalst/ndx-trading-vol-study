"""Prospectively fixed peak-age return comparison and guarded experiment runner."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import index_hinge_search as existing_inference
from . import orthogonal_round2 as inference
from . import peak_age_admission as admission
from . import peak_age_features as feature_builder
from . import peak_age_pipeline as pipeline
from .international_search import diagnostics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "peak_age.yaml"
REPORT = ROOT / "reports/peak_age"
OUT = ROOT / "data/peak_age"
CONTRASTS = tuple(("peak_age", c, 21) for c in ("baseline", "depth", "mean"))
WAVE_ALPHA = 0.05 / (22 * 23)
CONTRACT = {
    "study_id": "peak_age_wave22",
    "specified_on": "2026-09-08",
    "status": "specified_before_historical_admission_features_support_fits_or_scores",
    "wave": 22,
    "evidence_class": "exploratory_reused_archival_SPX_return_history",
    "objective": "Test trailing closing-price peak age as a sequential correction beyond drawdown "
    "depth, window return and fixed market controls",
    "sources": {
        "daily": "data/research_paths/spx_daily.parquet",
        "vix": "data/free_sources/raw/cboe/VIX_History.csv",
        "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
        "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
    },
    "source_contract": {
        "price": "raw SPX OHLC; Yahoo price index, no dividends or risk-free "
        "subtraction; no fund execution claim",
        "vintage": "existing archival Yahoo and Cboe extracts; historical revisions "
        "and exact release latency unverified",
        "vix9d": "January2011-October2013 back-calculated archival training values; "
        "not available at original observation dates",
        "missing": "preserve observed SPX reference calendar before rolling; no "
        "filling; strict complete windows",
        "timing": "every market feature ends at previous observed SPX session; entry "
        "weekday only calendar feature",
        "variance_proxy": "max(Garman-Klass daily variance,1e-10) plus squared raw "
        "overnight log return",
        "scale": "annualization252 is explicit convention; VIX30calendar-day variance "
        "and trailing22trading-session OHLC proxy differ",
        "interpretation": "log implied-versus-trailing-proxy gap; neither measured "
        "variance risk premium nor replication of high-frequency "
        "premium research",
    },
    "index": {
        "asset": "SPX",
        "source_end": "2025-10-20",
        "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04",
        "origin_end": "2025-10-17",
        "latest_target": "2025-10-20",
        "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "minimum_train": 1000,
        "market_lag": 1,
        "horizons": [21],
        "models": ["mean", "baseline", "depth", "peak_age"],
        "raw": [
            "const",
            "I",
            "R",
            "ret_d",
            "ret_w",
            "ret_m",
            "ret_q",
            "lr_d",
            "lr_w",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
        ],
        "baseline": [
            "const",
            "I",
            "R",
            "ret_d",
            "ret_w",
            "ret_m",
            "ret_q",
            "lr_d",
            "lr_w",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "I_square",
            "R_square",
        ],
        "depth": [
            "const",
            "I",
            "R",
            "ret_d",
            "ret_w",
            "ret_m",
            "ret_q",
            "lr_d",
            "lr_w",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "I_square",
            "R_square",
            "drawdown",
            "drawdown_sq",
            "window_return",
        ],
        "common": [
            "const",
            "I",
            "R",
            "ret_d",
            "ret_w",
            "ret_m",
            "ret_q",
            "lr_d",
            "lr_w",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "peak_age",
            "drawdown",
            "drawdown_sq",
            "window_return",
        ],
        "target": "log(raw_close[t+21]/raw_close[t]); target end and availability at observed "
        "session t+21 close",
        "fit_schedule": "First common-feature-complete origin of each month within origin/phase "
        "fences before query-label filtering; all four arms use identical "
        "mature training and full application origins",
        "training_availability": "Earlier common-complete origins with finite labels available "
        "no later than previous observed session of fit; whole-trial "
        "failure below1000",
        "missing": "Keep full reference calendar; exact252-close windows; no "
        "fill/compress/window shortening; no arithmetic-fault row dropping",
    },
    "peak": {
        "window_closes": 252,
        "return_intervals": 251,
        "cutoff": "Previous observed SPX session k; literal positions k-251..k",
        "tie": "Latest exact stored float64 close equal to window maximum P",
        "age": "(k-latest_peak_position)/251; fixed scale one",
        "depth": "Stable log ratio P/C[k]",
        "depth_square": "Single multiplication of returned depth",
        "window_return": "Stable log ratio C[k]/C[k-251]",
        "log_ratio": "Positive finite float64 x,y; exact equality returns0; frexp "
        "mantissas/exponents mx,ex,my,ey, d=ex-ey; abs(d)<=1 uses "
        "log1p((ldexp(mx,d)-my)/my), otherwise fsum(log(mx/my),d*log(2)); finite "
        "result must have correct nonzero sign for unequal prices",
        "arithmetic": "No overflow or nonzero-product underflow in required operations; "
        "incomplete source windows remain missing; arithmetic invalidity on "
        "otherwise observed inputs aborts trial",
        "interpretation": "Age of latest maximum inside rolling252-close window, not "
        "all-time-high or recovery duration",
    },
    "model": {
        "ridge": 0.01,
        "baseline_slopes": 16,
        "depth_slopes": 19,
        "curvature": "I_square and R_square centered on exact common training rows only",
        "scaling": "Population mean/std on common training rows; all declared nuisance scales "
        "finite and >1e-12; no unused hinge and no feature deletion",
        "intercept": "Unpenalized common training target mean",
        "objective": "Mean squared error plus .01 squared norm of standardized slopes; "
        "separately refit baseline and depth",
        "scalar": "Freeze current monthly depth fit; residuals are y minus its in-sample "
        "training predictions; age centered by math.fsum/n, exactconstant uses first "
        "age; beta=mean(centered_age*residual)/(mean(centered_age^2)+.01)",
        "application": "depth_prediction+beta*(query_age-training_age_mean); no additional "
        "intercept, no joint refit, no age standardization",
        "constant": "Exactconstant training age or exactlyzero numerator yields beta0 and exact "
        "copy of depth query predictions, even if query age differs",
        "attribution": "Sequential correction can retune ridge shrinkage inside nuisance span; "
        "no Frisch-Waugh residualization or orthogonality claim",
        "fallback": "None; any source/support/model/numerical/inference/publication failure "
        "aborts all three hypotheses at p1",
    },
    "comparisons": {
        "new_hypotheses": 3,
        "inherited_hypotheses": 137,
        "cumulative_hypotheses": 140,
        "inherited_sources": [
            "reports/orthogonal_round2/metrics.json",
            "reports/model_memory_study/combined_metrics.json",
            "reports/iterative_signal_search/metrics.json",
            "reports/international_volatility/metrics.json",
            "reports/overnight_index/metrics.json",
            "reports/macro_overnight/metrics.json",
            "reports/measurement_memory/metrics.json",
            "reports/index_hinge/metrics.json",
            "reports/tail_shape/metrics.json",
            "reports/calendar_variance/metrics.json",
            "reports/relative_risk/metrics.json",
            "reports/joint_risk/metrics.json",
            "reports/cross_moment/metrics.json",
            "reports/target_aligned/metrics.json",
            "reports/sign_memory/metrics.json",
            "reports/causal_pool/metrics.json",
            "reports/civil_quarter/metrics.json",
            "reports/civil_quarter_replay/metrics.json",
            "reports/profiled_quarter/metrics.json",
            "reports/range_alert/metrics.json",
            "reports/issued_calibration/metrics.json",
            "reports/event_cluster/metrics.json",
            "reports/event_cluster_replay/metrics.json",
        ],
        "contrasts": [
            ["peak_age", "baseline", 21],
            ["peak_age", "depth", 21],
            ["peak_age", "mean", 21],
        ],
        "gate": "All three controls, both phases >=.25percent relative MSE improvement; "
        "negative paired differences in both fixed later slices and every "
        "nonempty21-offset in both phases; wave and cumulative Holm",
    },
    "inference": {
        "loss": "squared 21-session cumulative log-price-return prediction error",
        "effect_relative": 0.0025,
        "blocks": [126, 252, 504],
        "hac_lags": 504,
        "minimum_phase_observations": 505,
        "bootstrap_draws": 399999,
        "seed": 20260928,
        "seed_rule": "seed+21*1000000+phase_code*10000+block; development0,evaluation1; "
        "same drawdesign across controls",
        "wave_alpha": 9.881422924901186e-05,
        "cumulative_alpha": 0.05,
        "p_phase": "Maximum two-sided centered-null circular block bootstrap p and Bartlett "
        "HAC504 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm3 against wave_alpha and cumulativeHolm140 against .05",
        "offsets": "All offsets0..20 anchored to full bounded SPX calendar position "
        "modulo21 before any filtering; each must be nonempty and improve",
        "resolution": "1/400000 below one tenth strictest raw Holm3 wave cutoff1/30360",
        "limits": "Repeated archival selection, overlapping targets and original "
        "source-vintage limits; no untouched confirmation or executable trading "
        "claim",
    },
    "verification": {
        "ridge_method": "Independent augmented least squares",
        "scalar_method": "Independent one-column augmented least squares conditional on "
        "validated saved depth fit",
        "gradient_max_abs": 1e-10,
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-12,
        "forecast_rtol": 1e-08,
        "forecast_atol": 1e-12,
        "derived_rtol": 1e-10,
        "derived_atol": 1e-12,
        "date_rule": "Native naive ms/us/ns normalized-midnight dates, exact instants "
        "and NaT masks, checked lossless ns conversion with roundtrip; all "
        "nondates keep declared types and finite-value checks",
        "coverage": "Entire bounded feature/target/state tables, every monthly common "
        "training/application cohort, all saved coefficients and gradients, "
        "all unscored applications, scored predictions/losses, independent "
        "three-contrast inference, complete ledger and "
        "input/output/preservation identities",
        "feature_rtol": 1e-10,
        "feature_atol": 1e-12,
        "target_rtol": 1e-10,
        "target_atol": 1e-13,
    },
    "upstream": {
        "anchors": {
            "event_cluster_replay.yaml": "b3eeea877c536a25cde0edca69cd7cc1474a425a2458ac36978adfeacee7e76f",
            "reports/event_cluster_replay/publication_audit.json": "1e962011d0ccf66776a05a09ba09e8420f5d48f4c646e6bc8616ec07c81bdac5",
        },
        "role": "Verified terminal wave21 plus complete transitive source and failed-family "
        "preservation closure; decode only checked bounded SPX OHLC and Cboe "
        "buffers; no old forecasts used as newly fitted controls",
    },
    "outputs": {"data": "data/peak_age", "reports": "reports/peak_age"},
}


def validate(p):
    if json.dumps(p, sort_keys=True, allow_nan=False) != json.dumps(
        CONTRACT, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Frozen peak-age specification differs")


def inherited(p, pins=None):
    rows = []
    for source_name in p["comparisons"]["inherited_sources"]:
        payload = (ROOT / source_name).read_bytes()
        signature = hashlib.sha256(payload).hexdigest()
        if pins is not None and pins.get(source_name) != signature:
            raise ValueError(
                "Pinned inherited metrics changed before decoding: " + source_name
            )
        for number, row in enumerate(json.loads(payload)["rows"]):
            rows.append(
                {
                    "study": row.get(
                        "study", Path(source_name).parent.name or Path(source_name).stem
                    ),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source_name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{name: row[name] for name in ["measure", "score"] if name in row},
                }
            )
    if len(rows) != 137:
        raise ValueError("All137 inherited comparisons required")
    if any(
        isinstance(row["p_conservative"], bool)
        or not isinstance(row["p_conservative"], (int, float))
        or not np.isfinite(row["p_conservative"])
        or not 0 <= row["p_conservative"] <= 1
        for row in rows
    ):
        raise ValueError("Finite inherited probabilities in [0,1] required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    ps = [row["p_holm_wave"], row["p_holm_cumulative"]]
    if any(
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not np.isfinite(x)
        or not 0 <= x <= 1
        for x in ps
    ):
        return False
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        ps[0] < WAVE_ALPHA
        and ps[1] < 0.05
        and all(
            x["n"] >= 505 and x["delta"] < 0 and x["gain_relative"] >= 0.0025 for x in phases
        )
        and len(evaluation["stability"]) == 2
        and all(x["n"] > 0 and x["delta"] < 0 for x in evaluation["stability"])
        and all(
            len(x["nonoverlap_phases"]) == 21
            and {part["phase"] for part in x["nonoverlap_phases"]} == set(range(21))
            and all(
                part["n"] > 0 and part["delta"] is not None and part["delta"] < 0
                for part in x["nonoverlap_phases"]
            )
            for x in phases
        )
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["horizon"]) for x in rows]
    if len(keys) != 3 or set(keys) != set(CONTRASTS):
        raise ValueError("Exact complete three-contrast21-session family required")
    return [21] if all(passes(x) for x in rows) else []


def evaluate(panel, calendar, p, monthly_schedules, application_n, *, prior=None):
    pipeline.validate_panel(panel)
    dates = pd.DatetimeIndex(calendar)
    pipeline._dates(dates)
    n = int(panel.origin.nunique())
    if (
        type(monthly_schedules) is not int
        or monthly_schedules <= 0
        or type(application_n) is not int
        or application_n < n
        or monthly_schedules > application_n
    ):
        raise ValueError("Literal positive monthly and complete application counts required")
    if not panel.origin.isin(dates).all():
        raise ValueError("Scored origins must belong to full reference calendar")
    clock = pd.Series(dates, index=dates)
    for name, shift in [
        ("feature_cutoff_date", 1),
        ("target_end", -21),
        ("available_date", -21),
    ]:
        if not np.array_equal(
            panel[name].to_numpy(), clock.shift(shift).loc[panel.origin].to_numpy()
        ):
            raise ValueError("Exact previous-session and21-session scored clocks required")
    dev = panel.origin.between(*p["index"]["development"])
    ev = panel.origin.between(*p["index"]["evaluation"])
    if (
        not (dev | ev).all()
        or not np.array_equal(panel.phase, np.where(dev, "development", "evaluation"))
        or not panel.train_n.ge(p["index"]["minimum_train"]).all()
        or not panel.available_date.le(pd.Timestamp(p["index"]["latest_target"])).all()
        or not panel.loc[dev, "available_date"]
        .le(pd.Timestamp(p["index"]["development_target_available_by"]))
        .all()
    ):
        raise ValueError("Declared phase, maturity and training support required")
    prior = inherited(p) if prior is None else prior
    if len(prior) != 137:
        raise ValueError("All137 inherited comparisons required")
    rows = []
    for candidate, control, horizon in CONTRASTS:
        phases = []
        for code, name in enumerate(("development", "evaluation")):
            frame = panel.loc[panel.phase == name]
            a = frame.loc[frame.model == candidate].set_index("origin")
            b = frame.loc[frame.model == control].set_index("origin")
            if len(a) < p["inference"]["minimum_phase_observations"]:
                raise ValueError("INSUFFICIENT_DATA: phase below505 paired observations")
            phase = existing_inference.paired_inference(
                a.loss.to_numpy(),
                b.loss.to_numpy(),
                p,
                p["inference"]["seed"] + horizon * 1000000 + code * 10000,
            )
            phase.update(
                name=name,
                first_origin=str(a.index[0].date()),
                last_origin=str(a.index[-1].date()),
            )
            phase.update(
                diagnostics(
                    a.index,
                    a.loss.to_numpy() - b.loss.to_numpy(),
                    dates,
                    horizon,
                    p["index"]["evaluation_stability"] if name == "evaluation" else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "peak_age",
                "candidate": candidate,
                "control": control,
                "horizon": horizon,
                "score": "mse",
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    wave = inference.holm_adjust([x["p_conservative"] for x in rows])
    cumulative = inference.holm_adjust([x["p_conservative"] for x in prior + rows])[-3:]
    for one, pw, pc in zip(rows, wave, cumulative, strict=True):
        one.update(p_holm_wave=float(pw), p_holm_cumulative=float(pc))
        one["verdict"] = "COMPARISON_GATE_PASS" if passes(one) else "DOES_NOT_QUALIFY"
    return {
        "rows": rows,
        "inherited_rows": prior,
        "leads": candidate_leads(rows),
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 140,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "newly_generated_forecasts": len(panel),
        "common_scored_origins": n,
        "common_application_origins": application_n,
        "full_application_predictions": 4 * application_n,
        "monthly_schedules": monthly_schedules,
        "ridge_models_fitted": 2 * monthly_schedules,
        "scalar_models_fitted": monthly_schedules,
        "training_means_computed": monthly_schedules,
    }


def failure_metrics(error, signature):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 3,
        "cumulative_hypothesis_count": 140,
        "protocol_sha256": signature,
        "rows": [
            {
                "study": "peak_age",
                "candidate": candidate,
                "control": control,
                "horizon": horizon,
                "score": "mse",
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "phases": [],
                "status": status,
                "error": str(error),
            }
            for candidate, control, horizon in CONTRASTS
        ],
    }


def produce(daily, iv, p):
    features, targets, feature_states = feature_builder.build_features(daily, iv)
    panel, fits, application_states, support = pipeline.build_panel(
        features, targets, feature_states, p
    )
    return features, targets, feature_states, panel, fits, application_states, support


def validate_upstream(root=ROOT, registered_pins=None):
    return admission.admit_upstream(
        root, expected=CONTRACT["upstream"]["anchors"], registered_pins=registered_pins
    )


def report(metrics):
    lines = [
        "# Trailing peak age and SPX21-session returns",
        "",
        "Positive relative MSE gain means improvement; all three controls and both periods required.",
        "",
        "| Control | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics["rows"]:
        d, e = row["phases"]
        lines.append(
            f"| {row['control']} | {d['gain_relative']:.4%} | {e['gain_relative']:.4%} | {row['p_holm_wave']:.8f} | {row['p_holm_cumulative']:.8f} | {row['verdict']} |"
        )
    lines += [
        "",
        "Passing horizons: " + json.dumps(metrics["leads"]),
        "",
        "Sequential age correction may retune ridge shrinkage; this is not orthogonalized attribution. Reused archival price-index history, not untouched confirmation or a trading-profit test.",
        "",
    ]
    _write_current(REPORT / "results.md", "\n".join(lines).encode())


def _replace_ledger(rows):
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=REPORT, prefix=".trial-ledger-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(REPORT / "trial_ledger.jsonl")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ledger(rows):
    path = REPORT / "trial_ledger.jsonl"
    previous = (
        [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    )
    _replace_ledger(previous + rows)


def recover_failure_ledger(registrations, prior, failure):
    path = REPORT / "trial_ledger.jsonl"
    if path.exists():
        # Preserve any partial or successful writes before producing the canonical terminal record.
        _write_current(REPORT / "interrupted_trial_ledger.jsonl", path.read_bytes())
    rows = registrations + [{"event": "inherited", **row} for row in (prior or [])]
    rows += [{"event": "unevaluable", **row} for row in failure["rows"]]
    _replace_ledger(rows)


def validate_freeze(p, signature):
    freeze = json.loads((REPORT / "freeze_record.json").read_bytes())
    if freeze["protocol_sha256"] != signature or inference.digest(PROTOCOL) != signature:
        raise ValueError("Pre-run protocol freeze identity differs")
    actual = {
        str(f.relative_to(ROOT)) for d in ["src", "tests"] for f in (ROOT / d).rglob("*.py")
    }
    if actual != set(freeze["code"]):
        raise ValueError("Pre-run code inventory differs")
    for group in ["code", "prefit_design"]:
        for name, expected in freeze[group].items():
            if inference.digest(ROOT / name) != expected:
                raise ValueError("Pre-run frozen artifact changed: " + name)
    if (
        inference.digest(REPORT / "full_repository_tests.txt")
        != freeze["checks"]["full_log_sha256"]
    ):
        raise ValueError("Full pre-run test log changed")
    return freeze


def input_paths(p):
    paths = set(admission.collect_input_pins(ROOT, expected=p["upstream"]["anchors"]))
    paths.update(p["comparisons"]["inherited_sources"])
    paths.add("reports/peak_age/freeze_record.json")
    freeze = json.loads((REPORT / "freeze_record.json").read_bytes())
    paths.update(freeze["prefit_design"])
    return paths


def _replace_current(path, writer):
    """Write a new regular leaf, then replace the destination without following it."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".peak-age-output-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_current(path, payload):
    _replace_current(path, lambda stream: stream.write(payload))


def _dump_current(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    _write_current(path, payload.encode())


def require_bound_upstream(pins):
    successful = (
        "index_hinge",
        "range_alert",
        "issued_calibration",
        "event_cluster_replay",
        "peak_age",
    )

    def absent():
        for name in successful:
            marker = ROOT / f"reports/{name}/failure.json"
            if marker.exists() or marker.is_symlink():
                raise ValueError(
                    "Required successful ancestor or current failure marker: " + name
                )

    absent()
    names = tuple(
        "reports/event_cluster/" + n
        for n in ("failure.json", "metrics.json", "verification.json")
    ) + tuple(
        "reports/event_cluster_replay/" + n
        for n in ("metrics.json", "verification.json", "publication_audit.json")
    )
    for name in names:
        path = ROOT / name
        if (
            name not in pins
            or not path.is_file()
            or path.is_symlink()
            or inference.digest(path) != pins[name]
        ):
            raise ValueError("Required final upstream pin changed: " + name)
    absent()


def run():
    payload = PROTOCOL.read_bytes()
    signature = hashlib.sha256(payload).hexdigest()
    p = yaml.safe_load(payload)
    validate(p)
    validate_freeze(p, signature)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    canonical = (
        "manifest.json",
        "trial_ledger.jsonl",
        "failure.json",
        "metrics.json",
        "verification.json",
        "publication_audit.json",
    )
    if (
        REPORT.is_symlink()
        or OUT.is_symlink()
        or any((REPORT / name).exists() or (REPORT / name).is_symlink() for name in canonical)
        or any(OUT.iterdir())
    ):
        raise ValueError("Refusing to overwrite an existing peak-age attempt")
    modules = [
        "tests.test_peak_age_admission",
        "tests.test_peak_age_features",
        "tests.test_peak_age_models",
        "tests.test_peak_age_pipeline",
        "tests.test_peak_age_verification",
        "tests.test_peak_age_search",
        "tests.test_verify_peak_age",
        "tests.test_peak_age_publication",
        "tests.test_plot_peak_age",
    ]
    checked = subprocess.run(
        [sys.executable, "-m", "unittest", *modules, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_current(REPORT / "pre_run_checks.txt", (checked.stdout + checked.stderr).encode())
    if checked.returncode:
        raise RuntimeError(
            "Prewritten peak-age tests failed; no registration or historical admission"
        )
    validate_freeze(p, signature)
    registrations = [
        {
            "event": "registered",
            "study": "peak_age",
            "candidate": candidate,
            "control": control,
            "horizon": horizon,
            "score": "mse",
            "protocol_sha256": signature,
        }
        for candidate, control, horizon in CONTRASTS
    ]
    metrics = None
    prior = None
    try:
        ledger(registrations)
        code = [f for directory in ("src", "tests") for f in (ROOT / directory).rglob("*.py")]
        inputs = input_paths(p)
        preserved = [
            f
            for f in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
            if f.is_file()
            and f != PROTOCOL
            and REPORT not in f.parents
            and str(f.relative_to(ROOT)) not in inputs
        ]
        backend = io.StringIO()
        with redirect_stdout(backend):
            np.show_config()
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": signature,
            "environment": {
                "python": sys.version,
                "packages": {
                    name: version(name)
                    for name in ("numpy", "pandas", "scipy", "pyarrow", "PyYAML")
                },
                "numpy_backend": backend.getvalue(),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "LOKY_MAX_CPU_COUNT",
                    )
                },
            },
            "code": {str(f.relative_to(ROOT)): inference.digest(f) for f in sorted(code)},
            "inputs": {name: inference.digest(ROOT / name) for name in sorted(inputs)},
            "preserved": {
                str(f.relative_to(ROOT)): inference.digest(f) for f in sorted(preserved)
            },
        }
        _dump_current(REPORT / "manifest.json", manifest)
        prior = inherited(p, manifest["inputs"])
        if len(prior) != 137:
            raise ValueError("Complete inherited137 required")
        ledger([{"event": "inherited", **row} for row in prior])
        audit, daily, iv = validate_upstream(ROOT, manifest["inputs"])
        _dump_current(OUT / "upstream_admission.json", audit)
        features, targets, feature_states, panel, fits, application_states, support = produce(
            daily, iv, p
        )
        for name, frame in [
            ("features", features),
            ("targets", targets),
            ("feature_states", feature_states),
            ("forecasts", panel),
            ("application_states", application_states),
        ]:
            _replace_current(OUT / (name + ".parquet"), frame.to_parquet)
        _dump_current(OUT / "fits.json", fits)
        _dump_current(OUT / "support_audit.json", support)
        metrics = evaluate(
            panel, features.index, p, len(fits), len(application_states), prior=prior
        )
        keys = [(row["candidate"], row["control"], row["horizon"]) for row in metrics["rows"]]
        if (
            len(keys) != 3
            or set(keys) != set(CONTRASTS)
            or metrics["hypothesis_count"] != 3
            or metrics["cumulative_hypothesis_count"] != 140
        ):
            raise ValueError("Incomplete or changed scored three-contrast family")
        _dump_current(REPORT / "metrics.json", metrics)
        report(metrics)
        validate_freeze(p, signature)
        for group in ("code", "inputs", "preserved"):
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        if inference.digest(PROTOCOL) != signature:
            raise ValueError("Frozen protocol changed during run")
        require_bound_upstream(manifest["inputs"])
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, signature)
        failure["inherited_rows_available"] = prior is not None
        failure["inherited_rows_reconstructed"] = len(prior) if prior is not None else 0
        _dump_current(REPORT / "metrics.json", failure)
        _dump_current(REPORT / "failure.json", failure)
        _write_current(
            REPORT / "results.md",
            b"# Peak age and SPX returns\n\nUNEVALUABLE: all three comparisons retained with p=1; no lead.\n",
        )
        recover_failure_ledger(registrations, prior, failure)
        if metrics is not None:
            _dump_current(
                REPORT / "unpublished_scored_metrics.json",
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": metrics,
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION",
                "newly_generated_forecasts": len(panel),
                "monthly_schedules": len(fits),
                "common_application_origins": len(application_states),
            }
        )
    )


if __name__ == "__main__":
    run()
