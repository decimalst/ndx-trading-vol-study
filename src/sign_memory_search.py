"""Preregistered directional-agreement memory and immutable-source publication."""

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

from . import cross_moment_score as cs
from . import cross_moment_search as old_inference
from . import orthogonal_round2 as inference
from . import sign_memory_features as sf
from . import sign_memory_models as sm
from .international_search import diagnostics
from .verify_sign_memory import validate_upstream

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "sign_memory.yaml"
REPORT = ROOT / "reports/sign_memory"
OUT = ROOT / "data/sign_memory"
CONTRASTS = (("memory", "baseline", "brier"), ("memory", "frequency", "brier"))
WAVE_ALPHA = 0.05 / (13 * 14)
EFFECT = 0.0005


CONTRACT = {
    "sources": {
        "daily": "data/research_paths/spx_daily.parquet",
        "vix": "data/free_sources/raw/cboe/VIX_History.csv",
        "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
        "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
        "qqq": "data/raw/daily_ohlc.parquet",
        "vxn": "data/free_sources/raw/cboe/VXN_History.csv",
    },
    "index": {
        "asset": "QQQ_ETF_and_SPX_price_index",
        "source_end": "2025-10-20",
        "sealed_start": "2025-11-03",
        "origin_start": "2016-01-04",
        "origin_end": "2025-10-17",
        "latest_target": "2025-10-20",
        "development": ["2016-01-04", "2019-12-31"],
        "development_target_available_by": "2019-12-31",
        "evaluation": ["2020-01-02", "2025-10-17"],
        "evaluation_stability": [["2020-01-02", "2022-12-31"], ["2023-01-01", "2025-10-17"]],
        "horizons": [1],
        "market_lag": 1,
        "minimum_train": 1000,
        "models": ["frequency", "baseline", "memory"],
        "all_features": [
            "const",
            "qqq_lg_d",
            "qqq_lg_w",
            "qqq_lg_m",
            "qqq_lt_d",
            "qqq_lt_w",
            "qqq_lt_m",
            "qqq_neg_d",
            "qqq_neg_w",
            "qqq_neg_m",
            "spx_lg_d",
            "spx_lg_w",
            "spx_lg_m",
            "spx_lt_d",
            "spx_lt_w",
            "spx_lt_m",
            "spx_neg_d",
            "spx_neg_w",
            "spx_neg_m",
            "lvxn",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "corr22",
            "qqq_day_d",
            "qqq_day_w",
            "qqq_day_m",
            "spx_day_d",
            "spx_day_w",
            "spx_day_m",
            "qqq_pos22",
            "qqq_neg22",
            "spx_pos22",
            "spx_neg22",
            "independent22",
            "excess22",
        ],
        "minimum_train_per_class": 50,
    },
    "measurement": {
        "gk_floor": 1e-10,
        "formula": "0.5*log(high/low)^2-(2*log(2)-1)*log(close/open)^2",
        "admission_pool": "For each asset separately, every observed complete OHLC row after "
        "alignment to full bounded SPX reference calendar; before ANY "
        "correlation, feature, target or sample mask",
        "gate": "Every complete observed raw GK strictly greater than1e-10; every observed "
        "open/close pair must yield finite signed log(close/open), including high/low "
        "gaps. Audit precedes any feature/target/sample mask.",
        "failure": "Save measurement audit before requiring PASS; no feature/target "
        "construction or fits on floor failure; retain both registered hypotheses "
        "as UNEVALUABLEp1",
        "missing": "Absent or partially missing OHLC is counted separately and remains "
        "unknown; never converted to zero or floored",
        "invalid": "Invalid observed prices/ranges or nonfinite computed GK fail wholewave. "
        "Any observed open/close pair producing nonfinite signed log return fails "
        "even when high/low is missing; missing open/close stays unknown.",
        "zero_target": "Any exact zero raw intraday return is retained as strict nonagreement; "
        "any missing pair remains unknown",
        "diagnostics": "Full inherited GK/signed return gate before any feature/target mask; "
        "binary counts only after pretests and registration",
        "claims": "No standalone index direction, volatility magnitude, covariance, copula "
        "identification or profit claim",
    },
    "source_contract": {
        "instruments": "QQQ ETF vendor raw OHLC and Yahoo ^GSPC SPX price index; no "
        "audited exact Nasdaq100 OHLC source is claimed",
        "fields": "open,high,low,close only; no adjusted close, volume, inferred "
        "distributions or corporate-action substitution",
        "reference_calendar": "Retain every bounded observed SPX date; reindex QQQ without "
        "first intersecting calendars; no next-common-date labels",
        "previous_close": "Each asset raw overnight and close return requires its actual "
        "source predecessor date equal to preceding SPX reference date; "
        "otherwise unknown",
        "vintage": "Archival Yahoo and Cboe extracts; revisions, synchronized auctions and "
        "exact historical publication latency unverified",
        "vix9d": "Back-calculated January2011-October2013 training history; forecasts "
        "begin2016",
        "iv_match": "VXN underlying Nasdaq100 and SPX/VIX controls; no implied covariance "
        "or exact synchronous index/ETF auction data claimed",
        "missing": "Missing observed quotes or absent exact-date sources remain unknown "
        "through strict rolling windows; no filling or compressed rolling",
        "limitations": "Intraday ratios cancel common within-session units, but ETF "
        "tracking, index construction and vendor conventions remain; raw "
        "intersession controls retain distribution effects",
        "immutable_read": "Copy each six declared raw file and its documentary manifest "
        "from a single hash-checked byte snapshot into an isolated "
        "temporary directory at same relative path; frozen loader bounds "
        "dates before numerical parsing; source audits retain original "
        "registered path/hash; no source writes",
    },
    "correlation": {
        "window": 22,
        "returns": "raw log(close/open) per matched SPX date, common within-session units "
        "cancel",
        "calculation": "Centered two-pass Pearson covariance divided by centered norms over "
        "exactly22 complete paired returns; ends at previous SPX session",
        "missing": "Any missing pair in window or exact zero centered denominator makes "
        "correlation unknown for all models",
        "roundoff_tolerance": 1e-12,
        "bounds": "Retain finite rho including tiny overshoot with abs(rho)<=1+1e-12; reject "
        "larger violation; no clipping, epsilon denominator, alternate window or "
        "Fisher transform",
    },
    "study_id": "sign_memory_wave13",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_binary_outcomes_counts_or_fits",
    "wave": 13,
    "wave_alpha": 0.0002747252747252747,
    "objective": "Test one lagged excess directional-agreement memory increment for QQQ ETF and SPX price "
    "index",
    "evidence_class": "exploratory_reused_history_archival_QQQ_SPX_raw_sign_agreement",
    "upstream": {
        "protocol": "target_aligned.yaml",
        "reports": "reports/target_aligned",
        "data": "data/target_aligned",
        "protocol_sha256": "ca11ec1c90f8d83867b6dce4efaaac4f0e5ed11e23143e2ab1f28d78a9f136b3",
        "manifest_sha256": "996e9f8f081a69a66dad9c4c611feb6775fc69c5ad9e3630230d55ad97b426d9",
        "verification_sha256": "dfdc8224a8b7455622e6a16e2671bc70fe01a9a2600e78f39814ad824510fbd0",
        "verifier_sha256": "93e8ac7f88624fec0933600a85062397c9b0b2306afe87d0b4e3ee4af0f751d2",
        "required_status": "VERIFIED",
        "admission": "Reconstruct original VERIFIED record via frozen read-only functions; all "
        "prior pins and full outputs included in new manifest; historical inventory "
        "anchored by original manifest hash, not current-tree old coverage",
    },
    "target": {
        "event": "Both next observed SPX-session raw log(close/open) returns strictly positive, or "
        "both strictly negative",
        "zero": "Either or both exactly zero gives class0; complement includes opposite directions "
        "and ties",
        "missing": "Any missing return gives unknown, never class0; reject infinity or invalid "
        "observed source",
        "sign_arithmetic": "Direct comparisons, never multiplication of returns or epsilon sign "
        "threshold",
        "positive_rescaling": "Within-session positive price unit changes preserve each raw return; "
        "no total-return or executable auction equivalence",
        "fit_schedule": "Monthly first feature-complete origin before future query-label filtering; "
        "expanding training labels mature by prior SPX close; retain unscored "
        "applications and folds",
    },
    "features": {
        "window": 22,
        "bounded": ["qqq_pos22", "qqq_neg22", "spx_pos22", "spx_neg22", "independent22"],
        "memory": "excess22",
        "formula": "independent22=P_Q+*P_S+ + P_Q-*P_S-; "
        "excess22=joint_strict_agreement22-independent22",
        "history": "All five component fractions require identical complete paired 22-session "
        "window ending t-1 on full SPX calendar; no compressed dates or filling",
        "interpretation": "Empirical excess agreement memory beyond marginal signs; not an "
        "unbiased dependence estimator or copula parameter",
    },
    "fitting": {
        "penalty": 0.01,
        "maximum_iterations": 200,
        "maximum_backtracks": 60,
        "armijo": 0.0001,
        "gradient_tolerance": 1e-08,
        "scale_minimum": 1e-12,
        "baseline": "Unpenalized intercept and normalized logistic NLL plus .01 squared slopes; "
        "original33nonintercept+training-centered corr22 square population "
        "standardized, five bounded controls centered with fixedscale1",
        "memory": "Freeze baseline logit; add b*(excess22-training_mean) with .01*b², no intercept "
        "or joint augmented refit",
        "constant_bounded": "Predeclared exact all-equal test centers column exactly to zero, "
        "retains column with canonical penalized coefficient zero; fixedscale1",
        "constant_memory": "Predeclared exact all-equal excess22 yields centered memory zero and "
        "canonical b0 with retained model/fold",
        "frequency": "Mean binary training label on exact same complete mature sample",
        "arithmetic": "Finite real inputs, logits, derivatives, objectives, coefficients and "
        "probabilities; stable logaddexp/expit; numeric probability endpoints "
        "allowed; no posthoc probability clipping or optimizer fallback",
        "optimizer": "Deterministic damped Newton with positive Hessian and Armijo from fixed "
        "initialization; audit objective/fullgradient and independent stationary "
        "convex optimum before interpretation",
        "optimization_claim": "Each stage convex with unique slope optimum due positive ridge; "
        "staged pair is not the jointly optimized augmented model",
        "initialization": "Baseline intercept logit(training frequency), all slopes0; memoryb0; no "
        "restarts",
        "stable_logistic": "NLL mean(logaddexp(0,(1-2y)*eta)); score expit(eta) fory0 and "
        "-expit(-eta) fory1; curvature expit(eta)*expit(-eta)",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), after "
        "finite individual squared loss validation",
        "arithmetic": "Binary finite y and finite probabilities in[0,1]; loss in[0,1]; exactzero "
        "errors valid; nonzero squares/products underflowing tozero reject; "
        "coherence64epsilon/downwardULP no arbitrary absolute floor",
        "functional": "Conditional strict-agreement probability; expected Brier pi(1-pi)+(p-pi)^2; "
        "bounded loss needs no return fourth moments",
        "effect_reference": "Absolute squared-probability error decrease .0005 bothphases; fixed "
        "statistical reference, not profit or direct probability-point "
        "improvement",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 117,
        "cumulative_hypotheses": 119,
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
        ],
        "controls": ["baseline", "frequency"],
        "contrasts": [["memory", "baseline", "brier"], ["memory", "frequency", "brier"]],
        "candidate_gate": "Both registered contrasts must pass all fixed "
        "phase/effect/stability/multiplicity gates",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "bootstrap_draws": 99999,
        "seed": 20260919,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap p and Bartlett "
        "HAC126 p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(13*14), separately cumulativeHolm119 at.05",
        "gate": "Bothphase absolute Brier decrease>=.0005 against BOTHcontrols; negative "
        "differences in bothfixed evaluation slices; waveHolm<.05/182,cumulativeHolm<.05",
        "support": "Atleast50events/50nonevents perfit;30/class perphase;15/class perfixed "
        "evaluationslice; any insufficient support aborts wholefamily, no "
        "droppedfolds",
        "power": "Nominal HAC80percent MDE diagnostic only, divided by fixed .0005 effect; no "
        "postscore power gate or equivalence claim",
        "resolution": "Minimum p1/100000 below one tenth strictest two-comparison raw wave "
        "cutoff1/7280",
        "failure_rule": "All2new hypotheses UNEVALUABLEp1 after any "
        "admission/measurement/support/fit/scoring/verification/publication "
        "failure; preserve all diagnostics and old artifacts; no "
        "outcome-dependent repair",
        "calibration": "Descriptive calibration in the large only: perphase/model n, observed "
        "binary frequency, mean issued probability, probability-minus-frequency "
        "gap and Brier. No bins, calibration fitting, hypothesis, selection or "
        "promotion gate; not a full conditional calibration claim.",
    },
    "verification": {
        "coefficient_relative_tolerance": 1e-07,
        "coefficient_absolute_tolerance": 1e-06,
        "independent_probability_relative_tolerance": 1e-07,
        "independent_probability_absolute_tolerance": 1e-06,
        "saved_probability_relative_tolerance": 1e-10,
        "saved_probability_absolute_tolerance": 1e-12,
        "objective_relative_tolerance": 1e-10,
        "objective_absolute_tolerance": 1e-12,
        "saved_gradient_tolerance": 1e-08,
        "gradient_roundoff_allowance": 1e-12,
        "independent_gradient_tolerance": 1e-10,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Independent raw source, feature, "
        "sign/tie/rolling/maturity/support/transformation/convex objective, "
        "fit, probability, score, fourphase inference and complete119family "
        "ledger; no new producer import",
        "independent_optimizer": {
            "baseline_method": "trust-exact",
            "baseline_initialization": "all_zero",
            "baseline_gradient_target": 1e-10,
            "baseline_maximum_iterations": 500,
            "memory_method": "brentq",
            "memory_bracket": "symmetric plus/minus (mean(abs(centered_memory))+1)/0.02",
            "memory_absolute_tolerance": 1e-12,
            "memory_relative_tolerance": 1e-14,
            "memory_maximum_iterations": 200,
            "accepted_gradient_tolerance": 1.0001e-08,
            "acceptance": "Require finite objectives, parameters and "
            "probabilities plus independently recomputed "
            "full gradient within the accepted tolerance. "
            "Record optimizer status; a tighter target "
            "being unmet does not override the declared "
            "accepted gradient criterion. Compare "
            "independent coefficients/probabilities at "
            "their separate fixed tolerances; reconstruct "
            "primary scores from stricter "
            "saved-coefficient replay.",
            "arithmetic": "Independently evaluate stable signed-label "
            "logaddexp NLL, signed logistic score and "
            "expit(eta)*expit(-eta) curvature. Finite "
            "probability endpoints are valid; no "
            "probability clipping or optimization "
            "retries.",
        },
    },
    "outputs": {
        "data": "data/sign_memory",
        "reports": "reports/sign_memory",
        "features": "data/sign_memory/features.parquet",
        "targets": "data/sign_memory/targets.parquet",
        "forecasts": "data/sign_memory/forecasts.parquet",
        "fits": "data/sign_memory/fits.json",
        "admission": "data/sign_memory/upstream_admission.json",
    },
}


def validate(p):
    if (
        p["wave"] != 13
        or p["wave_alpha"] != WAVE_ALPHA
        or any(p.get(g) != v for g, v in CONTRACT.items())
    ):
        raise ValueError("Fixed sign-memory protocol differs")


def paired_difference(candidate, control, actual):
    a, b, y = (cs._finite(v, "Brier input") for v in [candidate, control, actual])
    if any(v.ndim != 1 for v in [a, b, y]) or not a.shape == b.shape == y.shape:
        raise ValueError("Aligned one-dimensional Brier inputs required")
    if not np.isin(y, [0.0, 1.0]).all() or any(((v < 0) | (v > 1)).any() for v in [a, b]):
        raise ValueError("Binary labels and probabilities in[0,1] required")
    return cs.paired_difference(a, b, y)


def paired_inference(candidate, control, difference, p, seed):
    result = old_inference.paired_inference(candidate, control, difference, p, seed)
    result["nominal_mde_effect_ratio"] = result["hac126"]["mde80_nominal"] / EFFECT
    return result


def inherited(p):
    rows = []
    for source in p["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((ROOT / source).read_text())["rows"]):
            rows.append(
                {
                    "study": row.get("study", Path(source).parent.name or Path(source).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source,
                    "source_sha256": inference.digest(ROOT / source),
                    "source_row_index": number,
                    **{name: row[name] for name in ["measure", "score"] if name in row},
                }
            )
    if len(rows) != 117:
        raise ValueError("All117 inherited comparisons required")
    return rows


def passes(row):
    phases = row["phases"]
    if len(phases) != 2 or {x["name"] for x in phases} != {"development", "evaluation"}:
        return False
    evaluation = next(x for x in phases if x["name"] == "evaluation")
    return (
        row["p_holm_wave"] < WAVE_ALPHA
        and row["p_holm_cumulative"] < 0.05
        and all(x["n"] > 0 and x["delta"] <= -0.0005 for x in phases)
        and len(evaluation["stability"]) == 2
        and all(x["delta"] < 0 for x in evaluation["stability"])
    )


def candidate_leads(rows):
    keys = [(x["candidate"], x["control"], x["score"]) for x in rows]
    if len(keys) != 2 or set(keys) != set(CONTRASTS):
        raise ValueError("Complete two-comparison family required")
    return ["memory"] if all(passes(row) for row in rows) else []


def require_support(y, minimum, label):
    values = np.asarray(y)
    if values.ndim != 1 or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Binary support labels required")
    events = int((values == 1).sum())
    nonevents = int((values == 0).sum())
    if min(events, nonevents) < minimum:
        raise ValueError("INSUFFICIENT_DATA: binary class support " + label)
    return {"n": len(values), "events": events, "nonevents": nonevents}


def validate_support(panel, p):
    base = panel.loc[panel.model.eq("frequency")].set_index("origin").sort_index()
    support = {}
    for name in ["development", "evaluation"]:
        first, last = p["index"][name]
        y = base.loc[first:last, "y"].to_numpy()
        support[name] = require_support(y, p["inference"]["minimum_phase_per_class"], name)
        if len(y) < p["inference"]["minimum_phase_observations"]:
            raise ValueError("INSUFFICIENT_DATA: literal inference bandwidth " + name)
    support["evaluation_slices"] = []
    for first, last in p["index"]["evaluation_stability"]:
        one = require_support(
            base.loc[first:last, "y"].to_numpy(),
            p["inference"]["minimum_slice_per_class"],
            first + ".." + last,
        )
        support["evaluation_slices"].append({"start": first, "end": last, **one})
    return support


def evaluate(panel, calendar, p, monthly_fits):
    sm.validate_panel(panel)
    dates = pd.DatetimeIndex(calendar)
    if dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("Complete ordered reference calendar required")
    expected = pd.Series(dates, index=dates)
    if not panel.origin.isin(dates).all():
        raise ValueError("Every forecast origin must belong to the reference calendar")
    for column, shift in [("feature_cutoff_date", 1), ("target_end", -1)]:
        if not np.array_equal(
            panel[column].to_numpy(), expected.shift(shift).loc[panel.origin].to_numpy()
        ):
            raise ValueError("Exact source-calendar predecessor and target session required")
    development = panel.origin.between(*p["index"]["development"])
    evaluation = panel.origin.between(*p["index"]["evaluation"])
    if (
        not (development | evaluation).all()
        or not np.array_equal(panel.phase, np.where(development, "development", "evaluation"))
        or not panel.train_n.ge(p["index"]["minimum_train"]).all()
        or not panel.loc[development, "available_date"]
        .le(pd.Timestamp(p["index"]["development_target_available_by"]))
        .all()
    ):
        raise ValueError(
            "Declared training support and development/evaluation fences required"
        )
    support = validate_support(panel, p)
    calibration = {}
    for phase in ["development", "evaluation"]:
        calibration[phase] = {}
        for model in sf.MODELS:
            one = panel.loc[panel.phase.eq(phase) & panel.model.eq(model)]
            observed, probability = float(one.y.mean()), float(one.probability.mean())
            calibration[phase][model] = {
                "n": len(one),
                "observed_frequency": observed,
                "mean_probability": probability,
                "calibration_gap": probability - observed,
                "brier": float(one.loss.mean()),
            }
    rows = []
    for candidate, control, score in CONTRASTS:
        phases = []
        for code, name in enumerate(["development", "evaluation"]):
            first, last = p["index"][name]
            frame = panel.loc[panel.origin.between(first, last)]
            models = {
                m: frame.loc[frame.model.eq(m)].set_index("origin").sort_index()
                for m in sf.MODELS
            }
            a, b = models[candidate], models[control]
            d = paired_difference(
                a.probability.to_numpy(), b.probability.to_numpy(), a.y.to_numpy()
            )
            phase = paired_inference(
                a.loss.to_numpy(),
                b.loss.to_numpy(),
                d,
                p,
                p["inference"]["seed"] + code * 10000,
            )
            phase.update(
                name=name,
                first_origin=str(a.index[0].date()),
                last_origin=str(a.index[-1].date()),
                class_support=support[name],
            )
            phase.update(
                diagnostics(
                    a.index,
                    d,
                    calendar,
                    1,
                    p["index"]["evaluation_stability"] if name == "evaluation" else [],
                )
            )
            phases.append(phase)
        rows.append(
            {
                "study": "sign_memory",
                "candidate": candidate,
                "control": control,
                "score": score,
                "horizon": 1,
                "phases": phases,
                "p_conservative": max(x["p_conservative"] for x in phases),
            }
        )
    prior = inherited(p)
    wave = inference.holm_adjust([r["p_conservative"] for r in rows])
    family = inference.holm_adjust([r["p_conservative"] for r in prior + rows])[-len(rows) :]
    for row, pw, pc in zip(rows, wave, family, strict=True):
        row.update(p_holm_wave=float(pw), p_holm_cumulative=float(pc))
        row["verdict"] = "PASSES_ALL_GATES" if passes(row) else "DOES_NOT_QUALIFY"
    return {
        "rows": rows,
        "inherited_rows": prior,
        "leads": candidate_leads(rows),
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 119,
        "protocol_sha256": inference.digest(PROTOCOL),
        "evidence_class": p["evidence_class"],
        "new_monthly_fits": monthly_fits,
        "new_forecasts": len(panel),
        "common_scored_origins": panel.origin.nunique(),
        "class_support": support,
        "calibration": calibration,
    }


def failure_metrics(error, protocol_hash):
    status = "INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in str(error) else "INVALID_RUN"
    return {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 119,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "sign_memory",
                "candidate": a,
                "control": b,
                "score": s,
                "horizon": 1,
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "phases": [],
                "status": status,
                "error": str(error),
            }
            for a, b, s in CONTRASTS
        ],
    }


def report(metrics):
    lines = [
        "# QQQ-SPX directional agreement memory",
        "",
        "Negative Brier difference means improvement; both controls required.",
        "",
        "| Control | Development | Evaluation | Wave Holm p | Cumulative Holm p | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics["rows"]:
        d, e = row["phases"]
        lines.append(
            f"| {row['control']} | {d['delta']:+.8f} | {e['delta']:+.8f} | {row['p_holm_wave']:.8f} | {row['p_holm_cumulative']:.8f} | {row['verdict']} |"
        )
    lines += [
        "",
        "Passing candidates: " + json.dumps(metrics["leads"]),
        "",
        "Reused archival history and raw strict sign-agreement probability; no volatility magnitude, true correlation, standalone direction or trading-profit claim.",
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(lines))


def ledger(rows):
    with (REPORT / "trial_ledger.jsonl").open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def input_paths(p):
    manifest = json.loads((ROOT / p["upstream"]["reports"] / "manifest.json").read_text())
    paths = set(manifest["inputs"]) | {p["upstream"]["protocol"]}
    for directory in [p["upstream"]["reports"], p["upstream"]["data"]]:
        paths.update(
            str(f.relative_to(ROOT)) for f in (ROOT / directory).rglob("*") if f.is_file()
        )
    return paths


def load_pinned_sources(p, manifest):
    names = set(p["sources"].values())
    for name in list(names):
        names.update(x for x in manifest["inputs"] if x == name + ".manifest.json")
    with tempfile.TemporaryDirectory(prefix="sign-memory-sources-") as directory:
        staged = Path(directory)
        for name in sorted(names):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Relative registered source paths required")
            payload = (ROOT / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != manifest["inputs"][name]:
                raise ValueError("Pinned raw source changed before decoding: " + name)
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        qqq, spx, iv, audit = sf.load_sources(p, staged)
        for key, source in audit["sources"].items():
            source["source_path"] = str(ROOT / p["sources"][key])
        return qqq, spx, iv, audit


def run():
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    p = yaml.safe_load(protocol_bytes)
    validate(p)
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if (REPORT / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a registered sign-memory experiment")
    modules = [
        "tests.test_sign_memory_features",
        "tests.test_sign_memory_models",
        "tests.test_sign_memory_integration",
        "tests.test_sign_memory_search",
        "tests.test_sign_memory_publication",
        "tests.test_verify_sign_memory",
        "tests.test_plot_sign_memory",
        "tests.test_round2_inference",
    ]
    checked = subprocess.run(
        [sys.executable, "-m", "unittest", *modules, "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (REPORT / "pre_run_checks.txt").write_text(checked.stdout + checked.stderr)
    if checked.returncode:
        raise RuntimeError("Prewritten sign-memory tests failed; no registration or fit")
    if inference.digest(PROTOCOL) != protocol_hash:
        raise ValueError("Frozen protocol changed during pretests; no registration")
    code = list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    inputs = input_paths(p)
    preserved = [
        file
        for file in list(ROOT.glob("*.yaml")) + list((ROOT / "reports").rglob("*"))
        if file.is_file()
        and file != PROTOCOL
        and REPORT not in file.parents
        and str(file.relative_to(ROOT)) not in inputs
    ]
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_hash,
        "environment": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ["numpy", "pandas", "scipy", "pyarrow", "PyYAML"]
            },
            "numpy_backend": backend.getvalue(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]
            },
        },
        "code": {str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(code)},
        "inputs": {name: inference.digest(ROOT / name) for name in sorted(inputs)},
        "preserved": {
            str(file.relative_to(ROOT)): inference.digest(file) for file in sorted(preserved)
        },
    }
    inference.dump(REPORT / "manifest.json", manifest)
    metrics = None
    try:
        ledger([{"event": "inherited", **row} for row in inherited(p)])
        ledger(
            [
                {
                    "event": "registered",
                    "study": "sign_memory",
                    "candidate": a,
                    "control": b,
                    "score": s,
                    "horizon": 1,
                    "protocol_sha256": manifest["protocol_sha256"],
                }
                for a, b, s in CONTRASTS
            ]
        )
        proof = validate_upstream(ROOT)
        inference.dump(OUT / "upstream_admission.json", proof)
        qqq, spx, iv, audit = load_pinned_sources(p, manifest)
        inference.dump(REPORT / "source_audit.json", audit)
        measurement = sf.measurement_audit(qqq, spx)
        inference.dump(REPORT / "measurement_audit.json", measurement)
        sf.require_measurement(measurement)
        features, targets = sf.build_features(qqq, spx, iv)
        features.to_parquet(OUT / "features.parquet")
        targets.to_parquet(OUT / "targets.parquet")
        forecasts, fits = sm.forecast_panel(features, targets, p["index"])
        forecasts.to_parquet(OUT / "forecasts.parquet")
        inference.dump(OUT / "fits.json", fits)
        metrics = evaluate(forecasts, features.index, p, len(fits))
        if inference.digest(PROTOCOL) != manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during sign-memory run")
        for group in ["code", "inputs", "preserved"]:
            for name, expected in manifest[group].items():
                if inference.digest(ROOT / name) != expected:
                    raise ValueError("Frozen artifact changed: " + name)
        inference.dump(REPORT / "metrics.json", metrics)
        report(metrics)
        ledger([{"event": "evaluated", **row} for row in metrics["rows"]])
    except Exception as error:
        failure = failure_metrics(error, manifest["protocol_sha256"])
        inference.dump(REPORT / "metrics.json", failure)
        inference.dump(REPORT / "failure.json", failure)
        (REPORT / "results.md").write_text(
            "# Directional agreement memory experiment\n\nUNEVALUABLE: all2comparisons retained with p=1; no lead.\n"
        )
        ledger([{"event": "unevaluable", **row} for row in failure["rows"]])
        if metrics is not None:
            diagnostic = {
                "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                "not_for_inherited_inference_or_promotion": True,
            }
            try:
                json.dumps(metrics, allow_nan=False)
                diagnostic["scored_metrics"] = metrics
            except (TypeError, ValueError):
                diagnostic["nonserializable_metrics_repr"] = repr(metrics)
            inference.dump(REPORT / "unpublished_scored_metrics.json", diagnostic)
        raise
    print(
        json.dumps(
            {
                "status": "SCORED_AWAITING_INDEPENDENT_VERIFICATION",
                "forecasts": len(forecasts),
                "new_monthly_fits": len(fits),
                "new_forecasts": metrics["new_forecasts"],
                "leads": metrics["leads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
