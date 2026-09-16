"""Independent range-alert construction, convex solves, full-clock states and publication.

Only frozen independent helpers are reused. No wave18 producer is imported.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import root
from scipy.special import expit

from . import verify_causal_pool as clock
from . import verify_sign_memory as logistic
from . import verify_tail_shape as source
from .verify_cross_moment import (
    _finite,
    compare_tree,
    digest,
    explicit_bootstrap_means,
    holm,
    independent_hac,
    paired_difference,
    same_tree,
)

ROOT = Path(__file__).resolve().parents[1]
ALL_FEATURES = source.RAW + (
    "intraday",
    "intraday_sq",
    "overnight",
    "overnight_sq",
    "range_extremity",
)
BASE = source.BASE + ("intraday", "intraday_sq", "overnight", "overnight_sq")
MODELS = ("baseline", "recent_frequency", "location")
COMPARISONS = (("location", "baseline"), ("location", "recent_frequency"))
PANEL_COLUMNS = logistic.PANEL_COLUMNS
STATE_COLUMNS = clock.STATE_COLUMNS
WAVE_ALPHA = 0.05 / (18 * 19)
EFFECT = 0.0005
GRADIENT_TOLERANCE = 1e-8 + 1e-12
real = logistic.real
binary = logistic.binary
close = logistic.close
support = logistic.support
logistic_objective = logistic.logistic_objective
location_objective = logistic.memory_objective
independent_location_fit = logistic.independent_memory_fit
explicit_states = clock.explicit_states
normalized_difference = logistic.normalized_difference
brier_loss = logistic.brier_loss
read_snapshot = clock.read_snapshot
read_json_snapshot = clock.read_json_snapshot


def known_binary(left, right, operation, label):
    a, b = np.broadcast_arrays(real(left, missing=True), real(right, missing=True))
    valid = np.isfinite(a) & np.isfinite(b)
    out = np.full(a.shape, np.nan)
    with np.errstate(all="ignore"):
        values = operation(a[valid], b[valid])
    if not np.isfinite(values).all():
        raise ValueError(label + ": nonfinite computed value")
    out[valid] = values
    return out


def ratio(a, b, label):
    a, b = np.broadcast_arrays(real(a, missing=True), real(b, missing=True))
    out = known_binary(a, b, np.divide, label)
    if ((a != 0) & np.isfinite(a) & np.isfinite(b) & (out == 0)).any():
        raise ValueError(label + ": nonzero quotient underflow")
    return out


def logarithm(a, label):
    a = real(a, missing=True)
    valid = np.isfinite(a)
    if (a[valid] <= 0).any():
        raise ValueError(label + ": positive log operand required")
    out = np.full(a.shape, np.nan)
    with np.errstate(all="ignore"):
        out[valid] = np.log(a[valid])
    if not np.isfinite(out[valid]).all():
        raise ValueError(label + ": nonfinite log")
    return out


def product(a, b, label):
    a, b = np.broadcast_arrays(real(a, missing=True), real(b, missing=True))
    out = known_binary(a, b, np.multiply, label)
    if ((a != 0) & (b != 0) & np.isfinite(a) & np.isfinite(b) & (out == 0)).any():
        raise ValueError(label + ": nonzero multiplication underflow")
    return out


def measurements(daily):
    source.validate_dates(daily.index)
    prices = daily.loc[:, ["open", "high", "low", "close"]]
    values = real(prices, missing=True)
    if ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Observed OHLC must be positive")
    o, h, low, c = (prices[name].to_numpy(float) for name in ("open", "high", "low", "close"))
    for a, b in ((h, low), (h, o), (h, c), (o, low), (c, low)):
        if (a < b).any():
            raise ValueError("Observed OHLC range is unordered")
    day = logarithm(ratio(c, o, "C/O"), "log(C/O)")
    day2 = product(day, day, "day square")
    highlow = logarithm(ratio(h, low, "H/L"), "log(H/L)")
    hl2 = product(highlow, highlow, "range square")
    raw = known_binary(
        product(0.5, hl2, "half range square"),
        product(2 * np.log(2) - 1, day2, "GK day adjustment"),
        np.subtract,
        "raw GK",
    )
    gk = np.maximum(raw, 1e-10)
    prev = np.r_[np.nan, c[:-1]]
    overnight = logarithm(ratio(o, prev, "O/previous C"), "overnight log")
    overnight2 = product(overnight, overnight, "overnight square")
    risk = known_binary(gk, overnight2, np.add, "positive daily risk")
    # Evaluate every available primitive even if another OHLC operand is absent.
    cl = logarithm(ratio(c, low, "C/L"), "log(C/L)")
    hc = logarithm(ratio(h, c, "H/C"), "log(H/C)")
    numerator = known_binary(cl, hc, np.subtract, "location numerator")
    positive = (h > low) & np.isfinite(highlow)
    if (highlow[positive] <= 0).any():
        raise ValueError("Positive observed range requires positive logarithmic width")
    eligible = positive & np.isfinite(numerator)
    z = np.full(len(daily), np.nan)
    z[eligible] = ratio(numerator[eligible], highlow[eligible], "location Z")
    if (np.abs(z[eligible]) > 1).any():
        raise ValueError("Z outside exact unit interval")
    e = product(z, z, "location square")
    if ((e[eligible] < 0) | (e[eligible] > 1)).any():
        raise ValueError("E outside exact unit interval")
    e[~np.isfinite(values).all(axis=1)] = np.nan
    return pd.DataFrame(
        {
            "intraday": day,
            "intraday_sq": day2,
            "overnight": overnight,
            "overnight_sq": overnight2,
            "range_extremity": e,
            "risk": risk,
        },
        index=daily.index,
    )


def reference_mean(risk):
    values = real(risk, missing=True)
    if ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Positive daily risk required")
    result = np.full(len(values), np.nan)
    for position in range(22, len(values)):
        window = values[position - 22 : position]
        if np.isfinite(window).all():
            try:
                total = math.fsum(map(float, window))
            except OverflowError as error:
                raise ValueError("Compensated reference sum overflow") from error
            mean = _finite(total / 22)
            if mean <= 0:
                raise ValueError("Strict risk reference underflow")
            result[position] = mean
    return pd.Series(result, index=risk.index)


def event_targets(risk):
    source.validate_dates(risk.index)
    average = reference_mean(risk)
    threshold = product(2.0, average, "twice reference risk")
    next_risk = risk.shift(-1).to_numpy()
    known = np.isfinite(threshold) & np.isfinite(next_risk)
    target = np.full(len(risk), np.nan)
    target[known] = (next_risk[known] > threshold[known]).astype(float)
    dates = pd.Series(risk.index, index=risk.index).shift(-1)
    return pd.DataFrame(
        {"y": target, "target_end": dates, "available_date": dates}, index=risk.index
    )


def feature_target_tables(daily, iv):
    m = measurements(daily)
    source.validate_dates(iv.index)
    quotes = real(iv.loc[:, ["vix", "vix9d", "vvix", "skew"]], missing=True)
    aligned = iv.reindex(daily.index)
    if ((quotes <= 0) & np.isfinite(quotes)).any():
        raise ValueError("Positive observed IV required")
    # Reproduce the frozen baseline formulas, validating each computed primitive.
    result = pd.DataFrame(index=daily.index)
    result["const"] = 1.0
    v = ratio(aligned.vix, 100.0, "VIX units")
    iv2 = product(v, v, "VIX squared")
    result["I"] = pd.Series(logarithm(iv2, "log IV"), index=daily.index).shift()
    risk = m.risk
    for column, width in (("R", 22), ("lr_d", 1), ("lr_w", 5)):
        avg = risk.rolling(width, min_periods=width).mean()
        known = (
            pd.Series(np.isfinite(risk), index=risk.index, name="known")
            .rolling(width, min_periods=width)
            .sum()
            == width
        )
        if not np.isfinite(avg[known]).all() or (avg[known] <= 0).any():
            raise ValueError("Computed complete legacy risk window invalid")
        result[column] = pd.Series(
            logarithm(product(252.0, avg, "annualized risk"), "log risk"), index=daily.index
        ).shift()
    logc = logarithm(daily.close, "log close")
    ret = pd.Series(logc, index=daily.index).diff()
    real(ret, missing=True)
    for label, width in (("d", 1), ("w", 5), ("m", 22), ("q", 63)):
        result["ret_" + label] = ret.rolling(width, min_periods=width).mean().shift()
    negative = -pd.Series(
        logarithm(ratio(daily.close, daily.close.shift(), "close ratio"), "close ratio log"),
        index=daily.index,
    )
    negative = negative.clip(lower=0)
    for label, width in (("d", 1), ("w", 5), ("m", 22)):
        result["neg_" + label] = negative.rolling(width, min_periods=width).mean().shift()
    result["term"] = pd.Series(
        logarithm(ratio(aligned.vix9d, aligned.vix, "IV term"), "log IV term"),
        index=daily.index,
    ).shift()
    result["lvvix"] = pd.Series(logarithm(aligned.vvix, "log VVIX"), index=daily.index).shift()
    result["skew"] = aligned["skew"].shift()
    for dow in range(1, 5):
        result[f"entry_dow_{dow}"] = (daily.index.dayofweek == dow).astype(float)
    for name in ALL_FEATURES[-5:]:
        result[name] = m[name].shift()
    result["feature_cutoff_date"] = pd.Series(daily.index, index=daily.index).shift()
    real(result.loc[:, ALL_FEATURES], missing=True)
    return result.loc[:, ALL_FEATURES + ("feature_cutoff_date",)], event_targets(risk)


def transform(train, application):
    a = train.loc[:, ALL_FEATURES].copy()
    b = application.loc[:, ALL_FEATURES].copy()
    real(a)
    real(b)
    if len(a) < 2 or not (a.const == 1).all() or not (b.const == 1).all():
        raise ValueError("Literal intercept and complete training design required")
    centers = {name: float(a[name].mean()) for name in ("I", "R", "skew")}
    for name, center in centers.items():
        for frame in (a, b):
            frame[name + "_square"] = product(
                frame[name] - center, frame[name] - center, "training curvature"
            )
    a, b = a.loc[:, BASE], b.loc[:, BASE]
    means = np.r_[0.0, a.loc[:, BASE[1:]].mean().to_numpy()]
    scales = np.r_[1.0, a.loc[:, BASE[1:]].std(ddof=0).to_numpy()]
    real(means)
    real(scales)
    if (scales <= 1e-12).any():
        raise ValueError("All25 baseline scales must exceed1e-12")
    x = (a.to_numpy() - means) / scales
    q = (b.to_numpy() - means) / scales
    if ((train.range_extremity < 0) | (train.range_extremity > 1)).any() or (
        (application.range_extremity < 0) | (application.range_extremity > 1)
    ).any():
        raise ValueError("Strict range extremity domain before geometry")
    e = train.range_extremity.to_numpy()
    constant = bool((e == e[0]).all())
    center = float(e[0] if constant else e.mean())
    audit = {
        "train_n": len(train),
        "columns": list(BASE),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "curvature_means": centers,
        "range_extremity_mean": center,
        "range_extremity_scale": 1.0,
        "range_extremity_constant": constant,
    }
    return (
        real(x),
        real(q),
        real(e - center),
        real(application.range_extremity.to_numpy() - center),
        audit,
    )


def independent_baseline_fit(x, y):
    x, y = real(x), binary(y)
    if x.ndim != 2 or x.shape[1] != 26 or not (x[:, 0] == 1).all() or not 0 < y.mean() < 1:
        raise ValueError("26-column supported normalized logistic design required")
    result = root(
        lambda b: logistic_objective(b, x, y)[1],
        np.zeros(26),
        jac=lambda b: logistic_objective(b, x, y)[2],
        method="hybr",
        options={"xtol": 1e-10, "maxfev": 2000, "factor": 1.0},
    )
    value, g, _ = logistic_objective(result.x, x, y)
    maximum = float(np.max(abs(g)))
    if not result.success or maximum > GRADIENT_TOLERANCE:
        raise ValueError(
            "Independent root baseline failed success/full-gradient gate: "
            f"status={result.status}; gradient={maximum}; {result.message}"
        )
    return {
        "beta": real(result.x),
        "objective": value,
        "gradient_max_abs": maximum,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "jacobian_evaluations": int(result.njev),
        "start": [0.0] * 26,
        "method": "scipy_root_hybr",
    }


def invalidate_publication(root, error):
    report = Path(root) / "reports/range_alert"
    report.mkdir(parents=True, exist_ok=True)
    path = report / "metrics.json"
    original_bytes = path.read_bytes() if path.exists() else None
    previous, invalid_bytes = None, None
    if original_bytes is not None:
        try:
            decoded = json.loads(original_bytes)
            if isinstance(decoded, dict):
                # JSON's permissive parser accepts NaN/Infinity. These cannot
                # enter either the canonical failure record or its JSON backup.
                json.dumps(decoded, allow_nan=False)
                previous = decoded
            else:
                invalid_bytes = original_bytes
        except (UnicodeDecodeError, ValueError):
            invalid_bytes = original_bytes
    protocol_hash = previous.get("protocol_sha256") if previous else None
    message = "Independent verification failed: " + type(error).__name__ + ": " + str(error)
    failure = {
        "status": "UNEVALUABLE",
        "whole_wave_aborted": True,
        "leads": [],
        "hypothesis_count": 2,
        "cumulative_hypothesis_count": 129,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "range_alert",
                "candidate": candidate,
                "control": control,
                "score": "brier",
                "horizon": 1,
                "phases": [],
                "p_conservative": 1.0,
                "p_holm_wave": 1.0,
                "p_holm_cumulative": 1.0,
                "status": "INVALID_RUN",
                "verdict": "UNEVALUABLE",
                "error": message,
            }
            for candidate, control in COMPARISONS
        ],
    }
    serialized = json.dumps(failure, indent=2, allow_nan=False) + "\n"
    path.write_text(serialized)
    (report / "failure.json").write_text(serialized)
    (report / "results.md").write_text(
        "# SPX daily risk-event range location\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
        )
        + "\n"
    )
    with (report / "trial_ledger.jsonl").open("a") as stream:
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    backup = report / "unpublished_scored_metrics.json"
    if (
        previous is not None
        and previous.get("status") != "UNEVALUABLE"
        and not backup.exists()
    ):
        backup.write_text(
            json.dumps(
                {
                    "status": "UNPUBLISHED_DIAGNOSTIC_ONLY",
                    "not_for_inherited_inference_or_promotion": True,
                    "scored_metrics": previous,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    invalid_backup = report / "unpublished_invalid_metrics.txt"
    if invalid_bytes is not None and not invalid_backup.exists():
        invalid_backup.write_bytes(invalid_bytes)
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


CONTRACT = {
    "study_id": "range_alert_wave18",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_features_counts_or_fits",
    "wave": 18,
    "wave_alpha": 0.00014619883040935673,
    "objective": "Test whether prior closing location within the high-low range adds conditional "
    "information about next-session SPX risk exceeding twice its prior22-session "
    "mean",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_event",
    "sources": {
        "daily": "data/research_paths/spx_daily.parquet",
        "vix": "data/free_sources/raw/cboe/VIX_History.csv",
        "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
        "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
        "skew": "data/raw/SKEW_History.csv",
        "skew_source": "data/raw/skew_daily_source.json",
        "skew_derived": "data/raw/skew_daily.parquet",
    },
    "source_contract": {
        "skew_sha256": "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492",
        "skew_anchor": {"date": "2018-08-13", "close": 159.03, "tolerance": 0.01},
        "skew_derived": "bounded raw and existing derived values must agree; raw "
        "CSV is the predictor source; no fetch or vintage "
        "substitution",
        "historical_availability": "archival current-vintage Yahoo and Cboe "
        "extracts; revisions and exact release "
        "latency unverified",
        "vix9d": "January2011-October2013 values back-calculated; no "
        "original-date historical availability claim",
        "missing": "preserve observed SPX reference calendar and strict windows; "
        "no filling; numerical CSV values parsed only after date "
        "cutoff",
        "interpretation": "SKEW is an option-implied30day construction; it is "
        "not a physical one-day crash probability",
    },
    "source_files": {
        "data/free_sources/raw/cboe/VIX9D_History.csv": "0d6f600ee71bf6ffb5069d0c583cbe0b1ec97df6da4e4e439d5f0f22f5616abe",
        "data/free_sources/raw/cboe/VIX9D_History.csv.manifest.json": "aee15817f60f0e3cde73a55fa99b807ca4dd10b6b045d8f9b87683df7050cab0",
        "data/free_sources/raw/cboe/VIX_History.csv": "a34aabce269632f30904cf482986dd50b6d4cf51f2203dc51a0a9f460f3c90b2",
        "data/free_sources/raw/cboe/VIX_History.csv.manifest.json": "51caa998f0dadc1a707182a0222f793626da93da535a31f49d66b79db0ab2917",
        "data/free_sources/raw/cboe/VVIX_History.csv": "f6bc726455fa3859c662875a005e97ba656b9536ae58fa6977ad24c6228e4f6a",
        "data/free_sources/raw/cboe/VVIX_History.csv.manifest.json": "f79ecf39b83642b0df23743a1f3aa0077bbf8fc8628fdc0b5de49f0a8ccd0a52",
        "data/raw/SKEW_History.csv": "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492",
        "data/raw/skew_daily.parquet": "159deae9944941c92a454e80ff32d33eddc94e6e4f2e79a64678bd0429ade5a7",
        "data/raw/skew_daily_source.json": "d45a9e3080116655b8f613b131903de33eb23543037893c8603624e84d4121d7",
        "data/research_paths/source_manifest.json": "457afd656244fc527535984874ee11313fd3156815f0d982d0c9f29ab85d244c",
        "data/research_paths/spx_daily.parquet": "3958fbb1eb36689df1596c26b9f0e02e3f1d4fa3b0032381e5287a2a03697bf0",
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
        "horizons": [1],
        "market_lag": 1,
        "minimum_train": 1000,
        "models": ["baseline", "recent_frequency", "location"],
        "minimum_train_per_class": 50,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "minimum_phase_observations": 127,
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
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "I_square",
            "R_square",
            "skew_square",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
        ],
        "all_features": [
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
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
            "range_extremity",
        ],
    },
    "feature": {
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
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
            "range_extremity",
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
            "neg_d",
            "neg_w",
            "neg_m",
            "skew",
            "I_square",
            "R_square",
            "skew_square",
            "intraday",
            "intraday_sq",
            "overnight",
            "overnight_sq",
        ],
        "location": "Z=(log(C/L)-log(H/C))/log(H/L);E=Z*Z;use E[t-1] only",
        "zero_range": "unknown E shared common-sample mask; retain full reference "
        "calendar and independently known risk-event labels",
        "domain": "positive ordered OHLC, finite real arithmetic,abs(Z)<=1,0<=E<=1; "
        "reject violations; no clipping or zero-fill",
        "nuisance": "raw log(C/O),its square,raw log(O/C_previous),its square;all "
        "shifted one observed session",
        "curvature": "(I-training_mean_I)^2,(R-training_mean_R)^2,(skew-training_mean_skew)^2 "
        "on common mature rows",
        "scaling": "all25 baseline slopes use training population mean/std; scale>1e-12; "
        "no dropping duplicates or collinear columns; unit intercept remains",
        "location_scaling": "training-centered E, fixed scale1; exactconstant E "
        "center=firstvalue gives exactzero with foldretention; audit "
        "constant condition",
        "nonredundancy": "algebraic distinctness motivates one increment; no empirical "
        "rank-based feature selection; coefficient ridge handles linear "
        "dependencies and retained zero correction cannot improve "
        "scores",
    },
    "target": {
        "risk": "V_s=max(0.5*log(H_s/L_s)^2-(2*log(2)-1)*log(C_s/O_s)^2,1e-10)+log(O_s/C_previous)^2",
        "event": "y_t=1{V[t+1]>2*mean(V[t-22:t-1])};exact22 observed sessions ending "
        "prior session",
        "threshold": 2.0,
        "reference_sessions": 22,
        "ties": "equality is0; missing risk or anyreference value unknown, never0",
        "maturity": "target_end=available_date=nextobservedSPXsessionclose; cutoff "
        "priorobservedclose",
        "measurement": "dailyOHLCriskproxy relative to trailingmean, not "
        "integratedvariance or doubling versus immediatepreviousday",
        "schedule": "first feature-complete monthly application before future querylabel "
        "mask; expanding mature commontraining>=1000 and>=50eachclass; "
        "everymonth preflight before anyfit; retain unscoredapplications and "
        "fits",
        "reference_arithmetic": "Checked math.fsum of exactly22 finite positive "
        "session-risk values dividedby22, then checked2*mean; no "
        "epsilon at equality. Legacy rolling baseline features "
        "unchanged.",
    },
    "model": {
        "alpha": 0.01,
        "objective": "mean(logaddexp(0,(1-2*y)*eta))+.01*sum(slope^2);unpenalizedbaselineintercept",
        "baseline": "26-column normalized ridge logistic, trainingfrequencylogit intercept "
        "start,zero slopes",
        "candidate": "holdbaselineeta fixed; one penalized scalar b*centeredE, start0; "
        "exactconstantE canonicalzero",
        "producer": {
            "method": "newton_armijo",
            "maximum_iterations": 200,
            "maximum_backtracks": 60,
            "armijo": 0.0001,
            "gradient_tolerance": 1e-08,
        },
        "probabilities": "finite expit(logit) in[0,1]; endpoints valid; no clipping; "
        "alloriginalgradients mandatory",
    },
    "frequency": {
        "half_life_sessions": 63,
        "decay": "2**(-1/63)",
        "seed": "firstfit eligible commontraining eventmean S=f0,W=1 at "
        "firstapplication prior-sessioncutoff k0; no labelavailable<=k0 is fed "
        "again",
        "update": "on every later fullreference session decay S,W bydelta, then "
        "add(1-delta)*y and(1-delta) ifoneknownlabelmatures; missing "
        "addsnothing; no resets acrossphase orfeaturegaps",
        "labels": "fulltargetcalendar includingfeatureincomplete or unscoredorigins; "
        "exactunique nextsessionavailability",
        "arithmetic": "fixedfloat64 order with checked nonzero multiplication/division "
        "underflow;0<=S<=W<=1,W>0; q=S/W in[0,1]; no clipping or reset",
        "outputs": "stateaudit everyapplication; baseline/location "
        "applicationprobabilities included in everyfit; scoredorigins get "
        "all3modelrows",
    },
    "scoring": {
        "loss": "brier",
        "effect_threshold_absolute": 0.0005,
        "paired_difference": "(p_candidate-p_control)*((p_candidate-y)+(p_control-y)), "
        "afterfiniteindividualsquaredlossvalidation",
        "arithmetic": "finitebinaryy "
        "andprobabilitiesin[0,1],boundedloss;rejectnonzerosquares/productsunderflowtozero; "
        "coherence64epsilon/downwardULP noabsolutefloor",
        "interpretation": "conditionalprobability ofdefinedevent; absolute Brierscore "
        "gain is notprofit or probabilitypointgain",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 127,
        "cumulative_hypotheses": 129,
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
        ],
        "controls": ["baseline", "recent_frequency"],
        "contrasts": [
            ["location", "baseline", "brier"],
            ["location", "recent_frequency", "brier"],
        ],
        "candidate_gate": "bothcontrols pass bothphases/effect/stability "
        "andbothmultiplicitygates; no "
        "selectiveevent/subperiod/calibrationbin promotion",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "minimum_phase_per_class": 30,
        "minimum_slice_per_class": 15,
        "bootstrap_draws": 199999,
        "seed": 20260924,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block;development0,evaluation1;common "
        "drawdesign forbothcontrols",
        "p_phase": "maximum two-sided centered-null circularblock bootstrap "
        "andBartlettHAC126p",
        "p_hypothesis": "maximumdevelopment/evaluationp",
        "multiplicity": "Holm2 at.05/(18*19);cumulativeHolm129 at.05 includingallfailedprior",
        "gate": "bothphasesmeanBrierdecrease>=.0005 againstbothcontrols;bothlate "
        "slicesnegative;bothHolmgates",
        "power": "ordinary5percentHAC80percentMDE diagnostic "
        "dividedby.0005;notadjustedpower or equivalencegate",
        "failure_rule": "anysource,arithmetic,support,optimizer,forecast,score,verification,publicationfailure "
        "keeps bothnewhypothesesUNEVALUABLEp1; "
        "nooutcome-dependentrepair",
        "calibration": "descriptivein-the-large perphase/model "
        "n,eventfrequency,meanprobability,gap,Brier only;no "
        "fitting,bins orpromotion",
        "resolution": "Minimum p=1/200000 below one tenth strictest two-comparison raw "
        "wave18 cutoff .05/(18*19*2).",
    },
    "verification": {
        "baseline": {
            "method": "scipy_root_hybr",
            "start": "all26zero",
            "xtol": 1e-10,
            "maxfev": 2000,
            "factor": 1.0,
            "success_required": True,
            "full_gradient_tolerance": 1.0001e-08,
            "fallback": "none",
        },
        "scalar": {
            "method": "brentq",
            "bracket": "plus/minus(mean(abs(centeredE))+1)/.02",
            "xtol": 1e-12,
            "rtol": 1e-14,
            "maxiter": 200,
            "full_gradient_tolerance": 1.0001e-08,
            "constant": "exactzero canonicalcoefficient",
        },
        "coefficient_rtol": 1e-07,
        "coefficient_atol": 1e-06,
        "prediction_rtol": 1e-07,
        "prediction_atol": 1e-06,
        "saved_replay_rtol": 1e-10,
        "saved_replay_atol": 1e-12,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "state_roundoff_multiplier": 64,
        "independent_state": "explicitcompensatedsum seed*delta^k plus "
        "known(1-delta)*delta^age weights;domainandzero masks "
        "mandatory;64*(k+1)*max(eps*absexpected,downwardULP) "
        "tolerance",
        "reconstruction": "independentboundedsourcevalues,features/targets,allmonthlytraining/transform/fit/application/state/score/contrast/ledger;immutableinputsandoutputs "
        "rehashedatend",
    },
    "upstream": {
        "protocol": "profiled_quarter.yaml",
        "reports": "reports/profiled_quarter",
        "data": "data/profiled_quarter",
        "protocol_sha256": "6e7821d33c13524ffbfafa666e8c85c1fab7aa48cdd94563989dcd735d73af0c",
        "manifest_sha256": "051c96e0f04bb4f42553ba735a750ffe27d589730d15d734f038debf0a361d8d",
        "verification_sha256": "d76047352010750e117c928a9fbd41c3f361be6bc345db50fcddeaeb878d3b7f",
        "verifier_sha256": "dc7a3a04941594b7e52c2ec3d3da04c16aa9306c3f30e65130b950e2239b4f91",
        "required_status": "VERIFIED",
        "publication_audit_sha256": "84613cd856c92994d9b23e0e74f88fdc457a00ce07b5095601ea5906a51a6bda",
        "prospectus": "reports/profiled_quarter/NEXT_RESEARCH_DIRECTION.md",
        "prospectus_sha256": "c59a79477858e9d58862b26761c98d8c9c8dbead1ed8f9a9b7d74394e5e18560",
    },
    "source_anchor": {
        "protocol": "tail_shape.yaml",
        "reports": "reports/tail_shape",
        "data": "data/tail_shape",
        "protocol_sha256": "0d7184c2d966f28d6ab1602ee6ff6bce4de54019ed0d601688ed7281011366ee",
        "manifest_sha256": "136cfa954874231d305af552d3f89d7e77bee76d3638a9152f764b425509644d",
        "verification_sha256": "d4f63b1848a4d33170101c8717c50c5bf0ad07b254d262ffda83517735fa89ff",
        "verifier_sha256": "18916219840f3a2b01cfaf9b1ada77e7f3bd83219912e70ae4bf5c5e10aadd48",
        "required_status": "VERIFIED",
    },
    "admission": "Hashcheck complete frozen17manifest/code/input/preserved closure, "
    "originaltailshape proof andsource11pins, current17published artifacts; "
    "preserve allpreviousfailedattempts. Priorhistoricalfits/inference are anchored "
    "by their exact successfulrecords, not newlyrefit; new study independently "
    "reconstructs its own inputs andforecasts. Source numbers decoded onlyfrom "
    "singlehashchecked byte snapshots.",
    "outputs": {"data": "data/range_alert", "reports": "reports/range_alert"},
}


def validate_protocol(p):
    if json.dumps(p, sort_keys=True, allow_nan=False) != json.dumps(
        CONTRACT, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Frozen wave18 range-alert contract differs")


def eligible_entries(features, targets, section):
    if not features.index.equals(targets.index):
        raise AssertionError("Identical full calendars required")
    complete = (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
    )
    permitted = pd.Series(False, index=features.index)
    for name in ("development", "evaluation"):
        first, last = section[name]
        permitted |= (features.index >= first) & (features.index <= last)
    application = features.index[
        complete
        & permitted
        & (features.index >= section["origin_start"])
        & (features.index <= section["origin_end"])
    ]
    known = (
        targets.y.isin([0.0, 1.0])
        & targets.available_date.notna()
        & (targets.available_date <= section["latest_target"])
    )
    known &= (features.index > section["development"][1]) | (
        targets.available_date <= section["development_target_available_by"]
    )
    return application, application[known.loc[application]]


def training_mask(features, targets, entry):
    if not features.index.equals(targets.index):
        raise AssertionError("Identical full calendars required")
    position = features.index.get_loc(entry)
    if position < 1:
        raise ValueError("Predecessor session required")
    return (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & features.feature_cutoff_date.notna()
        & targets.y.isin([0.0, 1.0])
        & (features.index < entry)
        & (targets.available_date <= features.index[position - 1])
    )


def validate_panel(panel):
    if (
        panel.empty
        or tuple(panel.columns) != PANEL_COLUMNS
        or panel.duplicated(["origin", "model"]).any()
        or set(panel.model) != set(MODELS)
    ):
        raise AssertionError("Three full model cohorts and exact schema required")
    shared = [
        column for column in PANEL_COLUMNS if column not in ("model", "probability", "loss")
    ]
    first = (
        panel.loc[panel.model == MODELS[0], shared]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    for name in MODELS[1:]:
        if not first.equals(
            panel.loc[panel.model == name, shared].sort_values("origin").reset_index(drop=True)
        ):
            raise AssertionError("Exact cross-model metadata and labels required")
    for column in (
        "origin",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        "fit_origin",
        "fit_cutoff_date",
        "train_last_target",
        "train_last_available",
    ):
        dates = pd.DatetimeIndex(panel[column])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise AssertionError("Normalized finite dates required")
    if not np.array_equal(brier_loss(panel.probability, panel.y), real(panel.loss)):
        raise AssertionError("Exact issued Brier score required")


def verify_fit(train, targets, app, audit):
    x, query, m, querym, geometry = transform(train, app)
    y = binary(targets.y)
    expected_support = support(y, 50)
    means = np.asarray(geometry["means"])
    scales = np.asarray(geometry["scales"])
    same_tree(audit["transform"], geometry, "Independent training-only geometry")
    if (
        audit["train_n"] != len(train)
        or audit["application_n"] != len(app)
        or audit["support"] != expected_support
    ):
        raise AssertionError("Exact common mature train/support/application audit differs")
    frequency = float(y.mean())
    same_tree(
        audit["frequency"],
        {"probability": frequency, "train_n": len(y), "application_n": len(app)},
        "Same-sample frequency control",
    )
    base, mem = audit["baseline"], audit["location"]
    for one in (base, mem):
        if (
            one["alpha"] != 0.01
            or one["train_n"] != len(y)
            or one["application_n"] != len(app)
            or one["success"] is not True
            or not isinstance(one["iterations"], int)
            or not 0 <= one["iterations"] <= 200
            or not isinstance(one["backtracks"], int)
            or not 0 <= one["backtracks"] <= 59 * one["iterations"]
            or not 0 <= float(real(one["gradient_max_abs"])) <= 1e-8
        ):
            raise AssertionError("Saved fixed numerical fit contract differs")
    if base["columns"] != list(BASE):
        raise AssertionError("Declared baseline coefficient order differs")
    close(base["means"], means, "Saved baseline means")
    close(base["scales"], scales, "Saved baseline scales")
    start = np.zeros(len(BASE))
    start[0] = np.log(frequency / (1 - frequency))
    close(base["start"], start, "Fixed producer initialization", rtol=0, atol=0)
    beta = real(base["beta"])
    value, g, _ = logistic_objective(beta, x, y)
    close(base["objective"], value, "Saved normalized Bernoulli objective")
    close(
        base["gradient"],
        g,
        "Saved independently recomputed full gradient",
        rtol=1e-7,
        atol=1e-12,
    )
    close(
        base["gradient_max_abs"], np.max(abs(g)), "Saved gradient norm", rtol=1e-7, atol=1e-12
    )
    if np.max(abs(g)) > GRADIENT_TOLERANCE:
        raise AssertionError("Saved baseline is not stationary")
    independent = independent_baseline_fit(x, y)
    close(
        beta, independent["beta"], "Independent convex baseline optimum", rtol=1e-7, atol=1e-6
    )
    eta, appeta = real(x @ beta), real(query @ beta)
    independent_p = expit(query @ independent["beta"])
    close(
        expit(appeta), independent_p, "Independent baseline probability", rtol=1e-7, atol=1e-6
    )
    if (
        mem["column"] != "range_extremity"
        or mem["scale"] != 1.0
        or mem["baseline_frozen"] is not True
        or mem["status"]
        != ("EXACT_CONSTANT_INPUT" if geometry["range_extremity_constant"] else "FITTED")
    ):
        raise AssertionError("Frozen baseline offset or constant-memory convention differs")
    close(mem["mean"], geometry["range_extremity_mean"], "Training-only memory mean")
    close(mem["start"], [0.0], "Fixed scalar initialization", rtol=0, atol=0)
    b = float(real(mem["b"]))
    if geometry["range_extremity_constant"] and b != 0:
        raise AssertionError("Exact constant memory coefficient must be zero")
    mvalue, mg, _ = location_objective(b, eta, y, m)
    close(mem["objective"], mvalue, "Saved scalar normalized Bernoulli objective")
    close(mem["gradient"], [mg], "Independent saved scalar gradient", rtol=1e-7, atol=1e-12)
    close(
        mem["gradient_max_abs"], abs(mg), "Saved scalar gradient norm", rtol=1e-7, atol=1e-12
    )
    if abs(mg) > GRADIENT_TOLERANCE:
        raise AssertionError("Saved scalar memory is not stationary")
    scalar = independent_location_fit(eta, y, m)
    close(b, scalar["b"], "Independent monotone scalar root", rtol=1e-7, atol=1e-6)
    issued = expit(real(appeta + b * querym))
    close(
        issued,
        expit(real(appeta + scalar["b"] * querym)),
        "Independent scalar probability",
        rtol=1e-7,
        atol=1e-6,
    )
    # Replay with the saved geometry as well, so audit rounding never changes
    # the authoritative emitted probability used for the proper score.
    raw = app.loc[:, ALL_FEATURES].copy()
    for name in ("I", "R", "skew"):
        center = audit["transform"]["curvature_means"][name]
        raw[name + "_square"] = product(
            app[name] - center, app[name] - center, "saved curvature replay"
        )
    saved_query = (raw.loc[:, BASE].to_numpy() - real(base["means"])) / real(base["scales"])
    saved_eta = real(saved_query @ beta)
    saved_m = app["range_extremity"].to_numpy() - mem["mean"]
    probabilities = {
        "baseline": expit(saved_eta),
        "location": expit(real(saved_eta + b * saved_m)),
    }
    close(probabilities["baseline"], expit(appeta), "Independent geometry probability replay")
    close(probabilities["location"], issued, "Independent memory geometry replay")
    for values in probabilities.values():
        real(values)
        if ((values < 0) | (values > 1)).any():
            raise AssertionError(
                "Finite application probabilities required even without labels"
            )
    return probabilities, {
        "constant_location": geometry["range_extremity_constant"],
        "baseline_solver": {k: v for k, v in independent.items() if k != "beta"},
        "independent_baseline_gradient": independent["gradient_max_abs"],
        "independent_location_gradient": scalar["gradient_max_abs"],
    }


def alignment(features, targets):
    reference = clock.target_alignment(targets, features.index)
    if tuple(features.columns) != ALL_FEATURES + ("feature_cutoff_date",):
        raise ValueError("Exact24-feature schema required")
    real(features.loc[:, ALL_FEATURES], missing=True)
    if not features.feature_cutoff_date.equals(
        pd.Series(reference, index=reference).shift().rename("feature_cutoff_date")
    ):
        raise ValueError("Exact full-calendar previous-session cutoffs required")
    return reference


def metadata(features, targets, app, mask):
    entry = app[0]
    dates = features.index
    return {
        "fit_origin": str(entry.date()),
        "fit_cutoff_date": str(dates[dates.get_loc(entry) - 1].date()),
        "train_n": int(mask.sum()),
        "train_first_origin": str(dates[mask][0].date()),
        "train_last_origin": str(dates[mask][-1].date()),
        "train_last_target": str(targets.loc[mask, "target_end"].max().date()),
        "train_last_available": str(targets.loc[mask, "available_date"].max().date()),
        "application_n": len(app),
    }


def preflight(features, targets, section):
    alignment(features, targets)
    apps, scored = eligible_entries(features, targets, section)
    if not len(apps):
        raise ValueError("INSUFFICIENT_DATA: no common applications")
    fits = []
    for month in apps.to_period("M").unique():
        app = apps[apps.to_period("M") == month]
        mask = training_mask(features, targets, app[0])
        if mask.sum() < section["minimum_train"]:
            raise ValueError("INSUFFICIENT_DATA: mature training support")
        *_, geometry = transform(features.loc[mask], features.loc[app])
        fits.append(
            {
                **metadata(features, targets, app, mask),
                "support": support(targets.loc[mask, "y"], 50),
                "transform": geometry,
            }
        )
    phases = []
    for name in ("development", "evaluation"):
        first, last = section[name]
        dates = scored[(scored >= first) & (scored <= last)]
        if len(dates) < 127:
            raise ValueError("INSUFFICIENT_DATA: phase bandwidth")
        phase = {
            "name": name,
            "n": len(dates),
            "first_origin": str(dates[0].date()),
            "last_origin": str(dates[-1].date()),
            "support": support(targets.loc[dates, "y"], 30),
            "slices": [],
        }
        if name == "evaluation":
            covered = []
            for start, end in section["evaluation_stability"]:
                selected = dates[(dates >= start) & (dates <= end)]
                covered.extend(selected)
                phase["slices"].append(
                    {
                        "start": start,
                        "end": end,
                        "n": len(selected),
                        "support": support(targets.loc[selected, "y"], 15),
                    }
                )
            if not pd.DatetimeIndex(covered).equals(dates):
                raise ValueError("Exact two evaluation slices must partition phase")
        phases.append(phase)
    return {
        "common_application_origins": len(apps),
        "common_scored_origins": len(scored),
        "monthly_fits": len(fits),
        "fits": fits,
        "phases": phases,
    }


def verify_forecasts(features, targets, panel, fits, states, protocol):
    dates = alignment(features, targets)
    validate_panel(panel)
    section = protocol["index"]
    applications, scored = eligible_entries(features, targets, section)
    groups = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    if any(not part.index.equals(scored) for part in groups.values()):
        raise AssertionError("Complete scored common cohort differs")
    if tuple(states.columns) != STATE_COLUMNS or not pd.DatetimeIndex(states.origin).equals(
        applications
    ):
        raise AssertionError("Every application state including unscored required")
    months = applications.to_period("M").unique()
    if len(fits) != len(months):
        raise AssertionError("All monthly fits including unscored months required")
    solver_audits = []
    trainrows = flat = 0
    source_fit = {}
    maxlocation = 0.0
    for number, month in enumerate(months):
        app = applications[applications.to_period("M") == month]
        entry = app[0]
        mask = training_mask(features, targets, entry)
        if mask.sum() < section["minimum_train"]:
            raise AssertionError("Fixed mature minimum failed")
        expected = metadata(features, targets, app, mask)
        saved = fits[number]
        extras = {"model_audit", "application_origins", "application_probabilities"}
        if {k: v for k, v in saved.items() if k not in extras} != expected:
            raise AssertionError("Exact monthly metadata differs")
        if saved["application_origins"] != [str(d.date()) for d in app]:
            raise AssertionError("All application date audits differ")
        replay, checked = verify_fit(
            features.loc[mask], targets.loc[mask], features.loc[app], saved["model_audit"]
        )
        if set(saved["application_probabilities"]) != {"baseline", "location"}:
            raise AssertionError("Both full application probability vectors required")
        for model in ("baseline", "location"):
            values = real(saved["application_probabilities"][model])
            if values.shape != (len(app),) or ((values < 0) | (values > 1)).any():
                raise AssertionError("Strict application probability domain")
            close(values, replay[model], "All saved application probabilities")
        trainrows += int(mask.sum())
        flat += int(checked["constant_location"])
        solver_audits.append(
            {"fit_origin": str(entry.date()), "audit": checked["baseline_solver"]}
        )
        maxlocation = max(maxlocation, checked["independent_location_gradient"])
        chosen = app[app.isin(scored)]
        keep = app.isin(scored)
        for date in app:
            source_fit[date] = entry
        for model in MODELS:
            rows = groups[model].loc[chosen]
            expected_fields = {
                "horizon": np.ones(len(chosen), int),
                "feature_cutoff_date": features.loc[chosen, "feature_cutoff_date"].to_numpy(),
                "target_end": targets.loc[chosen, "target_end"].to_numpy(),
                "available_date": targets.loc[chosen, "available_date"].to_numpy(),
                "y": targets.loc[chosen, "y"].to_numpy(),
                "fit_origin": np.full(len(chosen), entry.to_datetime64()),
                "fit_cutoff_date": np.full(
                    len(chosen), pd.Timestamp(expected["fit_cutoff_date"]).to_datetime64()
                ),
                "train_n": np.full(len(chosen), expected["train_n"]),
                "train_last_target": np.full(
                    len(chosen), pd.Timestamp(expected["train_last_target"]).to_datetime64()
                ),
                "train_last_available": np.full(
                    len(chosen), pd.Timestamp(expected["train_last_available"]).to_datetime64()
                ),
                "phase": np.where(
                    chosen <= pd.Timestamp(section["development"][1]),
                    "development",
                    "evaluation",
                ),
            }
            for column, values in expected_fields.items():
                if not np.array_equal(rows[column], values):
                    raise AssertionError("Full timing/sample " + column + " differs")
            if model != "recent_frequency":
                close(
                    rows.probability, replay[model][keep], "Saved coefficient forecast replay"
                )
    seed = fits[0]
    cutoff = pd.Timestamp(seed["fit_cutoff_date"])
    last = pd.Timestamp(seed["train_last_available"])
    seedp = float(seed["model_audit"]["frequency"]["probability"])
    full = explicit_states(
        targets,
        dates,
        cutoff,
        seedp,
        last,
        features.loc[applications[-1], "feature_cutoff_date"],
    )
    for row in states.itertuples(index=False):
        origin = pd.Timestamp(row.origin)
        one = full.loc[features.loc[origin, "feature_cutoff_date"]]
        elapsed = int(one.elapsed_sessions)
        for key, value in {
            "feature_cutoff_date": features.loc[origin, "feature_cutoff_date"],
            "source_fit_origin": source_fit[origin],
            "seed_fit_origin": applications[0],
            "seed_cutoff_date": cutoff,
            "seed_last_available": last,
            "seed_train_n": seed["train_n"],
            "latest_consumed_available": one.latest_consumed_available,
            "cumulative_updates": int(one.cumulative_updates),
            "elapsed_sessions": elapsed,
            "scored": origin in scored,
        }.items():
            if getattr(row, key) != value:
                raise AssertionError("Full-calendar state " + key + " differs")
        clock.exact_float(row.seed_probability, seedp, "First eligible sample seed")
        s, w, q = map(_finite, (row.S, row.W, row.recent_frequency))
        if not 0 <= s <= w <= 1 or not w > 0 or not 0 <= q <= 1:
            raise AssertionError("Strict saved state domain")
        for name, value in [("S", s), ("W", w), ("recent_frequency", q)]:
            clock.state_equal(value, one[name], elapsed, name)
        clock.exact_float(q, clock.divide(s, w), "Exact saved quotient replay")
        if origin in scored:
            clock.exact_float(
                groups["recent_frequency"].loc[origin, "probability"],
                q,
                "Exact issued frequency",
            )
    return {
        "monthly_fits_verified": len(fits),
        "independent_convex_fits_verified": 2 * len(fits),
        "forecasts_verified": len(panel),
        "common_scored_origins": len(scored),
        "common_application_origins": len(applications),
        "application_states_verified": len(states),
        "full_calendar_states_reconstructed": len(full),
        "post_seed_label_updates_verified": int(full.iloc[-1].cumulative_updates),
        "constant_location_fits_retained": flat,
        "training_rows_reconstructed": trainrows,
        "baseline_solver_audits": solver_audits,
        "maximum_independent_baseline_gradient": max(
            a["audit"]["gradient_max_abs"] for a in solver_audits
        ),
        "maximum_independent_location_gradient": maxlocation,
    }


def pins_checked(root, pins):
    for name, expected in pins.items():
        read_snapshot(root, name, expected)


def admit_anchor(root, info):
    read_snapshot(root, info["protocol"], info["protocol_sha256"])
    old = read_json_snapshot(root, info["reports"] + "/manifest.json", info["manifest_sha256"])
    verified = read_json_snapshot(
        root, info["reports"] + "/verification.json", info["verification_sha256"]
    )
    if (
        verified.get("status") != info["required_status"]
        or verified.get("protocol_sha256") != info["protocol_sha256"]
        or verified.get("verifier_sha256") != info["verifier_sha256"]
        or old.get("protocol_sha256") != info["protocol_sha256"]
    ):
        raise AssertionError("Original successful protocol/verifier proof identity differs")
    read_snapshot(
        root, "src/verify_" + Path(info["protocol"]).stem + ".py", info["verifier_sha256"]
    )
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, old[group])
    return {
        "status": "VERIFIED_RECORD_PINNED",
        "protocol_sha256": info["protocol_sha256"],
        "manifest_sha256": info["manifest_sha256"],
        "verification_sha256": info["verification_sha256"],
        "entries_checked": {
            group: len(old[group]) for group in ("code", "inputs", "preserved")
        },
    }


def validate_upstream(root=ROOT):
    root = Path(root)
    payload = (root / "range_alert.yaml").read_bytes()
    p = yaml.safe_load(payload)
    validate_protocol(p)
    anchors = {key: admit_anchor(root, p[key]) for key in ("upstream", "source_anchor")}
    info = p["upstream"]
    publication = read_json_snapshot(
        root, info["reports"] + "/publication_audit.json", info["publication_audit_sha256"]
    )
    if (
        publication["status"] != "VERIFIED_REPORT_AUDITED"
        or publication["verification_sha256"] != info["verification_sha256"]
    ):
        raise AssertionError("Original publication audit identity differs")
    for name, signature in publication["report_artifact_hashes"].items():
        read_snapshot(root, info["reports"] + "/" + name, signature)
    read_snapshot(root, info["prospectus"], info["prospectus_sha256"])
    pins_checked(root, p["source_files"])
    return {
        "status": "PINNED_PRIOR_VERIFIED_PROOFS",
        "anchors": anchors,
        "source_files_checked": len(p["source_files"]),
        "prior_publication_sha256": info["publication_audit_sha256"],
        "prior_files_written": False,
        "scope": "Prior successful evidence and original input closure rehashed; prior fits and inference are not newly recomputed.",
    }


def load_source_tables(root, protocol, manifest):
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="independent-range-sources-") as directory:
        stage = Path(directory)
        for name, signature in protocol["source_files"].items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or manifest["inputs"].get(name) != signature
            ):
                raise AssertionError("Exact registered safe source path/hash required")
            payload = read_snapshot(root, name, signature)
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        daily, iv, audit = source.load_source_tables(stage, protocol)
    for name, one in audit["sources"].items():
        one["source_path"] = str(root / protocol["sources"][name])
    audit.pop("gap_interpretation", None)
    audit.update(
        raw_columns=list(ALL_FEATURES),
        baseline_columns=list(BASE),
        normalization="binary target uses checked fsum of22 prior daily risk values dividedby22; legacy log-risk controls remain unchanged",
        target_interpretation="next-session SPX floored-GK-plus-raw-overnight-square proxy strictly exceeds twice its prior22-session mean; equality0, missingunknown",
        timing_assumption="all market and location predictors through prior observed SPX session close; current entry weekday only; next actual session dates used only for target maturity",
        range_extremity_interpretation="squared logarithmic close location within complete valid OHLC range; zero range unknown; vendor bar statistic, not order flow or an executable quote",
        arithmetic_policy="strict finite observed operands and computed domains; checked ratio/square underflow; no range clipping; existing GKfloor1e-10 retained",
    )
    return daily, iv, audit


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    code = {
        str(p.relative_to(root))
        for folder in ("src", "tests")
        for p in (root / folder).rglob("*.py")
    }
    inputs = set(protocol["source_files"]) | set(protocol["comparisons"]["inherited_sources"])
    for key in ("upstream", "source_anchor"):
        info = protocol[key]
        old = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        inputs.update(old["inputs"])
        inputs.add(info["protocol"])
        if not set(old["code"]) <= code:
            raise AssertionError("All old frozen code must remain")
        for folder in (info["reports"], info["data"]):
            inputs.update(
                str(p.relative_to(root)) for p in (root / folder).rglob("*") if p.is_file()
            )
    name = "reports/range_alert/freeze_record.json"
    frozen = read_json_snapshot(root, name, manifest["inputs"][name])
    inputs.add(name)
    inputs.update(frozen["prefit_design"])
    if (
        frozen["protocol_sha256"] != manifest["protocol_sha256"]
        or frozen["code"] != manifest["code"]
    ):
        raise AssertionError(
            "All prefit source code and protocol pins must match registration"
        )
    pins_checked(root, frozen["prefit_design"])
    read_snapshot(
        root,
        "reports/range_alert/full_repository_tests.txt",
        frozen["checks"]["full_log_sha256"],
    )
    preserved = {
        str(p.relative_to(root))
        for p in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if p.is_file()
        and p != root / "range_alert.yaml"
        and root / "reports/range_alert" not in p.parents
        and str(p.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
    ):
        raise AssertionError(
            "Complete new and historical artifact inventories must be registered"
        )
    if inputs & preserved:
        raise AssertionError("No double-counted inputs/publications")


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/range_alert/manifest.json").read_bytes())["inputs"]
    output = []
    for name in protocol["comparisons"]["inherited_sources"]:
        signature = pins[name]
        decoded = read_json_snapshot(root, name, signature)
        for number, row in enumerate(decoded["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(name).parent.name or Path(name).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": name,
                    "source_sha256": signature,
                    "source_row_index": number,
                    **{key: row[key] for key in ("measure", "score") if key in row},
                }
            )
    if len(output) != 127:
        raise AssertionError("All127 inherited comparisons must remain identified")
    return output


def verify_metrics(
    root, panel, protocol, metrics, monthly_fits, application_n, *, admitted_inputs=None
):
    validate_panel(panel)
    if (
        metrics["new_monthly_fits"] != monthly_fits
        or metrics["new_forecasts"] != len(panel)
        or metrics["common_application_origins"] != application_n
        or metrics["common_scored_origins"] != panel.origin.nunique()
    ):
        raise AssertionError("Exact new monthly fit/forecast/application counts differ")
    section = protocol["index"]
    counts = {}
    calibration = {}
    phase_union = np.zeros(len(panel), dtype=bool)
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        phase_union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        base = panel.loc[mask & panel.model.eq("recent_frequency")]
        counts[name] = support(base.y, 30)
        if len(base) < 127:
            raise AssertionError("Literal phase bandwidth unsupported")
        calibration[name] = {}
        for model in MODELS:
            rows = panel.loc[mask & panel.model.eq(model)]
            frequency = float(rows.y.mean())
            probability = float(rows.probability.mean())
            calibration[name][model] = {
                "n": len(rows),
                "observed_frequency": frequency,
                "mean_probability": probability,
                "calibration_gap": probability - frequency,
                "brier": float(rows.loss.mean()),
            }
    if (
        not phase_union.all()
        or not panel.horizon.eq(1).all()
        or (panel.train_n < section["minimum_train"]).any()
        or (
            panel.loc[panel.phase.eq("development"), "available_date"]
            > section["development_target_available_by"]
        ).any()
        or (panel.available_date > section["latest_target"]).any()
    ):
        raise AssertionError("Literal inference sample support/date fences differ")
    counts["evaluation_slices"] = []
    for start, end in section["evaluation_stability"]:
        values = panel.loc[
            panel.origin.between(start, end) & panel.model.eq("recent_frequency"), "y"
        ]
        counts["evaluation_slices"].append({"start": start, "end": end, **support(values, 15)})
    same_tree(metrics["class_support"], counts, "Declared phase and slice class support")
    compare_tree(metrics["calibration"], calibration, "Descriptive calibration in the large")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both fixed Brier controls required")
    probabilities = []
    effects = []
    for row in rows:
        if row["study"] != "range_alert" or row["horizon"] != 1 or row["score"] != "brier":
            raise AssertionError("Strict-sign Brier comparison identity differs")
        phases = [
            phase_statistics(panel, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        compare_tree(row["phases"], phases, "Independent Brier phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        compare_tree(row["p_conservative"], probability, "Conjunction of both phases", "p")
        probabilities.append(probability)
        effects.append(
            all(phase["delta"] <= -EFFECT for phase in phases)
            and len(phases[1]["stability"]) == 2
            and all(one["delta"] < 0 for one in phases[1]["stability"])
        )
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    same_tree(metrics["inherited_rows"], prior, "All inherited comparisons retained")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        compare_tree(row["p_holm_wave"], wave[number], "Two-hypothesis wave correction", "p")
        compare_tree(
            row["p_holm_cumulative"],
            cumulative[number],
            "129-hypothesis cumulative correction",
            "p",
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed statistical/effect/stability gate differs")
        passed.append(one)
    leads = ["location"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 129
    ):
        raise AssertionError("Complete candidate and hypothesis family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 129,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "calibration_rows_verified": 6,
        "class_support": counts,
        "leads": leads,
    }


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[
            selected.available_date <= section["development_target_available_by"]
        ]
    groups = {
        name: selected.loc[selected.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    rows, benchmark = groups["location"], groups[control]
    if any(not part.index.equals(rows.index) for part in groups.values()) or len(rows) < 127:
        raise AssertionError(
            "Complete original cohort and at least127 phase observations required"
        )
    difference = paired_difference(rows.probability, benchmark.probability, rows.y)
    values = normalized_difference(difference)
    mean = float(values.mean())
    delta = _finite(mean * 1.0)
    hac = independent_hac(values, maxlags=126)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + code * 10000 + width
        samples = explicit_bootstrap_means(
            values, width, protocol["inference"]["bootstrap_draws"], seed
        )[:, 0]
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite independent normalized bootstrap samples")
        probability = float(
            (1 + np.count_nonzero(abs(samples - mean) >= abs(mean))) / (len(samples) + 1)
        )
        blocks[str(width)] = {
            "p": probability,
            "ci95": [_finite(value * 1.0) for value in np.quantile(samples, [0.025, 0.975])],
        }
    hac = {
        "se": _finite(hac["se"] * 1.0),
        "p": _finite(hac["p"]),
        "ci95": [_finite(value * 1.0) for value in hac["ci95"]],
        "mde80_nominal": _finite(hac["mde80_nominal"] * 1.0),
    }
    intervals = [hac["ci95"], *(row["ci95"] for row in blocks.values())]
    result = {
        "name": phase,
        "first_origin": str(rows.index[0].date()),
        "last_origin": str(rows.index[-1].date()),
        "n": len(rows),
        "class_support": support(rows.y, 30),
        "delta": delta,
        "candidate_loss": _finite(rows.loss.mean()),
        "control_loss": _finite(benchmark.loss.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "ci95_envelope": [
            min(pair[0] for pair in intervals),
            max(pair[1] for pair in intervals),
        ],
        "p_conservative": max(hac["p"], *(row["p"] for row in blocks.values())),
        "nominal_mde_effect_ratio": _finite(hac["mde80_nominal"] / EFFECT),
        "annual": [],
        "stability": [],
        "nonoverlap_phases": [{"phase": 0, "n": len(rows), "delta": delta}],
    }
    for year in sorted(set(rows.index.year)):
        mask = rows.index.year == year
        result["annual"].append(
            {
                "year": int(year),
                "n": int(mask.sum()),
                "delta": _finite(values[mask].mean() * 1.0),
            }
        )
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (rows.index >= start) & (rows.index <= end)
            if not mask.any():
                raise AssertionError("Fixed evaluation slice empty")
            result["stability"].append(
                {
                    "start": start,
                    "end": end,
                    "n": int(mask.sum()),
                    "delta": _finite(values[mask].mean() * 1.0),
                }
            )
    return result


def verify_ledger(root, metrics, prior, final_event, *, signature=None):
    report = Path(root) / "reports/range_alert"
    payload = (report / "trial_ledger.jsonl").read_bytes()
    if signature is not None and sha256(payload).hexdigest() != signature:
        raise AssertionError("Trial ledger bytes changed before decoding")
    ledger = [json.loads(line) for line in payload.splitlines() if line]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        compare_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 131
        or len(registered) != 2
        or len(prior) != 127
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "range_alert"
            or row["horizon"] != 1
            or row["score"] != "brier"
            for row in registered
        )
    ):
        raise AssertionError("Complete127inherited+2registered+2terminal ledger required")
    return {"inherited": 127, "registered": 2, final_event: 2}


def compare_targets(actual, expected):
    if (
        tuple(actual.columns) != ("y", "target_end", "available_date")
        or not actual.index.equals(expected.index)
        or not np.array_equal(actual.y.to_numpy(), expected.y.to_numpy(), equal_nan=True)
    ):
        raise AssertionError("Exact binary/missing target values and full calendar required")
    for name in ("target_end", "available_date"):
        if not actual[name].equals(expected[name]):
            raise AssertionError("Exact next-session target " + name + " differs")


def verify(root=ROOT):
    root = Path(root)
    protocol_path = root / "range_alert.yaml"
    payload = protocol_path.read_bytes()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    signature = sha256(payload).hexdigest()
    report = root / "reports/range_alert"
    out = root / "data/range_alert"
    mpath = report / "manifest.json"
    mpayload = mpath.read_bytes()
    mhash = sha256(mpayload).hexdigest()
    manifest = json.loads(mpayload)
    if manifest["protocol_sha256"] != signature:
        raise AssertionError("Protocol and registration identity differs")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    paths = [
        out / name
        for name in (
            "upstream_admission.json",
            "source_audit.json",
            "support_audit.json",
            "features.parquet",
            "targets.parquet",
            "forecasts.parquet",
            "states.parquet",
            "fits.json",
        )
    ] + [report / "metrics.json", report / "trial_ledger.jsonl"]
    snapshots = {str(path.relative_to(root)): digest(path) for path in paths}

    def decoded(path):
        name = str(path.relative_to(root))
        return read_json_snapshot(root, name, snapshots[name])

    def parquet(path):
        name = str(path.relative_to(root))
        return pd.read_parquet(io.BytesIO(read_snapshot(root, name, snapshots[name])))

    proof = validate_upstream(root)
    same_tree(
        decoded(out / "upstream_admission.json"),
        proof,
        "Exact read-only anchored source proof",
    )
    daily, iv, audit = load_source_tables(root, protocol, manifest)
    same_tree(
        decoded(out / "source_audit.json"), audit, "Independent original bounded source audit"
    )
    features, targets = feature_target_tables(daily, iv)
    clock.table_equal(
        parquet(out / "features.parquet"),
        features,
        ALL_FEATURES,
        "Independent24raw feature values",
    )
    compare_targets(parquet(out / "targets.parquet"), targets)
    support_proof = preflight(features, targets, protocol["index"])
    same_tree(
        decoded(out / "support_audit.json"),
        support_proof,
        "Complete pre-optimization support and geometry",
    )
    panel = parquet(out / "forecasts.parquet")
    states = parquet(out / "states.parquet")
    fits = decoded(out / "fits.json")
    metrics = decoded(report / "metrics.json")
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != signature
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("Only intact newly scored publication can be verified")
    forecast = verify_forecasts(features, targets, panel, fits, states, protocol)
    result = {
        "status": "VERIFIED",
        "protocol_sha256": signature,
        "verifier_sha256": digest(Path(__file__)),
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "source_reconstruction": {
            "bounded_reference_rows": len(features),
            "raw_features_verified": len(ALL_FEATURES),
            "target_rows_verified": len(targets),
            "numeric_post_cutoff_values_parsed": False,
        },
        "forecast_reconstruction": forecast,
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": 2
        * forecast.get("common_scored_origins", len(states)),
        "inference": verify_metrics(
            root,
            panel,
            protocol,
            metrics,
            len(fits),
            len(states),
            admitted_inputs=manifest["inputs"],
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots[str((report / "trial_ledger.jsonl").relative_to(root))],
        ),
        "verified_output_hashes": snapshots,
        "artifact_hashes_checked": {
            **{group: len(manifest[group]) for group in ("code", "inputs", "preserved")},
            "outputs": len(snapshots),
        },
        "limitations": [
            "Archival SPX daily floored-GK-plus-overnight-square threshold event, not measured high-frequency variance, a physical crash probability, executable return, or profit.",
            "Prior closing location is an OHLC bar statistic; adjustment for declared magnitude and leverage controls does not establish order-flow causation or general orthogonality.",
            "The baseline and one frozen-logit scalar correction are distinct staged convex fits; exact-constant location retains the canonical zero correction.",
            "The recent event-rate filter consumes all known full-calendar labels after the seed cutoff, including feature-incomplete and previously unscored origins; old seed labels are never re-added.",
            "Prior successes are preserved by anchored proof and original artifact hashes; current private prior outputs are frozen for preservation, not independently revalidated by new old-model fits.",
            "Repeated exploratory reuse of historical archival data remains adaptive; no independent future validation sample, original historical vintage certification, or conditional calibration claim.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        pins_checked(root, manifest[group])
    pins_checked(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest)
    if digest(protocol_path) != signature or digest(mpath) != mhash:
        raise AssertionError("Protocol/manifest changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
