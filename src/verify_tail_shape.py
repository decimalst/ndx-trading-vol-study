"""Independent signed-tail source, shared moments and skew-density verifier.

The density is independently expressed through scipy's ordinary Student-t.
Formula reference: https://arch.readthedocs.io/en/latest/_modules/arch/univariate/distribution.html#SkewStudent
Synthetic quadrature checks fix its normalization and first two moments.
No new producer functions are imported. Fixed verification tolerances are
mean ridge forecasts1e-8 relative/1e-12 absolute, variance BFGS coefficients
1e-6 absolute/forecasts1e-6 relative, independent shape parameters3e-5 absolute,
producer projected KKT1e-7 plus numerical allowance1e-8, independent KKT2e-7,
and independent density probabilities1e-6 relative/1e-8 absolute.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import gamma, stdtr
from scipy.stats import t

from .verify_index_hinge import RAW as PRIOR_RAW
from .verify_index_hinge import raw_features, ridge_prediction, validate_dates
from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_macro_overnight import penalized_objective, second_moment_prediction
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
RAW = PRIOR_RAW + ("neg_d", "neg_w", "neg_m", "skew")
BASE = RAW + ("I_square", "R_square", "skew_square")
MODELS = ("frequency", "constant_shape", "skew_shape")
COMPARISONS = (("skew_shape", "constant_shape", "brier"), ("skew_shape", "frequency", "brier"),
               ("skew_shape", "constant_shape", "nll"))
NU = 8.
WAVE_ALPHA = .05 / (7 * 8)
SKEW_SHA256 = "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492"


def parse_skew(raw, cutoff="2025-10-20"):
    csv = pd.read_csv(io.BytesIO(raw), dtype=str)
    csv.columns = [str(column).strip().upper() for column in csv]
    available = set(csv) & {"SKEW", "CLOSE"}
    if "DATE" not in csv or len(available) != 1 or len(set(csv)) != len(csv.columns):
        raise ValueError("Unambiguous SKEW date and measurement columns required")
    dates = pd.to_datetime(csv.DATE, format="%m/%d/%Y")
    if dates.isna().any() or dates.duplicated().any() or not len(dates):
        raise ValueError("Unique nonempty SKEW date tokens required")
    selected = dates <= pd.Timestamp(cutoff)
    values = pd.to_numeric(csv.loc[selected, next(iter(available))]).to_numpy(float)
    series = pd.Series(values, index=pd.DatetimeIndex(dates.loc[selected], name="date"), name="skew").sort_index()
    validate_dates(series.index)
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Observed SKEW values must be finite and positive")
    anchor = pd.Timestamp("2018-08-13")
    if anchor not in series.index or abs(float(series.loc[anchor]) - 159.03) > .01:
        raise ValueError("Pinned historical SKEW anchor differs")
    return series


def event_indicator(values):
    values = pd.Series(values, copy=False)
    return values.lt(-1.5).astype(float).where(np.isfinite(values))


def feature_target_tables(daily, iv):
    result = raw_features(daily, iv)
    returns = np.log(daily.close / daily.close.shift())
    negative = (-returns).clip(lower=0)
    for label, width in (("d", 1), ("w", 5), ("m", 22)):
        result[f"neg_{label}"] = negative.rolling(width, min_periods=width).mean().shift()
    result["skew"] = iv["skew"].reindex(daily.index).shift()
    gk = np.maximum(.5 * np.log(daily.high / daily.low)**2
                    - (2 * np.log(2) - 1) * np.log(daily.close / daily.open)**2, 1e-10)
    variance = gk + np.log(daily.open / daily.close.shift())**2
    mean = returns.rolling(22, min_periods=22).mean().shift()
    scale = np.sqrt(variance.rolling(22, min_periods=22).mean()).shift()
    result = result.loc[:, (*RAW, "feature_cutoff_date")]
    result["normalization_mean"], result["normalization_scale"] = mean, scale
    raw_return = np.log(daily.close.shift(-1) / daily.close)
    normalized = ((raw_return - mean) / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
    maturity = pd.Series(daily.index, index=daily.index).shift(-1)
    targets = pd.DataFrame({"y": normalized, "raw_return": raw_return.where(np.isfinite(raw_return)),
                            "event": event_indicator(normalized), "target_end": maturity, "available_date": maturity})
    return result, targets


def derived_design(training, application):
    train, apply = training.loc[:, RAW].copy(), application.loc[:, RAW].copy()
    if len(train) < 2 or not np.isfinite(train).all().all() or not np.isfinite(apply).all().all():
        raise ValueError("Complete finite raw design required")
    audit = {"train_n": len(train)}
    for column in ("I", "R", "skew"):
        center = float(train[column].mean())
        audit[column + "_mean"] = center
        for frame in (train, apply):
            frame[column + "_square"] = (frame[column] - center)**2
    if (not np.isfinite(train).all().all() or not np.isfinite(apply).all().all()
            or (train.loc[:, BASE[1:]].std(ddof=0) <= 1e-12).any()):
        raise ValueError("All21 shared moment slopes must have finite positive training scale")
    return train, apply, audit


def training_mask(complete, targets, origin, dates):
    cutoff = pd.Series(dates, index=dates).shift().loc[origin]
    return (complete & np.isfinite(targets.y) & np.isfinite(targets.raw_return) & np.isfinite(targets.event)
            & (targets.index < pd.Timestamp(origin)) & targets.available_date.notna() & (targets.available_date <= cutoff))


def eligible_entries(features, targets, section):
    columns = (*RAW, "normalization_mean", "normalization_scale")
    complete = pd.Series(np.isfinite(features.loc[:, columns]).all(axis=1), index=features.index)
    complete &= features.normalization_scale > 0
    phases = pd.Series(False, index=features.index)
    for name in ("development", "evaluation"):
        first, last = section[name]
        phases |= (features.index >= first) & (features.index <= last)
    entries = features.index[complete & phases & (features.index >= section["origin_start"])
                             & (features.index <= section["origin_end"])]
    ready = np.isfinite(targets[["y", "raw_return", "event"]]).all(axis=1) & targets.available_date.notna()
    ready &= targets.available_date <= section["latest_target"]
    ready &= (features.index > section["development"][1]) | (targets.available_date <= section["development_target_available_by"])
    return entries, entries[ready.loc[entries].to_numpy()], complete


def skew_constants(lam):
    lam = np.asarray(lam, float)
    if not np.isfinite(lam).all() or (np.abs(lam) >= 1).any():
        raise ValueError("Finite interior skew parameter required")
    c = gamma((NU + 1) / 2) / (gamma(NU / 2) * np.sqrt(np.pi * (NU - 2)))
    a = lam * (4 * c * (NU - 2) / (NU - 1))
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    return a, b


def skew_logpdf(z, lam):
    z, lam = np.broadcast_arrays(np.asarray(z, float), np.asarray(lam, float))
    a, b = skew_constants(lam)
    middle = b * z + a
    stretch = np.where(middle < 0, 1 - lam, 1 + lam)
    ordinary = middle / stretch * np.sqrt(NU / (NU - 2))
    return t.logpdf(ordinary, df=NU) + np.log(b) + .5 * np.log(NU / (NU - 2))


def skew_cdf(z, lam):
    z, lam = np.broadcast_arrays(np.asarray(z, float), np.asarray(lam, float))
    a, b = skew_constants(lam)
    middle = b * z + a
    ordinary = middle / np.where(middle < 0, 1 - lam, 1 + lam) * np.sqrt(NU / (NU - 2))
    return np.where(middle < 0, (1 - lam) * stdtr(NU, ordinary), 1 - (1 + lam) * stdtr(NU, -ordinary))


def tail_probability(mu, variance, lam):
    mu, variance = np.asarray(mu, float), np.asarray(variance, float)
    if not np.isfinite(mu).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Finite shared location and positive conditional variance required")
    return skew_cdf((-1.5 - mu) / np.sqrt(variance), lam)


def shape_objective(theta, z, standardized_skew):
    theta, z = np.asarray(theta, float), np.asarray(z, float)
    design = np.ones((len(z), 1)) if standardized_skew is None else np.c_[np.ones(len(z)), standardized_skew]
    eta = design @ theta
    tanh = np.tanh(eta)
    lam = .95 * tanh
    logpdf = skew_logpdf(z, lam)
    step = 2e-5
    derivative = (skew_logpdf(z, lam - 2*step) - 8*skew_logpdf(z, lam - step)
                  + 8*skew_logpdf(z, lam + step) - skew_logpdf(z, lam + 2*step)) / (12*step)
    gradient = -design.T @ (derivative * .95 * (1 - tanh**2)) / len(z)
    penalty = .01 * np.sum(theta[1:]**2)
    gradient[1:] += .02 * theta[1:]
    return float(-logpdf.mean() + penalty), gradient


def projected_gradient(theta, gradient):
    theta, gradient = np.asarray(theta, float), np.asarray(gradient, float).copy()
    gradient[((theta <= -3) & (gradient > 0)) | ((theta >= 3) & (gradient < 0))] = 0
    return gradient


def fit_shape(z, standardized_skew, constant=None):
    start = np.array([0.]) if standardized_skew is None else np.array([float(constant), 0.])
    result = minimize(shape_objective, start, args=(np.asarray(z, float), standardized_skew), method="L-BFGS-B", jac=True,
                      bounds=[(-3., 3.)] * len(start), options={"maxiter": 1000, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9})
    value, gradient = shape_objective(result.x, z, standardized_skew)
    maximum = float(np.max(np.abs(projected_gradient(result.x, gradient))))
    if not result.success or not np.isfinite(value) or maximum > 2e-7:
        raise AssertionError("Independent skew likelihood failed fixed projected KKT/convergence contract")
    return {"theta": result.x, "objective": value, "gradient": gradient, "projected_gradient_max_abs": maximum,
            "success": bool(result.success), "iterations": int(result.nit), "start": start}


def event_support(event, minimum):
    values = np.asarray(event, float)
    if not np.isfinite(values).all() or not np.isin(values, [0., 1.]).all():
        raise ValueError("Complete binary event labels required")
    counts = {"events": int(values.sum()), "nonevents": int(len(values) - values.sum())}
    if min(counts.values()) < minimum:
        raise ValueError("INSUFFICIENT_DATA: fixed minimum event and nonevent support")
    return counts


def load_source_tables(root, protocol):
    """Read source identity and bounded inputs without constructing event labels."""
    paths, section = protocol["sources"], protocol["index"]
    limit = pd.Timestamp(section["source_end"])
    if limit > pd.Timestamp("2025-10-20") or limit >= pd.Timestamp(section["sealed_start"]):
        raise AssertionError("Protected source fence crossed")
    daily = pd.read_parquet(root / paths["daily"], columns=["open", "high", "low", "close"], filters=[("date", "<=", limit)])
    validate_dates(daily.index)
    def audit_source(path, frame):
        return {"source_path": str(path), "source_sha256": digest(path), "bounded_rows": len(frame),
                "first_date": str(frame.index.min().date()), "last_date": str(frame.index.max().date()),
                "missing_values": {name: int(frame[name].isna().sum()) for name in frame},
                "missing_reference_dates": [str(day.date()) for day in daily.index.difference(frame.index)],
                "outside_reference_dates": [str(day.date()) for day in frame.index.difference(daily.index)]}
    sources = {"daily": audit_source(root / paths["daily"], daily)}
    series = {}
    for name in ("vix", "vix9d", "vvix"):
        column = "VVIX" if name == "vvix" else "CLOSE"
        csv = pd.read_csv(root / paths[name], usecols=["DATE", column], dtype=str)
        dates = pd.to_datetime(csv.DATE, format="%m/%d/%Y")
        keep = dates <= limit
        one = pd.Series(pd.to_numeric(csv.loc[keep, column]).to_numpy(float), index=pd.DatetimeIndex(dates.loc[keep], name="date"), name=name)
        validate_dates(one.index)
        series[name] = one
        sources[name] = {**audit_source(root / paths[name], one.to_frame()), "provider_date_field": "DATE", "provider_value_field": column}
    source = root / paths["skew"]
    if digest(source) != SKEW_SHA256 or protocol["source_contract"]["skew_sha256"] != SKEW_SHA256:
        raise AssertionError("Only fixed legacy SKEW bytes can enter the new study")
    metadata = json.loads((root / paths["skew_source"]).read_text())
    if (metadata["sha256"] != digest(source) or metadata["url"] != "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
            or metadata["anchor_date"] != "2018-08-13" or abs(metadata["anchor_close"] - 159.03) > .01):
        raise AssertionError("Original SKEW provenance or anchor differs")
    raw = source.read_bytes()
    skew = parse_skew(raw, section["source_end"])
    tokens = pd.read_csv(io.BytesIO(raw), dtype=str)
    tokens.columns = [str(name).strip().lower() for name in tokens]
    dates = pd.to_datetime(tokens.date, format="%m/%d/%Y")
    if (metadata["rows"] != len(dates) or metadata["first_date"] != str(dates.min().date())
            or metadata["last_date"] != str(dates.max().date())):
        raise AssertionError("Full date-only source extent differs from original metadata")
    derived = pd.read_parquet(root / paths["skew_derived"], columns=["close"], filters=[("date", "<=", limit)])
    validate_dates(derived.index)
    if not derived.index.equals(skew.index) or not np.array_equal(derived.close.to_numpy(), skew.to_numpy(), equal_nan=True):
        raise AssertionError("Bounded raw and historical derived SKEW dates/values differ")
    sources["skew"] = {**audit_source(source, skew.to_frame()), "provider_date_field": "date",
                       "provider_value_field": next(name for name in ("close", "skew") if name in tokens),
                       "source_rows_from_date_tokens": len(dates), "source_first_date": str(dates.min().date()),
                       "source_last_date": str(dates.max().date())}
    sources["skew_derived"] = audit_source(root / paths["skew_derived"], derived)
    sources["skew_source"] = {"source_path": str(root / paths["skew_source"]), "source_sha256": digest(root / paths["skew_source"])}
    audit = {"sources": sources, "source_end": section["source_end"], "sealed_start": section["sealed_start"],
             "reference_calendar": {"definition": "bounded observed SPX raw-OHLC sessions", "sessions": len(daily),
                                    "first_date": str(daily.index.min().date()), "last_date": str(daily.index.max().date())},
             "raw_columns": list(RAW), "numeric_post_cutoff_values_parsed": False, "historical_vintage_certified": False,
             "timing_assumption": "all market predictors through prior observed SPX session close; current entry weekday only",
             "vix9d_caveat": "prelaunch January2011-October2013 values are back-calculated archival training inputs",
             "gap_interpretation": "log implied-versus-trailing-OHLC-proxy gap; not a measured variance risk premium",
             "normalization": "strict prior-session trailing22 raw logreturn mean and square root of daily GK-plus-raw-overnight variance mean; no annualization",
             "target_interpretation": "next-session SPX raw-close log price return normalized by prior information; event u<-1.5, not a universal VaR probability",
             "skew_provenance": {"url": metadata["url"], "fetched_at_utc": metadata.get("fetched_at_utc"),
                 "registered_raw_sha256": metadata["sha256"], "raw_matches_manifest": True, "raw_matches_fixed_pin": True,
                 "raw_derived_bounded_exact_equal": True, "anchor_date": "2018-08-13", "anchor_expected": 159.03,
                 "anchor_observed": float(skew.loc["2018-08-13"]), "anchor_tolerance": .01,
                 "predictor_source": "raw SKEW CSV; existing derived parquet is audit-only"}}
    series["skew"] = skew
    return daily, pd.DataFrame(series).sort_index(), audit


def verify_moments(training, y, application, audits):
    combined = pd.concat([training, application])
    expected_mean, rebuilt = ridge_prediction(training, y, combined)
    saved = audits["mean"]
    if tuple(saved["columns"]) != BASE or saved["alpha"] != .01 or saved["train_n"] != len(y):
        raise AssertionError("Shared location must include all22 fixed features on the common sample")
    for field in ("means", "scales", "beta"):
        same(saved[field], rebuilt[field], "Independent shared ridge mean " + field, rtol=1e-7, atol=1e-12)
    standardized = (combined.to_numpy() - np.asarray(saved["means"])) / np.asarray(saved["scales"])
    mean = standardized @ np.asarray(saved["beta"])
    same(mean, expected_mean, "Independent shared mean on training and application", rtol=1e-8, atol=1e-12)
    gradient_mean = standardized[:len(y)].T @ (mean[:len(y)] - y) / len(y)
    gradient_mean[1:] += .01 * np.asarray(saved["beta"])[1:]
    if np.max(np.abs(gradient_mean)) > 1e-10 or saved["gradient_max_abs"] > 1e-10:
        raise AssertionError("Shared mean violates independently reconstructed ridge KKT")
    residual = np.asarray(y) - mean[:len(y)]
    residual_squares = residual**2
    variance_prediction, independent = second_moment_prediction(training, residual_squares, combined)
    var = audits["variance"]
    if tuple(var["columns"]) != BASE or var["alpha"] != .01 or var["train_n"] != len(y):
        raise AssertionError("Shared variance must use the same22 features and training rows")
    for field in ("means", "scales", "train_mean"):
        same(var[field], independent[field], "Training-residual variance " + field, rtol=1e-9, atol=1e-12)
    same(var["beta"], independent["beta"], "Independent BFGS shared variance coefficients", rtol=0, atol=1e-6)
    z = np.c_[np.ones(len(combined)), (combined.to_numpy()[:, 1:] - np.asarray(var["means"])) / np.asarray(var["scales"])]
    beta = np.asarray(var["scaled_beta"])
    restored = beta.copy()
    restored[0] += np.log(residual_squares.mean())
    same(var["beta"], restored, "Exact shared variance target-unit restoration", rtol=1e-9, atol=1e-12)
    objective, gradient = penalized_objective(beta, z[:len(y)], residual_squares / residual_squares.mean())
    kkt = float(np.max(np.abs(gradient)))
    if kkt > 1e-8 + 1e-12 or var["iterations"] >= 200 or var["backtracks"] < 0:
        raise AssertionError("Shared variance fails frozen numerical contract")
    same(var["gradient_max_abs"], kkt, "Shared variance saved KKT", rtol=1e-4, atol=1e-12)
    same(var["objective"], objective + np.log(residual_squares.mean()), "Shared variance mean proper-score objective", rtol=1e-9, atol=1e-12)
    variance = np.exp(np.log(residual_squares.mean()) + z @ beta)
    same(variance, variance_prediction, "Independent shared conditional variance", rtol=1e-6, atol=1e-12)
    return mean, variance, {"variance_kkt": kkt, "independent_variance_kkt": independent["gradient_max_abs"]}


def verify_shapes(innovation, standardized_skew, audits):
    result, maximum = {}, 0.
    for model in ("constant_shape", "skew_shape"):
        saved = audits[model]
        theta = np.asarray(saved["theta"], float)
        is_constant = model == "constant_shape"
        start = np.array([0.]) if is_constant else np.array([audits["constant_shape"]["theta"][0], 0.])
        if (theta.shape != start.shape or not np.isfinite(theta).all() or (np.abs(theta) > 3).any()
                or saved["nu"] != 8. or saved["penalty"] != (0. if is_constant else .01)
                or not saved["success"] or not 0 <= saved["iterations"] <= 1000 or saved["bounds"] != [[-3., 3.]] * len(theta)):
            raise AssertionError("Shape audit violates fixed parameter, optimizer or penalty contract")
        same(saved["start"], start, "Single fixed shape optimizer initialization", rtol=0, atol=0)
        s = None if is_constant else standardized_skew
        objective, gradient = shape_objective(theta, innovation, s)
        kkt = float(np.max(np.abs(projected_gradient(theta, gradient))))
        if kkt > 1.1e-7 or saved["projected_gradient_max_abs"] > 1e-7:
            raise AssertionError("Saved shape coefficients fail independent numerical projected KKT")
        same(saved["objective"], objective, "Independent standardized innovation likelihood", rtol=1e-10, atol=1e-12)
        same(saved["gradient"], gradient, "Independent five-point shape gradient", rtol=1e-3, atol=1e-8)
        independent = fit_shape(innovation, s, None if is_constant else float(start[0]))
        same(theta, independent["theta"], "Independent shape optimum", rtol=0, atol=3e-5)
        maximum = max(maximum, kkt)
        result[model] = independent
    return result, maximum


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {"origin", "model", "horizon", "prediction", "mu", "variance", "lambda", "log_density", "y", "event",
                "raw_return", "target_end", "available_date", "feature_cutoff_date", "normalization_mean", "normalization_scale",
                "fit_origin", "fit_cutoff_date", "train_n", "train_event_count", "train_last_target", "train_last_available", "phase"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(MODELS) or set(forecasts.horizon) != {1}:
        raise AssertionError("All three registered tail-forecast schemas required")
    if forecasts.duplicated(["origin", "model", "horizon"]).any():
        raise AssertionError("Duplicate tail forecasts")
    entries, scored, complete = eligible_entries(features, targets, section)
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Tail models must have identical independently completed common rows")
        same(rows[["y", "raw_return", "event"]], targets.loc[scored, ["y", "raw_return", "event"]],
             "Independent normalized return and strict downside event", rtol=1e-10, atol=1e-12)
        if not np.array_equal(rows.event.to_numpy(), event_indicator(rows.y).to_numpy()):
            raise AssertionError("A forecast model changed the candidate-independent target event")
        for name in ("target_end", "available_date"):
            if not np.array_equal(rows[name].to_numpy(), targets.loc[scored, name].to_numpy()):
                raise AssertionError("Tail target maturity differs")
        if not np.array_equal(rows.feature_cutoff_date.to_numpy(), features.loc[scored, "feature_cutoff_date"].to_numpy()):
            raise AssertionError("Tail inputs violate prior-session cutoff")
        for name in ("normalization_mean", "normalization_scale"):
            same(rows[name], features.loc[scored, name], "Candidate-independent prior22 return normalizer", rtol=1e-10, atol=1e-12)
        if not np.array_equal(rows.phase.to_numpy(), np.where(scored <= section["development"][1], "development", "evaluation")):
            raise AssertionError("Tail phase fences differ")
    density0 = forecasts.loc[forecasts.model == "constant_shape"].sort_values("origin")
    density1 = forecasts.loc[forecasts.model == "skew_shape"].sort_values("origin")
    if not np.array_equal(density0[["mu", "variance"]].to_numpy(), density1[["mu", "variance"]].to_numpy()):
        raise AssertionError("Candidate and baseline density moments must be exactly identical")
    lookup = {pd.Timestamp(row["fit_origin"]): row for row in fits}
    expected_fits = entries[~entries.to_period("M").duplicated()]
    if len(lookup) != len(fits) or set(lookup) != set(expected_fits):
        raise AssertionError("Monthly tail fits must be selected by features before target readiness")
    count, shape_count, variance_max, independent_var_max, shape_max = 0, 0, 0., 0., 0.
    min_events, min_nonevents = np.inf, np.inf
    for origin in expected_fits:
        record = lookup[origin]
        mask = training_mask(complete, targets, origin, features.index)
        train_origins, n = features.index[mask], int(mask.sum())
        application = entries[entries.to_period("M") == origin.to_period("M")]
        query = scored[scored.to_period("M") == origin.to_period("M")]
        cutoff = features.loc[origin, "feature_cutoff_date"]
        counts = event_support(targets.loc[mask, "event"], 50)
        min_events, min_nonevents = min(min_events, counts["events"]), min(min_nonevents, counts["nonevents"])
        if n < section["minimum_train"] or record["train_n"] != n or record["application_n"] != len(application):
            raise AssertionError("Tail common training/application counts differ")
        if record["train_event_count"] != counts["events"] or record["train_nonevent_count"] != counts["nonevents"]:
            raise AssertionError("Saved training event support differs")
        dates = {"fit_cutoff_date": cutoff, "train_first_origin": train_origins[0], "train_last_origin": train_origins[-1],
                 "train_last_target": targets.loc[mask, "target_end"].max(),
                 "train_last_available": targets.loc[mask, "available_date"].max()}
        if any(pd.Timestamp(record[key]) != expected for key, expected in dates.items()) or dates["train_last_available"] > cutoff:
            raise AssertionError("Tail training labels were unavailable at the required cutoff")
        train, apply, centers = derived_design(features.loc[mask], features.loc[application])
        same_tree(record["transform_audit"], centers, "Shared training-only curvature centers")
        u = targets.loc[mask, "y"].to_numpy(float)
        mu, h, audit = verify_moments(train, u, apply, record["moment_audit"])
        variance_max = max(variance_max, audit["variance_kkt"])
        independent_var_max = max(independent_var_max, audit["independent_variance_kkt"])
        standardized_skew = (train["skew"] - train["skew"].mean()) / train["skew"].std(ddof=0)
        same(record["skew_mean"], train["skew"].mean(), "Training-only shape-input center", rtol=1e-12)
        same(record["skew_scale"], train["skew"].std(ddof=0), "Training-only population shape-input scale", rtol=1e-12)
        independent, maximum = verify_shapes((u - mu[:n]) / np.sqrt(h[:n]), standardized_skew.to_numpy(), record["shape_audit"])
        shape_count += 2
        shape_max = max(shape_max, maximum)
        application_positions = application.get_indexer(query)
        query_mu, query_h = mu[n:][application_positions], h[n:][application_positions]
        query_s = ((apply.loc[query, "skew"] - record["skew_mean"]) / record["skew_scale"]).to_numpy()
        frequency = float(targets.loc[mask, "event"].mean())
        same(record["frequency"], frequency, "Exact common training frequency")
        for model in MODELS:
            rows = forecasts.loc[(forecasts.model == model) & forecasts.origin.isin(query)].sort_values("origin")
            if (not rows.fit_origin.eq(origin).all() or not rows.fit_cutoff_date.eq(cutoff).all()
                    or not rows.train_n.eq(n).all() or not rows.train_event_count.eq(counts["events"]).all()):
                raise AssertionError("Tail row-level training provenance differs")
            for name in ("train_last_target", "train_last_available"):
                if not rows[name].eq(dates[name]).all():
                    raise AssertionError("Tail forecast label-maturity audit differs")
            if model == "frequency":
                same(rows.prediction, np.repeat(frequency, len(query)), "Historical frequency probability")
                if not rows[["mu", "variance", "lambda", "log_density"]].isna().all().all():
                    raise AssertionError("Frequency control cannot claim a conditional density or moments")
            else:
                theta = np.asarray(record["shape_audit"][model]["theta"])
                linear = np.repeat(theta[0], len(query)) if model == "constant_shape" else theta[0] + theta[1] * query_s
                lam = .95 * np.tanh(linear)
                probability = tail_probability(query_mu, query_h, lam)
                log_density = skew_logpdf((targets.loc[query, "y"].to_numpy() - query_mu) / np.sqrt(query_h), lam) - .5 * np.log(query_h)
                same(rows.mu, query_mu, "Exact shared conditional mean", rtol=1e-10, atol=1e-12)
                same(rows.variance, query_h, "Exact shared conditional variance", rtol=1e-9, atol=1e-12)
                same(rows["lambda"], lam, "Training-standardized SKEW shape map", rtol=1e-10, atol=1e-12)
                same(rows.prediction, probability, "Independent Student-CDF tail probability", rtol=1e-10, atol=1e-12)
                same(rows.log_density, log_density, "Independent full-u log density including external scale", rtol=1e-10, atol=1e-12)
                other = independent[model]["theta"]
                other_lam = .95 * np.tanh(other[0] if model == "constant_shape" else other[0] + other[1] * query_s)
                other_probability = tail_probability(query_mu, query_h, other_lam)
                same(rows.prediction, other_probability, "Independently optimized shape probabilities", rtol=1e-6, atol=1e-8)
            count += len(rows)
    return {"forecasts_verified": count, "common_scored_origins": len(scored), "feature_complete_applications": len(entries),
            "monthly_fits_verified": len(expected_fits), "shape_optimizations_verified": shape_count,
            "models_verified": len(MODELS), "shared_variance_kkt_max": variance_max,
            "independent_variance_kkt_max": independent_var_max, "shape_numerical_projected_kkt_max": shape_max,
            "minimum_training_events": int(min_events), "minimum_training_nonevents": int(min_nonevents)}


def support_record(event, minimum):
    counts = event_support(event, minimum)
    return {"n": len(event), **counts, "event_rate": float(np.mean(event))}


def support_summary(features, targets, section):
    entries, scored, _ = eligible_entries(features, targets, section)
    output = {"feature_complete_applications": len(entries), "common_scored_origins": len(scored), "phases": []}
    for phase in ("development", "evaluation"):
        first, last = section[phase]
        days = scored[(scored >= first) & (scored <= last)]
        summary = {"name": phase, **support_record(targets.loc[days, "event"], 30), "slices": []}
        if phase == "evaluation":
            for start, end in section["evaluation_stability"]:
                sub = days[(days >= start) & (days <= end)]
                summary["slices"].append({"start": start, "end": end, **support_record(targets.loc[sub, "event"], 15)})
        output["phases"].append(summary)
    return output


def calibration(probability, event):
    p, y = np.asarray(probability, float), np.asarray(event, float)
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (p >= lower) & ((p < upper) | ((index == 9) & (p <= upper)))
        bins.append({"lower": lower, "upper": upper, "n": int(mask.sum()),
                     "mean_probability": float(p[mask].mean()) if mask.any() else None,
                     "event_rate": float(y[mask].mean()) if mask.any() else None})
    return {"mean_probability": float(p.mean()), "event_rate": float(y.mean()), "brier": float(np.mean((p-y)**2)), "bins": bins}


def effect_passes(phases, score):
    threshold = .0005 if score == "brier" else .005
    return (len(phases) == 2 and all(row["delta"] <= -threshold for row in phases)
            and len(phases[1]["stability"]) == 2 and all(row["delta"] < 0 for row in phases[1]["stability"]))


def phase_statistics(panel, control, score, phase, code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[selected.available_date <= section["development_target_available_by"]]
    wide = selected.pivot(index="origin", columns="model", values="prediction").sort_index()
    frames = {model: selected.loc[selected.model == model].set_index("origin").reindex(wide.index) for model in MODELS}
    event = frames[control].event
    support = support_record(event, 30)
    if score == "brier":
        candidate_loss, control_loss = (event - wide.skew_shape).to_numpy()**2, (event - wide[control]).to_numpy()**2
    else:
        candidate_loss, control_loss = -frames["skew_shape"].log_density.to_numpy(), -frames[control].log_density.to_numpy()
    difference = candidate_loss - control_loss
    if len(difference) <= 126 or not np.isfinite(difference).all():
        raise AssertionError("Insufficient phase observations for literal HAC126 or invalid paired loss")
    delta, hac = float(difference.mean()), independent_hac(difference, maxlags=126)
    blocks = {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + code * 10000 + width
        samples = explicit_bootstrap_means(difference, width, protocol["inference"]["bootstrap_draws"], seed)[:, 0]
        probability = float((1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1))
        blocks[str(width)] = {"p": probability, "ci95": np.quantile(samples, [.025, .975]).tolist()}
    intervals = [hac["ci95"], *(part["ci95"] for part in blocks.values())]
    output = {"name": phase, "first_origin": str(wide.index[0].date()), "last_origin": str(wide.index[-1].date()),
              "n": len(wide), "delta": delta, "candidate_loss": float(candidate_loss.mean()), "control_loss": float(control_loss.mean()),
              "block_inference": blocks, "hac126": hac, "ci95_envelope": [min(row[0] for row in intervals), max(row[1] for row in intervals)],
              "p_conservative": max(hac["p"], *(part["p"] for part in blocks.values())), "event_support": support,
              "annual": [], "stability": [], "nonoverlap_phases": [{"phase": 0, "n": len(wide), "delta": delta}]}
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        output["annual"].append({"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            support_record(event.loc[mask], 15)
            output["stability"].append({"start": start, "end": end, "n": int(mask.sum()), "delta": float(difference[mask].mean())})
    return output


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((root / source).read_text())["rows"]):
            output.append({"study": row.get("study", Path(source).parent.name or Path(source).stem), "candidate": row["candidate"],
                           "control": row.get("control", "baseline"), "horizon": row["horizon"], "p_conservative": row["p_conservative"],
                           "source": source, "source_sha256": digest(root / source), "source_row_index": number,
                           **{name: row[name] for name in ("measure", "score") if name in row}})
    if len(output) != 103:
        raise AssertionError("All103 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if [(row["candidate"], row["control"], row["score"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("All three tail comparisons required in declared order")
    probabilities, effects = [], []
    for row in rows:
        if row["study"] != "tail_shape" or row["horizon"] != 1:
            raise AssertionError("Tail comparison identity differs")
        phases = [phase_statistics(panel, row["control"], row["score"], phase, code, protocol)
                  for code, phase in enumerate(("development", "evaluation"))]
        same_tree(row["phases"], phases, "Independent event/density phase inference and support")
        p = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], p, "Both-phase tail conjunction probability")
        probabilities.append(p)
        effects.append(effect_passes(phases, row["score"]))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "Complete tail cumulative trial identities")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-3:]
    passed = []
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Three-way tail wave Holm")
        same(row["p_holm_cumulative"], cumulative[number], "106-way cumulative Holm")
        eligible = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < .05)
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Tail score-specific effect or statistical gate differs")
        passed.append(eligible)
    leads = ["skew_shape"] if all(passed) else []
    if metrics["leads"] != leads or metrics["hypothesis_count"] != 3 or metrics["cumulative_hypothesis_count"] != 106:
        raise AssertionError("Shape candidate must pass all three fixed contrasts")
    expected_calibration = {}
    for phase in ("development", "evaluation"):
        first, last = protocol["index"][phase]
        selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
        if phase == "development":
            selected = selected.loc[selected.available_date <= protocol["index"]["development_target_available_by"]]
        expected_calibration[phase] = {model: calibration(rows.prediction, rows.event)
                                       for model in MODELS for rows in [selected.loc[selected.model == model].sort_values("origin")]}
    same_tree(metrics["calibration"], expected_calibration, "Fixed probability bins and Brier calibration")
    return {"new_hypotheses_verified": 3, "cumulative_hypotheses_verified": 106, "phase_comparisons_verified": 6,
            "bootstrap_runs_verified": 18, "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
            "calibration_bins_verified": 60, "leads": leads}


def verify(root=ROOT):
    root = Path(root)
    output, report = root / "data/tail_shape", root / "reports/tail_shape"
    protocol_file = root / "tail_shape.yaml"
    protocol = yaml.safe_load(protocol_file.read_text())
    manifest = json.loads((report / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol_file):
        raise AssertionError("Frozen tail protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError("Frozen source, code or earlier artifact changed: " + path)
    daily, iv, audit = load_source_tables(root, protocol)
    features, targets = feature_target_tables(daily, iv)
    saved_features = pd.read_parquet(output / "features.parquet")
    saved_targets = pd.read_parquet(output / "targets.parquet")
    for actual, expected, numeric in ((saved_features, features, (*RAW, "normalization_mean", "normalization_scale")),
                                      (saved_targets, targets, ("y", "raw_return", "event"))):
        if not actual.index.equals(expected.index) or tuple(actual.columns) != tuple(expected.columns):
            raise AssertionError("Independent tail table schema or dates differ")
        same(actual.loc[:, numeric], expected.loc[:, numeric], "Every independent raw tail input and normalized label", rtol=1e-10, atol=1e-12)
        for name in set(expected) - set(numeric):
            if not actual[name].equals(expected[name]):
                raise AssertionError("Independent tail table maturity differs: " + name)
    same_tree(json.loads((output / "source_audit.json").read_text()), audit, "Independent legacy SKEW and market provenance")
    support = support_summary(features, targets, protocol["index"])
    same_tree(json.loads((output / "support_audit.json").read_text()), support, "Pre-model class-support gates")
    forecasts = pd.read_parquet(output / "forecasts.parquet")
    fits = json.loads((output / "fits.json").read_text())
    result = {"status": "VERIFIED", "protocol_sha256": digest(protocol_file), "verifier_sha256": digest(Path(__file__)),
              "raw_feature_rows_verified": len(features), "raw_feature_columns_verified": len(RAW),
              "shared_moment_columns_verified": len(BASE), "event_support": support,
              "forecast_reconstruction": verify_forecasts(features, targets, forecasts, fits, protocol)}
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_file) or metrics["evidence_class"] != protocol["evidence_class"]:
        raise AssertionError("Tail metric provenance differs")
    result["inference"] = verify_metrics(root, forecasts, protocol, metrics)
    ledger = [json.loads(line) for line in (report / "trial_ledger.jsonl").read_text().splitlines() if line]
    for event, expected in (("inherited", metrics["inherited_rows"]), ("evaluated", metrics["rows"])):
        actual = [{key: value for key, value in row.items() if key != "event"} for row in ledger if row["event"] == event]
        same_tree(actual, expected, "Tail " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (len(ledger) != 109 or len(registered) != 3
            or [(row["candidate"], row["control"], row["score"]) for row in registered] != list(COMPARISONS)
            or any(row["protocol_sha256"] != digest(protocol_file) or row["horizon"] != 1 for row in registered)):
        raise AssertionError("Complete103inherited+3registered+3evaluated ledger required")
    result["ledger_events_verified"] = {"inherited": 103, "registered": 3, "evaluated": 3}
    result["limitations"] = ["Equal fitted moments isolate this parametric increment but do not eliminate location/scale or tail-thickness misspecification.",
                            "One-day physical return events differ from option-implied30day SKEW construction.",
                            "Historical reuse, archival revisions and back-calculated VIX9D remain exploratory limitations."]
    (report / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
