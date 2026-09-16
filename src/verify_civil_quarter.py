"""Independent civil-quarter source, calendar, convex-fit and publication audit.

No current producer imports. Old verification entry points that publish files
are never called. Numerical optimization and civil arithmetic are independent.
"""

from __future__ import annotations

import calendar
import json
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, minimize

from . import verify_calendar_variance as calendar_old
from . import verify_causal_pool as previous
from . import verify_cross_moment as scalar
from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
OLD = calendar_old.ALL_FEATURES
MONTHS = tuple(f"month_{month}" for month in range(2, 13))
NUISANCE = MONTHS + ("month_end5", "year_end5")
MEMORY = "quarter_end5"
CIVIL = NUISANCE + (MEMORY,)
BASE = OLD + NUISANCE
ALL_FEATURES = BASE + (MEMORY,)
MODELS = ("mean", "baseline", "quarter")
COMPARISONS = (("quarter", "baseline"), ("quarter", "mean"))
WAVE_ALPHA = 0.05 / (15 * 16)
EFFECT = 0.005
read_snapshot = previous.read_snapshot
read_json_snapshot = previous.read_json_snapshot
table_equal = previous.table_equal
compare_tree = scalar.compare_tree


def finite(value):
    if np.iscomplexobj(value):
        raise ValueError("Real finite values required")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Finite values required")
    return result


def positive(value):
    result = finite(value)
    if (result <= 0).any():
        raise ValueError("Strictly positive values required")
    return result


def multiply(a, b):
    a, b = np.broadcast_arrays(finite(a), finite(b))
    with np.errstate(all="ignore"):
        value = a * b
    finite(value)
    if ((a != 0) & (b != 0) & (value == 0)).any():
        raise ValueError("Nonzero multiplication underflow")
    return value


def divide(a, b):
    a, b = np.broadcast_arrays(finite(a), finite(b))
    if (b == 0).any():
        raise ValueError("Zero denominator")
    with np.errstate(all="ignore"):
        value = a / b
    finite(value)
    if ((a != 0) & (value == 0)).any():
        raise ValueError("Nonzero division underflow")
    return value


def exp_checked(value):
    with np.errstate(all="ignore"):
        result = np.exp(finite(value))
    return positive(result)


def checked_mean(values):
    values = finite(values)
    if not values.size:
        raise ValueError("Nonempty arithmetic mean required")
    with np.errstate(all="ignore"):
        total = values.sum()
    return float(divide(total, values.size))


def design_product(design, beta):
    multiply(design, beta)
    return finite(design @ beta)


def civil_features(origins):
    origins = pd.DatetimeIndex(origins)
    if (
        origins.has_duplicates
        or not origins.is_monotonic_increasing
        or origins.hasnans
        or origins.tz is not None
        or not origins.equals(origins.normalize())
    ):
        raise ValueError("Ordered distinct normalized civil dates required")
    records = []
    for timestamp in origins:
        date = timestamp.date() + timedelta(days=1)
        while date.weekday() in (5, 6):
            date += timedelta(days=1)
        final = date.day >= calendar.monthrange(date.year, date.month)[1] - 4
        records.append(
            [float(date.month == month) for month in range(2, 13)]
            + [
                float(final),
                float(final and date.month == 12),
                float(final and date.month in (3, 6, 9)),
            ]
        )
    return pd.DataFrame(records, index=origins, columns=CIVIL, dtype=float)


def augment_features(original):
    if tuple(original.columns) != OLD + ("feature_cutoff_date",):
        raise ValueError("Original18-column table and cutoff required")
    expected = (
        pd.Series(original.index, index=original.index).shift().rename("feature_cutoff_date")
    )
    if not original.feature_cutoff_date.equals(expected):
        raise ValueError("Original preceding reference-session cutoff required")
    result = original.loc[:, OLD].join(civil_features(original.index))
    result["feature_cutoff_date"] = original.feature_cutoff_date
    return result.loc[:, ALL_FEATURES + ("feature_cutoff_date",)]


def civil_support(frame, scope="train"):
    if scope not in ("train", "phase", "slice"):
        raise ValueError("Unknown support scope")
    values = finite(frame.loc[:, CIVIL])
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Unmasked binary civil values required")
    minimums = {
        **dict.fromkeys(MONTHS, 10 if scope == "slice" else 20),
        "month_end5": 20,
        "year_end5": 5,
        MEMORY: {"train": 20, "phase": 30, "slice": 15}[scope],
    }
    groups = {}
    for j, column in enumerate(CIVIL):
        ones = int(np.count_nonzero(values[:, j]))
        zeros = len(values) - ones
        groups[column] = {"ones": ones, "zeros": zeros, "minimum_per_class": minimums[column]}
        if min(ones, zeros) < minimums[column]:
            raise ValueError(
                "INSUFFICIENT_DATA: fixed civil support failed: " + scope + " " + column
            )
    return {"scope": scope, "n": len(frame), "groups": groups}


def civil_rank(frame):
    values = finite(frame.loc[:, CIVIL])
    if not np.isin(values, [0, 1]).all() or not np.array_equal(
        frame.const, np.ones(len(frame))
    ):
        raise ValueError("Unit intercept and binary civil geometry required")
    singular = np.linalg.svd(np.column_stack([np.ones(len(frame)), values]), compute_uv=False)
    if not len(singular) or not np.isfinite(singular).all() or singular[0] <= 0:
        raise ValueError("INSUFFICIENT_DATA: no finite civil geometry")
    threshold = float(singular[0]) * 1e-10
    rank = int((singular > threshold).sum())
    if rank != 15:
        raise ValueError("INSUFFICIENT_DATA: civil rank15 required")
    return {
        "columns": ["const", *CIVIL],
        "n": len(frame),
        "rank": rank,
        "relative_threshold": 1e-10,
        "threshold": threshold,
        "singular_values": singular.tolist(),
    }


def transform(training, application):
    civil_support(training)
    civil_rank(training)
    values = finite(training.loc[:, BASE])
    query = finite(application.loc[:, BASE])
    if not np.array_equal(values[:, 0], np.ones(len(training))) or not np.array_equal(
        query[:, 0], np.ones(len(application))
    ):
        raise ValueError("Literal unit intercept required")
    means = np.zeros(len(BASE))
    scales = np.ones(len(BASE))
    means[1:18] = training.loc[:, OLD[1:]].mean().to_numpy(float)
    scales[1:18] = training.loc[:, OLD[1:]].std(ddof=0).to_numpy(float)
    if (scales[1:18] <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: original component population scale")
    means[18:] = values[:, 18:].mean(axis=0)
    x = divide(values - means, scales)
    a = divide(query - means, scales)
    z = training[MEMORY].to_numpy(float) - float(training[MEMORY].mean())
    coefficients, _, rank, _ = np.linalg.lstsq(x, z, rcond=1e-12)
    residual = z - x @ coefficients
    norm, residual_norm = float(np.linalg.norm(z)), float(np.linalg.norm(residual))
    relative = float(divide(residual_norm, norm))
    if relative <= 1e-8:
        raise ValueError("INSUFFICIENT_DATA: quarter increment in baseline span")
    identification = {
        "columns": list(BASE),
        "ols_rcond": 1e-12,
        "baseline_rank": int(rank),
        "residual_norm": residual_norm,
        "centered_norm": norm,
        "relative_residual_norm": relative,
        "minimum_relative_norm": 1e-8,
    }
    return (
        x,
        a,
        {
            "columns": list(BASE),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "quarter_mean": float(training[MEMORY].mean()),
            "quarter_residual_relative_norm": relative,
            "identification": identification,
        },
    )


def positive_objective(beta, design, normalized_y):
    beta, design, q = finite(beta), finite(design), positive(normalized_y)
    if (
        design.ndim != 2
        or beta.shape != (design.shape[1],)
        or q.shape != (len(design),)
        or not len(design)
        or not np.array_equal(design[:, 0], np.ones(len(design)))
    ):
        raise ValueError("Aligned nonempty design with literal intercept required")
    eta = design_product(design, beta)
    ratio = exp_checked(np.log(q) - eta)
    slopes = beta.copy()
    slopes[0] = 0
    value = float(checked_mean(eta + ratio) + multiply(0.01, multiply(slopes, slopes).sum()))
    multiply(design, (1 - ratio)[:, None])
    gradient = divide(design.T @ (1 - ratio), len(q)) + multiply(0.02, slopes)
    hessian = divide(design.T @ multiply(design, ratio[:, None]), len(q))
    hessian += np.diag([0.0] + [0.02] * (len(beta) - 1))
    finite(value)
    finite(gradient)
    finite(hessian)
    return value, gradient, hessian


def scalar_objective(b, eta, q, z):
    b = float(finite(b))
    adjusted = finite(eta + multiply(b, z))
    ratio = exp_checked(np.log(positive(q)) - adjusted)
    value = float(checked_mean(adjusted + ratio) + multiply(0.01, multiply(b, b)))
    gradient = float(checked_mean(multiply(z, 1 - ratio)) + multiply(0.02, b))
    curvature = float(checked_mean(multiply(multiply(z, z), ratio)) + 0.02)
    finite([value, gradient, curvature])
    return value, gradient, curvature


def independent_fit(training, y, application):
    x, a, geometry = transform(training, application)
    y = positive(y)
    unit = float(positive(checked_mean(y)))
    q = divide(y, unit)
    start = np.zeros(len(BASE))
    solved = minimize(
        lambda beta: positive_objective(beta, x, q)[:2],
        start,
        method="trust-exact",
        jac=True,
        hess=lambda beta: positive_objective(beta, x, q)[2],
        options={"gtol": 1e-10, "maxiter": 500},
    )
    beta = finite(solved.x)
    value, gradient, _ = positive_objective(beta, x, q)
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-8 + 1e-12:
        raise AssertionError("Independent baseline stationarity failed")
    eta = design_product(x, beta)
    z = training[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    g0 = scalar_objective(0, eta, q, z)[1]
    radius = max(1.0, abs(g0) / 0.02)
    endpoints = [scalar_objective(b, eta, q, z)[1] for b in (-radius, radius)]
    if endpoints[0] > 0 or endpoints[1] < 0:
        raise AssertionError("Independent strictly-convex root bracket failed")
    b = float(
        brentq(
            lambda b: scalar_objective(b, eta, q, z)[1],
            -radius,
            radius,
            xtol=1e-12,
            rtol=1e-14,
            maxiter=200,
        )
    )
    quarter_value, quarter_gradient, curvature = scalar_objective(b, eta, q, z)
    if abs(quarter_gradient) > 1e-8 + 1e-12:
        raise AssertionError("Independent quarter stationarity failed")
    az = application[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    predictions = {
        "mean": np.repeat(unit, len(application)),
        "baseline": exp_checked(np.log(unit) + design_product(a, beta)),
        "quarter": exp_checked(np.log(unit) + design_product(a, beta) + multiply(b, az)),
    }
    return {
        "beta": beta,
        "b": b,
        "geometry": geometry,
        "train_mean": unit,
        "baseline_objective": value,
        "baseline_gradient_max_abs": maximum,
        "quarter_objective": quarter_value,
        "quarter_gradient": quarter_gradient,
        "quarter_curvature": curvature,
        "bracket": [-radius, radius],
        "bracket_gradients": endpoints,
        "predictions": predictions,
    }


def proper_score(y, h):
    y, h = positive(y), positive(h)
    if y.ndim != 1 or y.shape != h.shape or not len(y):
        raise ValueError("Exact nonempty aligned risk arrays required")
    result = np.log(h) + divide(y, h)
    return finite(result)


def paired_difference(a, b, y):
    a, b, y = positive(a), positive(b), positive(y)
    if a.ndim != 1 or a.shape != b.shape or a.shape != y.shape or not len(a):
        raise ValueError("Exact nonempty aligned paired risk arrays required")
    proper_score(y, a)
    proper_score(y, b)
    delta = divide(finite(a - b), np.maximum(a, b))
    ratio = divide(y, np.minimum(a, b))
    ratio_gap = -multiply(delta, ratio)
    with np.errstate(all="ignore"):
        log_gap = np.where(
            np.abs(delta) <= 0.5,
            np.where(delta >= 0, -np.log1p(-delta), np.log1p(delta)),
            np.log(a) - np.log(b),
        )
    result = finite(log_gap + ratio_gap)
    # Check representational coherence using each actual log/ratio term's ULP.
    direct = finite(proper_score(y, a) - proper_score(y, b))
    terms = [np.log(a), divide(y, a), np.log(b), divide(y, b), result, direct]
    bound = np.zeros_like(result)
    for term in terms:
        magnitude = np.abs(term)
        bound += 64 * np.maximum(
            np.finfo(float).eps * magnitude, magnitude - np.nextafter(magnitude, 0)
        )
    finite(bound)
    if (np.abs(result - direct) > bound).any():
        raise AssertionError("Paired proper score incoherence")
    return result


def normalized_equal(actual, expected, unit, label, *, independent=False):
    a = divide(positive(actual), positive(unit))
    b = divide(positive(expected), positive(unit))
    same(a, b, label, rtol=1e-7 if independent else 1e-10, atol=1e-6 if independent else 1e-12)


def verify_fit(training, y, application, audits):
    if set(audits) != set(MODELS):
        raise AssertionError("Every registered model audit required")
    independently = independent_fit(training, y, application)
    x, ax, geometry = transform(training, application)
    unit = independently["train_mean"]
    q = divide(positive(y), unit)
    for name, audit in audits.items():
        if audit["train_n"] != len(training) or audit["application_n"] != len(application):
            raise AssertionError("Training/application audit counts differ")
        same(
            audit["train_mean"], unit, "Shared exact target normalization", rtol=1e-12, atol=0
        )
        if name != "mean" and (
            audit["alpha"] != 0.01
            or not audit["start"] == [0.0] * (31 if name == "baseline" else 1)
            or not 0 <= audit["iterations"] < 200
            or audit["backtracks"] < 0
            or audit["numerical_trial_rejections"] < 0
            or not 0 <= float(finite(audit["gradient_max_abs"])) <= 1e-8
        ):
            raise AssertionError("Fixed optimizer and saved stationarity contract differs")
    mean = audits["mean"]
    if mean["columns"] != ["const"] or mean["gradient_max_abs"] != 0:
        raise AssertionError("Same-row arithmetic mean control differs")
    same(mean["beta"], [np.log(unit)], "Mean log-risk intercept", rtol=1e-12, atol=1e-12)
    base = audits["baseline"]
    if base["columns"] != list(BASE):
        raise AssertionError("Full31-column baseline required")
    same(
        base["means"], geometry["means"], "Training-only mixed centers", rtol=1e-12, atol=1e-14
    )
    same(
        base["scales"],
        geometry["scales"],
        "Old population/new fixed-one scales",
        rtol=1e-12,
        atol=1e-14,
    )
    beta = finite(base["scaled_beta"])
    if beta.shape != (31,):
        raise AssertionError("Full31 normalized coefficients required")
    native_beta = beta.copy()
    native_beta[0] += np.log(unit)
    same(
        base["beta"],
        native_beta,
        "Native/normalized intercept identity",
        rtol=1e-10,
        atol=1e-12,
    )
    value, gradient, _ = positive_objective(beta, x, q)
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-8 + 1e-12:
        raise AssertionError("Saved baseline fails independent stationarity")
    same(
        base["gradient"],
        gradient,
        "Independent saved baseline gradient",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        base["gradient_max_abs"],
        maximum,
        "Saved baseline full gradient",
        rtol=1e-4,
        atol=1e-12,
    )
    same(
        base["objective_scaled"],
        value,
        "Normalized penalized baseline objective",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        base["objective"],
        value + np.log(unit),
        "Native penalized baseline objective",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        beta,
        independently["beta"],
        "Separate trust-exact baseline optimum",
        rtol=1e-7,
        atol=1e-6,
    )
    quarter = audits["quarter"]
    if (
        quarter["columns"] != [MEMORY]
        or quarter["scale"] != 1
        or quarter["baseline_frozen"] is not True
    ):
        raise AssertionError("One fixed-scale slope on frozen baseline required")
    same(
        quarter["mean"],
        geometry["quarter_mean"],
        "Training-only quarter center",
        rtol=1e-12,
        atol=1e-14,
    )
    same_tree(quarter["support"], civil_support(training), "Independent train civil support")
    same_tree(quarter["civil_rank"], civil_rank(training), "Independent rank15 geometry")
    same_tree(
        quarter["identification"], geometry["identification"], "Independent novelty residual"
    )
    eta = design_product(x, beta)
    z = training[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    b = float(finite(quarter["b"]))
    value, gradient, curvature = scalar_objective(b, eta, q, z)
    g0 = scalar_objective(0.0, eta, q, z)[1]
    radius = max(1.0, abs(g0) / 0.02)
    endpoints = [scalar_objective(v, eta, q, z)[1] for v in (-radius, radius)]
    if (
        endpoints[0] > 0
        or endpoints[1] < 0
        or abs(gradient) > 1e-8 + 1e-12
        or curvature < 0.02
    ):
        raise AssertionError("Saved scalar strict-convex certificate failed")
    same(
        quarter["bracket"],
        [-radius, radius],
        "Deterministic derivative bracket",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["bracket_gradients"],
        endpoints,
        "Independent endpoint derivative signs",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["initial_gradient"],
        g0,
        "Independent zero-slope gradient",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["gradient"],
        [gradient],
        "Independent saved scalar gradient",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["gradient_max_abs"],
        abs(gradient),
        "Saved scalar stationarity",
        rtol=1e-4,
        atol=1e-12,
    )
    same(
        quarter["curvature"],
        curvature,
        "Independent positive scalar curvature",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["objective_scaled"],
        value,
        "Normalized scalar objective",
        rtol=1e-10,
        atol=1e-12,
    )
    same(
        quarter["objective"],
        value + np.log(unit),
        "Native scalar objective",
        rtol=1e-10,
        atol=1e-12,
    )
    # This root uses the saved and checked baseline, isolating scalar tolerance.
    separate_b = brentq(
        lambda v: scalar_objective(v, eta, q, z)[1],
        -radius,
        radius,
        xtol=1e-12,
        rtol=1e-14,
        maxiter=200,
    )
    same(b, separate_b, "Separate brent scalar optimum", rtol=1e-7, atol=1e-6)
    az = application[MEMORY].to_numpy(float) - geometry["quarter_mean"]
    replay = {
        "mean": np.repeat(unit, len(application)),
        "baseline": exp_checked(np.log(unit) + design_product(ax, beta)),
        "quarter": exp_checked(np.log(unit) + design_product(ax, beta) + multiply(b, az)),
    }
    for name in MODELS:
        normalized_equal(
            replay[name],
            independently["predictions"][name],
            unit,
            "Independent normalized " + name + " predictions",
            independent=True,
        )
    return replay, {
        "models_verified": 3,
        "producer_baseline_gradient": maximum,
        "independent_baseline_gradient": independently["baseline_gradient_max_abs"],
        "producer_quarter_gradient": abs(gradient),
        "independent_quarter_gradient": abs(scalar_objective(separate_b, eta, q, z)[1]),
    }


PANEL_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "target_end",
    "available_date",
    "y",
    "prediction",
    "fit_origin",
    "fit_cutoff_date",
    "train_n",
    "train_last_target",
    "train_last_available",
    "phase",
)


def eligible_entries(features, targets, section):
    if (
        tuple(features.columns) != ALL_FEATURES + ("feature_cutoff_date",)
        or tuple(targets.columns) != ("y", "target_end", "available_date")
        or not features.index.equals(targets.index)
    ):
        raise ValueError("Exact common full-calendar feature/target schema required")
    calendar = pd.DatetimeIndex(features.index)
    civil_features(calendar)
    dates = pd.Series(calendar, index=calendar)
    for frame, column, shift in (
        (features, "feature_cutoff_date", 1),
        (targets, "target_end", -1),
        (targets, "available_date", -1),
    ):
        if not frame[column].equals(dates.shift(shift).rename(column)):
            raise ValueError("Exact one-step full-reference timing required")
    known = targets.y.notna()
    positive(targets.loc[known, "y"])
    if (known & targets.available_date.isna()).any():
        raise ValueError("Risk observations require explicit maturity")
    complete = (
        pd.Series(np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1), index=calendar)
        & features.feature_cutoff_date.notna()
    )
    phase_union = np.zeros(len(calendar), dtype=bool)
    for phase in ("development", "evaluation"):
        first, last = section[phase]
        phase_union |= (calendar >= first) & (calendar <= last)
    entries = calendar[
        complete
        & phase_union
        & (calendar >= section["origin_start"])
        & (calendar <= section["origin_end"])
    ]
    label = (
        known
        & (targets.available_date <= section["latest_target"])
        & (
            (calendar > section["development"][1])
            | (targets.available_date <= section["development_target_available_by"])
        )
    )
    return entries, entries[label.loc[entries]], complete


def validate_panel(panel):
    if (
        tuple(panel.columns) != PANEL_COLUMNS
        or set(panel.model) != set(MODELS)
        or not panel.horizon.eq(1).all()
        or panel.duplicated(["origin", "model"]).any()
    ):
        raise AssertionError("Exact three-model proper-risk panel required")
    positive(panel.y)
    positive(panel.prediction)
    for name in set(PANEL_COLUMNS) - {
        "y",
        "prediction",
        "model",
        "horizon",
        "train_n",
        "phase",
    }:
        if panel[name].isna().any():
            raise AssertionError("Explicit finite timing metadata required")
    cohorts = {
        name: panel.loc[panel.model == name].set_index("origin").sort_index()
        for name in MODELS
    }
    base = cohorts["baseline"]
    for name, part in cohorts.items():
        if not part.index.equals(base.index):
            raise AssertionError("Exact common model origins required")
        for column in set(base.columns) - {"model", "prediction"}:
            if not part[column].equals(base[column]):
                raise AssertionError(
                    "Exact paired target/metadata required: " + name + " " + column
                )


def verify_forecasts(features, targets, panel, fits, protocol):
    validate_panel(panel)
    section = protocol["index"]
    entries, scored, complete = eligible_entries(features, targets, section)
    for name in MODELS:
        rows = panel.loc[panel.model == name].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError("Exact common scored cohort differs")
        same(rows.y, targets.loc[scored, "y"], "Saved unchanged risk target", rtol=0, atol=0)
        for column in ("available_date", "target_end"):
            if not np.array_equal(rows[column], targets.loc[scored, column]):
                raise AssertionError("Target maturity differs")
        if not np.array_equal(
            rows.feature_cutoff_date, features.loc[scored, "feature_cutoff_date"]
        ):
            raise AssertionError("Prior-session market cutoff differs")
        if not np.array_equal(
            rows.phase,
            np.where(scored <= section["development"][1], "development", "evaluation"),
        ):
            raise AssertionError("Exact fixed phases differ")
    fit_origins = entries[~entries.to_period("M").duplicated()]
    if len(fits) != len(fit_origins) or [
        pd.Timestamp(record["fit_origin"]) for record in fits
    ] != list(fit_origins):
        raise AssertionError(
            "All monthly feature-admitted fits required, including unscored months"
        )
    maxima = {
        "producer_baseline_gradient": 0.0,
        "independent_baseline_gradient": 0.0,
        "producer_quarter_gradient": 0.0,
        "independent_quarter_gradient": 0.0,
    }
    train_total = 0
    for entry, record in zip(fit_origins, fits, strict=True):
        applications = entries[entries.to_period("M") == entry.to_period("M")]
        query = scored[scored.to_period("M") == entry.to_period("M")]
        cutoff = features.loc[entry, "feature_cutoff_date"]
        mask = (
            complete
            & targets.y.notna()
            & (features.index < entry)
            & (targets.available_date <= cutoff)
        )
        origins = features.index[mask]
        if (
            len(origins) < section["minimum_train"]
            or record["train_n"] != len(origins)
            or record["application_n"] != len(applications)
        ):
            raise AssertionError("Exact mature common training/application counts differ")
        expected = {
            "fit_cutoff_date": cutoff,
            "train_first_origin": origins[0],
            "train_last_origin": origins[-1],
            "train_last_target": targets.loc[mask, "target_end"].max(),
            "train_last_available": targets.loc[mask, "available_date"].max(),
        }
        if any(pd.Timestamp(record[key]) != value for key, value in expected.items()):
            raise AssertionError("Exact mature training date evidence differs")
        replay, proof = verify_fit(
            features.loc[mask],
            targets.loc[mask, "y"],
            features.loc[applications],
            record["model_audit"],
        )
        train_total += len(origins)
        for key in maxima:
            maxima[key] = max(maxima[key], proof[key])
        for name in MODELS:
            rows = panel.loc[panel.model.eq(name) & panel.origin.isin(query)].sort_values(
                "origin"
            )
            expected_predictions = replay[name][applications.get_indexer(query)]
            normalized_equal(
                rows.prediction,
                expected_predictions,
                record["model_audit"][name]["train_mean"],
                "Strict saved native forecast replay in normalized units",
            )
            if not rows.fit_origin.eq(entry).all() or not rows.train_n.eq(len(origins)).all():
                raise AssertionError("Issued monthly fit identity differs")
            for key in ("fit_cutoff_date", "train_last_target", "train_last_available"):
                if not rows[key].eq(expected[key]).all():
                    raise AssertionError("Issued causal fit evidence differs")
    return {
        "forecasts_verified": len(panel),
        "monthly_fits_verified": len(fits),
        "common_scored_origins": len(scored),
        "common_application_origins": len(entries),
        "training_observation_instances_verified": train_total,
        "independent_baseline_fits_verified": len(fits),
        "independent_scalar_fits_verified": len(fits),
        **maxima,
    }


def preflight(features, targets, protocol):
    section = protocol["index"]
    entries, scored, complete = eligible_entries(features, targets, section)
    if not len(entries):
        raise ValueError("INSUFFICIENT_DATA: no complete application origins")
    rows = []
    for entry in entries[~entries.to_period("M").duplicated()]:
        cutoff = features.loc[entry, "feature_cutoff_date"]
        mask = (
            complete
            & targets.y.notna()
            & (features.index < entry)
            & (targets.available_date <= cutoff)
        )
        origins = features.index[mask]
        if len(origins) < section["minimum_train"]:
            raise ValueError("INSUFFICIENT_DATA: fixed mature training minimum")
        training = features.loc[mask]
        application = entries[entries.to_period("M") == entry.to_period("M")]
        _, _, geometry = transform(training, features.loc[application])
        rows.append(
            {
                "fit_origin": str(entry.date()),
                "fit_cutoff_date": str(cutoff.date()),
                "train_n": len(origins),
                "train_first_origin": str(origins[0].date()),
                "train_last_origin": str(origins[-1].date()),
                "train_last_target": str(targets.loc[mask, "target_end"].max().date()),
                "train_last_available": str(targets.loc[mask, "available_date"].max().date()),
                "application_n": len(application),
                "support": civil_support(training),
                "civil_rank": civil_rank(training),
                "identification": geometry["identification"],
            }
        )
    phases = []
    for name in ("development", "evaluation"):
        first, last = section[name]
        one = scored[(scored >= first) & (scored <= last)]
        if len(one) < 127:
            raise ValueError("INSUFFICIENT_DATA: fixed phase bandwidth")
        phase = {
            "name": name,
            "n": len(one),
            "first_origin": str(one[0].date()),
            "last_origin": str(one[-1].date()),
            "support": civil_support(features.loc[one], "phase"),
            "slices": [],
        }
        if name == "evaluation":
            for start, end in section["evaluation_stability"]:
                sliced = one[(one >= start) & (one <= end)]
                phase["slices"].append(
                    {
                        "start": start,
                        "end": end,
                        "n": len(sliced),
                        "support": civil_support(features.loc[sliced], "slice"),
                    }
                )
        phases.append(phase)
    return {
        "common_application_origins": len(entries),
        "common_scored_origins": len(scored),
        "monthly_fits": len(rows),
        "fits": rows,
        "phases": phases,
    }


PREVIOUS_LIMITATIONS = [
    "One fixed pool of an unchanged conditional probability and a causal unconditional event-rate filter; no fitted calibration coefficient or new conditional feature.",
    "The initial mass summarizes only the first original eligible training subset; subsequent labels use the full calendar and mature target table, including previously unscored origins.",
    "Adaptively selected exploratory comparison on reused archival QQQ ETF/SPX price-index history; preserved multiplicity and causal replay do not provide a new untouched validation sample.",
    "Strict raw directional agreement, not volatility magnitude, covariance, executable direction or profit; calibration in the large and nominal detectable effect remain descriptive.",
]


def reconstruct_previous(root, info, pins):
    root = Path(root)
    oldp = yaml.safe_load(read_snapshot(root, info["protocol"], info["protocol_sha256"]))
    previous.validate_protocol(oldp)
    oldm = read_json_snapshot(
        root, info["reports"] + "/manifest.json", info["manifest_sha256"]
    )

    def saved_json(name):
        return read_json_snapshot(root, name, pins[name])

    def saved_parquet(name):
        return scalar.read_issued_snapshot(root, name, pins[name])

    nested = previous.validate_upstream(root)
    same_tree(
        saved_json(info["data"] + "/upstream_admission.json"),
        nested,
        "Complete previous original admission",
    )
    inputs = {
        key: saved_parquet(oldp["upstream"][key])
        for key in ("features", "targets", "forecasts")
    }
    fits = saved_json(oldp["upstream"]["fits"])
    panel = saved_parquet(info["data"] + "/forecasts.parquet")
    states = saved_parquet(info["data"] + "/states.parquet")
    metrics = saved_json(info["reports"] + "/metrics.json")
    if (
        metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != info["protocol_sha256"]
        or metrics["evidence_class"] != oldp["evidence_class"]
    ):
        raise AssertionError("Previous successful metric identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": nested["status"], "prior_files_written": False},
        "forecast_reconstruction": previous.verify_forecasts(
            inputs["features"],
            inputs["targets"],
            inputs["forecasts"],
            fits,
            panel,
            states,
            oldp,
        ),
        "primitive_brier_scores_verified": len(panel),
        "paired_brier_differences_verified": int(panel.origin.nunique()) * 2,
        "inference": previous.verify_metrics(
            root, panel, oldp, metrics, len(states), admitted_inputs=oldm["inputs"]
        ),
        "ledger_events_verified": previous.verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=pins[info["reports"] + "/trial_ledger.jsonl"],
        ),
        "protocol_sha256": info["protocol_sha256"],
        "verifier_sha256": info["verifier_sha256"],
        "limitations": PREVIOUS_LIMITATIONS,
    }
    return result, nested


def restore_documentary_paths(value, staged, root):
    if isinstance(value, list):
        return [restore_documentary_paths(item, staged, root) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if (
                key == "source_path"
                and isinstance(item, str)
                and item.startswith(str(staged) + "/")
            ):
                result[key] = str(root) + item[len(str(staged)) :]
            else:
                result[key] = restore_documentary_paths(item, staged, root)
        return result
    return value


def reconstruct_calendar(root, info, pins):
    root = Path(root)
    oldp = yaml.safe_load(read_snapshot(root, info["protocol"], info["protocol_sha256"]))
    calendar_old.validate_protocol(oldp)
    oldm = read_json_snapshot(
        root, info["reports"] + "/manifest.json", info["manifest_sha256"]
    )

    def saved_json(name):
        return read_json_snapshot(root, name, pins[name])

    def saved_parquet(name):
        return scalar.read_issued_snapshot(root, name, pins[name])

    with TemporaryDirectory(prefix="civil-quarter-old-source-") as directory:
        staged = Path(directory)
        staged_names = set(oldm["inputs"]) | set(oldp["comparisons"]["inherited_sources"])
        for name in staged_names:
            destination = staged / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise AssertionError("Only relative source paths may be staged")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(read_snapshot(root, name, pins[name]))
        features, targets, audit = calendar_old.reconstruct(staged, oldp)
        audit = restore_documentary_paths(audit, staged, root)
        table_equal(
            saved_parquet(info["data"] + "/features.parquet"),
            features,
            OLD,
            "Entire frozen wave8 feature table",
        )
        table_equal(
            saved_parquet(info["data"] + "/targets.parquet"),
            targets,
            ("y",),
            "Entire frozen wave8 risk target",
        )
        same_tree(
            saved_json(info["data"] + "/source_audit.json"),
            audit,
            "Original source and macro-known masks",
        )
        panel = saved_parquet(info["data"] + "/forecasts.parquet")
        fits = saved_json(info["data"] + "/fits.json")
        metrics = saved_json(info["reports"] + "/metrics.json")
        if (
            metrics.get("status") == "UNEVALUABLE"
            or metrics["protocol_sha256"] != info["protocol_sha256"]
            or metrics["evidence_class"] != oldp["evidence_class"]
        ):
            raise AssertionError("Original calendar successful metric identity differs")
        result = {
            "status": "VERIFIED",
            "protocol_sha256": info["protocol_sha256"],
            "verifier_sha256": info["verifier_sha256"],
            "raw_feature_rows_verified": len(features),
            "raw_feature_columns_verified": len(OLD),
            "source_availability_rows_verified": len(audit["plan_availability"]),
            "sources": audit["plans"],
            "forecast_reconstruction": calendar_old.verify_forecasts(
                features, targets, panel, fits, oldp
            ),
            "inference": calendar_old.verify_metrics(staged, panel, oldp, metrics),
        }
    ledger_name = info["reports"] + "/trial_ledger.jsonl"
    ledger = [
        json.loads(line)
        for line in read_snapshot(root, ledger_name, pins[ledger_name]).splitlines()
        if line
    ]
    for event, expected in (
        ("inherited", metrics["inherited_rows"]),
        ("evaluated", metrics["rows"]),
    ):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        same_tree(actual, expected, "Original calendar " + event + " ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 110
        or len(registered) != 2
        or [(row["candidate"], row["control"]) for row in registered]
        != list(calendar_old.COMPARISONS)
        or any(
            row["protocol_sha256"] != info["protocol_sha256"]
            or row["horizon"] != 1
            or row["score"] != "qlike"
            for row in registered
        )
    ):
        raise AssertionError("Complete original wave8 ledger identity differs")
    result["ledger_events_verified"] = {"inherited": 106, "registered": 2, "evaluated": 2}
    result["limitations"] = [
        "Original-plan source and timing replication of a previously tested broad calendar mechanism.",
        "Nominal civil windows ignore holidays and early closes; target maturity alone uses future observed sessions.",
        "Historical reuse, archival revisions, incomplete original-plan coverage and back-calculated VIX9D remain exploratory limitations.",
    ]
    return result


def validate_upstream(root=ROOT):
    root = Path(root)
    protocol_payload = (root / "civil_quarter.yaml").read_bytes()
    protocol = yaml.safe_load(protocol_payload)
    manifest_payload = (root / "reports/civil_quarter/manifest.json").read_bytes()
    captured = json.loads(manifest_payload)
    if captured["protocol_sha256"] != sha256(protocol_payload).hexdigest():
        raise AssertionError("New manifest must precede both read-only admissions")
    pins = {}
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
        for name, signature in captured[group].items():
            if name in pins and pins[name] != signature:
                raise AssertionError("Conflicting current artifact pins")
            pins[name] = signature

    def inventory(info):
        return {info["protocol"]} | {
            str(path.relative_to(root))
            for folder in (info["reports"], info["data"])
            for path in (root / folder).rglob("*")
            if path.is_file()
        }

    proofs = {}
    for key, module, reconstruct in (
        ("upstream", "causal_pool", reconstruct_previous),
        ("feature_source", "calendar_variance", reconstruct_calendar),
    ):
        info = protocol[key]
        required = inventory(info)
        if (
            not required.issubset(captured["inputs"])
            or (root / info["reports"] / "failure.json").exists()
        ):
            raise AssertionError(
                "All original successful output/publication bytes must be pinned"
            )
        read_snapshot(root, info["protocol"], info["protocol_sha256"])
        oldm = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        record = read_json_snapshot(
            root, info["reports"] + "/verification.json", info["verification_sha256"]
        )
        if (
            info["required_status"] != "VERIFIED"
            or record["status"] != "VERIFIED"
            or record["protocol_sha256"] != info["protocol_sha256"]
            or record["verifier_sha256"] != info["verifier_sha256"]
            or oldm["protocol_sha256"] != info["protocol_sha256"]
            or oldm["code"]["src/verify_" + module + ".py"] != info["verifier_sha256"]
        ):
            raise AssertionError("Original protocol/manifest/verifier identity differs")
        if not set(oldm["code"]).issubset(captured["code"]) or not set(
            oldm["inputs"]
        ).issubset(captured["inputs"]):
            raise AssertionError("Every original code/input pin must be registered")
        for group in ("code", "inputs", "preserved"):
            scalar._pins(root, oldm[group])
            if any(pins.get(name) != signature for name, signature in oldm[group].items()):
                raise AssertionError("Original preserved pin missing from new registration")
        rebuilt = reconstruct(root, info, pins)
        if key == "upstream":
            rebuilt, nested = rebuilt
        else:
            nested = None
        compare_tree(record, rebuilt, "Entire original " + module + " VERIFIED record")
        if inventory(info) != required:
            raise AssertionError("Original artifact inventory changed during admission")
        proofs[key] = {
            "original_verification": rebuilt,
            "previous_manifest_entries_verified": sum(
                len(oldm[g]) for g in ("code", "inputs", "preserved")
            ),
            "pinned_previous_artifacts": len(required),
            "protocol_sha256": info["protocol_sha256"],
            "manifest_sha256": info["manifest_sha256"],
            "verification_sha256": info["verification_sha256"],
            **({"nested_admission": nested} if nested is not None else {}),
        }
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, captured[group])
    if (
        digest(root / "civil_quarter.yaml") != sha256(protocol_payload).hexdigest()
        or digest(root / "reports/civil_quarter/manifest.json")
        != sha256(manifest_payload).hexdigest()
    ):
        raise AssertionError("Current protocol/manifest changed during admission")
    return {
        "status": "UPSTREAM_VERIFIED_READ_ONLY",
        "prior_files_written": False,
        "input_hashes": captured["inputs"],
        "proofs": proofs,
    }


def invalidate_publication(root, error):
    report = Path(root) / "reports/civil_quarter"
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
        "cumulative_hypothesis_count": 123,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "civil_quarter",
                "candidate": candidate,
                "control": control,
                "score": "proper_variance",
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
        "# SPX civil quarter-end risk increment\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
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


def inherited_rows(root, protocol, *, pins=None):
    root = Path(root)
    if pins is None:
        pins = json.loads((root / "reports/civil_quarter/manifest.json").read_bytes())[
            "inputs"
        ]
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
    if len(output) != 121:
        raise AssertionError("All121 inherited comparisons must remain identified")
    return output


def phase_statistics(panel, features, control, phase, code, protocol):
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
    rows, benchmark = groups["quarter"], groups[control]
    if any(not part.index.equals(rows.index) for part in groups.values()) or len(rows) < 127:
        raise AssertionError(
            "Complete original cohort and at least127 phase observations required"
        )
    difference = paired_difference(rows.prediction, benchmark.prediction, rows.y)
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
        "civil_support": civil_support(features.loc[rows.index], "phase"),
        "delta": delta,
        "candidate_loss": _finite(proper_score(rows.y, rows.prediction).mean()),
        "control_loss": _finite(proper_score(benchmark.y, benchmark.prediction).mean()),
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
    report = Path(root) / "reports/civil_quarter"
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
        len(ledger) != 125
        or len(registered) != 2
        or len(prior) != 121
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["study"] != "civil_quarter"
            or row["horizon"] != 1
            or row["score"] != "proper_variance"
            for row in registered
        )
    ):
        raise AssertionError("Complete121inherited+2registered+2terminal ledger required")
    return {"inherited": 121, "registered": 2, final_event: 2}


normalized_difference = previous.normalized_difference
_finite = scalar._finite


CONTRACT = {
    "study_id": "civil_quarter_wave15",
    "specified_on": "2026-09-07",
    "status": "specified_before_new_civil_features_counts_or_fits",
    "wave": 15,
    "wave_alpha": 0.00020833333333333335,
    "objective": "Test one civil quarter-end SPX risk increment beyond market, original macro-plan, "
    "month-end, December year-end and additive annual calendar controls",
    "evidence_class": "exploratory_reused_history_archival_SPX_daily_risk_civil_calendar_interaction",
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
        "models": ["mean", "baseline", "quarter"],
        "baseline": [
            "const",
            "lrv_d",
            "lrv_w",
            "lrv_m",
            "neg_d",
            "neg_w",
            "neg_m",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "nominal_hours",
            "cpi_plan",
            "nfp_plan",
            "fomc_plan",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
        ],
        "all_features": [
            "const",
            "lrv_d",
            "lrv_w",
            "lrv_m",
            "neg_d",
            "neg_w",
            "neg_m",
            "lvix",
            "term",
            "lvvix",
            "entry_dow_1",
            "entry_dow_2",
            "entry_dow_3",
            "entry_dow_4",
            "nominal_hours",
            "cpi_plan",
            "nfp_plan",
            "fomc_plan",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
            "quarter_end5",
        ],
    },
    "upstream": {
        "protocol": "causal_pool.yaml",
        "reports": "reports/causal_pool",
        "data": "data/causal_pool",
        "protocol_sha256": "1e329ee208612ed8db1fede873b92a938820d51a3fcf2ec583c813d37ac000ad",
        "manifest_sha256": "a775d4491b3dab27f2c36c037090d37a094dfad85c5ff67256acfa262d40f4c2",
        "verification_sha256": "0f2142d97e62e4ab1dbfc02c6a27bdf23dc02676bdd43bc41fa64e180a2d1010",
        "verifier_sha256": "cb34da06786f61e50b89f494e4071abb8d250539a52145d782cc1e122aeaa900",
        "required_status": "VERIFIED",
    },
    "feature_source": {
        "protocol": "calendar_variance.yaml",
        "reports": "reports/calendar_variance",
        "data": "data/calendar_variance",
        "protocol_sha256": "b3357641d874d22eb6bf0903049e8dcfd93df4e38b0748f717923447ff367ca3",
        "manifest_sha256": "521deb0c053c3eb724f2cd84c4644688863ec263ecce62d00c8ea129580bb4c8",
        "verification_sha256": "47f2a0a333df19d258ca95dfef9fbf403ee0ae6f6a675ec85903755b84943cb7",
        "verifier_sha256": "c213b47e91c25c7293a557064dbf8c62c583fc32bf58701cbf2f5afa61f78677",
        "required_status": "VERIFIED",
        "features": "data/calendar_variance/features.parquet",
        "targets": "data/calendar_variance/targets.parquet",
    },
    "prospectus": {
        "path": "reports/causal_pool/NEXT_CALENDAR_VARIANCE_DESIGN.md",
        "sha256": "0933f511c14df8c5bdb32382663b1472d5e09d7678158ca67f455edfb1e7254f",
    },
    "source_contract": {
        "admission": "Reconstruct complete original wave14 VERIFIED record and separately "
        "complete wave8 VERIFIED record through frozen read-only functions; "
        "historical inventories anchored by original manifest hashes, never "
        "expanded-current-tree old coverage",
        "immutable_read": "Read and hash each admitted parquet/JSON input once before "
        "decoding; inherited metrics and ledgers use same checked "
        "snapshots. For old wave8 raw reconstruction stage all its "
        "registered input bytes into a temporary same-relative-path "
        "tree, normalize only documentary source_path prefixes back to "
        "originalroot, and preserve old parser/gates",
        "sources": {
            "daily": "data/research_paths/spx_daily.parquet",
            "vix": "data/free_sources/raw/cboe/VIX_History.csv",
            "vix9d": "data/free_sources/raw/cboe/VIX9D_History.csv",
            "vvix": "data/free_sources/raw/cboe/VVIX_History.csv",
            "cpi": "data/source_discovery/macro_plans/cpi/ledger.json",
            "nfp": "data/source_discovery/macro_plans/nfp/ledger_verified.json",
            "fomc": "data/source_discovery/macro_plans/calendar/fomc_original_annual_plans.csv",
            "fomc_coverage": "data/source_discovery/macro_plans/calendar/fomc_annual_coverage.json",
        },
        "calendar": {
            "source_publication_start": "2010-01-01",
            "source_publication_end": "2025-10-20",
            "bls_policy": "next plan printed in preceding monthly release; "
            "original planned dates retained even if later canceled "
            "or changed",
            "bls_revision_policy": "current archives with explicit this-release "
            "reissue notices remain unadmitted for both "
            "CPI and payroll; retain original source "
            "ledger and separate pre-fit admission "
            "correction",
            "source_eligibility": "publication civil date strictly before "
            "preceding observed market-session date",
            "nominal_start": "entry civil date1600America/New_York",
            "nominal_end": "next Monday-through-Friday civil "
            "date1600America/New_York; skip weekends only",
            "duration": "nominal endpoint UTC difference in hours; includes DST "
            "elapsed-time change",
            "cpi": "eligible original CPI planned timestamp falls in open-left "
            "closed-right nominal window",
            "nfp": "eligible original payroll planned timestamp falls in "
            "open-left closed-right nominal window",
            "fomc": "eligible original annual meeting final DATE equals nominal "
            "ending date; date-only timing control, no invented statement "
            "time",
            "fomc_annual": "exactly8original planned meetings per year from "
            "selected full annual announcement published before "
            "covered year",
            "coverage": "each nominal-window calendar month requires its eligible "
            "explicit original BLS plan; full eligible annual FOMC "
            "plan required",
            "missing": "pending retrieval blocks execution; intrinsically absent "
            "or ambiguous source statements remain unknown and enter "
            "all-arm complete-sample mask; no date/year inference or "
            "zero fill",
            "limitations": "nominal window ignores holidays and early closes; "
            "neither actual nor historically planned exchange "
            "holding interval; no actual next market date enters a "
            "predictor",
            "provenance": "exact currently captured official-document tool text "
            "and extraction hashes; raw provider bytes and "
            "immutable historical web vintages unverified",
            "replication": "Legacy repository calendars already forecast "
            "next-session variance; this tests original-plan "
            "admission with stronger matched SPX controls, not a "
            "first calendar mechanism",
        },
        "measurement": "Exact frozen wave8 daily SPX target/market transformations; "
        "max(GK,1e-10) is part of that preexisting risk proxy, not the "
        "later strict paired-asset raw-GK gate",
        "missing": "Keep full bounded SPX reference calendar, strict old rolling history "
        "and exact original-plan known-month masks; no zero filling, removed "
        "macro controls, holiday repair or next-common-date labels",
        "limitations": "Reused Yahoo/Cboe archival values, unverified historical "
        "publication latency and revisions, original-plan documentary "
        "coverage, nominal civil windows and early back-calculated VIX9D "
        "persist; no source acquisition or institutional mechanism claim",
    },
    "target": {
        "formula": "max(Garman-Klass[next],1e-10)+log(raw_open[next]/raw_close[entry])^2",
        "availability": "Next actual observed SPX close, target_end=available_date; strictly "
        "positive finite known risk labels; missing stays unknown",
        "interpretation": "Native squared-log-return daily OHLC full-session risk proxy, not "
        "measured high-frequency integrated variance",
        "cohort": "All32complete predictors and exact common mature labels across everymodel; "
        "monthly firstfeaturecomplete application before futurequerylabelmask; no "
        "event-only evaluation or dropped unscored month",
    },
    "civil": {
        "month_columns": [
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
        ],
        "nuisance_columns": [
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "month_end5",
            "year_end5",
        ],
        "candidate": "quarter_end5",
        "nominal_date": "First Monday-through-Friday civil date strictly after origin; skip weekends "
        "only; no future observed prices/holidays/earlycloses",
        "month_end5": "Nominal date lies within final5 civil dates of its Gregorian month, inclusive",
        "year_end5": "month_end5 times nominal December indicator",
        "quarter_end5": "month_end5 times nominal month in March,June,September; December belongs to "
        "year-end control",
        "seasonality": "Eleven nominal-month dummies omitting January",
        "training": "Original18 retain old geometry; all13 new nuisance civil columns and candidate "
        "train-centered on exact admitted rows with fixedscale1; query means unchanged",
        "constant": "Exact all-equal new civil values center exactly zero in mathematical "
        "primitives; whole-wave scientific support and rank gates still reject "
        "unsupported empirical folds",
    },
    "support": {
        "minimum_train": 1000,
        "minimum_phase_observations": 127,
        "per_class": {
            "train": {"quarter_end5": 20, "month_end5": 20, "year_end5": 5, "month_dummy": 20},
            "phase": {"quarter_end5": 30, "month_end5": 20, "year_end5": 5, "month_dummy": 20},
            "slice": {"quarter_end5": 15, "month_end5": 20, "year_end5": 5, "month_dummy": 10},
        },
        "civil_rank": "Everytraining [const,11month,E5,Y5,G5] block must have rank15 by "
        "singularvalues>1e-10*largest; no dropping or recoding",
        "civil_rank_relative_tolerance": 1e-10,
        "novelty": "Regress centeredG5 on exact transformedBASE31 using lstsq rcond1e-12; "
        "normresidual/normcenteredG5 must exceed1e-8; fullmarket rank not required "
        "because slopes regularized",
        "novelty_lstsq_rcond": 1e-12,
        "minimum_relative_residual_norm": 1e-08,
        "timing": "Audit everymonthly training group/rank/novelty and allphase/slice groups on "
        "exact commoncohorts before any optimization; any failure aborts both "
        "hypotheses, no support-dependent deletion",
    },
    "fitting": {
        "penalty": 0.01,
        "old_scale_minimum": 1e-12,
        "new_civil_scale": 1.0,
        "baseline": "31-column normalized mean eta+exp(log(q/meanq)-eta) plus.01squaredslopes; "
        "unpenalizedintercept, old17populationmean/std, new13fixedscale1center",
        "normalization": "Trainmean=arithmetic mean of positivey; qscaled=y/trainmean; "
        "nativeforecast exp(log(trainmean)+design@scaled_beta)",
        "quarter": "Freeze baselineeta; z=G5-trainmeanG5; fitonly b via "
        "mean(eta0+bz+exp(logqscaled-eta0-bz))+.01b²; no extra intercept or jointrefit",
        "mean": "Exact arithmetic training mean on sameall32featurecomplete mature rows",
        "optimizer": "Deterministic Newton from allzero baselinecoefficients and quarterb0; "
        "atmost200states and60Armijo halvings perstep, armijo1e-4; accepted "
        "fullgradientmax<=1e-8",
        "maximum_iterations": 200,
        "maximum_backtracks": 60,
        "armijo": 0.0001,
        "gradient_tolerance": 1e-08,
        "scalar_bracket": "R=max(1,abs(initial_scalar_gradient)/.02); audit finite gradients at "
        "-R,+R enclosing zero; unique optimum follows curvature>=.02",
        "arithmetic": "Strictrealfinite designs/parameters/states/predictions; "
        "positivey/mean/scaledtargets/ratio/forecast; reject unsupported "
        "nonzeroproduct,quotient,exp underflow. Tentative invalid objectives may be "
        "rejected within the fixed Armijo schedule; accepted iterates and published "
        "forecasts cannot be repaired/clipped/restarted",
    },
    "scoring": {
        "loss": "proper_variance",
        "formula": "log(h)+y/h",
        "effect_threshold_absolute": 0.005,
        "paired_difference": "d=(hA-hB)/max(hA,hB); gap=logratio-d*(y/min(hA,hB)). For abs(d)<=.5 "
        "logratio=-log1p(-d) ifd>=0 else log1p(d); otherwise log(hA)-log(hB)",
        "arithmetic": "First validate every positive real input and finite individual score, "
        "reject unsupported quotient/product underflow. Coherence allowance64 times "
        "sum of unit(x) for loghA,y/hA,loghB,y/hB,stablegap,directgap; "
        "unit=max(eps*abs(x),abs(x)-nextafter(abs(x),0)), no unit floor",
        "effect_reference": "Absolute mean natural-log proper-score decrease.005 bothphases, not "
        "percentage of potentiallynegative rawscore or profit",
    },
    "comparisons": {
        "new_hypotheses": 2,
        "inherited_hypotheses": 121,
        "cumulative_hypotheses": 123,
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
        ],
        "controls": ["baseline", "mean"],
        "contrasts": [
            ["quarter", "baseline", "proper_variance"],
            ["quarter", "mean", "proper_variance"],
        ],
        "candidate_gate": "Both contrasts must pass "
        "everyfixedphase/effect/stability/multiplicity gate",
    },
    "inference": {
        "blocks": [21, 63, 126],
        "hac_lags": 126,
        "minimum_phase_observations": 127,
        "bootstrap_draws": 99999,
        "seed": 20260921,
        "inference_scale": 1.0,
        "seed_rule": "seed+phase_code*10000+block; development0,evaluation1",
        "p_phase": "Maximum two-sided centered-null circular-block bootstrap and BartlettHAC126 "
        "p",
        "p_hypothesis": "Maximum development/evaluation p",
        "multiplicity": "Holm2 at.05/(15*16), separately cumulativeHolm123 at.05",
        "gate": "Bothphase proper-score delta<=-.005 againstBOTHcontrols; bothfixed "
        "evaluationslice deltas<0; waveHolm<.05/240,cumulativeHolm<.05",
        "power": "Nominal HAC80percent minimumdetectableeffect dividedby.005; ordinary5percent "
        "diagnostic, noequivalence or adjustedpowerclaim",
        "failure_rule": "Any admission/support/fit/score/verification/publication failure leaves "
        "bothregistered hypotheses UNEVALUABLEp1; retain alloldresults and "
        "diagnosticfailures, no outcome-dependent repair",
    },
    "verification": {
        "baseline_method": "trust-exact",
        "baseline_initialization": "all_zero",
        "baseline_maximum_iterations": 500,
        "independent_gradient_target": 1e-10,
        "accepted_gradient_tolerance": 1.0001e-08,
        "quarter_method": "brentq",
        "scalar_absolute_tolerance": 1e-12,
        "scalar_relative_tolerance": 1e-14,
        "scalar_maximum_iterations": 200,
        "coefficient_relative_tolerance": 1e-07,
        "coefficient_absolute_tolerance": 1e-06,
        "independent_normalized_prediction_relative_tolerance": 1e-07,
        "independent_normalized_prediction_absolute_tolerance": 1e-06,
        "saved_normalized_prediction_relative_tolerance": 1e-10,
        "saved_normalized_prediction_absolute_tolerance": 1e-12,
        "feature_relative_tolerance": 1e-10,
        "feature_absolute_tolerance": 1e-12,
        "inference_relative_tolerance": 1e-09,
        "inference_absolute_tolerance": 1e-14,
        "reconstruction": "Independent "
        "originalupstreamproofs,civilcalendar,commonmaturity/support/rank/novelty,trainingtransforms,convexfits,strictsavedprediction "
        "replay "
        "normalizedbyexacttrainmean,primitiveproperlosses/stablepairedgaps,fourphaseinference "
        "and complete123family ledger; no newproducerimport",
    },
    "outputs": {
        "data": "data/civil_quarter",
        "reports": "reports/civil_quarter",
        "features": "data/civil_quarter/features.parquet",
        "targets": "data/civil_quarter/targets.parquet",
        "forecasts": "data/civil_quarter/forecasts.parquet",
        "fits": "data/civil_quarter/fits.json",
        "support_audit": "data/civil_quarter/support_audit.json",
        "admission": "data/civil_quarter/upstream_admission.json",
    },
}


def validate_protocol(protocol):
    if protocol != CONTRACT:
        raise AssertionError("Entire independent operative contract differs")


def inference_equal(actual, expected, label, key=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(label + ": mapping identity differs")
        for name, value in expected.items():
            inference_equal(actual[name], value, label + "." + name, name)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(label + ": sequence identity differs")
        for index, value in enumerate(expected):
            inference_equal(actual[index], value, label + f"[{index}]", key)
    elif isinstance(expected, (float, np.floating)):
        left, right = float(finite(actual)), float(finite(expected))
        bound = 1e-14 if key == "p" or key.startswith("p_") else 1e-14 + 1e-9 * abs(right)
        if abs(left - right) > bound:
            raise AssertionError(label + ": fixed numerical tolerance exceeded")
    elif actual != expected:
        raise AssertionError(label + ": literal value differs")


def verify_metrics(
    root,
    panel,
    features,
    protocol,
    metrics,
    monthly_fits,
    application_n,
    *,
    admitted_inputs=None,
):
    validate_panel(panel)
    reference = pd.DatetimeIndex(features.index)
    if (
        reference.has_duplicates
        or reference.hasnans
        or not reference.is_monotonic_increasing
        or not panel.origin.isin(reference).all()
    ):
        raise AssertionError("Every score origin requires its full ordered source calendar")
    values = features.loc[panel.origin, ALL_FEATURES].to_numpy()
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise AssertionError("All scored origins require complete real registered inputs")
    dates = pd.Series(reference, index=reference)
    for column, shift in (
        ("feature_cutoff_date", 1),
        ("target_end", -1),
        ("available_date", -1),
    ):
        if not np.array_equal(panel[column], dates.shift(shift).loc[panel.origin]):
            raise AssertionError("Exact reference-calendar score timing required")
    n = int(panel.origin.nunique())
    if (
        metrics["new_monthly_fits"] != monthly_fits
        or metrics["new_forecasts"] != len(panel)
        or metrics["common_scored_origins"] != n
        or metrics["common_application_origins"] != application_n
    ):
        raise AssertionError("Complete new monthly fit and forecast accounting differs")
    section = protocol["index"]
    union = np.zeros(len(panel), dtype=bool)
    support = {}
    for name in ("development", "evaluation"):
        first, last = section[name]
        mask = panel.origin.between(first, last)
        union |= mask
        if not panel.loc[mask, "phase"].eq(name).all():
            raise AssertionError("Literal phase membership differs")
        origins = pd.DatetimeIndex(panel.loc[mask & panel.model.eq("baseline"), "origin"])
        if len(origins) < 127:
            raise AssertionError("Fixed phase bandwidth unsupported")
        support[name] = civil_support(features.loc[origins], "phase")
    if (
        not union.all()
        or (panel.train_n < section["minimum_train"]).any()
        or (panel.available_date > section["latest_target"]).any()
        or (
            panel.loc[panel.phase.eq("development"), "available_date"]
            > section["development_target_available_by"]
        ).any()
    ):
        raise AssertionError("Fixed complete inference sample or maturity fence differs")
    support["evaluation_slices"] = []
    for start, end in section["evaluation_stability"]:
        origins = pd.DatetimeIndex(
            panel.loc[panel.model.eq("baseline") & panel.origin.between(start, end), "origin"]
        )
        support["evaluation_slices"].append(
            {"start": start, "end": end, **civil_support(features.loc[origins], "slice")}
        )
    same_tree(metrics["civil_support"], support, "Every fixed phase/slice civil class count")
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both fixed quarter contrasts required")
    probabilities, effects = [], []
    for row in rows:
        if (
            row["study"] != "civil_quarter"
            or row["horizon"] != 1
            or row["score"] != "proper_variance"
        ):
            raise AssertionError("Proper-risk comparison identity differs")
        phases = [
            phase_statistics(panel, features, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        inference_equal(row["phases"], phases, "Independent proper-risk phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        inference_equal(
            row["p_conservative"], probability, "Both-phase conjunction probability", "p"
        )
        probabilities.append(probability)
        effects.append(
            all(phase["delta"] <= -EFFECT for phase in phases)
            and len(phases[1]["stability"]) == 2
            and all(one["delta"] < 0 for one in phases[1]["stability"])
        )
    prior = inherited_rows(root, protocol, pins=admitted_inputs)
    same_tree(metrics["inherited_rows"], prior, "Entire prior121 hypothesis identity")
    wave = holm(probabilities)
    cumulative = holm([row["p_conservative"] for row in prior] + probabilities)[-2:]
    passed = []
    for number, row in enumerate(rows):
        inference_equal(row["p_holm_wave"], wave[number], "Wave15 Holm2", "p")
        inference_equal(
            row["p_holm_cumulative"], cumulative[number], "Cumulative Holm123", "p"
        )
        one = bool(effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05)
        if row["verdict"] != ("PASSES_ALL_GATES" if one else "DOES_NOT_QUALIFY"):
            raise AssertionError("Fixed effect/statistical/stability gates differ")
        passed.append(one)
    leads = ["quarter"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 123
    ):
        raise AssertionError("Both-control candidate and complete family required")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 123,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "civil_support": support,
        "leads": leads,
    }


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    code = {
        str(path.relative_to(root))
        for folder in ("src", "tests")
        for path in (root / folder).rglob("*.py")
    }
    inputs = set(protocol["comparisons"]["inherited_sources"])
    for key in ("upstream", "feature_source"):
        info = protocol[key]
        old = read_json_snapshot(
            root, info["reports"] + "/manifest.json", info["manifest_sha256"]
        )
        code.update(old["code"])
        inputs.update(old["inputs"])
        inputs.add(info["protocol"])
        for folder in (info["reports"], info["data"]):
            inputs.update(
                str(path.relative_to(root))
                for path in (root / folder).rglob("*")
                if path.is_file()
            )
    preserved = {
        str(path.relative_to(root))
        for path in [*root.glob("*.yaml"), *(root / "reports").rglob("*")]
        if path.is_file()
        and path != root / "civil_quarter.yaml"
        and root / "reports/civil_quarter" not in path.parents
        and str(path.relative_to(root)) not in inputs
    }
    if (
        set(manifest["code"]) != code
        or set(manifest["inputs"]) != inputs
        or set(manifest["preserved"]) != preserved
        or set(manifest["inputs"]) & set(manifest["preserved"])
    ):
        raise AssertionError(
            "Exact new code, dual-upstream inputs and preserved corpus required"
        )


def verify(root=ROOT):
    root = Path(root)
    payload = (root / "civil_quarter.yaml").read_bytes()
    protocol = yaml.safe_load(payload)
    validate_protocol(protocol)
    protocol_hash = sha256(payload).hexdigest()
    report, out = root / "reports/civil_quarter", root / "data/civil_quarter"
    manifest_payload = (report / "manifest.json").read_bytes()
    manifest_hash = sha256(manifest_payload).hexdigest()
    manifest = json.loads(manifest_payload)
    if manifest["protocol_sha256"] != protocol_hash:
        raise AssertionError("Frozen protocol and manifest differ")
    verify_manifest_coverage(root, protocol, manifest)
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    names = [
        str((out / name).relative_to(root))
        for name in (
            "upstream_admission.json",
            "support_audit.json",
            "features.parquet",
            "targets.parquet",
            "forecasts.parquet",
            "fits.json",
        )
    ] + [
        str((report / name).relative_to(root))
        for name in ("metrics.json", "trial_ledger.jsonl")
    ]
    snapshots = {name: digest(root / name) for name in names}

    def saved_json(name):
        return read_json_snapshot(root, name, snapshots[name])

    def saved_parquet(name):
        return scalar.read_issued_snapshot(root, name, snapshots[name])

    proof = validate_upstream(root)
    same_tree(
        saved_json("data/civil_quarter/upstream_admission.json"),
        proof,
        "Dual original admission proof",
    )
    source = protocol["feature_source"]
    original = scalar.read_issued_snapshot(
        root, source["features"], manifest["inputs"][source["features"]]
    )
    targets = scalar.read_issued_snapshot(
        root, source["targets"], manifest["inputs"][source["targets"]]
    )
    features = augment_features(original)
    table_equal(
        saved_parquet("data/civil_quarter/features.parquet"),
        features,
        ALL_FEATURES,
        "Independent raw civil augmentation",
    )
    table_equal(
        saved_parquet("data/civil_quarter/targets.parquet"),
        targets,
        ("y",),
        "Unchanged original target table",
    )
    support = preflight(features, targets, protocol)
    same_tree(
        saved_json("data/civil_quarter/support_audit.json"),
        support,
        "Every preoptimization support/rank/novelty check",
    )
    panel = saved_parquet("data/civil_quarter/forecasts.parquet")
    fits = saved_json("data/civil_quarter/fits.json")
    metrics = saved_json("reports/civil_quarter/metrics.json")
    if (
        (report / "failure.json").exists()
        or metrics.get("status") == "UNEVALUABLE"
        or metrics["protocol_sha256"] != protocol_hash
        or metrics["evidence_class"] != protocol["evidence_class"]
    ):
        raise AssertionError("New publication failed or identity differs")
    result = {
        "status": "VERIFIED",
        "upstream_admission": {"status": proof["status"], "prior_files_written": False},
        "raw_feature_rows_verified": len(features),
        "raw_feature_columns_verified": len(ALL_FEATURES),
        "support_audit_verified": {
            "monthly_fits": support["monthly_fits"],
            "phases": len(support["phases"]),
            "evaluation_slices": len(support["phases"][1]["slices"]),
        },
        "forecast_reconstruction": verify_forecasts(features, targets, panel, fits, protocol),
        "primitive_proper_scores_verified": len(panel),
        "paired_proper_differences_verified": int(panel.origin.nunique()) * 2,
        "inference": verify_metrics(
            root,
            panel,
            features,
            protocol,
            metrics,
            len(fits),
            support["common_application_origins"],
            admitted_inputs=manifest["inputs"],
        ),
        "ledger_events_verified": verify_ledger(
            root,
            metrics,
            metrics["inherited_rows"],
            "evaluated",
            signature=snapshots["reports/civil_quarter/trial_ledger.jsonl"],
        ),
        "protocol_sha256": protocol_hash,
        "verifier_sha256": digest(Path(__file__)),
        "limitations": [
            "One staged civil interaction conditional on a newly fitted market, macro, month, ordinary month-end and December year-end baseline; no joint refit in the quarter arm.",
            "Nominal next-weekday civil arithmetic ignores holidays and early closes; unchanged target maturity uses the next actual observed SPX session.",
            "The daily floored Garman-Klass plus overnight-squared-return target remains an OHLC risk proxy; original-plan missingness, archival revisions and back-calculated VIX9D persist.",
            "Adaptively selected exploratory reuse of prior historical periods; proper-score gains are absolute natural-log-score units, not percentages, causal mechanisms or profit.",
        ],
    }
    for group in ("code", "inputs", "preserved"):
        scalar._pins(root, manifest[group])
    scalar._pins(root, snapshots)
    verify_manifest_coverage(root, protocol, manifest)
    if (
        digest(root / "civil_quarter.yaml") != protocol_hash
        or digest(report / "manifest.json") != manifest_hash
    ):
        raise AssertionError("Protocol or manifest changed during verification")
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False), flush=True)
