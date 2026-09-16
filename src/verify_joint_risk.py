"""Independent joint-return source, staged-moment, matrix-score and certificate audit.

No wave10 feature, model, density or runner implementation is imported.
Frozen independent source/ridge/convex helpers are reused unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from . import verify_relative_risk as previous
from .verify_index_hinge import ridge_prediction
from .verify_international_volatility import same_tree
from .verify_iterative_signal_search import digest, holm, same
from .verify_macro_overnight import penalized_objective, second_moment_prediction
from .verify_model_memory_study import explicit_bootstrap_means, independent_hac

ROOT = Path(__file__).resolve().parents[1]
ALL_FEATURES = previous.ALL_FEATURES + tuple(
    f"{asset}_day_{suffix}" for asset in ("qqq", "spx") for suffix in ("d", "w", "m")
)
MARGINAL_COLUMNS = ALL_FEATURES + ("corr22_centered_sq",)
TARGETS = ("y_qqq", "y_spx")
MODELS = ("constant_matrix", "constant_correlation", "dynamic_correlation")
COMPARISONS = (
    ("dynamic_correlation", "constant_correlation"),
    ("dynamic_correlation", "constant_matrix"),
)
WAVE_ALPHA = 0.05 / (10 * 11)
FLOOR, CAP, BOUND, ALPHA = 1e-10, 0.995, 4.0, 0.01


def validate_protocol(protocol):
    expected = {
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
            "evaluation_stability": [
                ["2020-01-02", "2022-12-31"],
                ["2023-01-01", "2025-10-17"],
            ],
            "horizons": [1],
            "models": list(MODELS),
            "all_features": list(ALL_FEATURES),
            "market_lag": 1,
            "minimum_train": 1000,
            "penalty": 0.01,
            "effect_threshold_absolute": 0.005,
            "residual_staging": "current_fit_training_residuals",
            "transform": "training_centered_corr22_square",
            "mean_gradient_tolerance": 1e-10,
            "moment_gradient_tolerance": 1e-8,
        },
        "inference": {
            "blocks": [21, 63, 126],
            "hac_lags": 126,
            "minimum_phase_observations": 127,
            "bootstrap_draws": 99999,
            "seed": 20260916,
        },
        "comparisons": {
            "new_hypotheses": 2,
            "inherited_hypotheses": 110,
            "cumulative_hypotheses": 112,
            "controls": ["constant_correlation", "constant_matrix"],
            "contrasts": [[a, b, "matrix_qlike"] for a, b in COMPARISONS],
        },
        "verification": {
            "coefficient_relative_tolerance": 1e-7,
            "coefficient_absolute_tolerance": 1e-10,
            "forecast_relative_tolerance": 1e-7,
            "forecast_absolute_tolerance": 1e-10,
            "mean_gradient_tolerance": 1e-10,
            "moment_gradient_tolerance": 1e-8,
            "dependence_projected_gradient_tolerance": 1e-7,
            "global_value_gap": 1e-8,
            "independent_variance_coefficient_relative_tolerance": 1e-7,
            "independent_variance_coefficient_absolute_tolerance": 1e-6,
            "independent_variance_forecast_relative_tolerance": 1e-6,
            "independent_variance_forecast_absolute_tolerance": 1e-12,
        },
        "dependence": {
            "rho_max": CAP,
            "parameter_bound": BOUND,
            "slope_penalty": ALPHA,
            "minimum_correlation_eigenvalue": 1e-6,
            "global_value_gap": 1e-8,
            "maximum_splits": 32768,
            "projected_gradient_tolerance": 1e-7,
            "floating_inflation_eps_multiplier": 256,
            "constant_objective_tie_tolerance": 1e-12,
            "cubic_root_imaginary_tolerance": 1e-10,
            "cubic_root_scaled_residual_tolerance": 1e-10,
            "local_options": {"maxiter": 1000, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9},
        },
    }
    for section, values in expected.items():
        if any(protocol[section].get(name) != value for name, value in values.items()):
            raise AssertionError("Fixed joint-risk protocol differs in " + section)
    if (
        protocol["wave"] != 10
        or protocol["wave_alpha"] != WAVE_ALPHA
        or protocol["measurement"]["gk_floor"] != FLOOR
        or protocol["correlation"]["window"] != 22
        or protocol["correlation"]["roundoff_tolerance"] != 1e-12
    ):
        raise AssertionError("Fixed wave, measurement or prior-correlation contract differs")


def load_source_tables(root, protocol):
    qqq, spx, iv, audit = previous.load_source_tables(root, protocol)
    audit["raw_columns"] = list(ALL_FEATURES)
    audit["target_interpretation"] = (
        "paired next observed SPX session raw log(close/open) returns for QQQ ETF and SPX price index; signed and zero valid, not measured high-frequency covariance"
    )
    audit["measurement_gate"] = (
        "all complete OHLC rows on the full SPX reference calendar require finite raw GK strictly greater than 1e-10; every observed open/close pair requires finite signed log(close/open), including rows with missing high/low; zero signed returns valid; before any feature/label mask"
    )
    return qqq, spx, iv, audit


def measurement_audit(qqq, spx):
    audit = previous.measurement_audit(qqq, spx)
    for name, source in (("qqq", qqq), ("spx", spx)):
        observed = source.reindex(spx.index)
        complete = observed[["open", "high", "low", "close"]].notna().all(axis=1)
        day = np.log(observed.close / observed.open)
        paired = observed[["open", "close"]].notna().all(axis=1)
        if not np.isfinite(day.loc[paired]).all():
            raise ValueError("Nonfinite intraday return on any observed open/close pair")
        audit["per_asset"][name].update(
            finite_intraday_return_rows=int(np.isfinite(day.loc[complete]).sum()),
            zero_intraday_return_rows=int(day.loc[complete].eq(0).sum()),
        )
    return audit


def require_measurement(audit):
    previous.require_measurement(audit)
    if any(
        one["finite_intraday_return_rows"] != one["observed_complete_rows"]
        or not 0 <= one["zero_intraday_return_rows"] <= one["finite_intraday_return_rows"]
        for one in audit["per_asset"].values()
    ):
        raise ValueError("INSUFFICIENT_MEASUREMENT: invalid paired intraday-return audit")


def feature_target_tables(qqq, spx, iv):
    require_measurement(measurement_audit(qqq, spx))
    output, _ = previous.feature_target_tables(qqq, spx, iv)
    target = {}
    for asset, source in (("qqq", qqq), ("spx", spx)):
        frame = source.reindex(spx.index)
        day = np.log(frame.close / frame.open)
        known = frame[["close", "open"]].notna().all(axis=1)
        if not np.isfinite(day.loc[known]).all():
            raise ValueError("Nonfinite observed intraday return cannot be treated as missing")
        for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
            output[f"{asset}_day_{suffix}"] = (
                day.rolling(width, min_periods=width).mean().shift()
            )
        target["y_" + asset] = day.shift(-1)
    output = output.loc[:, [*ALL_FEATURES, "feature_cutoff_date"]]
    maturity = pd.Series(spx.index, index=spx.index).shift(-1)
    return output, pd.DataFrame(
        {**target, "target_end": maturity, "available_date": maturity}, index=spx.index
    )


def transform(train, application):
    first, second = train.loc[:, ALL_FEATURES].copy(), application.loc[:, ALL_FEATURES].copy()
    if (
        len(first) < 2
        or not np.isfinite(first).all(axis=None)
        or not np.isfinite(second).all(axis=None)
    ):
        raise ValueError("Finite shared input rows required")
    if not first.const.eq(1).all() or not second.const.eq(1).all():
        raise ValueError("Unit intercept required")
    center = float(first.corr22.mean())
    for frame in (first, second):
        frame["corr22_centered_sq"] = (frame.corr22 - center) ** 2
    if (
        not np.isfinite(first).all(axis=None)
        or not np.isfinite(second).all(axis=None)
        or (first.iloc[:, 1:].std(ddof=0) <= 1e-12).any()
    ):
        raise ValueError("INSUFFICIENT_DATA: invalid common scale or transformation")
    return first, second, {"corr22_mean": center}


def eligible_entries(features, targets, section):
    complete = pd.Series(
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1), index=features.index
    )
    phases = pd.Series(False, index=features.index)
    for name in ("development", "evaluation"):
        first, last = section[name]
        phases |= (features.index >= first) & (features.index <= last)
    applications = features.index[
        complete
        & phases
        & (features.index >= section["origin_start"])
        & (features.index <= section["origin_end"])
    ]
    known = np.isfinite(targets.loc[:, TARGETS]).all(axis=1) & targets.available_date.notna()
    known &= targets.available_date <= pd.Timestamp(section["latest_target"])
    known &= (features.index > section["development"][1]) | (
        targets.available_date <= section["development_target_available_by"]
    )
    return applications, applications[known.loc[applications]], complete


def training_mask(features, targets, entry):
    if not features.index.equals(targets.index):
        raise AssertionError("Identical source reference calendars required")
    position = features.index.get_loc(entry)
    if position < 1:
        raise ValueError("No predecessor for model origin")
    cutoff = features.index[position - 1]
    return (
        np.isfinite(features.loc[:, ALL_FEATURES]).all(axis=1)
        & np.isfinite(targets.loc[:, TARGETS]).all(axis=1)
        & (features.index < entry)
        & (targets.available_date <= cutoff)
    )


def matrix_score(residual, diagonal, correlation):
    e, h, rho = (
        np.asarray(residual, float),
        np.asarray(diagonal, float),
        np.asarray(correlation, float),
    )
    if (
        e.ndim != 2
        or e.shape[1] != 2
        or h.shape != e.shape
        or rho.shape != (len(e),)
        or not np.isfinite(e).all()
        or not np.isfinite(h).all()
        or not np.isfinite(rho).all()
        or (h <= 0).any()
        or (np.abs(rho) >= 1).any()
    ):
        raise ValueError(
            "Finite paired residuals, positive diagonals and nonsingular correlation required"
        )
    whitened = e / np.sqrt(h)
    first = whitened[:, 0]
    second = (whitened[:, 1] - rho * first) / np.sqrt(1 - rho**2)
    answer = np.log(h).sum(axis=1) + np.log1p(-(rho**2)) + first**2 + second**2
    if not np.isfinite(answer).all():
        raise ValueError("Nonfinite independent Cholesky score")
    return answer


def _arguments(innovation, z=None):
    u = np.asarray(innovation, float)
    if u.ndim != 2 or u.shape[1] != 2 or len(u) < 2 or not np.isfinite(u).all():
        raise ValueError("Finite paired standardized residuals required")
    state = np.ones(len(u)) if z is None else np.asarray(z, float)
    if state.shape != (len(u),) or not np.isfinite(state).all():
        raise ValueError("Aligned finite dependence signal required")
    return u, state


def dependence_objective(parameter, innovation, z=None, constant=None):
    u, state = _arguments(innovation, z)
    if not np.isfinite(parameter) or (constant is not None and not np.isfinite(constant)):
        raise ValueError("Finite dependence parameters required")
    eta = parameter * state + (0.0 if constant is None else constant)
    tangent = np.tanh(eta)
    rho = CAP * tangent
    exponential = np.exp(-2 * np.abs(eta))
    derivative = CAP * 4 * exponential / (1 + exponential) ** 2
    a, b = np.square(u).sum(axis=1), np.prod(u, axis=1)
    denominator = 1 - rho**2
    polynomial = rho**3 - b * rho**2 + (a - 1) * rho - b
    penalty = 0.0 if constant is None else ALPHA
    value = np.mean(matrix_score(u, np.ones_like(u), rho)) + penalty * parameter**2
    gradient = (
        np.mean(2 * polynomial / denominator**2 * derivative * state) + 2 * penalty * parameter
    )
    if not np.isfinite([value, gradient]).all():
        raise ValueError("Nonfinite scalar objective or gradient")
    return float(value), float(gradient)


def projected_gradient(parameter, gradient):
    if not np.isfinite([parameter, gradient]).all() or abs(parameter) > BOUND:
        raise ValueError("Finite bounded parameter required")
    return (
        0.0
        if (parameter == -BOUND and gradient > 0) or (parameter == BOUND and gradient < 0)
        else float(gradient)
    )


def constant_optimum(innovation):
    u, _ = _arguments(innovation)
    a, b = float(np.square(u).sum(axis=1).mean()), float(np.prod(u, axis=1).mean())

    def polynomial(rho):
        return ((rho - b) * rho + (a - 1)) * rho - b

    edge = CAP * np.tanh(BOUND)
    cuts = [-edge, edge]
    discriminant = b * b - 3 * (a - 1)
    if discriminant >= 0:
        for point in ((b - np.sqrt(discriminant)) / 3, (b + np.sqrt(discriminant)) / 3):
            if -edge < point < edge:
                cuts.append(float(point))
    cuts = sorted(set(cuts))
    roots = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if polynomial(left) * polynomial(right) < 0:
            roots.append(brentq(polynomial, left, right, xtol=1e-14, rtol=1e-14))
    roots.extend(
        point for point in cuts if abs(polynomial(point)) <= 1e-12 * (1 + abs(a) + abs(b))
    )
    candidates = []
    for rho in sorted({-edge, edge, *roots}):
        theta = float(np.arctanh(rho / CAP))
        if rho == -edge:
            theta = -BOUND
        if rho == edge:
            theta = BOUND
        value, gradient = dependence_objective(theta, u)
        candidates.append({"parameter": theta, "objective": value, "gradient": gradient})
    lowest = min(one["objective"] for one in candidates)
    tied = [one for one in candidates if one["objective"] <= lowest + 1e-12]
    best = min(tied, key=lambda one: one["parameter"])
    return {**best, "candidates": candidates}


def curvature_bound(left, right, innovation, z, constant):
    u, state = _arguments(innovation, z)
    eta1, eta2 = constant + left * state, constant + right * state
    low, high = np.minimum(eta1, eta2), np.maximum(eta1, eta2)
    tlow, thigh = np.tanh(low), np.tanh(high)
    radius = np.nextafter(CAP * np.maximum(abs(tlow), abs(thigh)), np.inf)
    denominator = 1 - radius**2
    a, b = np.square(u).sum(axis=1), np.abs(np.prod(u, axis=1))
    p = radius**3 + b * radius**2 + abs(a - 1) * radius + b
    dp = 3 * radius**2 + 2 * b * radius + abs(a - 1)
    g1 = 2 * p / denominator**2
    g2 = 2 * dp / denominator**2 + 8 * radius * p / denominator**3
    nearest = np.where(low > 0, low, np.where(high < 0, high, 0.0))
    e = np.exp(-2 * np.abs(nearest))
    r1 = CAP * 4 * e / (1 + e) ** 2
    el, eh = np.exp(-2 * np.abs(low)), np.exp(-2 * np.abs(high))
    r2 = (
        2
        * CAP
        * np.maximum(abs(tlow) * 4 * el / (1 + el) ** 2, abs(thigh) * 4 * eh / (1 + eh) ** 2)
    )
    critical = 1 / np.sqrt(3)
    contains = ((tlow <= critical) & (critical <= thigh)) | (
        (tlow <= -critical) & (-critical <= thigh)
    )
    r2 = np.where(contains, 4 * CAP / (3 * np.sqrt(3)), r2)
    value = float(np.mean(state**2 * (g2 * r1**2 + g1 * r2)) + 2 * ALPHA)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Nonfinite interval curvature bound")
    inflation = 256 * np.finfo(float).eps
    return float(np.nextafter(value * (1 + inflation) + inflation, np.inf))


def verify_dependence(innovation, z, audits):
    inflation = 256 * np.finfo(float).eps
    constant = audits["constant_correlation"]
    if set(audits) != {"constant_correlation", "dynamic_correlation"}:
        raise AssertionError("Exactly two conditional dependence arms required")
    independent = constant_optimum(innovation)
    same(
        constant["parameter"],
        independent["parameter"],
        "Independent cubic root/bracket constant coefficient",
        rtol=1e-7,
        atol=1e-10,
    )
    all_gaps, all_kkt, leaves_checked = [], [], 0
    for model in ("constant_correlation", "dynamic_correlation"):
        audit = audits[model]
        is_constant = model == "constant_correlation"
        intercept = None if is_constant else constant["parameter"]
        if (
            audit["bounds"] != [-4.0, 4.0]
            or audit["rho_max"] != CAP
            or audit["constant"] != intercept
            or audit["penalty"] != (0.0 if is_constant else ALPHA)
            or audit["success"] is not True
            or audit["global_value_tolerance"] != 1e-8
            or audit["projected_gradient_tolerance"] != 1e-7
            or not np.isfinite(audit["parameter"])
            or abs(audit["parameter"]) > 4
        ):
            raise AssertionError("Fixed scalar dependence contract differs")
        state = None if is_constant else z
        value, gradient = dependence_objective(
            audit["parameter"], innovation, state, intercept
        )
        projected = projected_gradient(audit["parameter"], gradient)
        if abs(projected) > 1e-7 or abs(audit["projected_gradient"]) > 1e-7:
            raise AssertionError("Independent bounded dependence KKT failed")
        same(
            audit["objective"],
            value,
            "Independent unhalved penalized matrix objective",
            rtol=1e-10,
            atol=1e-12,
        )
        same(
            audit["gradient"],
            gradient,
            "Independent analytic scalar gradient",
            rtol=1e-7,
            atol=1e-10,
        )
        same(
            audit["projected_gradient"],
            projected,
            "Independent projected KKT",
            rtol=1e-7,
            atol=1e-10,
        )
        if is_constant:
            if (
                audit["method"] != "all_real_stationary_cubic_roots_and_endpoints"
                or audit["interval_splits"] != 0
                or audit["local_attempts"] != []
            ):
                raise AssertionError("Constant must exhaust all bounded cubic candidates")
            candidates = audit["candidates"]
            for expected in independent["candidates"]:
                if not any(
                    abs(row["parameter"] - expected["parameter"]) <= 1e-7 for row in candidates
                ):
                    raise AssertionError(
                        "An independently isolated cubic stationary point or endpoint is missing"
                    )
            for row in candidates:
                f, g = dependence_objective(row["parameter"], innovation)
                same(
                    row["objective"],
                    f,
                    "Independent constant candidate objective",
                    rtol=1e-10,
                    atol=1e-12,
                )
                same(
                    row["gradient"],
                    g,
                    "Independent constant candidate derivative",
                    rtol=1e-7,
                    atol=1e-10,
                )
                same(
                    row["projected_gradient"],
                    projected_gradient(row["parameter"], g),
                    "Constant candidate projected KKT",
                    rtol=1e-7,
                    atol=1e-10,
                )
            minimum = min(one["objective"] for one in independent["candidates"])
            lower = float(np.nextafter(minimum - inflation * (1 + abs(minimum)), -np.inf))
        else:
            if (
                audit["method"] != "best_first_interval_curvature_certificate"
                or audit["maximum_interval_splits"] != 32768
                or not 0 <= audit["interval_splits"] <= 32768
                or audit["floating_inflation"] != inflation
                or audit["local_options"]
                != {"maxiter": 1000, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9}
            ):
                raise AssertionError("Fixed bounded global certificate settings differ")
            leaves = audit["certificate_intervals"]
            if (
                len(leaves) != audit["interval_splits"] + 1
                or leaves[0]["left"] != -4.0
                or leaves[-1]["right"] != 4.0
                or any(a["right"] != b["left"] for a, b in zip(leaves[:-1], leaves[1:]))
            ):
                raise AssertionError(
                    "Certificate leaves must partition the entire parameter bound without gaps"
                )
            bounds = []
            for leaf in leaves:
                left, right = leaf["left"], leaf["right"]
                if (
                    not np.isfinite([left, right, leaf["lower_bound"]]).all()
                    or not -4 <= left < right <= 4
                ):
                    raise AssertionError("Finite ordered certificate intervals required")
                midpoint, radius = (left + right) / 2, (right - left) / 2
                f, g = dependence_objective(midpoint, innovation, z, intercept)
                curve = curvature_bound(left, right, innovation, z, intercept)
                spread = abs(g) * radius + 0.5 * curve * radius**2
                bound = float(
                    np.nextafter(f - spread - inflation * (1 + abs(f) + spread), -np.inf)
                )
                same(
                    leaf["lower_bound"],
                    bound,
                    "Independent interval Taylor lower bound",
                    rtol=1e-10,
                    atol=1e-10,
                )
                bounds.append(bound)
            leaves_checked += len(leaves)
            lower = min(bounds)
            attempts = audit["local_attempts"]
            if (
                not attempts
                or attempts[0]["start"] != 0.0
                or len({row["start"] for row in attempts}) != len(attempts)
            ):
                raise AssertionError(
                    "Deterministic unique scalar refinements must begin at zero"
                )
            for attempt in attempts:
                if (
                    not -4 <= attempt["start"] <= 4
                    or not -4 <= attempt["parameter"] <= 4
                    or not 0 <= attempt["iterations"] <= 1000
                    or attempt["function_evaluations"] < 1
                ):
                    raise AssertionError("Invalid bounded local-refinement provenance")
                f, g = dependence_objective(attempt["parameter"], innovation, z, intercept)
                same(
                    attempt["objective"],
                    f,
                    "Independent refinement objective",
                    rtol=1e-10,
                    atol=1e-12,
                )
                same(
                    attempt["gradient"],
                    g,
                    "Independent refinement gradient",
                    rtol=1e-7,
                    atol=1e-10,
                )
                same(
                    attempt["projected_gradient"],
                    projected_gradient(attempt["parameter"], g),
                    "Independent refinement KKT",
                    rtol=1e-7,
                    atol=1e-10,
                )
        upper = value + inflation * (1 + abs(value))
        gap = upper - lower
        if (
            not np.isfinite(
                [
                    lower,
                    upper,
                    gap,
                    audit["global_lower_bound"],
                    audit["global_upper_bound"],
                    audit["global_value_gap"],
                ]
            ).all()
            or gap < 0
            or gap > 1e-8
            or not 0 <= audit["global_value_gap"] <= 1e-8
            or audit["global_lower_bound"] > audit["objective"]
            or audit["global_upper_bound"] < audit["objective"]
        ):
            raise AssertionError("Independent global value certificate failed")
        same(
            audit["global_lower_bound"],
            lower,
            "Reconstructed complete-domain lower bound",
            rtol=1e-10,
            atol=1e-10,
        )
        same(
            audit["global_upper_bound"],
            upper,
            "Reconstructed admissible objective upper bound",
            rtol=1e-10,
            atol=1e-12,
        )
        same(audit["global_value_gap"], gap, "Independent global gap", rtol=0, atol=1e-10)
        all_gaps.append(gap)
        all_kkt.append(abs(projected))
    return {
        "scalar_fits_verified": 2,
        "certificate_leaves_verified": leaves_checked,
        "global_gap_max": max(all_gaps),
        "dependence_kkt_max": max(all_kkt),
    }


def verify_moments(training, y, application, audits):
    combined = pd.concat([training, application])
    independent_mean, rebuilt = ridge_prediction(training, y, combined)
    mean = audits["mean"]
    if (
        tuple(mean["columns"]) != MARGINAL_COLUMNS
        or mean["alpha"] != 0.01
        or mean["train_n"] != len(y)
    ):
        raise AssertionError("All35 shared marginal features and same-row mean fit required")
    for name in ("means", "scales", "beta"):
        same(
            mean[name],
            rebuilt[name],
            "Independent augmented least-squares mean " + name,
            rtol=1e-7,
            atol=1e-10,
        )
    design = (combined.to_numpy() - np.asarray(mean["means"])) / np.asarray(mean["scales"])
    prediction = design @ np.asarray(mean["beta"])
    same(
        prediction,
        independent_mean,
        "Independent shared mean predictions",
        rtol=1e-7,
        atol=1e-10,
    )
    full_gradient = 2 * design[: len(y)].T @ (prediction[: len(y)] - y) / len(y)
    full_gradient[1:] += 2 * 0.01 * np.asarray(mean["beta"])[1:]
    maximum = float(np.max(abs(full_gradient)))
    if maximum > 1e-10 or mean["gradient_max_abs"] > 1e-10:
        raise AssertionError("Shared mean full-MSE gradient failed")
    same(mean["gradient_max_abs"], maximum, "Reconstructed mean KKT", rtol=1e-4, atol=1e-12)
    residual = np.asarray(y) - prediction[: len(y)]
    squared = residual**2
    independent_variance, rebuilt_variance = second_moment_prediction(
        training, squared, combined
    )
    variance = audits["variance"]
    if (
        tuple(variance["columns"]) != MARGINAL_COLUMNS
        or variance["alpha"] != 0.01
        or variance["train_n"] != len(y)
    ):
        raise AssertionError(
            "All35 shared variance predictors and same training residuals required"
        )
    for name in ("means", "scales", "train_mean"):
        same(
            variance[name],
            rebuilt_variance[name],
            "Independent residual variance geometry " + name,
            rtol=1e-9,
            atol=1e-12,
        )
    same(
        variance["beta"],
        rebuilt_variance["beta"],
        "Independent convex positive-moment coefficients",
        rtol=1e-7,
        atol=1e-6,
    )
    variance_design = np.c_[
        np.ones(len(combined)),
        (combined.to_numpy()[:, 1:] - np.asarray(variance["means"]))
        / np.asarray(variance["scales"]),
    ]
    beta = np.asarray(variance["scaled_beta"])
    restored = beta.copy()
    restored[0] += np.log(squared.mean())
    same(
        variance["beta"],
        restored,
        "Exact positive-moment target unit restoration",
        rtol=1e-9,
        atol=1e-12,
    )
    value, gradient = penalized_objective(
        beta, variance_design[: len(y)], squared / squared.mean()
    )
    kkt = float(np.max(abs(gradient)))
    if (
        kkt > 1e-8 + 1e-12
        or variance["gradient_max_abs"] > 1e-8
        or not 0 <= variance["iterations"] < 200
        or variance["backtracks"] < 0
    ):
        raise AssertionError("Shared positive-moment optimum or fixed budget failed")
    same(
        variance["gradient_max_abs"],
        kkt,
        "Reconstructed positive-moment KKT",
        rtol=1e-4,
        atol=1e-12,
    )
    same(
        variance["objective"],
        value + np.log(squared.mean()),
        "Reconstructed positive-moment objective",
        rtol=1e-9,
        atol=1e-12,
    )
    h = np.exp(np.log(squared.mean()) + variance_design @ beta)
    same(
        h,
        independent_variance,
        "Independent positive conditional predictions",
        rtol=1e-6,
        atol=1e-12,
    )
    return (
        prediction,
        h,
        residual,
        {
            "mean_kkt": maximum,
            "variance_kkt": kkt,
            "independent_variance_kkt": rebuilt_variance["gradient_max_abs"],
        },
    )


def verify_fit(training, targets, application, audit):
    tr, ap, transformation = transform(training, application)
    same_tree(
        audit["transform"], transformation, "Independent training-centered corr22 square"
    )
    if audit["residual_staging"] != "current_fit_training_residuals" or set(
        audit["moments"]
    ) != {"qqq", "spx"}:
        raise AssertionError("Fixed shared current-fit residual staging required")
    means, variances, residuals, innovation, numerical = [], [], [], [], []
    for asset in ("qqq", "spx"):
        mean, variance, residual, checks = verify_moments(
            tr, targets["y_" + asset].to_numpy(), ap, audit["moments"][asset]
        )
        means.append(mean[len(tr) :])
        variances.append(variance[len(tr) :])
        residuals.append(residual)
        innovation.append(residual / np.sqrt(variance[: len(tr)]))
        numerical.append(checks)
    residual = np.column_stack(residuals)
    matrix = np.einsum("ni,nj->ij", residual, residual) / len(residual)
    diagonal = np.diag(matrix)
    if not np.isfinite(matrix).all() or (diagonal <= 0).any():
        raise AssertionError("Positive finite constant residual moment matrix required")
    rho_matrix = float(matrix[0, 1] / np.sqrt(diagonal[0]) / np.sqrt(diagonal[1]))
    if abs(rho_matrix) > 1 - 1e-6:
        raise AssertionError("Normalized constant residual matrix ill-conditioned")
    constant_matrix = audit["constant_matrix"]
    same(
        constant_matrix["matrix"],
        matrix,
        "Uncentered current-residual outer product mean, denominator n",
        rtol=1e-7,
        atol=1e-12,
    )
    same(
        constant_matrix["rho"],
        rho_matrix,
        "Independent constant residual correlation",
        rtol=1e-7,
        atol=1e-10,
    )
    same(
        constant_matrix["minimum_correlation_eigenvalue"],
        1 - abs(rho_matrix),
        "Constant normalized minimum eigenvalue",
        rtol=1e-7,
        atol=1e-10,
    )
    if constant_matrix["train_n"] != len(training):
        raise AssertionError("Constant matrix must use identical complete training rows")
    center, scale = transformation["corr22_mean"], float(tr.corr22.std(ddof=0))
    same(
        audit["corr22_scale"],
        scale,
        "Training-only dependence signal scale",
        rtol=1e-10,
        atol=1e-12,
    )
    checks = verify_dependence(
        np.column_stack(innovation),
        (tr.corr22.to_numpy() - center) / scale,
        audit["dependence"],
    )
    a = audit["dependence"]["constant_correlation"]["parameter"]
    b = audit["dependence"]["dynamic_correlation"]["parameter"]
    mu, h = np.column_stack(means), np.column_stack(variances)
    forecasts = {
        "constant_matrix": {
            "mu": mu,
            "h": np.tile(diagonal, (len(ap), 1)),
            "rho": np.full(len(ap), rho_matrix),
        },
        "constant_correlation": {"mu": mu, "h": h, "rho": np.full(len(ap), CAP * np.tanh(a))},
        "dynamic_correlation": {
            "mu": mu,
            "h": h,
            "rho": CAP * np.tanh(a + b * (ap.corr22.to_numpy() - center) / scale),
        },
    }
    checks["mean_kkt_max"] = max(row["mean_kkt"] for row in numerical)
    checks["variance_kkt_max"] = max(row["variance_kkt"] for row in numerical)
    checks["independent_variance_kkt_max"] = max(
        row["independent_variance_kkt"] for row in numerical
    )
    return forecasts, checks


def verify_forecasts(features, targets, forecasts, fits, protocol):
    section = protocol["index"]
    required = {
        "origin",
        "model",
        "horizon",
        "feature_cutoff_date",
        "target_end",
        "available_date",
        *TARGETS,
        "mu_qqq",
        "mu_spx",
        "h_qqq",
        "h_spx",
        "rho",
        "loss",
        "fit_origin",
        "fit_cutoff_date",
        "train_n",
        "train_last_target",
        "train_last_available",
        "phase",
    }
    if (
        not required.issubset(forecasts)
        or set(forecasts.model) != set(MODELS)
        or set(forecasts.horizon) != {1}
        or forecasts.duplicated(["origin", "model", "horizon"]).any()
    ):
        raise AssertionError("All three unique joint-risk forecast arms and metadata required")
    entries, scored, _ = eligible_entries(features, targets, section)
    cuts = pd.Series(features.index, index=features.index).shift()
    aligned = {}
    for model in MODELS:
        rows = forecasts.loc[forecasts.model == model].sort_values("origin")
        if not pd.DatetimeIndex(rows.origin).equals(scored):
            raise AssertionError(
                "Exact common independently reconstructed scored origins required"
            )
        same(
            rows.loc[:, TARGETS],
            targets.loc[scored, TARGETS],
            "Independent signed paired next-session target",
            rtol=1e-10,
            atol=1e-12,
        )
        for name in ("target_end", "available_date"):
            if not np.array_equal(rows[name].to_numpy(), targets.loc[scored, name].to_numpy()):
                raise AssertionError("Same-session paired label maturity differs")
        if not np.array_equal(
            rows.feature_cutoff_date.to_numpy(), cuts.loc[scored].to_numpy()
        ):
            raise AssertionError("Prior-reference-session feature cutoff differs")
        if not np.array_equal(
            rows.phase,
            np.where(scored <= section["development"][1], "development", "evaluation"),
        ):
            raise AssertionError("Fixed phase labels differ")
        aligned[model] = rows
    for model in MODELS:
        for field in ("mu_qqq", "mu_spx"):
            if not np.array_equal(
                aligned[model][field], aligned["constant_correlation"][field]
            ):
                raise AssertionError("All arms must share exact mean predictions")
    for field in ("h_qqq", "h_spx"):
        if not np.array_equal(
            aligned["dynamic_correlation"][field], aligned["constant_correlation"][field]
        ):
            raise AssertionError("Dependence comparison must keep exact modeled diagonals")
    lookup = {pd.Timestamp(record["fit_origin"]): record for record in fits}
    expected_fits = entries[~entries.to_period("M").duplicated()]
    if len(lookup) != len(fits) or set(lookup) != set(expected_fits):
        raise AssertionError(
            "Every first feature-complete monthly origin must fit exactly once"
        )
    totals = {
        "scalar_fits_verified": 0,
        "certificate_leaves_verified": 0,
        "global_gap_max": 0.0,
        "dependence_kkt_max": 0.0,
        "mean_kkt_max": 0.0,
        "variance_kkt_max": 0.0,
        "independent_variance_kkt_max": 0.0,
    }
    for entry in expected_fits:
        record = lookup[entry]
        selected = training_mask(features, targets, entry)
        n = int(selected.sum())
        cutoff = cuts.loc[entry]
        apps = entries[entries.to_period("M") == entry.to_period("M")]
        metadata = {
            "fit_cutoff_date": cutoff,
            "train_first_origin": features.index[selected][0],
            "train_last_origin": features.index[selected][-1],
            "train_last_target": targets.loc[selected, "target_end"].max(),
            "train_last_available": targets.loc[selected, "available_date"].max(),
        }
        if (
            n < section["minimum_train"]
            or record["train_n"] != n
            or record["application_n"] != len(apps)
        ):
            raise AssertionError("Fixed mature common training/application counts differ")
        for name, date in metadata.items():
            if pd.Timestamp(record[name]) != date:
                raise AssertionError("Saved fit date provenance differs: " + name)
        replay, checks = verify_fit(
            features.loc[selected],
            targets.loc[selected],
            features.loc[apps],
            record["model_audit"],
        )
        keep = apps.isin(scored)
        for model in MODELS:
            rows = forecasts.loc[
                (forecasts.model == model) & forecasts.origin.isin(apps)
            ].sort_values("origin")
            if (
                not rows.fit_origin.eq(entry).all()
                or not rows.fit_cutoff_date.eq(cutoff).all()
                or not rows.train_n.eq(n).all()
            ):
                raise AssertionError("Joint forecast fit provenance differs")
            for name in ("train_last_target", "train_last_available"):
                if not rows[name].eq(metadata[name]).all():
                    raise AssertionError("Forecast mature-training provenance differs")
            rebuilt = replay[model]
            for stem in ("mu", "h"):
                same(
                    rows.loc[:, [stem + "_qqq", stem + "_spx"]],
                    rebuilt[stem][keep],
                    "Exact saved marginal replay " + stem,
                    rtol=1e-7,
                    atol=1e-10,
                )
            same(
                rows.rho,
                rebuilt["rho"][keep],
                "Exact fixed-intercept dependence replay",
                rtol=1e-7,
                atol=1e-10,
            )
            scored_loss = matrix_score(
                rows.loc[:, TARGETS].to_numpy() - rebuilt["mu"][keep],
                rebuilt["h"][keep],
                rebuilt["rho"][keep],
            )
            same(
                rows.loss,
                scored_loss,
                "Independent Cholesky full joint score, no penalty",
                rtol=1e-7,
                atol=1e-10,
            )
        for name, value in checks.items():
            totals[name] = (
                totals[name] + value
                if name.endswith("_verified")
                else max(totals[name], value)
            )
    return {
        "forecasts_verified": len(forecasts),
        "common_scored_origins": len(scored),
        "feature_complete_applications": len(entries),
        "monthly_fits_verified": len(expected_fits),
        "models_verified": len(MODELS),
        **totals,
    }


def effect_passes(phases):
    return (
        len(phases) == 2
        and all(one["delta"] <= -0.005 for one in phases)
        and len(phases[1]["stability"]) == 2
        and all(one["delta"] < 0 for one in phases[1]["stability"])
    )


def verify_manifest_coverage(root, protocol, manifest):
    root = Path(root)
    prior = json.loads((root / "reports/relative_risk/manifest.json").read_text())
    required_code = set(prior["code"])
    required_code.update(
        "src/" + name + ".py"
        for name in (
            "joint_risk_features",
            "joint_risk_models",
            "joint_risk_density",
            "joint_risk_search",
            "verify_joint_risk",
            "plot_joint_risk",
        )
    )
    required_code.update(
        "tests/test_" + name + ".py"
        for name in (
            "joint_risk_features",
            "joint_risk_models",
            "joint_risk_density",
            "joint_risk_search",
            "joint_risk_publication",
            "verify_joint_risk",
        )
    )
    if not required_code.issubset(manifest["code"]):
        raise AssertionError(
            "Entire prior Python source/test corpus and new modules must remain frozen"
        )
    required_inputs = set(protocol["sources"].values())
    required_inputs.update(
        protocol["sources"][name] + ".manifest.json"
        for name in ("vxn", "vix", "vix9d", "vvix")
    )
    required_inputs.update(
        [
            "data/research_paths/source_manifest.json",
            "data/history_extension/source_manifest.json",
        ]
    )
    if not required_inputs.issubset(manifest["inputs"]):
        raise AssertionError(
            "All paired market files and documentary source manifests required"
        )
    required_preserved = set(protocol["comparisons"]["inherited_sources"]) | {
        "reports/relative_risk/manifest.json"
    }
    if not required_preserved.issubset(manifest["preserved"]):
        raise AssertionError("Inherited result and prior registration provenance omitted")


def phase_statistics(panel, control, phase, code, protocol):
    section = protocol["index"]
    first, last = section[phase]
    selected = panel.loc[(panel.origin >= first) & (panel.origin <= last)]
    if phase == "development":
        selected = selected.loc[
            selected.available_date <= pd.Timestamp(section["development_target_available_by"])
        ]
    wide = selected.pivot(index="origin", columns="model", values="loss").sort_index()
    if set(wide.columns) != set(MODELS) or not np.isfinite(wide).all(axis=None):
        raise AssertionError("Exact complete common signed matrix scores required")
    recalculated = {}
    for name in MODELS:
        rows = selected.loc[selected.model == name].set_index("origin").reindex(wide.index)
        residual = rows.loc[:, TARGETS].to_numpy() - rows[["mu_qqq", "mu_spx"]].to_numpy()
        loss = matrix_score(residual, rows[["h_qqq", "h_spx"]].to_numpy(), rows.rho.to_numpy())
        same(
            rows.loss,
            loss,
            "Independent score reconstruction before paired inference",
            rtol=1e-7,
            atol=1e-10,
        )
        recalculated[name] = pd.Series(loss, index=wide.index)
    candidate_loss, control_loss = recalculated["dynamic_correlation"], recalculated[control]
    difference = candidate_loss - control_loss
    if len(difference) < 127:
        raise AssertionError("At least127 phase rows required; signed matrix score is valid")
    delta = float(difference.mean())
    hac, blocks = independent_hac(difference), {}
    for width in protocol["inference"]["blocks"]:
        seed = protocol["inference"]["seed"] + code * 10000 + width
        samples = explicit_bootstrap_means(
            difference, width, protocol["inference"]["bootstrap_draws"], seed
        )[:, 0]
        p = float(
            (1 + np.count_nonzero(np.abs(samples - delta) >= abs(delta))) / (len(samples) + 1)
        )
        blocks[str(width)] = {"p": p, "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    intervals = [hac["ci95"], *(value["ci95"] for value in blocks.values())]
    output = {
        "name": phase,
        "first_origin": str(wide.index[0].date()),
        "last_origin": str(wide.index[-1].date()),
        "n": len(wide),
        "delta": delta,
        "candidate_loss": float(candidate_loss.mean()),
        "control_loss": float(control_loss.mean()),
        "block_inference": blocks,
        "hac126": hac,
        "ci95_envelope": [
            min(item[0] for item in intervals),
            max(item[1] for item in intervals),
        ],
        "p_conservative": max(hac["p"], *(value["p"] for value in blocks.values())),
        "annual": [],
        "stability": [],
        "nonoverlap_phases": [{"phase": 0, "n": len(wide), "delta": delta}],
    }
    for year in sorted(set(wide.index.year)):
        mask = wide.index.year == year
        output["annual"].append(
            {"year": int(year), "n": int(mask.sum()), "delta": float(difference[mask].mean())}
        )
    if phase == "evaluation":
        for start, end in section["evaluation_stability"]:
            mask = (wide.index >= start) & (wide.index <= end)
            if not mask.any():
                raise AssertionError("Fixed evaluation slice empty")
            output["stability"].append(
                {
                    "start": start,
                    "end": end,
                    "n": int(mask.sum()),
                    "delta": float(difference[mask].mean()),
                }
            )
    return output


def inherited_rows(root, protocol):
    output = []
    for source in protocol["comparisons"]["inherited_sources"]:
        for number, row in enumerate(json.loads((root / source).read_text())["rows"]):
            output.append(
                {
                    "study": row.get("study", Path(source).parent.name or Path(source).stem),
                    "candidate": row["candidate"],
                    "control": row.get("control", "baseline"),
                    "horizon": row["horizon"],
                    "p_conservative": row["p_conservative"],
                    "source": source,
                    "source_sha256": digest(root / source),
                    "source_row_index": number,
                    **{name: row[name] for name in ("measure", "score") if name in row},
                }
            )
    if len(output) != 110:
        raise AssertionError("All110 inherited comparisons must remain identified")
    return output


def verify_metrics(root, panel, protocol, metrics):
    rows = metrics["rows"]
    if [(row["candidate"], row["control"]) for row in rows] != list(COMPARISONS):
        raise AssertionError("Both registered joint-risk comparisons required in fixed order")
    probabilities, effects = [], []
    for row in rows:
        if (
            row["study"] != "joint_risk"
            or row["horizon"] != 1
            or row["score"] != "matrix_qlike"
        ):
            raise AssertionError("Relative intraday-risk comparison identity differs")
        phases = [
            phase_statistics(panel, row["control"], phase, code, protocol)
            for code, phase in enumerate(("development", "evaluation"))
        ]
        same_tree(row["phases"], phases, "Independent joint-risk phase inference")
        probability = max(phase["p_conservative"] for phase in phases)
        same(row["p_conservative"], probability, "Both-phase conjunction probability")
        probabilities.append(probability)
        effects.append(effect_passes(phases))
    prior = inherited_rows(root, protocol)
    same_tree(metrics["inherited_rows"], prior, "Complete cumulative trial identities")
    wave, cumulative = (
        holm(probabilities),
        holm([row["p_conservative"] for row in prior] + probabilities)[-2:],
    )
    passed = []
    for number, row in enumerate(rows):
        same(row["p_holm_wave"], wave[number], "Two-comparison wave Holm")
        same(row["p_holm_cumulative"], cumulative[number], "112-comparison cumulative Holm")
        eligible = bool(
            effects[number] and wave[number] < WAVE_ALPHA and cumulative[number] < 0.05
        )
        if row["verdict"] != ("COMPARISON_GATE_PASS" if eligible else "DOES_NOT_QUALIFY"):
            raise AssertionError("Joint-risk statistical or fixed effect gate differs")
        passed.append(eligible)
    leads = ["dynamic_correlation"] if all(passed) else []
    if (
        metrics["leads"] != leads
        or metrics["hypothesis_count"] != 2
        or metrics["cumulative_hypothesis_count"] != 112
    ):
        raise AssertionError("One joint-return increment must pass both controls")
    return {
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 112,
        "phase_comparisons_verified": 4,
        "bootstrap_runs_verified": 12,
        "bootstrap_draws_per_run": protocol["inference"]["bootstrap_draws"],
        "leads": leads,
    }


def verify_ledger(root, metrics, prior, final_event):
    report = Path(root) / "reports/joint_risk"
    ledger = [
        json.loads(line)
        for line in (report / "trial_ledger.jsonl").read_text().splitlines()
        if line
    ]
    for event, expected in (("inherited", prior), (final_event, metrics["rows"])):
        actual = [
            {key: value for key, value in row.items() if key != "event"}
            for row in ledger
            if row["event"] == event
        ]
        same_tree(actual, expected, "Independent " + event + " trial ledger")
    registered = [row for row in ledger if row["event"] == "registered"]
    if (
        len(ledger) != 114
        or len(registered) != 2
        or len(prior) != 110
        or [(row["candidate"], row["control"]) for row in registered] != list(COMPARISONS)
        or any(
            row["protocol_sha256"] != metrics["protocol_sha256"]
            or row["horizon"] != 1
            or row["score"] != "matrix_qlike"
            for row in registered
        )
    ):
        raise AssertionError("Complete110inherited+2registered+2terminal ledger required")
    return {"inherited": 110, "registered": 2, final_event: 2}


def verify_terminal_measurement(root, measurement, metrics, prior):
    root = Path(root)
    output, report = root / "data/joint_risk", root / "reports/joint_risk"
    if (
        measurement["status"] != "INSUFFICIENT_MEASUREMENT"
        or metrics.get("status") != "UNEVALUABLE"
        or metrics.get("whole_wave_aborted") is not True
        or metrics.get("leads") != []
        or metrics.get("hypothesis_count") != 2
        or metrics.get("cumulative_hypothesis_count") != 112
    ):
        raise AssertionError(
            "Confirmed measurement failure must retain both unevaluable hypotheses"
        )
    if any(
        (output / name).exists()
        for name in ("features.parquet", "targets.parquet", "forecasts.parquet", "fits.json")
    ):
        raise AssertionError(
            "No feature, target or model artifact may follow failed measurement gate"
        )
    if [(row["candidate"], row["control"]) for row in metrics["rows"]] != list(COMPARISONS):
        raise AssertionError("Both measurement-gated hypotheses must remain")
    for row in metrics["rows"]:
        if (
            row["study"] != "joint_risk"
            or row["horizon"] != 1
            or row["score"] != "matrix_qlike"
            or row["phases"] != []
            or row["status"] not in {"INSUFFICIENT_MEASUREMENT", "INSUFFICIENT_DATA"}
            or any(
                row[key] != 1.0
                for key in ("p_conservative", "p_holm_wave", "p_holm_cumulative")
            )
        ):
            raise AssertionError(
                "Measurement failure cannot retain fitted results or informative p-values"
            )
    same_tree(
        json.loads((report / "failure.json").read_text()),
        metrics,
        "Canonical measurement failure publication",
    )
    return {
        "status": "VERIFIED_INSUFFICIENT_MEASUREMENT",
        "measurement_audit": measurement,
        "new_hypotheses_verified": 2,
        "cumulative_hypotheses_verified": 112,
        "leads": [],
        "ledger_events_verified": verify_ledger(root, metrics, prior, "unevaluable"),
    }


def verify(root=ROOT):
    root = Path(root)
    output, report = root / "data/joint_risk", root / "reports/joint_risk"
    protocol_path = root / "joint_risk.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    validate_protocol(protocol)
    manifest = json.loads((report / "manifest.json").read_text())
    verify_manifest_coverage(root, protocol, manifest)
    if manifest["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Frozen joint-risk protocol changed")
    for group in ("code", "inputs", "preserved"):
        for path, expected in manifest[group].items():
            if digest(root / path) != expected:
                raise AssertionError("Frozen source, code or prior artifact changed: " + path)
    qqq, spx, iv, source_audit = load_source_tables(root, protocol)
    same_tree(
        json.loads((output / "source_audit.json").read_text()),
        source_audit,
        "Independent paired raw source provenance",
    )
    measured = measurement_audit(qqq, spx)
    same_tree(
        json.loads((output / "measurement_audit.json").read_text()),
        measured,
        "Full-reference measurement gate before any sample mask",
    )
    metrics = json.loads((report / "metrics.json").read_text())
    if metrics["protocol_sha256"] != digest(protocol_path):
        raise AssertionError("Joint-risk metric protocol identity differs")
    if measured["status"] == "INSUFFICIENT_MEASUREMENT":
        result = verify_terminal_measurement(
            root, measured, metrics, inherited_rows(root, protocol)
        )
    else:
        require_measurement(measured)
        if metrics.get("status") == "UNEVALUABLE":
            raise AssertionError(
                "Measurement gate passed but completed fit/scoring outcome is absent"
            )
        features, targets = feature_target_tables(qqq, spx, iv)
        for filename, expected, numeric in (
            ("features", features, ALL_FEATURES),
            ("targets", targets, TARGETS),
        ):
            actual = pd.read_parquet(output / (filename + ".parquet"))
            if not actual.index.equals(expected.index) or tuple(actual.columns) != tuple(
                expected.columns
            ):
                raise AssertionError("Independent joint-risk table schema or dates differ")
            same(
                actual.loc[:, numeric],
                expected.loc[:, numeric],
                "Independent raw marginal/joint input and target cells",
                rtol=1e-10,
                atol=1e-12,
            )
            for column in set(expected) - set(numeric):
                if not actual[column].equals(expected[column]):
                    raise AssertionError("Independent joint-risk table maturity differs")
        forecasts = pd.read_parquet(output / "forecasts.parquet")
        fits = json.loads((output / "fits.json").read_text())
        if metrics["evidence_class"] != protocol["evidence_class"]:
            raise AssertionError("Exploratory ETF/index evidence classification differs")
        result = {
            "status": "VERIFIED",
            "raw_feature_rows_verified": len(features),
            "raw_feature_columns_verified": len(ALL_FEATURES),
            "measurement_audit": measured,
            "forecast_reconstruction": verify_forecasts(
                features, targets, forecasts, fits, protocol
            ),
            "inference": verify_metrics(root, forecasts, protocol, metrics),
            "ledger_events_verified": verify_ledger(
                root, metrics, metrics["inherited_rows"], "evaluated"
            ),
        }
    result.update(
        protocol_sha256=digest(protocol_path), verifier_sha256=digest(Path(__file__))
    )
    result["limitations"] = [
        "QQQ ETF and SPX price-index archival daily return residual second moments; no measured high-frequency covariance or exact Nasdaq100-index claim.",
        "Shared diagonal or mean misspecification can reward dynamic dependence without true correlation predictability; current-fit training residual staging.",
        "The GK measurement gate protects risk-history predictors; zero and signed return targets remain valid. No execution or profit claim.",
        "Reused history and back-calculated early VIX9D inputs remain exploratory limitations.",
    ]
    (report / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def invalidate_publication(root, error):
    report = Path(root) / "reports/joint_risk"
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
        "cumulative_hypothesis_count": 112,
        "protocol_sha256": protocol_hash,
        "rows": [
            {
                "study": "joint_risk",
                "candidate": candidate,
                "control": control,
                "score": "matrix_qlike",
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
        "# QQQ–SPX joint intraday residual second moments\n\nUNEVALUABLE: independent verification failed. Both comparisons retained with p=1; no lead.\n\n"
        + message
        + "\n"
    )
    (report / "verification.json").write_text(
        json.dumps(
            {"status": "FAILED", "error": message, "protocol_sha256": protocol_hash}, indent=2
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
    with (report / "trial_ledger.jsonl").open("a") as stream:
        for row in failure["rows"]:
            stream.write(
                json.dumps(
                    {"event": "verification_failed", **row}, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    return failure


def verify_with_failure_guard(root=ROOT):
    try:
        return verify(root)
    except Exception as error:
        invalidate_publication(root, error)
        raise


if __name__ == "__main__":
    print(json.dumps(verify_with_failure_guard(), indent=2, allow_nan=False))
