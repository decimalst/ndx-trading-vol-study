"""Independent shared-marginal copula and issued-forecast reconstruction.

No wave27 feature, model, pipeline, or density producer is imported. Frozen
independent joint-risk marginal reconstruction is explicitly reused.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, hyp2f1, ndtri_exp, stdtr

from . import verify_joint_risk as moments
from .verify_claims_release_forecasts import _compare

BOUND = 0.995
FAMILIES = ("t8", "gaussian")
GAP = 1e-8
ROUNDING = 256
MODELS = ("t8_copula", "gaussian_copula", "independence")
TARGETS = ("y_qqq", "y_spx")
APPLICATION_COLUMNS = (
    "origin",
    "model",
    "horizon",
    "feature_cutoff_date",
    "mu_qqq",
    "mu_spx",
    "h_qqq",
    "h_spx",
    "rho",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "phase",
    "offset",
)
PANEL_COLUMNS = (*APPLICATION_COLUMNS, "target_end", "available_date", "y_qqq", "y_spx")
COVERAGE_COLUMNS = (
    "origin",
    "phase",
    "feature_complete",
    "missing_features",
    "feature_cutoff_date",
    "offset",
    "target_end",
    "available_date",
    "target_observed",
    "target_within_phase",
    "issued",
    "scored",
    "status",
    "fit_origin",
    "training_cutoff",
)
SCHEDULE_COLUMNS = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "feature_complete_n",
    "application_n",
    "train_n",
)


def _z(value):
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or not len(array)
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError("Nonempty finite real standardized pairs required")
    return array.astype(float)


def _rho(value, n):
    array = np.asarray(value)
    if (
        array.dtype.kind not in "fiu"
        or array.ndim > 1
        or (array.ndim == 1 and array.shape != (n,))
        or not np.isfinite(array).all()
        or (abs(array) > BOUND).any()
    ):
        raise ValueError("Finite correlation in the fixed bounded domain required")
    return np.broadcast_to(array.astype(float), (n,))


def _log1psquare(values):
    with np.errstate(divide="ignore"):
        return np.logaddexp(0.0, 2 * np.log(abs(values)) - math.log(8))


def _normal_coordinates(z):
    """Invert log tails: I_w(4,1/2)=w^4 2F1(4,1/2;5;w)/(4B)."""
    magnitude = abs(z)
    logs = np.empty_like(z)
    modest = magnitude < math.sqrt(8)
    logs[modest] = np.log(stdtr(8, -magnitude[modest]))
    logw = -_log1psquare(magnitude[~modest])
    logs[~modest] = (
        4 * logw - math.log(8) - betaln(4, 0.5) + np.log(hyp2f1(4, 0.5, 5, np.exp(logw)))
    )
    result = -np.sign(z) * ndtri_exp(logs)
    if not np.isfinite(logs).all() or not np.isfinite(result).all():
        raise ValueError("Unrepresentable log-tail transform; no endpoint clipping")
    return result


def independent_log_copula(z, rho, family):
    values = _z(z)
    correlations = _rho(rho, len(values))
    if type(family) is not str or family not in FAMILIES:
        raise ValueError("Exact t8 or gaussian family required")
    d = (1 - correlations) * (1 + correlations)
    if family == "gaussian":
        w = _normal_coordinates(values)
        product = w[:, 0] * w[:, 1]
        result = (
            -0.5 * np.log(d)
            + (correlations * product - 0.5 * correlations**2 * (w * w).sum(axis=1)) / d
        )
    else:
        scale = np.max(abs(values), axis=1)
        divisor = np.where(scale > 0, scale, 1)
        normalized = values / divisor[:, None]
        quadratic = (normalized[:, 0] - normalized[:, 1]) ** 2 / (2 * (1 - correlations)) + (
            normalized[:, 0] + normalized[:, 1]
        ) ** 2 / (2 * (1 + correlations))
        with np.errstate(divide="ignore"):
            log_quadratic = 2 * np.log(scale) + np.log(quadratic)
        log_bivariate = (
            gammaln(5)
            - gammaln(4)
            - math.log(8 * math.pi)
            - 0.5 * np.log(d)
            - 5 * np.logaddexp(0, log_quadratic - math.log(8))
        )
        log_marginal = (
            gammaln(4.5)
            - gammaln(4)
            - 0.5 * math.log(8 * math.pi)
            - 4.5 * _log1psquare(values)
        )
        result = log_bivariate - log_marginal.sum(axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite independent copula density")
    return result


def _gaussian_candidates(z):
    w = _normal_coordinates(_z(z))
    s = float(np.mean(np.sum(w * w, axis=1)))
    c = float(np.mean(w[:, 0] * w[:, 1]))
    polynomial = [1.0, -c, s - 1, -c]
    roots = np.roots(polynomial)
    values = [-BOUND, BOUND]
    for root in roots:
        if abs(root.imag) <= 1e-10 and -BOUND <= root.real <= BOUND:
            r = float(root.real)
            scaled = abs(np.polyval(polynomial, r)) / (1 + sum(abs(v) for v in polynomial))
            if scaled > 1e-10:
                raise ValueError("Independent stationary root residual failed")
            values.append(r)
    return sorted(set(values))


def _scaled(z):
    values = _z(z)
    scale = np.maximum(1.0, np.max(abs(values), axis=1))
    normalized = values / scale[:, None]
    a = (normalized * normalized).sum(axis=1)
    b = normalized[:, 0] * normalized[:, 1]
    c = 8 * (1 / scale) ** 2
    return a, b, c, scale


def _t8_reduced(z, rho):
    values = _z(z)
    r = float(_rho(rho, len(values))[0])
    a, b, c, scale = _scaled(values)
    d = (1 - r) * (1 + r)
    w = c * d + a - 2 * r * b
    derivative = -2 * c * r - 2 * b
    if (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("Invalid positive t8 quadratic")
    value = 5 * float(np.mean(np.log(w) + 2 * np.log(scale))) - 4.5 * math.log(d)
    gradient = 5 * float(np.mean(derivative / w)) + 9 * r / d
    hessian = 5 * float(np.mean(-2 * c / w - (derivative / w) ** 2)) + 9 * (1 + r * r) / (
        d * d
    )
    if not np.isfinite([value, gradient, hessian]).all():
        raise ValueError("Nonfinite t8 objective derivatives")
    return value, gradient, hessian


def _t8_interval(z, left, right):
    if not np.isfinite([left, right]).all() or not -BOUND <= left < right <= BOUND:
        raise ValueError("Nonempty fixed-domain interval required")
    a, b, c, _ = _scaled(z)
    wl = c * ((1 - left) * (1 + left)) + a - 2 * left * b
    wr = c * ((1 - right) * (1 + right)) + a - 2 * right * b
    minimum = np.minimum(wl, wr)
    maximum_derivative = np.maximum(abs(-2 * c * left - 2 * b), abs(-2 * c * right - 2 * b))
    closest = 0 if left <= 0 <= right else min(abs(left), abs(right))
    positive = 9 * (1 + closest**2) / (1 - closest**2) ** 2
    curvature = (
        float(np.mean(-10 * c / minimum - 5 * (maximum_derivative / minimum) ** 2)) + positive
    )
    midpoint = (left + right) / 2
    value, gradient, _ = _t8_reduced(z, midpoint)
    radius = (right - left) / 2
    candidates = [-radius, radius]
    if curvature > 0:
        candidates.append(float(np.clip(-gradient / curvature, -radius, radius)))
    decrement = min(gradient * t + curvature * t * t / 2 for t in candidates)
    raw_bound = value + decrement
    inflation = (
        ROUNDING
        * np.finfo(float).eps
        * (1 + abs(value) + abs(gradient) * radius + abs(curvature) * radius * radius)
    )
    lower = raw_bound - inflation
    if not np.isfinite([curvature, midpoint, value, gradient, lower]).all():
        raise ValueError("Nonfinite interval certificate")
    return {
        "left": left,
        "right": right,
        "midpoint": midpoint,
        "value": value,
        "gradient": gradient,
        "curvature_lower": curvature,
        "lower_bound": lower,
    }


def _t8_lower_bound(z, left, right):
    return _t8_interval(z, left, right)["lower_bound"]


def _keys(value, required, label):
    if type(value) is not dict or set(value) != set(required):
        raise ValueError("Exact " + label + " schema required")


def _real(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("Literal finite real scalar required")
    if not np.isfinite(value):
        raise ValueError("Literal finite real scalar required")
    return float(value)


def _integer(value, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError("Literal bounded integer required")
    return value


def _close(actual, expected, label, *, atol=1e-10, rtol=1e-9):
    if not math.isclose(_real(actual), _real(expected), abs_tol=atol, rel_tol=rtol):
        raise AssertionError("Independent arithmetic differs: " + label)


def _objective_gradient(z, rho, family):
    value = -float(np.mean(independent_log_copula(z, rho, family)))
    if family == "t8":
        gradient = _t8_reduced(z, rho)[1]
    else:
        w = _normal_coordinates(z)
        s = float(np.mean(np.sum(w * w, axis=1)))
        c = float(np.mean(w[:, 0] * w[:, 1]))
        gradient = (rho**3 - c * rho * rho + (s - 1) * rho - c) / (1 - rho * rho) ** 2
    return value, gradient


def _projected(rho, gradient):
    if rho == -BOUND:
        return max(0.0, -gradient)
    if rho == BOUND:
        return max(0.0, gradient)
    return abs(gradient)


def verify_dependence(z, family, rho, audit):
    """Reconstruct each certificate over its full domain, without fitting producer code."""
    z = _z(z)
    rho = _real(rho)
    _rho(rho, len(z))
    _keys(
        audit,
        (
            "status",
            "family",
            "rho",
            "objective",
            "gradient",
            "projected_gradient",
            "train_n",
            "domain",
            "certificate",
        ),
        "dependence audit",
    )
    if (
        audit["status"] != "CERTIFIED_GLOBAL_NUMERICAL_OPTIMUM"
        or audit["family"] != family
        or audit["domain"] != [-BOUND, BOUND]
        or _integer(audit["train_n"], 1) != len(z)
        or _real(audit["rho"]) != rho
    ):
        raise ValueError("Wrong dependence family, parameter, domain or training membership")
    value, gradient = _objective_gradient(z, rho, family)
    projected = _projected(rho, gradient)
    for name, expected in (
        ("objective", value),
        ("gradient", gradient),
        ("projected_gradient", projected),
    ):
        _close(audit[name], expected, name)
    if projected > 1e-7 or _real(audit["projected_gradient"]) > 1e-7:
        raise ValueError("Independent projected stationarity failed")
    cert = audit["certificate"]
    common = {
        "method",
        "upper_bound",
        "lower_bound",
        "gap",
        "gap_tolerance",
        "roundoff_multiplier",
    }
    if family == "gaussian":
        _keys(
            cert,
            common | {"polynomial_roots", "polynomial", "candidates"},
            "Gaussian certificate",
        )
        if cert["method"] != "all_real_stationary_roots_and_endpoints":
            raise ValueError("Gaussian exhaustive-root certificate required")
        w = _normal_coordinates(z)
        s = float(np.mean(np.sum(w * w, axis=1)))
        c = float(np.mean(w[:, 0] * w[:, 1]))
        polynomial = [1.0, -c, s - 1, -c]
        if not isinstance(cert["polynomial"], list) or len(cert["polynomial"]) != 4:
            raise ValueError("Exact cubic polynomial required")
        for actual, expected in zip(cert["polynomial"], polynomial):
            _close(actual, expected, "Gaussian polynomial")
        roots = np.roots(polynomial)
        if type(cert["polynomial_roots"]) is not list or len(cert["polynomial_roots"]) != 3:
            raise ValueError("Every complex cubic root required")
        for actual, expected in zip(cert["polynomial_roots"], roots):
            if type(actual) is not list or len(actual) != 2:
                raise ValueError("Real and imaginary root components required")
            _close(actual[0], expected.real, "root real", atol=1e-10)
            _close(actual[1], expected.imag, "root imaginary", atol=1e-10)
        candidates = [-BOUND, 0.0, BOUND]
        for root in roots:
            if abs(root.imag) <= 1e-10 and -BOUND <= root.real <= BOUND:
                if abs(np.polyval(polynomial, root.real)) > 1e-10:
                    raise ValueError("Independent cubic residual failed")
                candidates.append(float(root.real))
        candidates.sort()
        if type(cert["candidates"]) is not list or len(cert["candidates"]) != len(candidates):
            raise ValueError("Every Gaussian candidate including endpoints required")
        objectives = []
        for actual, parameter in zip(cert["candidates"], candidates):
            _keys(actual, ("rho", "objective"), "Gaussian candidate")
            _close(actual["rho"], parameter, "candidate parameter", atol=1e-10)
            expected = -float(np.mean(independent_log_copula(z, parameter, family)))
            _close(actual["objective"], expected, "candidate objective")
            objectives.append(expected)
        minimum = min(objectives)
        winner = min(r for r, f in zip(candidates, objectives) if f <= minimum + 1e-12)
        _close(rho, winner, "globally best tie-broken Gaussian parameter", atol=1e-10)
        lower = minimum - ROUNDING * np.finfo(float).eps * (1 + abs(value))
        covered = len(candidates)
    elif family == "t8":
        _keys(
            cert,
            common | {"splits", "max_splits", "leaves", "local_optimizer"},
            "t8 certificate",
        )
        if cert["method"] != "interval_taylor_lower_bound":
            raise ValueError("Whole-domain Taylor certificate required")
        splits = _integer(cert["splits"], 0, 32768)
        if _integer(cert["max_splits"], 0, 32768) != 32768:
            raise ValueError("Fixed global certification budget required")
        leaves = cert["leaves"]
        if type(leaves) is not list or len(leaves) != splits + 1:
            raise ValueError("Every final interval required")
        endpoint = -BOUND
        bounds = []
        for leaf in leaves:
            _keys(
                leaf,
                (
                    "left",
                    "right",
                    "midpoint",
                    "value",
                    "gradient",
                    "curvature_lower",
                    "lower_bound",
                ),
                "t8 leaf",
            )
            left, right = _real(leaf["left"]), _real(leaf["right"])
            if left != endpoint:
                raise ValueError(
                    "Certificate intervals must cover the domain without gaps or overlaps"
                )
            reconstructed = _t8_interval(z, left, right)
            midpoint = reconstructed["midpoint"]
            full_value, full_gradient = _objective_gradient(z, midpoint, family)
            curvature = reconstructed["curvature_lower"]
            radius = (right - left) / 2
            steps = [-radius, radius]
            if curvature > 0:
                steps.append(max(-radius, min(radius, -full_gradient / curvature)))
            correction = min(
                full_gradient * step + curvature * step * step / 2 for step in steps
            )
            inflation = (
                ROUNDING
                * np.finfo(float).eps
                * (
                    1
                    + abs(full_value)
                    + abs(full_gradient) * radius
                    + 0.5 * abs(curvature) * radius * radius
                )
            )
            bound = full_value + correction - inflation
            for name, expected in (
                ("midpoint", midpoint),
                ("value", full_value),
                ("gradient", full_gradient),
                ("curvature_lower", curvature),
                ("lower_bound", bound),
            ):
                _close(leaf[name], expected, "leaf " + name)
            bounds.append(bound)
            endpoint = right
        if endpoint != BOUND:
            raise ValueError("Certificate misses upper domain endpoint")
        lower = min(bounds)
        _keys(
            cert["local_optimizer"],
            ("success", "message", "rho"),
            "local optimizer diagnostics",
        )
        local = cert["local_optimizer"]
        if local["success"] is not True or type(local["message"]) is not str:
            raise ValueError("Local optimizer success must be explicit")
        _rho(_real(local["rho"]), len(z))
        covered = len(leaves)
    else:
        raise ValueError("Exact fitted copula family required")
    if _real(cert["gap_tolerance"]) != GAP or _real(cert["roundoff_multiplier"]) != ROUNDING:
        raise ValueError("Fixed global gap and roundoff constants required")
    gap = value - lower
    if gap < 0 or gap > GAP or not 0 <= _real(cert["gap"]) <= GAP:
        raise ValueError("Independent entire-domain objective gap failed")
    for name, expected in (("upper_bound", value), ("lower_bound", lower), ("gap", gap)):
        _close(cert[name], expected, "certificate " + name)
    return {
        "status": "VERIFIED",
        "family": family,
        "global_value_gap": gap,
        "projected_gradient": projected,
        "domain_components_verified": covered,
    }


def _iso(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Literal ISO calendar date required")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Valid calendar date required")
    return result


def _config(value):
    if value is None:
        value = {
            "origin_start": "2016-01-04",
            "origin_end": "2025-10-17",
            "source_end": "2025-10-20",
            "development": ["2016-01-04", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "minimum_train": 1000,
        }
    _keys(
        value,
        (
            "origin_start",
            "origin_end",
            "source_end",
            "development",
            "evaluation",
            "minimum_train",
        ),
        "forecast configuration",
    )
    result = {name: _iso(value[name]) for name in ("origin_start", "origin_end", "source_end")}
    for phase in ("development", "evaluation"):
        if type(value[phase]) is not list or len(value[phase]) != 2:
            raise ValueError("Two literal phase dates required")
        result[phase] = tuple(_iso(x) for x in value[phase])
    if not (
        result["origin_start"]
        <= result["development"][0]
        <= result["development"][1]
        < result["evaluation"][0]
        <= result["origin_end"]
        <= result["evaluation"][1]
        <= result["source_end"]
        <= pd.Timestamp("2025-10-20")
    ):
        raise ValueError("Ordered source-bounded calendar configuration required")
    result["minimum_train"] = _integer(value["minimum_train"], 2)
    return result


def _calendar(index, ceiling):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or not len(index)
        or index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
        or index[-1] > ceiling
    ):
        raise ValueError("Unique increasing midnight source calendar within ceiling required")


def _source_preflight(qqq, spx, iv, config):
    # All calendars first, before any numeric or measurement operation.
    for source in (qqq, spx, iv):
        if not isinstance(source, pd.DataFrame) or not source.columns.is_unique:
            raise ValueError("Typed unique-column source frame required")
        _calendar(source.index, config["source_end"])
    for frame, columns in (
        (qqq, ("open", "high", "low", "close")),
        (spx, ("open", "high", "low", "close")),
        (iv, ("vxn", "vix", "vix9d", "vvix")),
    ):
        if not set(columns) <= set(frame):
            raise ValueError("Every declared source column required")
        for name in columns:
            if frame[name].dtype.kind not in "fiu":
                raise ValueError("Real numeric source values required")
            values = frame[name].to_numpy(float)
            if np.isinf(values).any() or np.any(values[np.isfinite(values)] <= 0):
                raise ValueError(
                    "Positive observed source values required; missing stays missing"
                )


def _phase(date, config):
    for name in ("development", "evaluation"):
        if config[name][0] <= date <= config[name][1]:
            return name
    return "outside_phase"


def _table(actual, expected, label, *, indexed=False):
    if (
        not isinstance(actual, pd.DataFrame)
        or tuple(actual.columns) != tuple(expected.columns)
        or len(actual) != len(expected)
        or (indexed and not actual.index.equals(expected.index))
    ):
        raise AssertionError("Complete ordered table mismatch: " + label)
    dates = {
        "origin",
        "feature_cutoff_date",
        "fit_origin",
        "training_cutoff",
        "target_end",
        "available_date",
    }
    counts = {
        "horizon",
        "offset",
        "train_n",
        "requested_n",
        "feature_complete_n",
        "application_n",
    }
    flags = {"feature_complete", "target_observed", "target_within_phase", "issued", "scored"}
    for name in expected:
        left, right = actual[name], expected[name]
        if name in dates:
            if (
                not pd.api.types.is_datetime64_any_dtype(left.dtype)
                or getattr(left.dt, "tz", None) is not None
                or not pd.DatetimeIndex(left).equals(pd.DatetimeIndex(right))
            ):
                raise AssertionError("Exact native date mismatch: " + label + "." + name)
        elif name in counts:
            if left.dtype.kind not in "iu":
                raise AssertionError("Literal integer count required: " + label + "." + name)
            for a, b in zip(left, right):
                _compare(a, b, label + "." + name)
        elif name in flags:
            if left.dtype.kind != "b" or not np.array_equal(
                left.to_numpy(), right.to_numpy(bool)
            ):
                raise AssertionError("Boolean flag mismatch: " + label + "." + name)
        elif right.dtype.kind in "iuf" or name.startswith(("mu_", "h_")) or name == "rho":
            if left.dtype.kind not in "fiu":
                raise AssertionError("Numeric table column required: " + label + "." + name)
            a, b = left.to_numpy(float), right.to_numpy(float)
            forecast = name.startswith(("mu_", "h_")) or name == "rho"
            if forecast and not np.isfinite(a).all():
                raise AssertionError("Finite issued forecast required: " + label + "." + name)
            if name.startswith("h_") and np.any(a <= 0):
                raise AssertionError(
                    "Positive issued variance required: " + label + "." + name
                )
            if name == "rho" and np.any(abs(a) > BOUND):
                raise AssertionError(
                    "Bounded issued dependence required: " + label + "." + name
                )
            atol, rtol = (1e-12, 1e-7) if forecast else (1e-12, 1e-9)
            if np.isinf(a).any() or not np.allclose(
                a, b, atol=atol, rtol=rtol, equal_nan=True
            ):
                raise AssertionError("Reconstructed values mismatch: " + label + "." + name)
        else:
            for a, b in zip(left, right):
                _compare(a, b, label + "." + name)
    if {"origin", "model", "mu_qqq", "mu_spx", "h_qqq", "h_spx", "rho"} <= set(actual):
        for _, group in actual.groupby("origin", sort=False):
            values = group[["mu_qqq", "mu_spx", "h_qqq", "h_spx"]].to_numpy(float)
            if len(values) != 3 or not np.array_equal(
                values, np.repeat(values[:1], 3, axis=0)
            ):
                raise AssertionError("Exactly shared issued marginals required: " + label)
        if not actual.loc[actual.model == "independence", "rho"].eq(0.0).all():
            raise AssertionError("Exact independence placeholder required: " + label)
        if actual.groupby(["fit_origin", "model"], sort=False).rho.nunique().gt(1).any():
            raise AssertionError(
                "One fixed dependence parameter per monthly arm required: " + label
            )


def _audit_marginal_schema(audit, n):
    _keys(audit, ("mean", "variance"), "marginal model")
    _keys(
        audit["mean"],
        ("columns", "means", "scales", "beta", "alpha", "train_n", "gradient_max_abs"),
        "mean audit",
    )
    _keys(
        audit["variance"],
        (
            "columns",
            "beta",
            "scaled_beta",
            "means",
            "scales",
            "train_mean",
            "objective",
            "gradient_max_abs",
            "iterations",
            "backtracks",
            "train_n",
            "alpha",
        ),
        "variance audit",
    )
    for which in ("mean", "variance"):
        one = audit[which]
        if _integer(one["train_n"], 2) != n or _real(one["alpha"]) != 0.01:
            raise ValueError("Fixed common marginal training and ridge penalty required")
        for name in ("means", "scales", "beta"):
            if type(one[name]) is not list:
                raise ValueError("Serialized marginal coefficient arrays required")
            for value in one[name]:
                _real(value)
    _integer(audit["variance"]["iterations"], 0, 199)
    _integer(audit["variance"]["backtracks"], 0)


def _verify_model(train, y, application, audit):
    _keys(audit, ("moments", "transform", "residual_staging", "dependence"), "shared model")
    _keys(audit["moments"], ("qqq", "spx"), "paired moments")
    _keys(audit["dependence"], ("t8_copula", "gaussian_copula"), "paired dependence fits")
    if audit["residual_staging"] != "current_fit_training_residuals":
        raise ValueError("Current-fit in-sample residual staging required")
    tr, ap, transform = moments.transform(train, application)
    _compare(audit["transform"], transform, "training_transform")
    mu, h, coordinates, diagnostics = [], [], [], []
    for asset in ("qqq", "spx"):
        _audit_marginal_schema(audit["moments"][asset], len(tr))
        mean, variance, residual, proof = moments.verify_moments(
            tr, y["y_" + asset].to_numpy(float), ap, audit["moments"][asset]
        )
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scale = np.sqrt(variance[: len(tr)] * 0.75)
            z = residual / scale
        if (
            not np.isfinite(variance).all()
            or np.any(variance <= 0)
            or np.any(scale <= 0)
            or not np.isfinite(z).all()
            or np.any((residual != 0) & (z == 0))
            or np.any((residual != 0) & (residual**2 == 0))
        ):
            raise ValueError(
                "Strictly positive scales and non-underflowed residual coordinates required"
            )
        mu.append(mean[len(tr) :])
        h.append(variance[len(tr) :])
        coordinates.append(z)
        diagnostics.append(proof)
    z = np.column_stack(coordinates)
    parameters, certificates = {"independence": 0.0}, []
    for name, family in (("t8_copula", "t8"), ("gaussian_copula", "gaussian")):
        item = audit["dependence"][name]
        _keys(item, ("rho", "audit"), "model dependence")
        rho = _real(item["rho"])
        certificates.append(verify_dependence(z, family, rho, item["audit"]))
        parameters[name] = rho
    return np.column_stack(mu), np.column_stack(h), parameters, certificates


def verify_forecasts(qqq, spx, iv, produced, config=None):
    config = _config(config)
    _source_preflight(qqq, spx, iv, config)
    _keys(
        produced,
        ("features", "targets", "applications", "panel", "coverage", "schedules", "fits"),
        "produced outputs",
    )
    features, targets = moments.feature_target_tables(qqq, spx, iv)
    _table(produced["features"], features, "features", indexed=True)
    _table(produced["targets"], targets, "targets", indexed=True)
    calendar = spx.index
    previous = pd.Series(calendar, index=calendar).shift(1)
    following = pd.Series(calendar, index=calendar).shift(-1)
    complete = np.isfinite(features.loc[:, moments.ALL_FEATURES].to_numpy(float)).all(axis=1)
    observed = np.isfinite(targets.loc[:, TARGETS].to_numpy(float)).all(axis=1)
    requested = (calendar >= config["origin_start"]) & (calendar <= config["origin_end"])
    scheduled, plans = [], []
    for month in pd.period_range(config["origin_start"], config["origin_end"], freq="M"):
        positions = np.flatnonzero(requested & (calendar.to_period("M") == month))
        applications = positions[complete[positions]]
        if len(applications):
            first = applications[0]
            cutoff = previous.iloc[first]
            train = np.flatnonzero(
                complete & observed & (calendar < calendar[first]) & (following <= cutoff)
            )
            if pd.isna(cutoff) or len(train) < config["minimum_train"]:
                raise ValueError(
                    "INSUFFICIENT_DATA: common mature pairs below fixed monthly floor"
                )
            row = {
                "month": str(month),
                "status": "fitted",
                "fit_origin": calendar[first],
                "training_cutoff": cutoff,
                "requested_n": len(positions),
                "feature_complete_n": len(applications),
                "application_n": len(applications),
                "train_n": len(train),
            }
            plans.append((row, train, applications))
        else:
            row = {
                "month": str(month),
                "status": "no_complete_origin" if len(positions) else "no_requested_origins",
                "fit_origin": pd.NaT,
                "training_cutoff": pd.NaT,
                "requested_n": len(positions),
                "feature_complete_n": 0,
                "application_n": 0,
                "train_n": None,
            }
        scheduled.append(row)
    if type(produced["fits"]) is not list or len(produced["fits"]) != len(plans):
        raise AssertionError("Every scheduled monthly fit must be retained")
    _table(
        produced["schedules"], pd.DataFrame(scheduled, columns=SCHEDULE_COLUMNS), "schedules"
    )
    applications, issued, certificates = [], {}, []
    for saved, (schedule, train, query) in zip(produced["fits"], plans):
        dates = lambda positions: [date.date().isoformat() for date in calendar[positions]]
        expected = {
            "month": schedule["month"],
            "status": "fitted",
            "fit_origin": schedule["fit_origin"].date().isoformat(),
            "training_cutoff": schedule["training_cutoff"].date().isoformat(),
            "train_origins": dates(train),
            "train_positions": train.tolist(),
            "train_n": len(train),
            "planned_application_origins": dates(query),
            "application_origins": dates(query),
        }
        _keys(saved, (*expected, "model_audit"), "monthly fit")
        _compare({name: saved[name] for name in expected}, expected, "fit_membership")
        mu, h, rho, certs = _verify_model(
            features.iloc[train],
            targets.iloc[train],
            features.iloc[query],
            saved["model_audit"],
        )
        certificates.extend(certs)
        for local, position in enumerate(query):
            rows = []
            for model in MODELS:
                row = {
                    "origin": calendar[position],
                    "model": model,
                    "horizon": 1,
                    "feature_cutoff_date": previous.iloc[position],
                    "mu_qqq": float(mu[local, 0]),
                    "mu_spx": float(mu[local, 1]),
                    "h_qqq": float(h[local, 0]),
                    "h_spx": float(h[local, 1]),
                    "rho": rho[model],
                    "fit_origin": schedule["fit_origin"],
                    "training_cutoff": schedule["training_cutoff"],
                    "train_n": len(train),
                    "phase": _phase(calendar[position], config),
                    "offset": int(position % 5),
                }
                rows.append(row)
                applications.append(row)
            issued[int(position)] = rows
    _table(
        produced["applications"],
        pd.DataFrame(applications, columns=APPLICATION_COLUMNS),
        "applications",
    )
    coverage, panel = [], []
    for position in np.flatnonzero(requested):
        date = calendar[position]
        phase = _phase(date, config)
        endpoint = following.iloc[position]
        within = (
            phase != "outside_phase" and pd.notna(endpoint) and endpoint <= config[phase][1]
        )
        if not complete[position]:
            status = "incomplete_features"
        elif phase == "outside_phase":
            status = "outside_phase"
        elif pd.isna(endpoint):
            status = "target_not_mature"
        elif not observed[position]:
            status = "missing_target"
        elif not within:
            status = "target_after_phase_cutoff"
        else:
            status = "scored"
        rowset = issued.get(int(position))
        row = {
            "origin": date,
            "phase": phase,
            "feature_complete": bool(complete[position]),
            "missing_features": "|".join(
                name for name in moments.ALL_FEATURES if pd.isna(features.loc[date, name])
            ),
            "feature_cutoff_date": previous.iloc[position],
            "offset": int(position % 5),
            "target_end": endpoint,
            "available_date": endpoint,
            "target_observed": bool(observed[position]),
            "target_within_phase": bool(within),
            "issued": rowset is not None,
            "scored": status == "scored",
            "status": status,
            "fit_origin": rowset[0]["fit_origin"] if rowset else pd.NaT,
            "training_cutoff": rowset[0]["training_cutoff"] if rowset else pd.NaT,
        }
        coverage.append(row)
        if status == "scored":
            for app in rowset:
                panel.append(
                    {
                        **app,
                        "target_end": endpoint,
                        "available_date": endpoint,
                        "y_qqq": float(targets.loc[date, "y_qqq"]),
                        "y_spx": float(targets.loc[date, "y_spx"]),
                    }
                )
    _table(produced["coverage"], pd.DataFrame(coverage, columns=COVERAGE_COLUMNS), "coverage")
    _table(produced["panel"], pd.DataFrame(panel, columns=PANEL_COLUMNS), "panel")
    return {
        "status": "VERIFIED",
        "calendar_rows": len(calendar),
        "coverage_origins": len(coverage),
        "application_forecasts_verified": len(applications),
        "forecasts_verified": len(panel),
        "monthly_fits": len(plans),
        "marginal_models_verified": 4 * len(plans),
        "dependence_fits_verified": len(certificates),
        "largest_independent_global_gap": max(
            (x["global_value_gap"] for x in certificates), default=None
        ),
        "limits": [
            "Frozen independent joint-risk feature and marginal reconstruction is reused.",
            "Current-fit training residuals are in-sample staged residuals, not issued historical errors.",
            "Global certificate values and stationarity are reconstructed; local optimizer messages are not independently reproduced.",
            "Source hash authentication, historical-vintage limitations and statistical inference are separate checks.",
        ],
    }
