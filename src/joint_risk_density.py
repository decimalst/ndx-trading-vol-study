"""Unhalved two-asset matrix score and bounded scalar dependence estimation.

Constant correlation is globally checked by all stationary cubic roots and
endpoints. The one-slope model uses deterministic interval branch-and-bound
with an analytic curvature lower bound. Its reported value certificate is
numerical, with conservative floating inflation, not exact interval arithmetic.
This module has no source reader or empirical entry point.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.optimize import minimize

RHO_MAX = 0.995
PARAMETER_BOUNDS = (-4.0, 4.0)
ALPHA = 0.01
GLOBAL_VALUE_TOLERANCE = 1e-8
PROJECTED_GRADIENT_TOLERANCE = 1e-7
MAX_INTERVAL_SPLITS = 32768
FLOAT_INFLATION = 256 * np.finfo(float).eps
LOCAL_OPTIONS = {"maxiter": 1000, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9}


def _residuals(values):
    array = np.asarray(values, float)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or not len(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Finite nonempty two-asset residual matrix required")
    return array


def matrix_score(residuals, h, rho):
    """Per-row logdet(H)+e' inv(H)e, H=diag(sqrt(h)) R(rho) diag(sqrt(h))."""
    e = _residuals(residuals)
    variance = np.asarray(h, float)
    correlation = np.asarray(rho, float)
    if variance.shape != e.shape:
        raise ValueError("One positive two-asset diagonal forecast per residual required")
    if correlation.ndim == 0:
        correlation = np.full(len(e), float(correlation))
    if (
        correlation.shape != (len(e),)
        or not np.isfinite(variance).all()
        or (variance <= 0).any()
        or not np.isfinite(correlation).all()
        or (np.abs(correlation) >= 1).any()
    ):
        raise ValueError("Finite positive diagonals and nonsingular correlations required")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        standardized = e / np.sqrt(variance)
        determinant = 1 - correlation**2
        quadratic = (
            standardized[:, 0] - correlation * standardized[:, 1]
        ) ** 2 / determinant + standardized[:, 1] ** 2
        result = np.log(variance).sum(axis=1) + np.log1p(-(correlation**2)) + quadratic
    if not np.isfinite(result).all():
        raise ValueError("Unrepresentable finite matrix score; no clipping or variance repair")
    return result


def full_matrix_score(residuals, covariance):
    """Score one common or row-specific finite symmetric positive-definite2x2 H."""
    e = _residuals(residuals)
    matrix = np.asarray(covariance, float)
    if matrix.shape == (2, 2):
        matrix = np.broadcast_to(matrix, (len(e), 2, 2))
    if (
        matrix.shape != (len(e), 2, 2)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix, matrix.transpose(0, 2, 1))
    ):
        raise ValueError("Finite exactly symmetric two-asset forecast matrices required")
    h = np.diagonal(matrix, axis1=1, axis2=2)
    if (h <= 0).any():
        raise ValueError("Positive matrix diagonal required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        correlation = (matrix[:, 0, 1] / np.sqrt(h[:, 0])) / np.sqrt(h[:, 1])
    return matrix_score(e, h, correlation)


def _sech_squared(eta):
    # This remains positive after tanh(eta) has rounded to1 in double precision.
    with np.errstate(under="ignore", over="ignore"):
        exponent = np.exp(-2 * np.abs(eta))
    return 4 * exponent / (1 + exponent) ** 2


def _prepared(residuals, z, constant):
    e = _residuals(residuals)
    with np.errstate(over="ignore", invalid="ignore"):
        squares = np.square(e).sum(axis=1)
        product = e[:, 0] * e[:, 1]
    if not np.isfinite(squares).all() or not np.isfinite(product).all():
        raise ValueError("Unrepresentable standardized residual moments")
    if constant is None:
        if z is not None:
            raise ValueError("Constant-correlation fit takes no slope signal")
        signal = None
    else:
        constant = float(constant)
        signal = np.asarray(z, float)
        if (
            not np.isfinite(constant)
            or not PARAMETER_BOUNDS[0] <= constant <= PARAMETER_BOUNDS[1]
            or signal.shape != (len(e),)
            or not np.isfinite(signal).all()
        ):
            raise ValueError(
                "Fixed bounded intercept and finite aligned slope signal required"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            finite_box = np.isfinite(constant + 4 * np.abs(signal)).all()
        if not finite_box:
            raise ValueError("Unrepresentable bounded signal index")
    return squares, product, signal, constant


def _value_gradient(parameter, squares, product, signal, constant):
    if not np.isfinite(parameter):
        raise ValueError("Finite scalar dependence parameter required")
    eta = parameter if constant is None else constant + parameter * signal
    rho = RHO_MAX * np.tanh(eta)
    determinant = 1 - rho * rho
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        losses = np.log1p(-rho * rho) + (squares - 2 * product * rho) / determinant
        polynomial = rho**3 - product * rho * rho + (squares - 1) * rho - product
        derivative = 2 * polynomial / (determinant * determinant)
        derivative = derivative * RHO_MAX * _sech_squared(eta)
        if constant is not None:
            derivative = derivative * signal
        value = float(losses.mean() + (ALPHA * parameter**2 if constant is not None else 0.0))
        gradient = float(
            derivative.mean() + (2 * ALPHA * parameter if constant is not None else 0.0)
        )
    if not np.isfinite([value, gradient]).all():
        raise ValueError("Nonfinite scalar score or analytic derivative")
    return value, gradient


def dependence_objective(parameter, standardized_residuals, z=None, constant=None):
    """Mean correlation-only score; candidate holds a0 fixed and penalizes b²."""
    squares, product, signal, intercept = _prepared(standardized_residuals, z, constant)
    return _value_gradient(float(parameter), squares, product, signal, intercept)


def projected_gradient(parameter, gradient):
    if parameter == PARAMETER_BOUNDS[0] and gradient >= 0:
        return 0.0
    if parameter == PARAMETER_BOUNDS[1] and gradient <= 0:
        return 0.0
    return float(gradient)


def _base_audit(parameter, value, gradient, constant):
    return {
        "parameter": float(parameter),
        "objective": float(value),
        "gradient": float(gradient),
        "projected_gradient": projected_gradient(parameter, gradient),
        "bounds": list(PARAMETER_BOUNDS),
        "rho_max": RHO_MAX,
        "penalty": 0.0 if constant is None else ALPHA,
        "constant": constant,
        "global_value_tolerance": GLOBAL_VALUE_TOLERANCE,
        "projected_gradient_tolerance": PROJECTED_GRADIENT_TOLERANCE,
    }


def _constant_fit(squares, product):
    A, B = float(squares.mean()), float(product.mean())
    coefficients = np.array([1.0, -B, A - 1.0, -B])
    roots = np.roots(coefficients)
    bound = RHO_MAX * np.tanh(PARAMETER_BOUNDS[1])
    stationary = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1e-10 and -bound < root.real < bound
    )
    for root in stationary:
        scale = 1 + sum(
            abs(coefficient) * abs(root) ** (3 - number)
            for number, coefficient in enumerate(coefficients)
        )
        if abs(np.polyval(coefficients, root)) > 1e-10 * scale:
            raise ValueError("Constant stationary cubic root fails numerical residual check")
    parameters = [
        PARAMETER_BOUNDS[0],
        *[float(np.arctanh(root / RHO_MAX)) for root in stationary],
        PARAMETER_BOUNDS[1],
    ]
    candidates = []
    for parameter in parameters:
        value, gradient = _value_gradient(parameter, squares, product, None, None)
        candidates.append(
            {
                "parameter": parameter,
                "objective": value,
                "gradient": gradient,
                "projected_gradient": projected_gradient(parameter, gradient),
            }
        )
    eligible = [
        row
        for row in candidates
        if abs(row["projected_gradient"]) <= PROJECTED_GRADIENT_TOLERANCE
    ]
    if not eligible:
        raise ValueError("No bounded stationary constant-correlation solution meets fixed KKT")
    # Objective ties within1e-12 use the smaller parameter, fixing sign-symmetric minima.
    minimum = min(row["objective"] for row in candidates)
    tied = [row for row in eligible if row["objective"] <= minimum + 1e-12]
    if not tied:
        raise ValueError("Global constant-correlation candidate does not meet fixed KKT")
    best = min(tied, key=lambda row: row["parameter"])
    audit = _base_audit(best["parameter"], best["objective"], best["gradient"], None)
    lower = float(np.nextafter(minimum - FLOAT_INFLATION * (1 + abs(minimum)), -np.inf))
    upper = best["objective"] + FLOAT_INFLATION * (1 + abs(best["objective"]))
    audit.update(
        method="all_real_stationary_cubic_roots_and_endpoints",
        stationary_rho_roots=stationary,
        candidates=candidates,
        global_lower_bound=lower,
        global_upper_bound=upper,
        global_value_gap=float(upper - lower),
        interval_splits=0,
        function_evaluations=len(candidates),
        local_attempts=[],
        success=True,
    )
    if not np.isfinite(audit["global_value_gap"]) or not (
        0 <= audit["global_value_gap"] <= GLOBAL_VALUE_TOLERANCE
    ):
        raise ValueError("Constant numerical global-value certificate exceeds tolerance")
    return audit


def interval_curvature_bound(left, right, squares, product, signal, constant):
    """Conservative upper bound on absolute f'' over a closed b interval."""
    eta_left, eta_right = constant + left * signal, constant + right * signal
    low, high = np.minimum(eta_left, eta_right), np.maximum(eta_left, eta_right)
    rabs = np.nextafter(
        np.maximum(np.abs(RHO_MAX * np.tanh(low)), np.abs(RHO_MAX * np.tanh(high))), np.inf
    )
    determinant = 1 - rabs * rabs
    pabs = rabs**3 + np.abs(product) * rabs**2 + np.abs(squares - 1) * rabs + np.abs(product)
    ppabs = 3 * rabs * rabs + 2 * np.abs(product) * rabs + np.abs(squares - 1)
    g1 = 2 * pabs / determinant**2
    g2 = 2 * ppabs / determinant**2 + 8 * rabs * pabs / determinant**3
    closest = np.where((low <= 0) & (high >= 0), 0.0, np.minimum(np.abs(low), np.abs(high)))
    first = RHO_MAX * _sech_squared(closest)
    second = np.maximum(
        2 * RHO_MAX * np.abs(np.tanh(low)) * _sech_squared(low),
        2 * RHO_MAX * np.abs(np.tanh(high)) * _sech_squared(high),
    )
    critical = np.arctanh(1 / np.sqrt(3))
    includes_extremum = ((low <= critical) & (high >= critical)) | (
        (low <= -critical) & (high >= -critical)
    )
    second = np.where(
        includes_extremum, np.maximum(second, 4 * RHO_MAX / (3 * np.sqrt(3))), second
    )
    with np.errstate(over="ignore", invalid="ignore"):
        bound = float(np.mean(signal**2 * (g2 * first**2 + g1 * second)) + 2 * ALPHA)
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("Nonfinite analytic curvature bound; certificate unavailable")
    return float(np.nextafter(bound * (1 + FLOAT_INFLATION) + FLOAT_INFLATION, np.inf))


def _candidate_fit(squares, product, signal, constant):
    cache, attempts, starts = {}, [], set()
    best = None

    def point(parameter):
        parameter = float(parameter)
        if parameter not in cache:
            cache[parameter] = _value_gradient(parameter, squares, product, signal, constant)
        return cache[parameter]

    def accept(parameter):
        nonlocal best
        if not PARAMETER_BOUNDS[0] <= parameter <= PARAMETER_BOUNDS[1]:
            return
        value, gradient = point(parameter)
        if abs(projected_gradient(parameter, gradient)) > PROJECTED_GRADIENT_TOLERANCE:
            return
        if best is None or value < best[1] or (value == best[1] and parameter < best[0]):
            best = float(parameter), value, gradient

    def refine(start):
        start = float(start)
        if start in starts:
            return
        starts.add(start)

        def objective(theta):
            value, gradient = point(theta[0])
            return value, np.array([gradient])

        fitted = minimize(
            objective,
            np.array([start]),
            jac=True,
            method="L-BFGS-B",
            bounds=[PARAMETER_BOUNDS],
            options=dict(LOCAL_OPTIONS),
        )
        parameter = float(fitted.x[0])
        value, gradient = point(parameter)
        attempts.append(
            {
                "start": start,
                "parameter": parameter,
                "objective": value,
                "gradient": gradient,
                "projected_gradient": projected_gradient(parameter, gradient),
                "success": bool(fitted.success),
                "status": int(fitted.status),
                "message": str(fitted.message),
                "iterations": int(fitted.nit),
                "function_evaluations": int(fitted.nfev),
            }
        )
        accept(parameter)

    for parameter in (*PARAMETER_BOUNDS, 0.0):
        accept(parameter)
    refine(0.0)

    def node(left, right):
        midpoint, radius = (left + right) / 2, (right - left) / 2
        value, gradient = point(midpoint)
        if best is None or value < best[1]:
            refine(midpoint)
        curvature = interval_curvature_bound(left, right, squares, product, signal, constant)
        spread = abs(gradient) * radius + 0.5 * curvature * radius * radius
        lower = value - spread - FLOAT_INFLATION * (1 + abs(value) + spread)
        lower = float(np.nextafter(lower, -np.inf))
        if not np.isfinite(lower):
            raise ValueError("Nonfinite interval lower bound")
        return lower, left, right

    active = [node(*PARAMETER_BOUNDS)]
    pruned, splits = [], 0
    while True:
        upper = np.inf if best is None else best[1] + FLOAT_INFLATION * (1 + abs(best[1]))
        lower = min(
            active[0][0] if active else np.inf, min((row[0] for row in pruned), default=np.inf)
        )
        if best is not None:
            gap = upper - lower
            if not np.isfinite(gap) or gap < 0:
                raise ValueError("Inconsistent numerical certificate bounds or nonfinite gap")
            if gap <= GLOBAL_VALUE_TOLERANCE:
                break
        if splits >= MAX_INTERVAL_SPLITS or not active:
            raise ValueError(
                f"Global numerical certificate budget exhausted: splits={splits}, gap={upper - lower}"
            )
        interval = heapq.heappop(active)
        if interval[0] >= upper:
            pruned.append(interval)
            continue
        _, left, right = interval
        midpoint = (left + right) / 2
        if midpoint in (left, right):
            raise ValueError("Floating interval cannot be subdivided; certificate unavailable")
        heapq.heappush(active, node(left, midpoint))
        heapq.heappush(active, node(midpoint, right))
        splits += 1
    leaves = sorted([*pruned, *active], key=lambda row: row[1])
    lower = min(row[0] for row in leaves)
    upper = best[1] + FLOAT_INFLATION * (1 + abs(best[1]))
    audit = _base_audit(*best, constant)
    audit.update(
        method="best_first_interval_curvature_certificate",
        global_lower_bound=lower,
        global_upper_bound=upper,
        global_value_gap=float(upper - lower),
        interval_splits=splits,
        maximum_interval_splits=MAX_INTERVAL_SPLITS,
        function_evaluations=len(cache),
        local_options=dict(LOCAL_OPTIONS),
        local_attempts=attempts,
        certificate_intervals=[
            {"left": left, "right": right, "lower_bound": bound}
            for bound, left, right in leaves
        ],
        floating_inflation=FLOAT_INFLATION,
        success=True,
    )
    if (
        abs(audit["projected_gradient"]) > PROJECTED_GRADIENT_TOLERANCE
        or not np.isfinite(audit["global_value_gap"])
        or not 0 <= audit["global_value_gap"] <= GLOBAL_VALUE_TOLERANCE
    ):
        raise ValueError(
            "Final bounded correlation solution fails fixed KKT or numerical certificate"
        )
    return audit


def fit_dependence(standardized_residuals, z=None, constant=None):
    """Fit a0 if constant=None; otherwise fit b with that a0 held fixed."""
    squares, product, signal, intercept = _prepared(standardized_residuals, z, constant)
    if intercept is None:
        return _constant_fit(squares, product)
    return _candidate_fit(squares, product, signal, intercept)
