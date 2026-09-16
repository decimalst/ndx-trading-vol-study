"""Independent intercept-profiled positive-score solver; no producer imports.

The fixed objective is mean(eta + q*exp(-eta)) + .01*||slopes||².
Directional derivative roots, not rounded objective reductions, choose steps.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

ALPHA = 0.01
INTERNAL_TOLERANCE = 1e-10
FULL_TOLERANCE = 1e-8 + 1e-12
MAX_ITERATIONS = 500
MAX_BRACKET_EVALUATIONS = 60
MAX_ROOT_ITERATIONS = 200
ROOT_XTOL = 1e-14
ROOT_RTOL = 1e-14
SMALL_DIFFERENCE = 0.5


def _finite(value):
    if np.asarray(value).dtype.kind not in "iuf":
        raise ValueError("Real numeric values required; no boolean/string/object conversion")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Finite values required")
    return result


def _multiply(left, right):
    left, right = np.broadcast_arrays(_finite(left), _finite(right))
    with np.errstate(all="ignore"):
        value = left * right
    _finite(value)
    if ((left != 0) & (right != 0) & (value == 0)).any():
        raise ValueError("Nonzero multiplication underflow")
    return value


def _divide(left, right):
    left, right = np.broadcast_arrays(_finite(left), _finite(right))
    if (right == 0).any():
        raise ValueError("Zero denominator")
    with np.errstate(all="ignore"):
        value = left / right
    _finite(value)
    if ((left != 0) & (value == 0)).any():
        raise ValueError("Nonzero division underflow")
    return value


def _sum(value):
    try:
        result = math.fsum(_finite(value).ravel())
    except (OverflowError, ValueError) as exc:
        raise ValueError("Finite compensated sum required") from exc
    return float(_finite(result))


def _mean(value):
    value = _finite(value)
    if not value.size:
        raise ValueError("Nonempty mean required")
    return float(_divide(_sum(value), value.size))


def _dot(left, right):
    return _sum(_multiply(left, right))


def _matvec(design, beta):
    _multiply(design, beta)
    with np.errstate(all="ignore"):
        value = design @ beta
    return _finite(value)


def _crossproduct(left, right):
    # Check every product before the BLAS reduction can conceal underflow.
    for column in left.T:
        _multiply(column[:, None], right)
    with np.errstate(all="ignore"):
        value = left.T @ right
    return _finite(value)


def _exp(log_values):
    with np.errstate(all="ignore"):
        value = np.exp(_finite(log_values))
    _finite(value)
    if (value <= 0).any():
        raise ValueError("Positive exponential underflow")
    return value


def _inputs(design, q):
    design, q = _finite(design), _finite(q)
    if design.ndim != 2 or design.shape[1] != 31 or not len(design):
        raise ValueError("Nonempty n by 31 design required")
    if q.shape != (len(design),) or (q <= 0).any():
        raise ValueError("Aligned strictly positive target required")
    if not np.array_equal(design[:, 0], np.ones(len(design))):
        raise ValueError("Literal intercept required")
    x = design[:, 1:]
    means = np.array([_mean(column) for column in x.T])
    return design, q, x, np.log(q), means


def _profile(b, prepared, *, curvature=True):
    design, q, x, logq, means = prepared
    b = _finite(b)
    if b.shape != (30,):
        raise ValueError("Thirty slopes required")
    linear = _matvec(x, b)
    logs = _finite(logq - linear)
    shift = float(np.max(logs))
    weights = _exp(logs - shift)
    total = _sum(weights)
    intercept = float(_finite(shift + math.log(float(_divide(total, len(q))))))
    # Check the original positive-ratio domain, not just normalized weights.
    _exp(logq - _matvec(design, np.r_[intercept, b]))
    weighted_mean = np.array([float(_divide(_dot(weights, column), total)) for column in x.T])
    gradient = _finite(means - weighted_mean + _multiply(2 * ALPHA, b))
    objective = _sum([1.0, intercept, _mean(linear), float(_multiply(ALPHA, _dot(b, b)))])
    result = {
        "objective": objective,
        "gradient": gradient,
        "intercept": intercept,
        "unnormalized_weights": weights,
        "weight_total": total,
    }
    if curvature:
        normalized_weights = _divide(weights, total)
        centered = _finite(x - weighted_mean)
        weighted = _multiply(centered, normalized_weights[:, None])
        hessian = _crossproduct(centered, weighted)
        result["hessian"] = _finite((hessian + hessian.T) * 0.5 + 2 * ALPHA * np.eye(30))
    return result


def profile_objective(slopes, design, normalized_y):
    """Profile value, gradient, covariance Hessian and recovered intercept."""
    return _profile(slopes, _inputs(design, normalized_y))


def full_objective(beta, design, normalized_y):
    """Reconstruct the original full criterion with stable expm1 residuals."""
    design, q, _, logq, _ = _inputs(design, normalized_y)
    beta = _finite(beta)
    if beta.shape != (31,):
        raise ValueError("Thirty-one full coefficients required")
    eta = _matvec(design, beta)
    logratio = _finite(logq - eta)
    ratio = _exp(logratio)
    with np.errstate(all="ignore"):
        residual = _finite(-np.expm1(logratio))
    gradient = np.array([_mean(_multiply(column, residual)) for column in design.T])
    gradient[1:] += _multiply(2 * ALPHA, beta[1:])
    value = _sum([_mean(eta + ratio), float(_multiply(ALPHA, _dot(beta[1:], beta[1:])))])
    weighted = _multiply(design, ratio[:, None])
    hessian = _divide(_crossproduct(design, weighted), len(q))
    hessian += np.diag([0.0] + [2 * ALPHA] * 30)
    return value, _finite(gradient), _finite(hessian)


def _difference(b, delta, prepared, state):
    projected = _matvec(prepared[2], delta)
    if np.max(np.abs(projected)) <= SMALL_DIFFERENCE:
        with np.errstate(all="ignore"):
            increments = _finite(np.expm1(-projected))
        change = float(
            _divide(_dot(state["unnormalized_weights"], increments), state["weight_total"])
        )
        if change <= -1:
            raise ValueError("Invalid logarithmic objective difference")
        log_change = math.log1p(change)
    else:
        next_state = _profile(_finite(b + delta), prepared, curvature=False)
        log_change = next_state["intercept"] - state["intercept"]
    return _sum(
        [
            _mean(projected),
            log_change,
            float(_multiply(2 * ALPHA, _dot(b, delta))),
            float(_multiply(ALPHA, _dot(delta, delta))),
        ]
    )


def objective_difference(slopes, delta, design, normalized_y):
    prepared = _inputs(design, normalized_y)
    slopes, delta = _finite(slopes), _finite(delta)
    if slopes.shape != (30,) or delta.shape != (30,):
        raise ValueError("Thirty-dimensional displacement required")
    return _difference(slopes, delta, prepared, _profile(slopes, prepared, curvature=False))


def _line_step(b, state, prepared):
    try:
        direction = -cho_solve(
            cho_factor(state["hessian"], lower=True, check_finite=True),
            state["gradient"],
            check_finite=True,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError("Profile covariance Cholesky solve failed") from exc
    direction = _finite(direction)
    norm = math.hypot(*direction)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Nonzero finite descent direction required")
    unit = _divide(direction, norm)
    derivative0 = _dot(state["gradient"], unit)
    if derivative0 >= 0:
        raise ValueError("Strict descent direction required")
    radius = float(_divide(_multiply(-2.0, derivative0), 2 * ALPHA))
    if radius <= 0:
        raise ValueError("Positive derivative-search radius required")
    trial = min(norm, radius)
    left, derivative_left, barrier = 0.0, derivative0, None
    trials = []
    invalid = 0

    def derivative(distance):
        candidate = _finite(b + _multiply(distance, unit))
        result = _profile(candidate, prepared, curvature=False)
        return _dot(result["gradient"], unit)

    for _ in range(MAX_BRACKET_EVALUATIONS):
        try:
            value = derivative(trial)
        except ValueError as exc:
            invalid += 1
            trials.append({"distance": trial, "valid": False, "error": str(exc)})
            barrier = trial
        else:
            trials.append({"distance": trial, "valid": True, "derivative": value})
            if value >= 0:
                right, derivative_right = trial, value
                break
            left, derivative_left = trial, value
        if barrier is not None:
            candidate_trial = left + 0.5 * (barrier - left)
        else:
            if trial == radius:
                raise ValueError("Analytic directional bracket contradicted")
            candidate_trial = min(float(_multiply(2.0, trial)), radius)
        if not left < candidate_trial or (
            barrier is not None and not candidate_trial < barrier
        ):
            raise ValueError("Derivative bracket stagnated before a valid sign change")
        trial = candidate_trial
    else:
        raise ValueError("Fixed derivative bracket budget exhausted")

    width = right - left
    if width <= 0:
        raise ValueError("Strict derivative bracket required")

    def unit_derivative(fraction):
        return derivative(left + float(_multiply(fraction, width)))

    if derivative_right == 0:
        fraction, root_iterations, root_calls = 1.0, 0, 0
    else:
        try:
            fraction, result = brentq(
                unit_derivative,
                0.0,
                1.0,
                xtol=ROOT_XTOL,
                rtol=ROOT_RTOL,
                maxiter=MAX_ROOT_ITERATIONS,
                full_output=True,
                disp=False,
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError("Fixed directional root solve failed") from exc
        if not result.converged or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Directional root did not converge")
        root_iterations, root_calls = int(result.iterations), int(result.function_calls)
        if not 0 <= root_iterations <= MAX_ROOT_ITERATIONS or root_calls < 0:
            raise ValueError("Invalid directional root iteration audit")
    distance = left + float(_multiply(fraction, width))
    candidate = _finite(b + _multiply(distance, unit))
    if np.array_equal(candidate, b):
        raise ValueError("Unchanged coefficients are not a stationarity certificate")
    delta = _finite(candidate - b)
    difference = _difference(b, delta, prepared, state)
    if difference > 0:
        raise ValueError("Directional root increased the stable objective difference")
    return candidate, {
        "newton_direction_norm": norm,
        "analytic_radius": radius,
        "bracket": [left, right],
        "bracket_derivatives": [derivative_left, derivative_right],
        "bracket_trials": trials,
        "invalid_bracket_trials": invalid,
        "root_iterations": root_iterations,
        "root_function_calls": root_calls,
        "root_fraction": float(fraction),
        "step_distance": distance,
        "objective_difference": difference,
    }


def solve_baseline(design, normalized_y):
    """Return full coefficients and JSON-safe independent convergence evidence."""
    prepared = _inputs(design, normalized_y)
    b = np.zeros(30)
    start_intercept = _profile(b, prepared, curvature=False)["intercept"]
    history = []
    for iteration in range(MAX_ITERATIONS):
        state = _profile(b, prepared)
        beta = np.r_[state["intercept"], b]
        value, full_gradient, _ = full_objective(beta, prepared[0], prepared[1])
        profile_max = float(np.max(np.abs(state["gradient"])))
        full_max = float(np.max(np.abs(full_gradient)))
        if profile_max <= INTERNAL_TOLERANCE:
            if full_max > FULL_TOLERANCE:
                raise ValueError("Original full gradient failed the unchanged gate")
            norm = math.hypot(*state["gradient"])
            return beta, {
                "method": "profiled_newton_directional_brent",
                "train_n": len(prepared[1]),
                "alpha": ALPHA,
                "start_slopes": [0.0] * 30,
                "start_intercept": start_intercept,
                "iterations": iteration,
                "success": True,
                "objective": value,
                "profile_objective": state["objective"],
                "gradient": full_gradient.tolist(),
                "gradient_max_abs": full_max,
                "profile_gradient": state["gradient"].tolist(),
                "profile_gradient_max_abs": profile_max,
                "slope_distance_bound": float(_divide(norm, 2 * ALPHA)),
                "objective_gap_bound": float(_divide(_multiply(norm, norm), 4 * ALPHA)),
                "controls": {
                    "internal_tolerance": INTERNAL_TOLERANCE,
                    "full_tolerance": FULL_TOLERANCE,
                    "max_iterations": MAX_ITERATIONS,
                    "max_bracket_evaluations": MAX_BRACKET_EVALUATIONS,
                    "max_root_iterations": MAX_ROOT_ITERATIONS,
                    "root_xtol": ROOT_XTOL,
                    "root_rtol": ROOT_RTOL,
                    "small_difference": SMALL_DIFFERENCE,
                },
                "history": history,
            }
        candidate, diagnostics = _line_step(b, state, prepared)
        history.append(
            {
                "iteration": iteration,
                "objective": value,
                "profile_gradient_max_abs": profile_max,
                "full_gradient_max_abs": full_max,
                **diagnostics,
            }
        )
        b = candidate
    raise ValueError("Fixed profiled iteration budget exhausted")
