"""Small convex, bounded precision gate; caller owns all historical eligibility."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize


def _vector(value, name, *, positive=False):
    try:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a one-dimensional real numeric vector")
        result = raw.astype(np.float64, copy=True)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name} vector") from error
    if not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _coefficients(value):
    result = _vector(value, "coefficients")
    if result.shape != (2,) or np.any((result < 0) | (result > 1)):
        raise ValueError("Exactly two coefficients in [0,1] required")
    return result


def _inputs(base, adaptive, state):
    b = _vector(base, "baseline", positive=True)
    a = _vector(adaptive, "adaptive", positive=True)
    s = _vector(state, "state")
    if not (len(b) == len(a) == len(s)):
        raise ValueError("Aligned baseline, adaptive and state vectors required")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = b / a
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("Baseline/adaptive ratios must be finite and positive")
    z = np.tanh(s)
    weights = np.column_stack(((1 - z) / 2, (1 + z) / 2))
    return b, ratio, weights


def _relative_precision(coefficients, ratio, weights):
    w = weights @ coefficients
    if not np.isfinite(w).all() or np.any((w < 0) | (w > 1)):
        raise ValueError("Precision weights outside the declared coefficient box")
    # The positive sum avoids losing a tiny ratio when w equals one.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        p = (1 - w) + w * ratio
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Relative precision must be finite and positive")
    return p


def _objective_gradient(coefficients, scaled_y, ratio, weights):
    """Objective and full two-coordinate derivative on already validated inputs."""
    try:
        p = _relative_precision(coefficients, ratio, weights)
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            objective = float(
                np.mean(scaled_y * p - np.log(p)) + 0.005 * np.dot(coefficients, coefficients)
            )
            gradient = (
                np.mean(((scaled_y - 1 / p) * (ratio - 1))[:, None] * weights, axis=0)
                + 0.01 * coefficients
            )
    except FloatingPointError as error:
        raise ValueError("Invalid precision-gate objective arithmetic") from error
    if not np.isfinite(objective) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite precision-gate objective or gradient")
    return objective, gradient


def predict_gate(base, adaptive, state, coefficients):
    """Return positive harmonic-mixture forecasts without any clipping or refit."""
    c = _coefficients(coefficients)
    b, ratio, weights = _inputs(base, adaptive, state)
    try:
        p = _relative_precision(c, ratio, weights)
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            prediction = b / p
    except FloatingPointError as error:
        raise ValueError("Invalid precision-gate prediction arithmetic") from error
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError("Precision-gate forecasts must be finite and positive")
    return prediction


def fit_gate(base, adaptive, y, state, constant=False):
    """Fit the fixed strictly convex box problem, returning coefficients and audit."""
    if type(constant) is not bool:
        raise ValueError("constant must be an exact bool")
    b, ratio, weights = _inputs(base, adaptive, state)
    target = _vector(y, "target", positive=True)
    if not len(b) or len(target) != len(b):
        raise ValueError("Nonempty aligned fitting vectors required")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        scaled_y = target / b
    if not np.isfinite(scaled_y).all() or np.any(scaled_y <= 0):
        raise ValueError("Target/baseline ratios must be finite and positive")

    def objective(theta):
        c = np.repeat(theta, 2) if constant else theta
        value, gradient = _objective_gradient(c, scaled_y, ratio, weights)
        return value, np.array([gradient.sum()]) if constant else gradient

    dimensions = 1 if constant else 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = minimize(
                objective,
                np.zeros(dimensions, dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1.0)] * dimensions,
                options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
            )
    except (FloatingPointError, RuntimeWarning) as error:
        raise ValueError("Precision-gate optimizer arithmetic failed; no fallback") from error
    if not result.success:
        raise ValueError(f"Precision-gate optimizer failed; no fallback: {result.message}")
    theta = _vector(result.x, "optimizer coefficients")
    if theta.shape != (dimensions,):
        raise ValueError("Invalid optimizer coefficient shape")
    c = _coefficients(np.repeat(theta, 2) if constant else theta)
    value, full_gradient = _objective_gradient(c, scaled_y, ratio, weights)
    gradient = np.array([full_gradient.sum()]) if constant else full_gradient
    projected = theta - np.clip(theta - gradient, 0.0, 1.0)
    projected_max = float(np.max(np.abs(projected)))
    if not np.isfinite(projected_max) or projected_max > 1e-8:
        raise ValueError("Precision-gate projected gradient exceeds 1e-8; no fallback")
    return {
        "coefficients": c.tolist(),
        "constant": constant,
        "objective": value,
        "gradient": full_gradient.tolist(),
        "optimization_gradient": gradient.tolist(),
        "projected_gradient_max_abs": projected_max,
        "n_train": len(b),
        "n_iterations": int(result.nit),
        "solver": "L-BFGS-B",
        "success": True,
    }
