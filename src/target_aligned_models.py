"""Two sequential bounded scalar fits for the issued residual cross moment.

Only one common positive unit is used for numerical conditioning. It does not
weight individual observations. An exactly flat slope objective has the
canonical coefficient zero; nonzero arithmetic underflow is never a flat fit.
"""

from __future__ import annotations

import math

import numpy as np

CORRELATION_BOUND = 0.995
SLOPE_BOUND = 1.0
KKT_EPS_MULTIPLIER = 512.0
MODELS = ("aligned_constant", "aligned_dynamic")


def _real(values, label):
    if np.iscomplexobj(values):
        raise ValueError("Real " + label + " required")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Finite " + label + " required")
    return array


def _multiply(left, right, label):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.multiply(left, right)
    if not np.isfinite(value).all():
        raise ValueError("Unrepresentable " + label + " multiplication")
    if np.any((np.asarray(left) != 0) & (np.asarray(right) != 0) & (value == 0)):
        raise ValueError("Nonzero " + label + " multiplication underflow")
    return value


def _divide(numerator, denominator, label):
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        value = np.divide(numerator, denominator)
    if not np.isfinite(value).all():
        raise ValueError("Unrepresentable " + label + " division")
    if np.any((np.asarray(numerator) != 0) & (value == 0)):
        raise ValueError("Nonzero " + label + " division underflow")
    return value


def _mean(values, label):
    flat = np.asarray(values).ravel()
    if not len(flat):
        raise ValueError("Nonempty " + label + " reduction required")
    try:
        total = math.fsum(float(value) for value in flat)
    except (OverflowError, ValueError) as error:
        raise ValueError("Unrepresentable " + label + " sum") from error
    return float(_divide(total, len(flat), label + " mean"))


def _subtract(left, right, label):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.subtract(left, right)
    return _real(value, label)


def _objective(residual):
    return _mean(
        _multiply(residual, residual, "normalized squared error"), "normalized squared error"
    )


def _moments(h, z):
    variance, state = _real(h, "marginal second moments"), _real(z, "standardized state")
    if (
        variance.ndim != 2
        or variance.shape[1] != 2
        or not len(variance)
        or state.shape != (len(variance),)
        or (variance <= 0).any()
    ):
        raise ValueError("Positive paired marginal moments and aligned state vector required")
    geometric = _multiply(
        np.sqrt(variance[:, 0]), np.sqrt(variance[:, 1]), "geometric marginal moment"
    )
    if (geometric <= 0).any():
        raise ValueError("Positive geometric marginal moments required")
    phi = np.tanh(state)
    if not np.isfinite(phi).all() or (abs(phi) > 1).any() or np.any((state != 0) & (phi == 0)):
        raise ValueError("Unrepresentable bounded state transformation")
    return geometric, phi


def projected_gradient(coefficient, gradient, bounds):
    coefficient, gradient = float(coefficient), float(gradient)
    lower, upper = map(float, bounds)
    if (
        not np.isfinite([coefficient, gradient, lower, upper]).all()
        or not lower <= coefficient <= upper
    ):
        raise ValueError("Finite feasible scalar coefficient and gradient required")
    if (coefficient == lower and gradient >= 0) or (coefficient == upper and gradient <= 0):
        return 0.0
    return gradient


def _gradient(coefficient, denominator, numerator, bounds):
    first = float(
        _multiply(
            2.0,
            _multiply(coefficient, denominator, "gradient coefficient norm"),
            "gradient first term",
        )
    )
    second = float(_multiply(2.0, numerator, "gradient second term"))
    gradient = float(_subtract(first, second, "full quadratic gradient"))
    allowance = 0.0
    for term in (first, second):
        magnitude = abs(term)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            unit = max(
                np.finfo(float).eps * magnitude, magnitude - np.nextafter(magnitude, 0.0)
            )
            allowance += KKT_EPS_MULTIPLIER * unit
    if not np.isfinite(allowance):
        raise ValueError("Nonfinite numerical KKT allowance")
    projected = projected_gradient(coefficient, gradient, bounds)
    if abs(projected) > allowance:
        raise ValueError("Bounded quadratic coefficient fails the fixed scale-aware KKT check")
    return gradient, projected, float(allowance)


def _scalar_fit(regressor, response, bound):
    squares = _multiply(regressor, regressor, "scalar regressor norm")
    denominator = _mean(squares, "scalar regressor norm")
    if denominator <= 0:
        raise ValueError("Nonpositive scalar regressor norm; no numerical fallback")
    numerator = _mean(
        _multiply(regressor, response, "scalar regressor response"), "scalar numerator"
    )
    unconstrained = float(_divide(numerator, denominator, "unconstrained coefficient"))
    coefficient = float(np.clip(unconstrained, -bound, bound))
    prediction = _multiply(coefficient, regressor, "scalar fitted value")
    residual = _subtract(response, prediction, "scalar fitted residual")
    gradient, projected, tolerance = _gradient(
        coefficient, denominator, numerator, [-bound, bound]
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "unconstrained_coefficient": unconstrained,
        "coefficient": coefficient,
        "bounds": [-bound, bound],
        "objective": _objective(residual),
        "gradient": gradient,
        "projected_gradient": projected,
        "gradient_tolerance": tolerance,
    }, residual


def fit_cross_moment(residual, h, z):
    """Fit the bounded constant first, then one slope with its intercept fixed."""
    e = _real(residual, "current-fit training residuals")
    geometric, phi = _moments(h, z)
    if e.shape != (len(geometric), 2) or len(e) < 2:
        raise ValueError("At least two aligned paired training residuals required")
    product = _multiply(e[:, 0], e[:, 1], "realized training cross product")
    common_unit = _mean(geometric, "common conditioning unit")
    if common_unit <= 0:
        raise ValueError("Positive finite common conditioning unit required")
    d = _divide(geometric, common_unit, "normalized geometric moment")
    y = _divide(product, common_unit, "normalized realized product")
    constant, remaining = _scalar_fit(d, y, CORRELATION_BOUND)
    a = constant["coefficient"]
    headroom = float(CORRELATION_BOUND - abs(a))
    w = _multiply(_multiply(headroom, d, "slope headroom design"), phi, "bounded slope design")
    if np.all(w == 0):
        status = "FLAT_OBJECTIVE"
        dynamic = {
            "numerator": 0.0,
            "denominator": 0.0,
            "unconstrained_coefficient": None,
            "coefficient": 0.0,
            "bounds": [-SLOPE_BOUND, SLOPE_BOUND],
            "objective": constant["objective"],
            "gradient": 0.0,
            "projected_gradient": 0.0,
            "gradient_tolerance": 0.0,
        }
    else:
        status = "IDENTIFIED"
        dynamic, _ = _scalar_fit(w, remaining, SLOPE_BOUND)
    return {
        "train_n": len(e),
        "common_unit": common_unit,
        "a": a,
        "b": dynamic["coefficient"],
        "headroom": headroom,
        "slope_status": status,
        "constant": constant,
        "dynamic": dynamic,
    }


def predict_cross_moment(h, z, audit):
    """Return the two bounded correlation arrays without fitting or rescaling."""
    geometric, phi = _moments(h, z)
    a, b, headroom = (float(audit[name]) for name in ("a", "b", "headroom"))
    if (
        not np.isfinite([a, b, headroom]).all()
        or not abs(a) <= CORRELATION_BOUND
        or not abs(b) <= SLOPE_BOUND
        or headroom != CORRELATION_BOUND - abs(a)
        or audit["slope_status"] not in ("IDENTIFIED", "FLAT_OBJECTIVE")
        or (audit["slope_status"] == "FLAT_OBJECTIVE" and b != 0)
        or (headroom == 0 and audit["slope_status"] != "FLAT_OBJECTIVE")
    ):
        raise ValueError("Valid fixed bounded scalar audit required")
    increment = _multiply(
        _multiply(b, headroom, "prediction slope headroom"),
        phi,
        "prediction bounded increment",
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        dynamic = a + increment
    if not np.isfinite(dynamic).all() or (abs(dynamic) > CORRELATION_BOUND).any():
        raise ValueError("Structurally bounded correlation forecast is not representable")
    return {"aligned_constant": np.full(len(geometric), a), "aligned_dynamic": dynamic}
