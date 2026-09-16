"""Checked positive-risk scores and stable candidate-minus-control gaps."""

from __future__ import annotations

import numpy as np

from .cross_moment_score import _finite, _product


def divide(numerator, denominator):
    a, b = _finite(numerator, "numerator"), _finite(denominator, "denominator")
    if np.any(b <= 0):
        raise ValueError("Strictly positive denominator required")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        value = a / b
    value = _finite(value, "quotient")
    if np.any((a != 0) & (value == 0)):
        raise ValueError("Nonzero quotient underflow; no zero replacement")
    return value


def components(y, prediction):
    y, h = _finite(y, "risk target"), _finite(prediction, "risk forecast")
    if y.ndim != 1 or h.shape != y.shape or not len(y) or np.any(y <= 0) or np.any(h <= 0):
        raise ValueError(
            "Aligned nonempty strictly positive real risk targets and forecasts required"
        )
    log = _finite(np.log(h), "log forecast")
    ratio = divide(y, h)
    loss = _finite(log + ratio, "proper variance score")
    return log, ratio, loss


def proper_score(y, prediction):
    return components(y, prediction)[2]


def unit(value):
    absolute = np.abs(_finite(value, "coherence component"))
    return np.maximum(np.finfo(float).eps * absolute, absolute - np.nextafter(absolute, 0.0))


def require_coherence(actual, expected, component_values):
    actual, expected = _finite(actual, "stable gap"), _finite(expected, "direct gap")
    if actual.shape != expected.shape or any(
        np.shape(v) != actual.shape for v in component_values
    ):
        raise ValueError("Aligned score-coherence components required")
    allowance = 64.0 * sum(
        (unit(v) for v in [*component_values, actual, expected]), start=np.zeros_like(actual)
    )
    allowance = _finite(allowance, "coherence allowance")
    if np.any(np.abs(actual - expected) > allowance):
        raise ValueError("Stable proper-score gap exceeds fixed component roundoff envelope")


def paired_difference(candidate, control, y):
    a, b, target = (_finite(v, "paired score input") for v in (candidate, control, y))
    if a.shape != b.shape or a.shape != target.shape or a.ndim != 1:
        raise ValueError("Aligned one-dimensional paired risk inputs required")
    log_a, ratio_a, loss_a = components(target, a)
    log_b, ratio_b, loss_b = components(target, b)
    difference = _finite(a - b, "forecast difference")
    relative = divide(difference, np.maximum(a, b))
    log_gap = np.empty_like(relative)
    close = np.abs(relative) <= 0.5
    positive = close & (relative >= 0)
    negative = close & (relative < 0)
    log_gap[positive] = -np.log1p(-relative[positive])
    log_gap[negative] = np.log1p(relative[negative])
    log_gap[~close] = log_a[~close] - log_b[~close]
    ratio_gap = _product(relative, divide(target, np.minimum(a, b)), "relative risk ratio gap")
    stable = _finite(log_gap - ratio_gap, "stable proper variance gap")
    direct = _finite(loss_a - loss_b, "direct proper variance gap")
    require_coherence(stable, direct, [log_a, ratio_a, log_b, ratio_b])
    return stable
