"""Fixed-offset nuisance fitting and scalar event-adjacency correction.

Pure supplied-table calculations; no file access, source admission or empirical
runner. Monthly historical offsets use that month's saved baseline fit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from . import event_cluster_features as feature
from . import range_alert_features as original_feature
from . import range_alert_models as original
from . import sign_memory_models as logistic
from .cross_moment_score import _product

ALPHA = 0.01
GRADIENT_TOLERANCE = 1e-8
XTOL = 1e-12
RTOL = 1e-14
MAX_ITERATIONS = 200
REPLAY_RTOL = 1e-10
REPLAY_ATOL = 1e-12
_newton = logistic._newton


def finite(value, label):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError("Finite real " + label + " required")
    return a.astype(float)


def scalar(value, label):
    a = finite(value, label)
    if a.ndim != 0:
        raise ValueError("Scalar " + label + " required")
    return float(a)


def summation(values):
    return scalar(math.fsum(finite(values, "summands").ravel()), "sum")


def _frame(frame):
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != feature.FEATURES
        or not len(frame)
        or frame.index.has_duplicates
    ):
        raise ValueError("Exact nonempty event-history feature schema required")
    a = finite(frame.to_numpy(), "event-history features")
    if (
        (a[:, :4] < 0).any()
        or (a[:, :4] > 1).any()
        or (np.abs(a[:, 4]) > 1).any()
        or (a[:, 5] < 0).any()
        or (a[:, 5] > 1).any()
    ):
        raise ValueError("Bounded event-history coordinates required")
    return a


def transform(train, apply):
    tr, app = _frame(train), _frame(apply)
    if len(tr) < 2:
        raise ValueError("At least two training rows required")
    constants = np.all(tr == tr[0], axis=0)
    centers = np.array(
        [float(tr[0, j]) if constants[j] else summation(tr[:, j]) / len(tr) for j in range(6)]
    )
    centered = finite(tr - centers, "centered training coordinates")
    query = finite(app - centers, "centered application coordinates")
    x = np.column_stack([np.ones(len(tr)), centered[:, :5]])
    q = np.column_stack([np.ones(len(app)), query[:, :5]])
    audit = {
        "columns": ["const", *feature.NUISANCE],
        "means": [0.0, *centers[:5].tolist()],
        "scales": [1.0] * 6,
        "nuisance_constant": constants[:5].tolist(),
        "memory_mean": float(centers[5]),
        "memory_scale": 1.0,
        "memory_constant": bool(constants[5]),
        "train_n": len(tr),
    }
    return x, q, centered[:, 5], query[:, 5], audit


def baseline_logits(frame, audit):
    base, geometry = audit["baseline"], audit["transform"]
    if (
        base["columns"] != list(original_feature.BASE)
        or geometry["columns"] != list(original_feature.BASE)
        or base["means"] != geometry["means"]
        or base["scales"] != geometry["scales"]
    ):
        raise ValueError("Saved baseline geometry must retain exact original identity")
    means, scales, beta = (finite(base[name], name) for name in ("means", "scales", "beta"))
    if (
        means.shape != (26,)
        or scales.shape != (26,)
        or beta.shape != (26,)
        or means[0] != 0
        or scales[0] != 1
        or (scales[1:] <= 1e-12).any()
    ):
        raise ValueError("Exact original26-coordinate geometry required")
    raw = frame.loc[:, original_feature.RAW].copy()
    finite(raw.to_numpy(), "original raw coordinates")
    for name in ("I", "R", "skew"):
        center = scalar(geometry["curvature_means"][name], "original curvature center")
        delta = finite(raw[name].to_numpy() - center, "original curvature difference")
        raw[name + "_square"] = _product(delta, delta, "original curvature square")
    numerator = raw.loc[:, original_feature.BASE] - pd.Series(
        means, index=original_feature.BASE
    )
    with np.errstate(all="ignore"):
        design = numerator / pd.Series(scales, index=original_feature.BASE)
    x = finite(design.to_numpy(), "original saved transformation")
    if ((numerator.to_numpy() != 0) & (x == 0)).any():
        raise ValueError("Original-coordinate division underflow")
    return original._logit(x, beta), design


def scalar_objective(coefficient, z, y, offset):
    b = scalar(coefficient, "cluster coefficient")
    z, y, eta = finite(z, "memory"), original._binary(y), finite(offset, "offset")
    if z.shape != y.shape or eta.shape != y.shape or (np.abs(z) > 2).any():
        raise ValueError("Aligned bounded centered memory, labels and offset required")
    eta = finite(eta + _product(b, z, "scalar correction"), "corrected logits")
    signed = (1 - 2 * y) * eta
    with np.errstate(all="ignore"):
        terms = np.logaddexp(0, signed)
        residual = np.where(y == 1, -expit(-eta), expit(eta))
    if (terms == 0).any() or (residual == 0).any():
        raise ValueError("Finite-logit likelihood or residual underflow")
    objective = summation(terms) / len(y) + float(
        _product(ALPHA, _product(b, b, "ridge square"), "ridge")
    )
    contribution = _product(z, residual, "scalar derivative contributions")
    gradient = summation(contribution) / len(y) + float(
        _product(2 * ALPHA, b, "ridge gradient")
    )
    return scalar(objective, "scalar objective"), scalar(gradient, "scalar gradient")


def fit_scalar(z, y, offset):
    z, y, eta = finite(z, "memory"), original._binary(y), finite(offset, "offset")
    _, zero_gradient = scalar_objective(0.0, z, y, eta)
    radius = (summation(np.abs(z)) / len(y) + 1) / (2 * ALPHA)
    lower = scalar_objective(-radius, z, y, eta)[1]
    upper = scalar_objective(radius, z, y, eta)[1]
    if not lower < 0 < upper:
        raise ValueError("Guaranteed scalar bracket failed")
    iterations, calls = 0, 0
    if np.array_equal(z, np.zeros(len(z))):
        b, status = 0.0, "EXACT_CONSTANT_INPUT"
    elif zero_gradient == 0:
        b, status = 0.0, "BALANCED_AT_ZERO"
    else:
        try:
            b, solved = brentq(
                lambda c: scalar_objective(c, z, y, eta)[1],
                -radius,
                radius,
                xtol=XTOL,
                rtol=RTOL,
                maxiter=MAX_ITERATIONS,
                full_output=True,
            )
        except Exception as exc:
            raise ValueError("Fixed scalar solve failed; no retry") from exc
        if (
            solved.converged is not True
            or isinstance(solved.iterations, bool)
            or not isinstance(solved.iterations, (int, np.integer))
            or not 0 <= solved.iterations <= MAX_ITERATIONS
            or isinstance(solved.function_calls, bool)
            or not isinstance(solved.function_calls, (int, np.integer))
            or solved.function_calls < 2
        ):
            raise ValueError("Fixed scalar convergence record required")
        iterations, calls, status = (
            int(solved.iterations),
            int(solved.function_calls),
            "FITTED",
        )
    objective, gradient = scalar_objective(b, z, y, eta)
    if not -radius <= b <= radius or abs(gradient) > GRADIENT_TOLERANCE:
        raise ValueError("Original full scalar gradient gate failed")
    return float(b), {
        "coefficient": float(b),
        "status": status,
        "alpha": ALPHA,
        "objective": objective,
        "gradient": gradient,
        "gradient_max_abs": abs(gradient),
        "bracket": [-radius, radius],
        "bracket_gradients": [lower, upper],
        "gradient_at_zero": zero_gradient,
        "iterations": iterations,
        "function_calls": calls,
        "success": True,
        "method": "brentq",
        "xtol": XTOL,
        "rtol": RTOL,
        "maxiter": MAX_ITERATIONS,
    }


def fit_stages(train, y, apply, train_eta, apply_eta, baseline_probability):
    if not isinstance(y, pd.Series) or not y.index.equals(train.index):
        raise ValueError("Exactly aligned mature training rows and labels required")
    labels = original._binary(y.to_numpy())
    support = original._support(labels, 50)
    x, q, z, qz, geometry = transform(train, apply)
    train_eta, apply_eta = (
        finite(train_eta, "training offsets"),
        finite(apply_eta, "query offsets"),
    )
    if train_eta.shape != (len(train),) or apply_eta.shape != (len(apply),):
        raise ValueError("Exact offset dimensions required")
    baseline = original._probabilities(baseline_probability, len(apply))
    if not np.allclose(expit(apply_eta), baseline, rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
        raise ValueError("Original query probability does not replay")
    beta, solver = _newton(x, labels, np.zeros(6), offset=train_eta)
    beta = original._check_solver(beta, solver, x, labels, offset=train_eta)
    for j, constant in enumerate(geometry["nuisance_constant"], start=1):
        if constant and beta[j] != 0:
            raise ValueError("Constant nuisance slope must remain canonically zero")
    nuisance_train = finite(train_eta + original._logit(x, beta), "nuisance training logits")
    correction = original._logit(q, beta)
    nuisance_query = finite(apply_eta + correction, "nuisance query logits")
    p_nuisance = np.where(correction == 0, baseline, expit(nuisance_query))
    b, scalar_fit = fit_scalar(z, labels, nuisance_train)
    delta = _product(b, qz, "query cluster correction")
    query = finite(nuisance_query + delta, "cluster query logits")
    p_cluster = np.where(delta == 0, p_nuisance, expit(query))
    if geometry["memory_constant"] and b != 0:
        raise ValueError("Constant memory must retain zero correction")
    predictions = {
        "nuisance": original._probabilities(p_nuisance, len(apply)),
        "cluster": original._probabilities(p_cluster, len(apply)),
    }
    return predictions, {
        "transform": geometry,
        "support": support,
        "baseline_frozen": True,
        "training_offset_kind": "current_month_saved_fit_in_sample",
        "nuisance": {"beta": beta.tolist(), "alpha": ALPHA, **solver},
        "cluster": scalar_fit,
    }
