"""Shared conditional moments and a single state-dependent skew-t shape slope."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import tail_shape_density as density
from . import tail_shape_features as features
from .macro_second_moment import fit_second_moment

NU = 8.0
ALPHA = 0.01
BOUND = 3.0
LAMBDA_BOUND = 0.95
THRESHOLD = -1.5
KKT_TOLERANCE = 1e-7


def projected_gradient(theta, gradient):
    theta, gradient = np.asarray(theta, float), np.asarray(gradient, float)
    if theta.shape != gradient.shape or not np.isfinite([theta, gradient]).all():
        raise ValueError("Finite aligned parameter and gradient vectors required")
    fixed = ((theta <= -BOUND) & (gradient > 0)) | ((theta >= BOUND) & (gradient < 0))
    return np.where(fixed, 0.0, gradient)


def shape_objective(theta, z, s):
    theta, z, s = np.asarray(theta, float), np.asarray(z, float), np.asarray(s, float)
    if (
        theta.shape not in ((1,), (2,))
        or z.ndim != 1
        or z.shape != s.shape
        or len(z) < 2
        or not np.isfinite(theta).all()
        or not np.isfinite(z).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError(
            "Finite innovation, standardized SKEW and one/two shape coefficients required"
        )
    design = np.ones((len(z), len(theta)))
    if len(theta) == 2:
        design[:, 1] = s
    linear = design @ theta
    tangent = np.tanh(linear)
    lam = LAMBDA_BOUND * tangent
    value = float(-np.mean(density.logpdf(z, NU, lam)) + ALPHA * np.sum(theta[1:] ** 2))
    derivative = -density.dlogpdf_dlambda(z, NU, lam) * LAMBDA_BOUND * (1 - tangent**2)
    gradient = design.T @ derivative / len(z)
    gradient[1:] += 2 * ALPHA * theta[1:]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite shape objective or gradient; no fallback")
    return value, gradient


def fit_shape(z, s, constant=None):
    if constant is not None and (not np.isfinite(constant) or abs(constant) > BOUND):
        raise ValueError("Invalid constant-shape starting coefficient")
    start = np.array([0.0]) if constant is None else np.array([float(constant), 0.0])
    # One deterministic optimizer attempt; a success flag alone is insufficient.
    fit = minimize(
        shape_objective,
        start,
        args=(z, s),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-BOUND, BOUND)] * len(start),
        options={"maxiter": 1000, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9},
    )
    theta = np.asarray(fit.x, float)
    if (
        not fit.success
        or theta.shape != start.shape
        or not np.isfinite(theta).all()
        or (np.abs(theta) > BOUND).any()
    ):
        raise ValueError(
            "Shape optimizer did not converge to finite bounded coefficients: "
            + str(fit.message)
        )
    objective, gradient = shape_objective(theta, z, s)
    pg = projected_gradient(theta, gradient)
    if np.max(np.abs(pg)) > KKT_TOLERANCE:
        raise ValueError("Shape optimizer projected gradient exceeds fixed tolerance")
    return {
        "theta": theta.tolist(),
        "nu": NU,
        "penalty": ALPHA if len(theta) == 2 else 0.0,
        "objective": objective,
        "gradient": gradient.tolist(),
        "projected_gradient_max_abs": float(np.max(np.abs(pg))),
        "iterations": int(fit.nit),
        "success": bool(fit.success),
        "start": start.tolist(),
        "bounds": [[-BOUND, BOUND]] * len(start),
    }


def _serializable(audit):
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in audit.items()
    }


def fit_predict(train, targets, apply):
    if not train.index.equals(targets.index):
        raise ValueError("Training labels and feature rows must have identical ordering")
    u = targets.y.to_numpy(float)
    event = targets.event.to_numpy(float)
    if (
        len(u) < 2
        or not np.isfinite(u).all()
        or not np.isfinite(event).all()
        or not np.array_equal(event, (u < THRESHOLD).astype(float))
    ):
        raise ValueError(
            "Finite labels with exact candidate-independent event definition required"
        )
    tr, ap, transform_audit = features.transform(train, apply)
    columns = features.BASE[1:]
    x, query = tr.loc[:, columns].to_numpy(float), ap.loc[:, columns].to_numpy(float)
    means, scales = x.mean(axis=0), x.std(axis=0, ddof=0)
    z, application = (x - means) / scales, (query - means) / scales
    u_mean = float(u.mean())
    beta = np.linalg.solve(
        z.T @ z / len(z) + ALPHA * np.eye(z.shape[1]), z.T @ (u - u_mean) / len(z)
    )
    train_mu = u_mean + z @ beta
    mu = u_mean + application @ beta
    mean_gradient = z.T @ (z @ beta - (u - u_mean)) / len(z) + ALPHA * beta
    if not np.isfinite([*train_mu, *mu, *beta, *mean_gradient]).all():
        raise ValueError("Nonfinite shared mean fit")
    mean_audit = {
        "columns": ["const", *columns],
        "means": [0.0, *means.tolist()],
        "scales": [1.0, *scales.tolist()],
        "beta": [u_mean, *beta.tolist()],
        "alpha": ALPHA,
        "train_n": len(u),
        "gradient_max_abs": float(np.max(np.abs(mean_gradient))),
    }
    residual = u - train_mu
    q = residual**2
    variance_fit = fit_second_moment(x, q, np.concatenate([x, query], axis=0))
    all_h = variance_fit.pop("prediction")
    train_h, h = all_h[: len(x)], all_h[len(x) :]
    innovations = residual / np.sqrt(train_h)
    skew_mean = float(tr["skew"].mean())
    skew_scale = float(tr["skew"].std(ddof=0))
    s = (tr["skew"].to_numpy(float) - skew_mean) / skew_scale
    s_apply = (ap["skew"].to_numpy(float) - skew_mean) / skew_scale
    constant = fit_shape(innovations, s)
    candidate = fit_shape(innovations, s, constant=constant["theta"][0])
    lam0 = np.full(len(ap), LAMBDA_BOUND * np.tanh(constant["theta"][0]))
    lam1 = LAMBDA_BOUND * np.tanh(candidate["theta"][0] + candidate["theta"][1] * s_apply)
    cutoff = (THRESHOLD - mu) / np.sqrt(h)
    probability0 = density.cdf(cutoff, NU, lam0)
    probability1 = density.cdf(cutoff, NU, lam1)
    frequency = float(event.mean())
    predictions = {
        "frequency": np.full(len(ap), frequency),
        "constant_shape": probability0,
        "skew_shape": probability1,
    }
    if any(
        not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in predictions.values()
    ):
        raise ValueError("Invalid event probabilities; no clipping fallback")
    return {
        "predictions": predictions,
        "conditional": {
            "mu": {"constant_shape": mu.copy(), "skew_shape": mu.copy()},
            "variance": {"constant_shape": h.copy(), "skew_shape": h.copy()},
            "lambda": {"constant_shape": lam0, "skew_shape": lam1},
        },
        "moment_audit": {
            "mean": mean_audit,
            "variance": {"columns": ["const", *columns], **_serializable(variance_fit)},
        },
        "shape_audit": {"constant_shape": constant, "skew_shape": candidate},
        "transform_audit": transform_audit,
        "skew_mean": skew_mean,
        "skew_scale": skew_scale,
        "frequency": frequency,
    }


def log_density(u, mu, variance, lam):
    u, mu, variance, lam = np.broadcast_arrays(
        *[np.asarray(x, float) for x in (u, mu, variance, lam)]
    )
    if u.ndim != 1 or not np.isfinite([u, mu, variance, lam]).all() or (variance <= 0).any():
        raise ValueError("Finite density arguments and positive conditional variance required")
    output = density.logpdf((u - mu) / np.sqrt(variance), NU, lam) - 0.5 * np.log(variance)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite full-return log density")
    return output
