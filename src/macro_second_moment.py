"""Positive conditional second moments with zero-compatible proper scoring.

Fixed Newton/Armijo solver: mean quasi-likelihood plus .01 standardized slope
penalty, unpenalized intercept. No empirical execution entrypoint exists here.
"""
from __future__ import annotations

import numpy as np

ALPHA = .01
MAX_ITER = 200
MAX_BACKTRACK = 60
GRADIENT_TOLERANCE = 1e-8
ARMIJO = 1e-4


def proper_score(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if (y.shape != prediction.shape or y.ndim != 1 or not np.isfinite(y).all()
            or not np.isfinite(prediction).all() or (y < 0).any() or (prediction <= 0).any()):
        raise ValueError("Finite nonnegative second moments and positive forecasts required")
    with np.errstate(over="ignore"):
        result = np.log(prediction)+y/prediction
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite proper score")
    return result


def objective(beta, design, scaled_y, alpha=ALPHA):
    """Mean eta + q exp(-eta), its gradient and Hessian in scaled target units."""
    beta, design, y = np.asarray(beta, float), np.asarray(design, float), np.asarray(scaled_y, float)
    if (design.ndim != 2 or beta.shape != (design.shape[1],) or y.shape != (len(design),)
            or len(y) == 0 or not np.isfinite(beta).all() or not np.isfinite(design).all()
            or not np.isfinite(y).all() or (y < 0).any() or not (design[:, 0] == 1).all()
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("Invalid finite nonnegative objective inputs")
    eta = design@beta
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        ratio = np.exp(np.log(y)-eta)
    if not np.isfinite(ratio).all():
        return np.inf, np.full(beta.shape, np.nan), np.full((len(beta), len(beta)), np.nan)
    slopes = beta.copy()
    slopes[0] = 0
    value = float(np.mean(eta+ratio) + alpha*(slopes@slopes))
    gradient = design.T@(1-ratio)/len(y) + 2*alpha*slopes
    hessian = design.T@(design*ratio[:, None])/len(y)
    hessian += np.diag(np.r_[0., np.full(len(beta)-1, 2*alpha)])
    return value, gradient, hessian


def fit_second_moment(train_x, y, apply_x):
    """Fit one fixed positive model, rejecting degenerate or unconverged fits."""
    x, query, y = np.asarray(train_x, float), np.asarray(apply_x, float), np.asarray(y, float)
    if (x.ndim != 2 or query.ndim != 2 or query.shape[1] != x.shape[1]
            or y.shape != (len(x),) or len(x) < 2 or not np.isfinite(x).all()
            or not np.isfinite(query).all() or not np.isfinite(y).all() or (y < 0).any()):
        raise ValueError("Aligned finite training/application data and nonnegative targets required")
    mean_y = float(y.mean())
    if not np.isfinite(mean_y) or mean_y <= 0:
        raise ValueError("INSUFFICIENT_DATA: positive finite training mean required")
    means, scales = x.mean(axis=0), x.std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale input, no fallback")
    design = np.c_[np.ones(len(x)), (x-means)/scales]
    application = np.c_[np.ones(len(query)), (query-means)/scales]
    scaled_y = y/mean_y
    beta = np.zeros(design.shape[1])
    converged, iterations, backtracks = False, 0, 0
    for iteration in range(MAX_ITER):
        value, gradient, hessian = objective(beta, design, scaled_y)
        if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise ValueError("Invalid objective or derivative during fit")
        if np.max(np.abs(gradient)) <= GRADIENT_TOLERANCE:
            converged = True
            iterations = iteration
            break
        direction = np.linalg.solve(hessian, gradient)
        descent = float(gradient@direction)
        if not np.isfinite(descent) or descent <= 0:
            raise ValueError("Non-descent Newton direction")
        accepted = False
        for trial in range(MAX_BACKTRACK):
            step = .5**trial
            trial_beta = beta-step*direction
            trial_value, _, _ = objective(trial_beta, design, scaled_y)
            if np.isfinite(trial_value) and trial_value <= value-ARMIJO*step*descent:
                beta = trial_beta
                backtracks += trial
                accepted = True
                break
        if not accepted:
            raise ValueError("Newton line search did not converge")
    if not converged:
        raise ValueError("Second-moment model did not converge within fixed iteration budget")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prediction = np.exp(np.log(mean_y)+application@beta)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Invalid positive forecast; no clipping fallback")
    actual_beta = beta.copy()
    actual_beta[0] += np.log(mean_y)
    return {
        "prediction": prediction, "beta": actual_beta, "scaled_beta": beta,
        "means": means, "scales": scales, "train_mean": mean_y,
        "objective": float(value+np.log(mean_y)), "gradient_max_abs": float(np.max(np.abs(gradient))),
        "iterations": iterations, "backtracks": backtracks, "train_n": len(y), "alpha": ALPHA,
    }
