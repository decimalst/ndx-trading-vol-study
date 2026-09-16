"""Fixed estimator arms for the additive model/memory study.

This module accepts already eligible training rows. The caller is responsible
for target maturity and temporal windows; no data source or score is read here.
Every transformation is fitted exclusively on X_train. All hyperparameters are
fixed before study scores, and unsuccessful optimizers fail without a fallback.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import SplineTransformer

BASE = (
    "const", "lrv_d", "lrv_w", "lrv_m", "lev_d", "lev_w", "lev_m",
    "liv", "lvix", "term", "xasset_stress", "market_stress",
)
MODEL_NAMES = (
    "baseline", "gamma", "adaptive_ols", "adaptive_gamma", "ridge_gamma", "spline_gamma",
)
SPLINE_COLUMNS = ("lrv_d", "lrv_w", "lrv_m", "liv", "lvix", "term")
HALF_LIFE = 252.0
GAMMA_ALPHA = 0.01
GAMMA_SOLVER = "newton-cholesky"
GAMMA_MAX_ITER = 2000
GAMMA_TOL = 1e-8
SCALE_FLOOR = 1e-12


def age_weights(ages):
    """Exponential weights using actual session ages at the fit origin."""
    age = np.asarray(ages, dtype=float)
    if age.ndim != 1 or not np.isfinite(age).all() or (age < 0).any():
        raise ValueError("Finite nonnegative one-dimensional session ages required")
    weights = np.exp2(-age / HALF_LIFE)
    if (weights <= 0).any():
        raise ValueError("Session ages underflow fixed exponential weights")
    return weights


def _validated_inputs(X_train, y_train, X_apply, ages):
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_apply, pd.DataFrame):
        raise ValueError("Training and apply features must be DataFrames")
    if tuple(X_train.columns) != BASE or tuple(X_apply.columns) != BASE:
        raise ValueError("Features must have exactly the declared baseline columns in order")
    train, apply = X_train.to_numpy(float), X_apply.to_numpy(float)
    y = np.asarray(y_train, dtype=float)
    weights = age_weights(ages)
    if (y.ndim != 1 or len(y) != len(train) or len(weights) != len(train)
            or len(train) <= len(BASE) or len(apply) < 1):
        raise ValueError("Aligned training targets and ages, sufficient rows and an apply row required")
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Positive finite training variance targets required")
    if not np.isfinite(train).all() or not np.isfinite(apply).all():
        raise ValueError("Finite feature values required")
    if not (train[:, 0] == 1).all() or not (apply[:, 0] == 1).all():
        raise ValueError("The const feature must equal one")
    return train[:, 1:], y, apply[:, 1:], weights


def _standardize(train, apply):
    center, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or (scale <= SCALE_FLOOR).any():
        raise ValueError("INSUFFICIENT_DATA: zero-scale nonconstant feature or spline basis")
    return (train - center) / scale, (apply - center) / scale, center, scale


def _positive_exp(log_prediction):
    with np.errstate(over="raise", under="raise", invalid="raise"):
        prediction = np.exp(log_prediction)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise RuntimeError("Estimator generated a nonpositive or nonfinite variance forecast")
    return prediction


def _ols(train, y, apply, weights, input_audit, adaptive):
    design = np.column_stack([np.ones(len(train)), train])
    query = np.column_stack([np.ones(len(apply)), apply])
    sqrt_weights = np.sqrt(weights)
    log_y = np.log(y)
    beta, _, rank, _ = np.linalg.lstsq(
        design * sqrt_weights[:, None], log_y * sqrt_weights, rcond=None,
    )
    if rank != design.shape[1]:
        raise ValueError("INSUFFICIENT_DATA: rank-deficient unpenalized OLS design")
    residual = log_y - design @ beta
    smear = float(np.average(_positive_exp(residual), weights=weights))
    prediction = _positive_exp(query @ beta + np.log(smear))
    gradient = design.T @ (-weights * residual / weights.sum())
    audit = {
        **input_audit,
        "converged": True,
        "objective": "weighted_squared_log_variance" if adaptive else "squared_log_variance",
        "n_train": len(train),
        "n_features": train.shape[1],
        "rank": int(rank),
        "coefficients": beta[1:].tolist(),
        "intercept": float(beta[0]),
        "smearing_factor": smear,
        "half_life_sessions": HALF_LIFE if adaptive else None,
        "weight_sum": float(weights.sum()),
        "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "gradient_inf_norm": float(np.abs(gradient).max()),
    }
    return prediction, audit


def _gamma(train, y, apply, weights, alpha, input_audit, adaptive):
    # Gamma/QLIKE is invariant to a common change in target and forecast units.
    # Normalizing tiny variances keeps an exact constant-target optimum at one,
    # avoiding a Newton line-search failure caused solely by log-scale rounding.
    target_scale = float(np.median(y))
    model = GammaRegressor(
        alpha=alpha, fit_intercept=True, solver=GAMMA_SOLVER,
        max_iter=GAMMA_MAX_ITER, tol=GAMMA_TOL,
    )
    try:
        with warnings.catch_warnings():
            # This also prohibits sklearn's warning-based solver substitution
            # when the Newton Hessian is singular or its line search fails.
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", LinAlgWarning)
            warnings.simplefilter("error", RuntimeWarning)
            model.fit(train, y / target_scale, sample_weight=weights)
    except Warning as error:
        raise RuntimeError(f"Gamma optimizer warning; no fallback permitted: {error}") from error
    if model.n_iter_ >= GAMMA_MAX_ITER:
        raise RuntimeError("Gamma optimizer exhausted its fixed iteration limit")
    intercept = float(model.intercept_ + np.log(target_scale))
    prediction = _positive_exp(apply @ model.coef_ + intercept)
    train_prediction = _positive_exp(train @ model.coef_ + intercept)
    weighted_score = weights * (1 - y / train_prediction) / weights.sum()
    gradient = np.r_[train.T @ weighted_score + alpha * model.coef_, weighted_score.sum()]
    gradient_inf_norm = float(np.abs(gradient).max())
    if not np.isfinite(gradient_inf_norm) or gradient_inf_norm > 2e-7:
        raise RuntimeError(f"Gamma objective gradient failed convergence check: {gradient_inf_norm}")
    audit = {
        **input_audit,
        "converged": True,
        "objective": "weighted_qlike" if adaptive else "qlike",
        "n_train": len(train),
        "n_features": train.shape[1],
        "coefficients": model.coef_.tolist(),
        "intercept": intercept,
        "target_scale": target_scale,
        "alpha": float(alpha),
        "solver": GAMMA_SOLVER,
        "max_iter": GAMMA_MAX_ITER,
        "tol": GAMMA_TOL,
        "n_iter": int(model.n_iter_),
        "half_life_sessions": HALF_LIFE if adaptive else None,
        "weight_sum": float(weights.sum()),
        "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "gradient_inf_norm": gradient_inf_norm,
    }
    return prediction, audit


def fit_models(X_train, y_train, X_apply, ages):
    """Fit six fixed arms and predict each apply row in its supplied order.

    Returns ``{"predictions": {model: positive_array}, "audit": {model: dict}}``.
    Ages must be the actual session distance from each training origin to the
    current fit origin; target eligibility belongs to the calling harness.
    """
    train, y, apply, adaptive_weights = _validated_inputs(X_train, y_train, X_apply, ages)
    train, apply, center, scale = _standardize(train, apply)
    input_audit = {
        "input_columns": list(BASE[1:]),
        "input_center": center.tolist(),
        "input_scale": scale.tolist(),
        "scaling_ddof": 0,
    }
    equal_weights = np.ones(len(train))
    predictions, audits = {}, {}
    for name, weights, adaptive in (
        ("baseline", equal_weights, False), ("adaptive_ols", adaptive_weights, True),
    ):
        predictions[name], audits[name] = _ols(train, y, apply, weights, input_audit, adaptive)
    for name, weights, alpha, adaptive in (
        ("gamma", equal_weights, 0., False),
        ("adaptive_gamma", adaptive_weights, 0., True),
        ("ridge_gamma", equal_weights, GAMMA_ALPHA, False),
    ):
        predictions[name], audits[name] = _gamma(
            train, y, apply, weights, alpha, input_audit, adaptive,
        )

    spline_indices = [BASE[1:].index(column) for column in SPLINE_COLUMNS]
    raw_indices = [i for i, column in enumerate(BASE[1:]) if column not in SPLINE_COLUMNS]
    transformer = SplineTransformer(
        n_knots=4, degree=3, knots="quantile", include_bias=False, extrapolation="linear",
    )
    spline_train = transformer.fit_transform(train[:, spline_indices])
    spline_apply = transformer.transform(apply[:, spline_indices])
    spline_train = np.column_stack([spline_train, train[:, raw_indices]])
    spline_apply = np.column_stack([spline_apply, apply[:, raw_indices]])
    spline_train, spline_apply, basis_center, basis_scale = _standardize(spline_train, spline_apply)
    spline_audit = {
        **input_audit,
        "spline_columns": list(SPLINE_COLUMNS),
        "spline_raw_columns": [BASE[1:][i] for i in raw_indices],
        "spline_n_knots": 4,
        "spline_degree": 3,
        "spline_include_bias": False,
        "spline_extrapolation": "linear",
        "spline_knot_rule": "quantile",
        "spline_knots": np.column_stack([spline.t for spline in transformer.bsplines_]).tolist(),
        "design_center": basis_center.tolist(),
        "design_scale": basis_scale.tolist(),
    }
    predictions["spline_gamma"], audits["spline_gamma"] = _gamma(
        spline_train, y, spline_apply, equal_weights, GAMMA_ALPHA, spline_audit, False,
    )
    return {
        "predictions": {name: predictions[name] for name in MODEL_NAMES},
        "audit": {name: audits[name] for name in MODEL_NAMES},
    }
