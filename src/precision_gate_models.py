"""Fixed common12 log-OLS experts with exact expert-specific Duan smearing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.claims_release_models import MARKET, _targets

RCOND = 1e-12
MINIMUM_SCALE = 1e-12


def _matrix(frame):
    if (
        not isinstance(frame, pd.DataFrame)
        or not frame.columns.is_unique
        or not frame.index.is_unique
        or not set(MARKET) <= set(frame.columns)
    ):
        raise ValueError("Unique frame with all12 market features required")
    for name in MARKET:
        dtype = frame[name].dtype
        if pd.api.types.is_bool_dtype(dtype) or not (
            pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
        ):
            raise ValueError("Real nonboolean features required")
    matrix = frame.loc[:, list(MARKET)].to_numpy(dtype=float, na_value=np.nan, copy=True)
    if not np.isfinite(matrix).all() or not np.all(matrix[:, 0] == 1):
        raise ValueError("Finite common12 features and exact intercept required")
    return matrix


def _fit(raw, log_y, query, weights):
    weight_sum = float(np.sum(weights))
    means = np.sum(weights[:, None] * raw[:, 1:], axis=0) / weight_sum
    scales = np.sqrt(np.sum(weights[:, None] * (raw[:, 1:] - means) ** 2, axis=0) / weight_sum)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= MINIMUM_SCALE)
    ):
        raise ValueError("Every declared feature requires training scale above 1e-12")
    design = np.column_stack((np.ones(len(raw)), (raw[:, 1:] - means) / scales))
    application = np.column_stack((np.ones(len(query)), (query[:, 1:] - means) / scales))
    square_roots = np.sqrt(weights)
    coefficients, _, rank, singular = np.linalg.lstsq(
        design * square_roots[:, None], log_y * square_roots, rcond=RCOND
    )
    if (
        rank != len(MARKET)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(singular).all()
    ):
        raise ValueError("Fixed expert design is rank deficient or nonfinite")
    residual = log_y - design @ coefficients
    log_smearing = float(logsumexp(np.log(weights) + residual) - np.log(weight_sum))
    prediction = np.exp(application @ coefficients + log_smearing)
    if (
        not np.isfinite(log_smearing)
        or not np.isfinite(prediction).all()
        or np.any(prediction <= 0)
    ):
        raise ValueError("Nonfinite or nonpositive exact-smearing expert prediction")
    return prediction, {
        "feature_names": list(MARKET),
        "coefficients": dict(zip(MARKET, coefficients.tolist())),
        "feature_means": dict(zip(MARKET[1:], means.tolist())),
        "feature_scales": dict(zip(MARKET[1:], scales.tolist())),
        "singular_values": singular.tolist(),
        "rank": int(rank),
        "rank_relative_cutoff": RCOND,
        "weights": weights.tolist(),
        "weight_sum": weight_sum,
        "log_smearing": log_smearing,
        "normal_equation_max_abs": float(
            np.max(np.abs(design.T @ (weights * residual) / weight_sum))
        ),
        "n_train": len(raw),
        "n_application": len(query),
    }


def fit_experts(
    train_features,
    train_y,
    apply_features,
    *,
    train_positions,
    cutoff_position,
    adaptive_half_life=252,
):
    """Caller selects common mature rows; full-calendar positions fix decay."""
    train, application = _matrix(train_features), _matrix(apply_features)
    target = _targets(train_y, train_features.index)
    positions = np.asarray(train_positions)
    if (
        positions.ndim != 1
        or positions.dtype.kind not in "iu"
        or len(positions) != len(train)
        or not len(train)
        or np.any(positions < 0)
        or np.any(np.diff(positions.astype(float)) <= 0)
        or type(cutoff_position) is not int
        or cutoff_position < 0
        or np.any(positions > cutoff_position)
        or type(adaptive_half_life) is not int
        or adaptive_half_life <= 0
    ):
        raise ValueError(
            "Ordered nonnegative training positions before exact cutoff and positive half-life required"
        )
    if len(train) < len(MARKET):
        raise ValueError("Too few rows for full expert rank")
    predictions, fits = {}, {}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            adaptive = np.exp2(
                -(cutoff_position - positions.astype(float)) / adaptive_half_life
            )
            if not np.isfinite(adaptive).all() or np.any(adaptive <= 0):
                raise ValueError(
                    "Adaptive weights underflowed or became nonfinite; no fallback"
                )
            for arm, weights in (("base", np.ones(len(train))), ("adaptive", adaptive)):
                predictions[arm], fits[arm] = _fit(train, np.log(target), application, weights)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Invalid expert fitting arithmetic; no fallback") from error
    return {"predictions": predictions, "fits": fits}
