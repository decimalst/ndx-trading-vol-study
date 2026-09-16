"""Three fixed, matched-cohort log-target OLS models with causal Duan smearing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)
MATCHED = MARKET + (
    "claim_m4",
    "claim_age",
    "entry_dow_1",
    "entry_dow_2",
    "entry_dow_3",
    "entry_dow_4",
)
ALL = (*MATCHED, "claim_x")
RANK_RCOND = 1e-12
MINIMUM_SCALE = 1e-12


def _matrix(frame):
    if (
        not isinstance(frame, pd.DataFrame)
        or not frame.columns.is_unique
        or not frame.index.is_unique
        or not set(ALL) <= set(frame.columns)
    ):
        raise ValueError("Unique DataFrame rows/columns and all19 named features required")
    for name in ALL:
        dtype = frame[name].dtype
        if pd.api.types.is_bool_dtype(dtype) or not (
            pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
        ):
            raise ValueError("Real numeric, nonboolean feature columns required")
    values = frame.loc[:, list(ALL)].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if not np.isfinite(values).all() or not np.all(values[:, 0] == 1.0):
        raise ValueError("All common features must be finite and intercept exactly one")
    return values


def _targets(target, index):
    if isinstance(target, pd.Series) and not target.index.equals(index):
        raise ValueError("Training target Series must align exactly with feature index")
    values = np.asarray(target)
    if values.ndim != 1 or values.shape[0] != len(index) or values.dtype.kind not in "iuf":
        raise ValueError("Aligned one-dimensional real numeric target required")
    values = values.astype(np.float64, copy=True)
    if not np.isfinite(values).all() or not np.all(values > 0):
        raise ValueError("Finite positive training targets required")
    return values


def _log_mean_exp(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty residual vector required")
    maximum = float(np.max(values))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = maximum + math.log(float(np.mean(np.exp(values - maximum))))
    if not math.isfinite(result):
        raise ValueError("Nonfinite log smearing")
    return result


def _fit_arm(train, log_y, application, names):
    columns = len(names)
    raw, query = train[:, :columns], application[:, :columns]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        means = np.mean(raw[:, 1:], axis=0)
        scales = np.std(raw[:, 1:], axis=0, ddof=0)
        if (
            not np.isfinite(means).all()
            or not np.isfinite(scales).all()
            or np.any(scales <= MINIMUM_SCALE)
        ):
            raise ValueError("Every declared feature must have finite positive training scale")
        design = np.column_stack((np.ones(len(raw)), (raw[:, 1:] - means) / scales))
        query_design = np.column_stack((np.ones(len(query)), (query[:, 1:] - means) / scales))
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design, log_y, rcond=RANK_RCOND
        )
        if (
            rank != columns
            or not np.isfinite(coefficients).all()
            or not np.isfinite(singular_values).all()
        ):
            raise ValueError("Declared model design is rank deficient or nonfinite")
        residuals = log_y - design @ coefficients
        log_smearing = _log_mean_exp(residuals)
        log_prediction = query_design @ coefficients + log_smearing
        prediction = np.exp(log_prediction)
        if not np.isfinite(prediction).all() or np.any(prediction <= 0):
            raise ValueError("Nonfinite or nonpositive OLS-smearing prediction")
        normal_equation = float(np.max(np.abs(design.T @ residuals / len(raw))))
    fit = {
        "feature_names": list(names),
        "coefficients": {name: float(value) for name, value in zip(names, coefficients)},
        "feature_means": {name: float(value) for name, value in zip(names[1:], means)},
        "feature_scales": {name: float(value) for name, value in zip(names[1:], scales)},
        "singular_values": singular_values.tolist(),
        "rank": int(rank),
        "rank_relative_cutoff": RANK_RCOND,
        "log_smearing": log_smearing,
        "normal_equation_max_abs": normal_equation,
        "n_train": len(raw),
        "n_application": len(query),
    }
    return prediction, fit


def fit_models(train_features: pd.DataFrame, train_y, apply_features: pd.DataFrame) -> dict:
    """Fit every fixed arm on supplied common rows; caller owns temporal selection."""
    train = _matrix(train_features)
    application = _matrix(apply_features)
    target = _targets(train_y, train_features.index)
    if len(train) < len(ALL):
        raise ValueError("Common training rows cannot support the declared candidate rank")
    log_y = np.log(target)
    predictions, fits = {}, {}
    try:
        for arm, names in (("market", MARKET), ("matched", MATCHED), ("candidate", ALL)):
            predictions[arm], fits[arm] = _fit_arm(train, log_y, application, names)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Invalid fixed OLS arithmetic; no fallback allowed") from error
    return {"predictions": predictions, "fits": fits}
