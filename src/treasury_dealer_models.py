"""Three fixed common32-row OLS arms, using the frozen generic smearing solver."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.claims_release_models import MARKET, _fit_arm, _targets

MATCHED = MARKET + (
    "tlt_ret",
    "tlt_r2",
    "tlt_lrv5",
    "tlt_lrv22",
    "weekday_1",
    "weekday_2",
    "weekday_3",
    "weekday_4",
    "auction_count",
    "tenor_3",
    "tenor_5",
    "tenor_7",
    "tenor_10",
    "tenor_30",
    "reopening_count",
    "log_offering_sum",
    "high_yield_sum",
    "bid_to_cover_sum",
    "prior_share_mean_sum",
)
ALL = MATCHED + ("dealer_surprise",)


def _matrix(frame):
    if (
        not isinstance(frame, pd.DataFrame)
        or not frame.index.is_unique
        or not frame.columns.is_unique
        or not set(ALL) <= set(frame.columns)
    ):
        raise ValueError("Unique frame rows/columns and all32 named features required")
    for name in ALL:
        dtype = frame[name].dtype
        if pd.api.types.is_bool_dtype(dtype) or not (
            pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
        ):
            raise ValueError("Real numeric nonboolean feature columns required")
    values = frame.loc[:, list(ALL)].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if not np.isfinite(values).all() or not np.all(values[:, 0] == 1.0):
        raise ValueError("All32 common features finite and intercept exactly one required")
    return values


def fit_models(train_features, train_y, apply_features):
    """Fit all fixed arms on exactly the supplied common rows; no row selection."""
    train = _matrix(train_features)
    application = _matrix(apply_features)
    target = _targets(train_y, train_features.index)
    if len(train) < len(ALL):
        raise ValueError("Common training rows cannot support the declared candidate rank")
    predictions, fits = {}, {}
    try:
        log_y = np.log(target)
        for arm, names in (("market", MARKET), ("matched", MATCHED), ("candidate", ALL)):
            predictions[arm], fits[arm] = _fit_arm(train, log_y, application, names)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Invalid fixed OLS arithmetic; no fallback allowed") from error
    return {"predictions": predictions, "fits": fits}
