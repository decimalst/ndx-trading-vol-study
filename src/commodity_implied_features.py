"""One-session-delayed commodity histories and option-implied log variances."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SOURCE_FLOOR = pd.Timestamp("2009-01-02")
SOURCE_CEILING = pd.Timestamp("2025-10-20")
CROSS_COLUMNS = ("hyg", "tlt", "gld", "uso", "uup")
IV_COLUMNS = ("OVX", "GVZ")
HISTORY_COLUMNS = (
    "uso_ret",
    "uso_r2",
    "uso_lrv5",
    "uso_lrv22",
    "gld_ret",
    "gld_r2",
    "gld_lrv5",
    "gld_lrv22",
)
VALUE_COLUMNS = HISTORY_COLUMNS + ("lovx", "lgvz")


def _calendar(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError("Unique ordered naive midnight calendar required")
    try:
        index.as_unit("ns")
    except (ValueError, OverflowError) as error:
        raise ValueError("Nanosecond-representable calendar required") from error
    if np.any(index > SOURCE_CEILING):
        raise ValueError("Calendar exceeds fixed source ceiling")


def _source(frame, columns):
    if (
        not isinstance(frame, pd.DataFrame)
        or not frame.columns.is_unique
        or set(frame.columns) != set(columns)
    ):
        raise ValueError("Exact source column names and unique columns required")
    _calendar(frame.index)


def _aligned_values(frame, names, calendar):
    # Read only the named commodity columns; other cross-asset values are opaque.
    aligned = frame.loc[:, list(names)].reindex(calendar).copy()
    for dtype in aligned.dtypes:
        if pd.api.types.is_bool_dtype(dtype) or not (
            pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
        ):
            raise ValueError("Real numeric nonboolean commodity columns required")
    aligned.loc[calendar < SOURCE_FLOOR, :] = np.nan
    values = aligned.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if np.isinf(values).any() or np.any(np.isfinite(values) & (values <= 0)):
        raise ValueError("Positive finite observed commodity prices and IV required")
    lagged = np.full(values.shape, np.nan)
    lagged[1:] = values[:-1]
    return lagged


def _strict_square(values):
    """Preserve unknown/zero returns; fail if a nonzero square cannot be represented."""
    values = np.asarray(values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Nonfinite return arithmetic")
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            squared = np.square(values)
    except FloatingPointError as error:
        raise ValueError(
            "Squared-return underflow or overflow; no zero substitution"
        ) from error
    observed = np.isfinite(values)
    if np.any(observed & (~np.isfinite(squared) | ((values != 0) & (squared == 0)))):
        raise ValueError("Invalid nonzero squared-return arithmetic")
    return squared


def _strict_log_mean(squared, window):
    result = np.full(len(squared), np.nan)
    if len(squared) < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(squared, window)
    complete = np.isfinite(windows).all(axis=1)
    positions = np.flatnonzero(complete)
    if not len(positions):
        return result
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            means = windows[positions].mean(axis=1)
            if (
                not np.isfinite(means).all()
                or np.any(means < 0)
                or np.any((means == 0) & np.any(windows[positions] != 0, axis=1))
            ):
                raise ValueError("Invalid strict variance mean")
            positive = means > 0
            result[positions[positive] + window - 1] = np.log(means[positive])
    except FloatingPointError as error:
        raise ValueError(
            "Strict variance arithmetic failed; no epsilon or row dropping"
        ) from error
    return result


def build_commodity_features(reference_calendar, cross, commodity_iv):
    """Retain the full reference calendar; unknown cells never borrow observations."""
    # Check every complete date envelope before examining selected numeric cells.
    _calendar(reference_calendar)
    _source(cross, CROSS_COLUMNS)
    _source(commodity_iv, IV_COLUMNS)
    prices = _aligned_values(cross, ("uso", "gld"), reference_calendar)
    implied = _aligned_values(commodity_iv, IV_COLUMNS, reference_calendar)
    output = {}
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            for column, asset in enumerate(("uso", "gld")):
                levels = np.log(prices[:, column])
                returns = np.full(len(reference_calendar), np.nan)
                returns[1:] = levels[1:] - levels[:-1]
                squared = _strict_square(returns)
                output[asset + "_ret"] = returns
                output[asset + "_r2"] = squared
                for window in (5, 22):
                    output[asset + f"_lrv{window}"] = _strict_log_mean(squared, window)
            for column, name in enumerate(("lovx", "lgvz")):
                output[name] = 2 * (np.log(implied[:, column]) - math.log(100)) - math.log(252)
    except FloatingPointError as error:
        raise ValueError(
            "Commodity feature arithmetic failed; no clipping or fallback"
        ) from error
    frame = pd.DataFrame(output, index=reference_calendar, columns=VALUE_COLUMNS)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Nonfinite commodity feature arithmetic")
    cutoff = pd.Series(reference_calendar, index=reference_calendar).shift(1)
    frame["commodity_cutoff_date"] = cutoff.where(cutoff >= SOURCE_FLOOR).astype(
        "datetime64[ns]"
    )
    return frame
