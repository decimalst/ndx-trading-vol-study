"""Fixed delayed TLT histories and known entry weekdays for auction controls."""

import numpy as np
import pandas as pd

from src.commodity_implied_features import (
    CROSS_COLUMNS,
    _calendar,
    _source,
    _strict_log_mean,
    _strict_square,
)

SOURCE_FLOOR = pd.Timestamp("2010-01-01")
VALUE_COLUMNS = ("tlt_ret", "tlt_r2", "tlt_lrv5", "tlt_lrv22") + tuple(
    f"weekday_{day}" for day in range(1, 5)
)


def build_market_context(reference_calendar, cross):
    """Use only prior-session TLT and preserve every full-reference-calendar gap.

    The calendar/envelope and strict arithmetic primitives are reused from the
    frozen commodity implementation. Only the TLT column is numerically read.
    Source provenance and snapshot authentication remain the caller's task.
    """
    _calendar(reference_calendar)
    _source(cross, CROSS_COLUMNS)
    values = cross.tlt.reindex(reference_calendar).copy()
    if pd.api.types.is_bool_dtype(values.dtype) or not (
        pd.api.types.is_integer_dtype(values.dtype)
        or pd.api.types.is_float_dtype(values.dtype)
    ):
        raise ValueError("Real nonboolean TLT prices required")
    values.loc[reference_calendar < SOURCE_FLOOR] = np.nan
    price = values.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if np.isinf(price).any() or np.any(np.isfinite(price) & (price <= 0)):
        raise ValueError("Positive finite observed TLT prices required")
    delayed = np.full(len(price), np.nan)
    delayed[1:] = price[:-1]
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            level = np.log(delayed)
            returns = np.full(len(level), np.nan)
            returns[1:] = level[1:] - level[:-1]
            square = _strict_square(returns)
            out = {
                "tlt_ret": returns,
                "tlt_r2": square,
                "tlt_lrv5": _strict_log_mean(square, 5),
                "tlt_lrv22": _strict_log_mean(square, 22),
            }
    except FloatingPointError as error:
        raise ValueError("TLT history arithmetic failed") from error
    for weekday in range(1, 5):
        out[f"weekday_{weekday}"] = (reference_calendar.weekday == weekday).astype(float)
    frame = pd.DataFrame(out, index=reference_calendar, columns=VALUE_COLUMNS)
    cutoff = pd.Series(reference_calendar, index=reference_calendar).shift(1)
    frame["treasury_cutoff_date"] = cutoff.where(cutoff >= SOURCE_FLOOR).astype(
        "datetime64[ns]"
    )
    return frame
