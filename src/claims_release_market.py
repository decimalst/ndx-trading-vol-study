"""Pure established market baseline and full-calendar five-session targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

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
DAILY = ("open", "high", "low", "close", "adj close", "volume")
CROSS = ("hyg", "tlt", "gld", "uso", "uup")
IV = ("vxn", "vix", "vix9d")
SOURCE_CEILING = pd.Timestamp("2025-10-20")


def _calendar(index, *, nonempty=False):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is not None
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
        or (nonempty and len(index) == 0)
    ):
        raise ValueError("Unique increasing timezone-free DatetimeIndex required")
    if not index.equals(index.normalize()):
        raise ValueError("Observed session dates must be midnight")
    if (index > SOURCE_CEILING).any():
        raise ValueError("Observed date exceeds fixed source ceiling")


def _numbers(frame, columns):
    if frame.columns.has_duplicates or set(frame.columns) != set(columns):
        raise ValueError("Exact required source columns without duplicates required")
    for dtype in frame.dtypes:
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError("Real numeric source columns required")
    values = frame.loc[:, list(columns)].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Observed infinity is invalid")
    return pd.DataFrame(values, index=frame.index, columns=columns)


def _prior_z(values):
    history = values.rolling(252, min_periods=126)
    mean = history.mean().shift(1)
    scale = history.std(ddof=1).shift(1).replace(0, np.nan)
    return (values - mean) / scale


def build_market_features(daily, cross, iv):
    """Preserve the complete daily calendar and genuine missing constituents."""
    for frame in (daily, cross, iv):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Daily, cross and IV DataFrames required")
    # Every source calendar is checked before converting any numerical values.
    _calendar(daily.index, nonempty=True)
    _calendar(cross.index)
    _calendar(iv.index)
    d, c, v = (_numbers(daily, DAILY), _numbers(cross, CROSS), _numbers(iv, IV))
    if (
        (d.loc[:, list(DAILY[:-1])] <= 0).any().any()
        or (c <= 0).any().any()
        or (v <= 0).any().any()
        or (d.volume < 0).any()
    ):
        raise ValueError("Positive observed prices/IV and nonnegative volume required")
    if (d.high < d[["open", "close", "low"]].max(axis=1)).any() or (
        d.low > d[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Observed OHLC geometry is inconsistent")

    intraday = (
        0.5 * np.log(d.high / d.low) ** 2 - (2 * np.log(2) - 1) * np.log(d.close / d.open) ** 2
    ).clip(lower=1e-10)
    overnight = np.log(d.open / d.close.shift(1)) ** 2
    variance = intraday + overnight
    f = pd.DataFrame(index=d.index)
    f["const"] = 1.0
    f["lrv_d"] = np.log(variance)
    f["lrv_w"] = np.log(variance.rolling(5).mean())
    f["lrv_m"] = np.log(variance.rolling(22).mean())
    returns = np.log(d["adj close"]).diff()
    f["lev_d"] = returns.clip(upper=0)
    f["lev_w"] = returns.rolling(5).mean().clip(upper=0)
    f["lev_m"] = returns.rolling(22).mean().clip(upper=0)
    delayed_iv = v.reindex(d.index).shift(1)
    f["liv"] = np.log(delayed_iv.vxn)
    f["lvix"] = np.log(delayed_iv.vix)
    f["term"] = np.log(delayed_iv.vix9d / delayed_iv.vix)
    asset_returns = np.log(c.reindex(d.index)).diff()
    z = _prior_z(asset_returns)
    f["xasset_stress"] = np.sqrt(z.pow(2).mean(axis=1, skipna=False))
    zvolume = _prior_z(np.log(d.volume.where(d.volume > 0))).clip(lower=0)
    zovernight = _prior_z((overnight / variance).clip(0, 1)).clip(lower=0)
    f["market_stress"] = np.sqrt((zvolume**2 + zovernight**2) / 2)
    f["rv_total"] = variance
    return f.replace([np.inf, -np.inf], np.nan)


def make_five_session_targets(rv):
    """Use exactly t+1 through t+5 full calendar rows without missing-value skips."""
    if not isinstance(rv, pd.Series):
        raise ValueError("Variance Series required")
    _calendar(rv.index, nonempty=True)
    values = _numbers(rv.to_frame(name="rv_total"), ("rv_total",))["rv_total"]
    if (values <= 0).any():
        raise ValueError("Observed variance must be strictly positive")
    forward = pd.concat([values.shift(-i) for i in range(1, 6)], axis=1)
    y = forward.mean(axis=1, skipna=False)
    target_end = pd.Series(rv.index, index=rv.index).shift(-5)
    return pd.DataFrame({"y": y, "target_end": target_end}, index=rv.index)
