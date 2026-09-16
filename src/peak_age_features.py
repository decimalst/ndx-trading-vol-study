"""Pure full-calendar peak-age inputs; no historical source access or runner."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RAW = ("const", "I", "R", "ret_d", "ret_w", "ret_m", "ret_q", "lr_d", "lr_w",
       "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
BASE = RAW + ("I_square", "R_square")
NEW = ("peak_age", "drawdown", "drawdown_sq", "window_return")
COMMON = RAW + NEW
OHLC = ("open", "high", "low", "close")
IV_FIELDS = ("vix", "vix9d", "vvix")
HORIZON = 21
WINDOW = 252


def _dates(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing or index.tz is not None
            or not index.equals(index.normalize())):
        raise ValueError("Unique sorted normalized timezone-naive reference sessions required")
    return index.as_unit("ns")


def _market(frame, columns):
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Observed market prices must be positive finite float64 values")
    return values


def stable_log_ratio(x, y):
    """Literal frexp/log1p/fsum algorithm on two stored positive float64 prices."""
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0):
        raise ValueError("Positive finite stored prices required for log ratio")
    if x == y:
        return 0.0
    mx, ex = math.frexp(x)
    my, ey = math.frexp(y)
    difference = ex - ey
    try:
        if abs(difference) <= 1:
            result = math.log1p((math.ldexp(mx, difference) - my) / my)
        else:
            result = math.fsum([math.log(mx / my), difference * math.log(2.0)])
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("Invalid finite log-ratio arithmetic") from exc
    if not math.isfinite(result) or result == 0 or (result > 0) != (x > y):
        raise ValueError("Log ratio lost stored-price ordering or finite nonzero value")
    return result


def _available(values, expected, label):
    observed = np.asarray(values, dtype=float)
    mask = np.asarray(expected, dtype=bool)
    if not np.isfinite(observed[mask]).all():
        raise ValueError("Arithmetic failure on fully observed " + label)
    return values


def _strict_mean(series, window, label):
    expected = series.notna().rolling(window, min_periods=window).sum() == window
    return _available(series.rolling(window, min_periods=window).mean(), expected, label)


def build_features(daily, iv):
    """Return raw15 plus A/D/D2/U, sole21 target, and auditable252-close state.

    Source NaNs remain on the supplied reference calendar. Complete-input
    arithmetic failures raise; no clipping or missing-value fallback is added.
    """
    d, implied = daily.copy(), iv.copy()
    d.index, implied.index = _dates(d.index), _dates(implied.index)
    _market(d, OHLC)
    _market(implied, IV_FIELDS)
    complete = d.loc[:, OHLC].notna().all(axis=1)
    if ((d.loc[complete, "high"] < d.loc[complete, ["open", "low", "close"]].max(axis=1)).any()
            or (d.loc[complete, "low"] > d.loc[complete, ["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Invalid observed OHLC ranges")
    aligned = implied.reindex(d.index)
    with np.errstate(all="ignore"):
        return_ready = d.close.notna() & d.close.shift().notna()
        returns = _available(np.log(d.close / d.close.shift()), return_ready, "raw close returns")
        day = _available(np.log(d.close / d.open), d.close.notna() & d.open.notna(), "intraday return")
        log_range = _available(np.log(d.high / d.low), d.high.notna() & d.low.notna(), "daily range")
        gk_raw = _available(.5 * log_range**2 - (2 * np.log(2) - 1) * day**2, complete, "GK variance")
        gk = gk_raw.clip(lower=1e-10)
        overnight_ready = d.open.notna() & d.close.shift().notna()
        overnight = _available(np.log(d.open / d.close.shift())**2, overnight_ready, "overnight variance")
        variance = _available(gk + overnight, complete & overnight_ready, "total variance")
        f = pd.DataFrame(index=d.index)
        f["I"] = _available(np.log((aligned.vix / 100)**2), aligned.vix.notna(), "implied variance")
        for name, width in [("R", 22), ("lr_d", 1), ("lr_w", 5)]:
            rolling = _strict_mean(variance, width, "rolling variance")
            f[name] = _available(np.log(252 * rolling), rolling.notna(), "annualized rolling variance")
        for suffix, width in [("d", 1), ("w", 5), ("m", 22), ("q", 63)]:
            f["ret_" + suffix] = _strict_mean(returns, width, "rolling raw return")
        f["term"] = _available(np.log(aligned.vix9d / aligned.vix),
                                aligned.vix9d.notna() & aligned.vix.notna(), "IV term ratio")
        f["lvvix"] = _available(np.log(aligned.vvix), aligned.vvix.notna(), "log VVIX")
    f = f.shift(1)
    f["const"] = 1.0
    for weekday in range(1, 5):
        f[f"entry_dow_{weekday}"] = (d.index.weekday == weekday).astype(float)
    f = f.loc[:, RAW]
    dates = pd.Series(d.index, index=d.index)
    for name in NEW:
        f[name] = np.nan
    f["feature_cutoff_date"] = dates.shift(1)

    state = pd.DataFrame(index=d.index)
    state["origin_position"] = pd.array(np.arange(len(d)), dtype="Int64")
    for name in ["cutoff_position", "window_start_position", "window_end_position", "peak_position", "peak_age_sessions"]:
        state[name] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    if len(d) > 1:
        state.loc[d.index[1:], "cutoff_position"] = np.arange(len(d) - 1)
    for name in ["window_start_date", "window_end_date", "peak_date"]:
        state[name] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    state["feature_cutoff_date"] = dates.shift(1)
    state["window_complete"] = False
    closes = d.close.to_numpy(dtype=np.float64)
    for i in range(WINDOW, len(d)):
        origin, start, end = d.index[i], i - WINDOW, i - 1
        state.loc[origin, ["window_start_position", "window_end_position"]] = [start, end]
        state.loc[origin, "window_start_date"] = d.index[start]
        state.loc[origin, "window_end_date"] = d.index[end]
        window = closes[start:i]
        if np.isnan(window).any():
            continue
        maximum = float(window.max())
        peak = start + int(np.flatnonzero(window == maximum)[-1])
        age = end - peak
        depth = stable_log_ratio(maximum, closes[end])
        total_return = stable_log_ratio(closes[end], closes[start])
        square = depth * depth
        if not math.isfinite(square) or (depth != 0 and square == 0) or depth < 0:
            raise ValueError("Invalid nonnegative depth-square arithmetic")
        f.loc[origin, list(NEW)] = [age / 251, depth, square, total_return]
        state.loc[origin, ["peak_position", "peak_age_sessions"]] = [peak, age]
        state.loc[origin, "peak_date"] = d.index[peak]
        state.loc[origin, "window_complete"] = True
    with np.errstate(all="ignore"):
        future = d.close.shift(-HORIZON)
        y = _available(np.log(future / d.close), future.notna() & d.close.notna(), "21-session raw target")
    target = pd.DataFrame({"y": y, "target_end": dates.shift(-HORIZON),
                           "available_date": dates.shift(-HORIZON)}, index=d.index)
    return f, target, state
