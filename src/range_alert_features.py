"""Causal archival range-location features and one fixed risk-alert target.

No empirical runner, class counting, prediction fitting or scoring lives here.
The caller registers source snapshots and owns common-row maturity and support.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_shape_features as tail

ROOT = Path(__file__).resolve().parents[1]
NUISANCE = ("intraday", "intraday_sq", "overnight", "overnight_sq")
LOCATION = "range_extremity"
RAW = tail.RAW + NUISANCE + (LOCATION,)
BASE = tail.BASE + NUISANCE
ALL_FEATURES = RAW
MODELS = ("baseline", "recent_frequency", "location")
OHLC = ("open", "high", "low", "close")
GK_FLOOR = 1e-10
REFERENCE_WINDOW = 22
THRESHOLD_MULTIPLIER = 2.0
SCALE_MINIMUM = 1e-12


def _numeric(frame, columns):
    if frame.columns.has_duplicates:
        raise ValueError("Unique numeric column names required")
    values = frame.loc[:, columns].to_numpy()
    if values.dtype.kind not in "iuf":
        raise ValueError("Real numeric columns required; no boolean/object/complex conversion")
    result = frame.loc[:, columns].astype(float).copy()
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Observed values must be finite; only NaN is unknown")
    return result


def _computed(value, known, description):
    if not np.isfinite(value.loc[known].to_numpy()).all():
        raise ValueError("Nonfinite computed " + description)
    return value.where(known)


def _divide(left, right, description):
    known = left.notna() & right.notna()
    if right.loc[known].eq(0).any():
        raise ValueError("Zero denominator in " + description)
    with np.errstate(all="ignore"):
        value = left / right
    value = _computed(value, known, description)
    if (left.loc[known].ne(0) & value.loc[known].eq(0)).any():
        raise ValueError("Nonzero division underflow in " + description)
    return value


def _log_positive(value, description):
    known = value.notna()
    if value.loc[known].le(0).any():
        raise ValueError("Positive logarithm argument required in " + description)
    with np.errstate(all="ignore"):
        result = np.log(value)
    return _computed(result, known, description)


def _log_ratio(left, right, description):
    return _log_positive(_divide(left, right, description), description)


def _square(value, description):
    known = value.notna()
    with np.errstate(all="ignore"):
        result = value**2
    result = _computed(result, known, description)
    if (value.loc[known].ne(0) & result.loc[known].eq(0)).any():
        raise ValueError("Nonzero square underflow in " + description)
    return result


def _scale(value, factor, description):
    known = value.notna()
    with np.errstate(all="ignore"):
        result = factor * value
    result = _computed(result, known, description)
    if factor != 0 and (value.loc[known].ne(0) & result.loc[known].eq(0)).any():
        raise ValueError("Nonzero multiplication underflow in " + description)
    return result


def _daily(daily):
    tail.index_hinge._dates(daily.index)
    prices = _numeric(daily, OHLC)
    if (prices.le(0) & prices.notna()).to_numpy().any():
        raise ValueError("Every observed OHLC value must be positive")
    # Check every available endpoint pair, including partially missing bars.
    for other in ("open", "low", "close"):
        if (prices.high < prices[other]).any():
            raise ValueError("Invalid observed OHLC upper range")
    for other in ("open", "high", "close"):
        if (prices.low > prices[other]).any():
            raise ValueError("Invalid observed OHLC lower range")
    return prices


def session_components(daily):
    """Current-session primitives on the full observed SPX calendar.

    All observed operand pairs are checked before any completeness mask. The
    location output requires a complete valid OHLC bar and a positive range.
    Zero range is unknown location but retains the intentional inherited GK
    floor and any otherwise observable daily variance. Missing bars are never
    removed, so overnight uses exactly the preceding reference-calendar close.
    """
    prices = _daily(daily)
    intraday = _log_ratio(prices.close, prices.open, "intraday log return")
    intraday_sq = _square(intraday, "intraday square")
    overnight = _log_ratio(prices.open, prices.close.shift(1), "overnight log return")
    overnight_sq = _square(overnight, "overnight square")
    _log_ratio(prices.close, prices.close.shift(1), "legacy close log return")
    width = _log_ratio(prices.high, prices.low, "high-low logarithmic range")
    width_sq = _square(width, "high-low logarithmic range square")
    raw = _scale(width_sq, 0.5, "GK range term") - _scale(
        intraday_sq, 2 * np.log(2) - 1, "GK intraday term"
    )
    raw = _computed(raw, width_sq.notna() & intraday_sq.notna(), "raw GK")
    gk = raw.clip(lower=GK_FLOOR)  # The existing measurement floor is intentional.
    with np.errstate(all="ignore"):
        variance = gk + overnight_sq
    variance = _computed(variance, gk.notna() & overnight_sq.notna(), "daily variance")
    if variance.dropna().le(0).any():
        raise ValueError("Computed daily variance must remain positive")

    close_low = _log_ratio(prices.close, prices.low, "close-low logarithm")
    high_close = _log_ratio(prices.high, prices.close, "high-close logarithm")
    numerator = _computed(
        close_low - high_close, close_low.notna() & high_close.notna(), "range numerator"
    )
    positive_range = prices.high.gt(prices.low)
    if width.loc[positive_range & width.notna()].le(0).any():
        raise ValueError("Positive observed range has invalid computed width")
    z = _divide(numerator.where(positive_range), width.where(positive_range), "range location")
    if z.dropna().abs().gt(1).any():
        raise ValueError("Computed range location exceeds its literal unit domain")
    extremity = _square(z, "range extremity")
    if extremity.dropna().lt(0).any() or extremity.dropna().gt(1).any():
        raise ValueError("Computed range extremity exceeds its literal unit domain")
    extremity = extremity.where(prices.notna().all(axis=1))
    return pd.DataFrame(
        {
            "intraday": intraday,
            "intraday_sq": intraday_sq,
            "overnight": overnight,
            "overnight_sq": overnight_sq,
            "variance": variance,
            LOCATION: extremity,
        },
        index=daily.index,
    )


def build_targets(variance):
    """Strict risk event; checked fsum defines equality at the doubled reference.

    At entry t, reference observations are exactly t-22 through t-1; the outcome
    is variance at the next actual reference-calendar session t+1. Known labels
    do not depend on completeness of predictors or location. Metadata preserves
    the full calendar even where a label is unknown.
    """
    tail.index_hinge._dates(variance.index)
    numeric = _numeric(variance.rename("variance").to_frame(), ("variance",)).variance
    if numeric.dropna().le(0).any():
        raise ValueError("Observed variance must be strictly positive")
    values = numeric.to_numpy(float)
    labels = np.full(len(values), np.nan)
    for entry in range(REFERENCE_WINDOW, len(values)):
        window = values[entry - REFERENCE_WINDOW : entry]
        if not np.isfinite(window).all():
            continue
        try:
            total = math.fsum(window)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Nonfinite compensated reference sum") from exc
        reference = total / REFERENCE_WINDOW
        threshold = THRESHOLD_MULTIPLIER * reference
        if (
            not np.isfinite([total, reference, threshold]).all()
            or min(reference, threshold) <= 0
        ):
            raise ValueError("Invalid computed reference or doubled threshold")
        if entry + 1 < len(values) and np.isfinite(values[entry + 1]):
            labels[entry] = float(values[entry + 1] > threshold)
    dates = pd.Series(variance.index, index=variance.index)
    return pd.DataFrame(
        {"y": labels, "target_end": dates.shift(-1), "available_date": dates.shift(-1)},
        index=variance.index,
    )


def _validate_legacy_arithmetic(daily, iv, variance):
    tail.index_hinge._dates(iv.index)
    values = _numeric(iv, ("vix", "vix9d", "vvix", "skew"))
    if (values.le(0) & values.notna()).to_numpy().any():
        raise ValueError("Observed IV and SKEW values must be positive")
    aligned = values.reindex(daily.index)
    decimal_iv = _divide(aligned.vix, pd.Series(100.0, index=daily.index), "decimal VIX")
    _log_positive(_square(decimal_iv, "implied variance"), "legacy implied log variance")
    _log_ratio(aligned.vix9d, aligned.vix, "legacy implied term ratio")
    _log_positive(aligned.vvix, "legacy log VVIX")
    for window in (1, 5, 22):
        average = variance.rolling(window, min_periods=window).mean()
        complete = variance.notna().rolling(window, min_periods=window).sum().eq(window)
        average = _computed(average, complete, "legacy strict rolling variance mean")
        _log_positive(
            _scale(average, 252.0, "legacy annualized variance"), "legacy log variance"
        )


def build_features(daily, iv):
    """Return RAW24 plus prior cutoff, and the independent fixed event labels."""
    components = session_components(daily)
    _validate_legacy_arithmetic(daily, iv, components.variance)
    # Reuse only its predictors. The old normalized-return target is irrelevant
    # and is neither returned, counted, filtered on, nor used as a control.
    legacy, _ = tail.build_features(daily, iv)
    frame = legacy.loc[:, tail.RAW].copy()
    for name in NUISANCE + (LOCATION,):
        frame[name] = components[name].shift(1)
    frame = frame.loc[:, RAW]
    frame["feature_cutoff_date"] = legacy.feature_cutoff_date
    return frame, build_targets(components.variance)


def transform(train, apply):
    """Training-only BASE26 scaling and fixed-unit centered location.

    Return (baseline_train, baseline_apply, location_train, location_apply,
    audit), preserving input rows. All 25 baseline population scales must exceed
    1e-12. Duplicate information/collinearity is retained for the ridge solver.
    Exact constant training location has a canonical exact-zero history; its
    scalar coefficient must be zero in the caller without discarding the fold.
    """
    tr, ap = _numeric(train, RAW), _numeric(apply, RAW)
    if (
        len(tr) < 2
        or not np.isfinite(tr.to_numpy()).all()
        or not np.isfinite(ap.to_numpy()).all()
    ):
        raise ValueError("Finite common raw24 rows and at least two training rows required")
    if not tr.const.eq(1).all() or not ap.const.eq(1).all():
        raise ValueError("Literal unit intercept required")
    for frame in (tr, ap):
        if frame[LOCATION].lt(0).any() or frame[LOCATION].gt(1).any():
            raise ValueError("Range extremity must lie in the literal unit domain")
    curvature = {}
    for name in ("I", "R", "skew"):
        center = float(tr[name].mean())
        if not np.isfinite(center):
            raise ValueError("Nonfinite training curvature center")
        curvature[name] = center
        tr[name + "_square"] = _square(tr[name] - center, "training centered curvature")
        ap[name + "_square"] = _square(ap[name] - center, "application centered curvature")
    tr_base, ap_base = tr.loc[:, BASE], ap.loc[:, BASE]
    means, scales = tr_base.mean(), tr_base.std(ddof=0)
    means["const"], scales["const"] = 0.0, 1.0
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or scales.loc[list(BASE[1:])].le(SCALE_MINIMUM).any()
    ):
        raise ValueError("INSUFFICIENT_DATA: every baseline population scale must exceed1e-12")
    with np.errstate(all="ignore"):
        x, a = (tr_base - means) / scales, (ap_base - means) / scales
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(a.to_numpy()).all():
        raise ValueError("Nonfinite common training/application scaled design")
    constant = bool(tr[LOCATION].eq(tr[LOCATION].iloc[0]).all())
    center = float(tr[LOCATION].iloc[0] if constant else tr[LOCATION].mean())
    e, ae = tr[LOCATION] - center, ap[LOCATION] - center
    audit = {
        "train_n": len(tr),
        "columns": list(BASE),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "curvature_means": curvature,
        "range_extremity_mean": center,
        "range_extremity_scale": 1.0,
        "range_extremity_constant": constant,
    }
    return x, a, e, ae, audit


def load_sources(protocol, root=ROOT):
    """Use the frozen archived source admission; only study descriptions change.

    The runner supplies immutable, hash-checked staged source bytes. The frozen
    loader checks bounded raw SKEW against its original hash, metadata, derived
    audit-only parquet and historical anchor; it parses no postfence numbers.
    """
    daily, iv, original = tail.load_sources(protocol, root)
    audit = {key: value for key, value in original.items() if key != "gap_interpretation"}
    audit.update(
        raw_columns=list(RAW),
        baseline_columns=list(BASE),
        normalization="binary target uses checked fsum of22 prior daily risk values dividedby22; legacy log-risk controls remain unchanged",
        target_interpretation="next-session SPX floored-GK-plus-raw-overnight-square proxy strictly exceeds twice its prior22-session mean; equality0, missingunknown",
        timing_assumption="all market and location predictors through prior observed SPX session close; current entry weekday only; next actual session dates used only for target maturity",
        range_extremity_interpretation="squared logarithmic close location within complete valid OHLC range; zero range unknown; vendor bar statistic, not order flow or an executable quote",
        arithmetic_policy="strict finite observed operands and computed domains; checked ratio/square underflow; no range clipping; existing GKfloor1e-10 retained",
    )
    return daily, iv, audit
