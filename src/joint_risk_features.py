"""Source-gated paired QQQ ETF and SPX index intraday-return observations.

This module constructs raw predictors and paired labels only. Shared marginal
models, training-only transforms, joint matrices, and scoring belong to callers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import relative_risk_features as relative

ROOT = relative.ROOT
BASE = relative.ALL_FEATURES
ADDITIONS = tuple(f"{asset}_day_{suffix}" for asset in ("qqq", "spx") for suffix in ("d", "w", "m"))
ALL_FEATURES = BASE+ADDITIONS
MODELS = ("constant_matrix", "constant_correlation", "dynamic_correlation")


def _signed_intraday(source):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return np.log(source.close/source.open)


def measurement_audit(qqq, spx):
    """Preserve the full-reference GK gate and check complete-OHLC day returns.

    Zero signed returns are valid. Missing OHLC stays unknown in this audit.
    Every observed open/close pair must yield a finite return, including rows
    with missing high/low, so such gaps cannot conceal numerical overflow.
    No complete reference row is selected using feature or label eligibility.
    The caller can save a finite GK failure before requiring the gate to pass.
    """
    old = relative.measurement_audit(qqq, spx)
    audit = {**old, "per_asset": {}}
    for asset, source in (("qqq", qqq), ("spx", spx)):
        aligned = source.reindex(spx.index)
        observed = aligned.loc[:, relative.index_hinge.OHLC].notna().all(axis=1)
        values = _signed_intraday(aligned)
        open_close_observed = aligned.loc[:, ["open", "close"]].notna().all(axis=1)
        if not np.isfinite(values.loc[open_close_observed]).all():
            raise ValueError(f"Nonfinite computed signed intraday return for {asset}")
        complete_values = values.loc[observed]
        audit["per_asset"][asset] = {
            **old["per_asset"][asset], "finite_intraday_return_rows": len(complete_values),
            "zero_intraday_return_rows": int(complete_values.eq(0).sum()),
        }
    return audit


def require_measurement(audit):
    """Reject failed or internally inconsistent measurement audits without fill."""
    relative.require_measurement(audit)
    for asset in ("qqq", "spx"):
        one = audit["per_asset"][asset]
        finite = one.get("finite_intraday_return_rows")
        zeros = one.get("zero_intraday_return_rows")
        if (finite is None or zeros is None or finite != one.get("observed_complete_rows")
                or not 0 <= zeros <= finite):
            raise ValueError("INSUFFICIENT_DATA: full-reference signed intraday return gate failed")


def build_features(qqq, spx, iv):
    """Keep the full SPX calendar, lag signed histories, and pair next-session labels.

    The inherited 28 raw controls retain their own source-predecessor checks.
    Each new signed return uses same-session raw close/open, so it requires no
    previous source date. High/low-only gaps affect GK controls, while a missing
    open or close makes the signed observation unknown. No row is compressed.
    """
    require_measurement(measurement_audit(qqq, spx))
    features, _ = relative.build_features(qqq, spx, iv)
    reference = spx.index
    dates = pd.Series(reference, index=reference)
    targets = pd.DataFrame(index=reference)
    for asset, source in (("qqq", qqq), ("spx", spx)):
        day = _signed_intraday(source.reindex(reference))
        day = day.where(np.isfinite(day))
        for suffix, width in (("d", 1), ("w", 5), ("m", 22)):
            features[f"{asset}_day_{suffix}"] = day.rolling(width, min_periods=width).mean().shift(1)
        targets[f"y_{asset}"] = day.shift(-1)
    targets["target_end"] = dates.shift(-1)
    targets["available_date"] = dates.shift(-1)
    return features.loc[:, ALL_FEATURES+("feature_cutoff_date",)], targets


def load_sources(protocol, root=ROOT):
    """Reuse the frozen bounded raw-source loader without substituting data."""
    qqq, spx, iv, old = relative.load_sources(protocol, root)
    audit = {
        **old, "raw_columns": list(ALL_FEATURES),
        "target_interpretation": "paired next observed SPX session raw log(close/open) returns for QQQ ETF and SPX price index; signed and zero valid, not measured high-frequency covariance",
        "measurement_gate": "all complete OHLC rows on the full SPX reference calendar require finite raw GK strictly greater than 1e-10; every observed open/close pair requires finite signed log(close/open), including rows with missing high/low; zero signed returns valid; before any feature/label mask",
    }
    return qqq, spx, iv, audit
