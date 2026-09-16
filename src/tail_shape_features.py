"""Fixed archival sources and causal normalized return targets for wave 7.

This module neither counts tail events nor fits empirical probability models.
The caller owns common-row maturity, class support, forecast scheduling, density
fitting, scoring, and inference after protocol freeze.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import index_hinge

ROOT = Path(__file__).resolve().parents[1]
RAW = index_hinge.RAW+("neg_d", "neg_w", "neg_m", "skew")
BASE = RAW+("I_square", "R_square", "skew_square")
ALL_FEATURES = BASE
MODELS = ("frequency", "constant_shape", "skew_shape")
SKEW_SHA256 = "becbf3f7510de66a736495df29a84ba1911d362402f05f6c46dedd5e9b971492"
SKEW_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
ANCHOR_DATE = "2018-08-13"
ANCHOR_CLOSE = 159.03
ANCHOR_TOLERANCE = .01
EVENT_THRESHOLD = -1.5


def build_features(daily, iv):
    """Return raw19 predictors, causal normalizers, and next-session return labels.

    The normalizer uses daily variance, without annualization. All market values
    end at the previous observed SPX session; only entry weekday is current.
    Missing observations retain their reference-calendar positions before strict
    rolling windows. Adjusted-close fields are never consumed.
    """
    index_hinge._daily_contract(daily)
    index_hinge._dates(iv.index)
    index_hinge._positive_market(iv, ("vix", "vix9d", "vvix", "skew"))
    aligned = iv.reindex(daily.index)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        returns = np.log(daily.close/daily.close.shift(1))
        negative = (-returns).clip(lower=0)
        day = np.log(daily.close/daily.open)
        gk = (.5*np.log(daily.high/daily.low)**2-(2*np.log(2)-1)*day**2).clip(lower=1e-10)
        variance = gk+np.log(daily.open/daily.close.shift(1))**2
        monthly_variance = variance.rolling(22, min_periods=22).mean()
        monthly_return = returns.rolling(22, min_periods=22).mean()
        frame = pd.DataFrame(index=daily.index)
        frame["I"] = np.log((aligned.vix/100)**2)
        frame["R"] = np.log(252*monthly_variance)
        for suffix, width in [("d", 1), ("w", 5), ("m", 22), ("q", 63)]:
            frame["ret_"+suffix] = returns.rolling(width, min_periods=width).mean()
        for suffix, width in [("d", 1), ("w", 5)]:
            frame["lr_"+suffix] = np.log(252*variance.rolling(width, min_periods=width).mean())
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            frame["neg_"+suffix] = negative.rolling(width, min_periods=width).mean()
        frame["term"] = np.log(aligned.vix9d/aligned.vix)
        frame["lvvix"] = np.log(aligned.vvix)
        frame["skew"] = aligned["skew"]
        normalization_scale = np.sqrt(monthly_variance).shift(1)
    frame = frame.shift(1)
    frame["const"] = 1.
    for weekday in range(1, 5):
        frame[f"entry_dow_{weekday}"] = (daily.index.weekday == weekday).astype(float)
    frame = frame.loc[:, RAW].replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(daily.index, index=daily.index)
    frame["feature_cutoff_date"] = dates.shift(1)
    frame["normalization_mean"] = monthly_return.shift(1).replace([np.inf, -np.inf], np.nan)
    frame["normalization_scale"] = normalization_scale.where(np.isfinite(normalization_scale) & normalization_scale.gt(0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        raw_return = np.log(daily.close.shift(-1)/daily.close)
        normalized = (raw_return-frame.normalization_mean)/frame.normalization_scale
    normalized = normalized.where(np.isfinite(normalized))
    event = normalized.lt(EVENT_THRESHOLD).astype(float).where(normalized.notna())
    targets = pd.DataFrame({"y": normalized, "raw_return": raw_return.where(np.isfinite(raw_return)),
                            "event": event, "target_end": dates.shift(-1), "available_date": dates.shift(-1)},
                           index=daily.index)
    return frame, targets


def transform(train, apply):
    """Center all three square terms using exactly the supplied training rows."""
    tr, ap = train.loc[:, RAW].astype(float).copy(), apply.loc[:, RAW].astype(float).copy()
    if (len(tr) < 2 or not np.isfinite(tr.to_numpy()).all() or not np.isfinite(ap.to_numpy()).all()
            or not tr.const.eq(1).all() or not ap.const.eq(1).all()):
        raise ValueError("Finite common designs with unit intercept and at least two training rows required")
    audit = {"train_n": len(tr)}
    with np.errstate(over="ignore", invalid="ignore"):
        for name in ["I", "R", "skew"]:
            center = float(tr[name].mean())
            audit[name+"_mean"] = center
            tr[name+"_square"] = (tr[name]-center)**2
            ap[name+"_square"] = (ap[name]-center)**2
    if (not np.isfinite(list(audit.values())).all() or not np.isfinite(tr.to_numpy()).all()
            or not np.isfinite(ap.to_numpy()).all()):
        raise ValueError("Nonfinite training-centered transform; no clipping fallback")
    scales = tr.loc[:, BASE[1:]].std(ddof=0).to_numpy(float)
    if not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("INSUFFICIENT_DATA: zero or nonfinite common feature scale; no fallback")
    return tr.loc[:, BASE], ap.loc[:, BASE], audit


def _read_skew_csv(raw, end):
    """Parse date tokens first; only bounded raw SKEW tokens become numbers."""
    frame = pd.read_csv(io.BytesIO(raw), dtype=str)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.has_duplicates or "date" not in frame:
        raise ValueError("Unambiguous SKEW date/value columns required")
    value_columns = [name for name in ["close", "skew"] if name in frame]
    if len(value_columns) != 1:
        raise ValueError("Unambiguous SKEW date/value columns required")
    value_column = value_columns[0]
    dates = pd.to_datetime(frame.date, format="%m/%d/%Y", errors="raise")
    if dates.isna().any() or dates.duplicated().any() or len(dates) == 0:
        raise ValueError("Unique nonempty SKEW source dates required")
    selected = dates <= end
    values = pd.to_numeric(frame.loc[selected, value_column], errors="raise").to_numpy(float)
    series = pd.Series(values, index=pd.DatetimeIndex(dates.loc[selected], name="date"), name="skew").sort_index()
    index_hinge._dates(series.index)
    index_hinge._positive_market(series.to_frame(), ["skew"])
    if series.empty:
        raise ValueError("INSUFFICIENT_DATA: no bounded SKEW observations")
    audit = {"provider_date_field": "date", "provider_value_field": value_column,
             "source_rows_from_date_tokens": len(dates),
             "source_first_date": str(dates.min().date()), "source_last_date": str(dates.max().date())}
    return series, audit


def load_sources(protocol, root=ROOT):
    """Reuse the four wave 6 sources and admit only the pinned legacy SKEW source.

    The raw CSV is the predictor source. Its existing derived parquet is checked
    for exact bounded equality and never substituted as predictor data. The
    original metadata, fixed raw hash, and historical anchor all remain binding.
    """
    root = Path(root)
    daily, iv, audit = index_hinge.load_sources(protocol, root)
    end = pd.Timestamp(protocol["index"]["source_end"])
    paths = {name: root/protocol["sources"][name] for name in ["skew", "skew_source", "skew_derived"]}
    raw = paths["skew"].read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash != SKEW_SHA256:
        raise ValueError("Raw SKEW hash differs from the fixed legacy source")
    metadata = json.loads(paths["skew_source"].read_text())
    if metadata.get("sha256") != raw_hash:
        raise ValueError("Raw SKEW hash differs from its original source metadata")
    if metadata.get("url") != SKEW_URL:
        raise ValueError("Original SKEW provider URL differs")
    if (metadata.get("anchor_date") != ANCHOR_DATE
            or not np.isfinite(float(metadata.get("anchor_close", np.nan)))
            or abs(float(metadata["anchor_close"])-ANCHOR_CLOSE) > ANCHOR_TOLERANCE):
        raise ValueError("Original SKEW historical anchor metadata differs")
    skew, date_audit = _read_skew_csv(raw, end)
    for registered, observed in [("rows", "source_rows_from_date_tokens"),
                                 ("first_date", "source_first_date"), ("last_date", "source_last_date")]:
        if metadata.get(registered) != date_audit[observed]:
            raise ValueError("SKEW source-date provenance differs from original metadata")
    anchor = pd.Timestamp(ANCHOR_DATE)
    if (anchor not in skew.index or not np.isfinite(skew.loc[anchor])
            or abs(float(skew.loc[anchor])-ANCHOR_CLOSE) > ANCHOR_TOLERANCE):
        raise ValueError("Pinned SKEW historical anchor differs or is unavailable within the source fence")
    derived = pd.read_parquet(paths["skew_derived"], columns=["close"], filters=[("date", "<=", end)]).loc[:end]
    index_hinge._dates(derived.index)
    index_hinge._positive_market(derived, ["close"])
    if (not skew.index.equals(derived.index)
            or not np.array_equal(skew.to_numpy(float), derived.close.to_numpy(float), equal_nan=True)):
        raise ValueError("SKEW raw and derived bounded dates/values differ")
    sources = {**audit["sources"],
               "skew": {**index_hinge._source_audit(paths["skew"], skew.to_frame(), daily.index), **date_audit},
               "skew_derived": index_hinge._source_audit(paths["skew_derived"], derived, daily.index),
               "skew_source": {"source_path": str(paths["skew_source"]),
                               "source_sha256": hashlib.sha256(paths["skew_source"].read_bytes()).hexdigest()}}
    audit = {**audit, "sources": sources, "raw_columns": list(RAW),
             "normalization": "strict prior-session trailing22 raw logreturn mean and square root of daily GK-plus-raw-overnight variance mean; no annualization",
             "target_interpretation": "next-session SPX raw-close log price return normalized by prior information; event u<-1.5, not a universal VaR probability",
             "skew_provenance": {
                 "url": metadata["url"], "fetched_at_utc": metadata.get("fetched_at_utc"),
                 "registered_raw_sha256": metadata["sha256"], "raw_matches_manifest": True,
                 "raw_matches_fixed_pin": True, "raw_derived_bounded_exact_equal": True,
                 "anchor_date": ANCHOR_DATE, "anchor_expected": ANCHOR_CLOSE,
                 "anchor_observed": float(skew.loc[anchor]), "anchor_tolerance": ANCHOR_TOLERANCE,
                 "predictor_source": "raw SKEW CSV; existing derived parquet is audit-only",
             }}
    return daily, pd.concat([iv, skew], axis=1).sort_index(), audit
