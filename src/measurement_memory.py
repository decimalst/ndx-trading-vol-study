"""Audited provider measurements, delayed inputs, and native squared targets.

This module performs source loading and deterministic feature/label construction
only. It has no estimator, score, empirical entrypoint, or output writer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_risklab_source import parse_equity_response

ROOT = Path(__file__).resolve().parents[1]
SOURCE_END = "2025-10-20"
SOURCE_SHA256 = "b7802ef21565b13831420c8fdd2b15176e57265fd5afe981be73693c8350e392"
BASE = ("const", "lq_d", "lq_w", "lq_m", "lvix", "livshape", "lvvix",
        "neg_d", "neg_w", "neg_m", "entry_dow_1", "entry_dow_2", "entry_dow_3", "entry_dow_4")
ALL_FEATURES = BASE+("width", "quality_memory")
MODELS = ("mean", "baseline", "width", "quality")
MEASURES = ("qmle", "rv5", "rv15")
PROVIDER_COLUMNS = {"qmle": "qmle_trade", "rv5": "rv5_trade", "rv15": "rv15_trade",
                    "ci_width": "ci_width_trade"}


def _dates(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing or index.tz is not None
            or not index.equals(index.normalize())):
        raise ValueError("Unique sorted normalized timezone-naive reference dates required")


def _cutoff(value):
    date = pd.Timestamp(value)
    if (pd.isna(date) or date.tz is not None or date != date.normalize()
            or date > pd.Timestamp(SOURCE_END)):
        raise ValueError("Invalid or protected numerical source cutoff")
    return date


def _positive_market(frame, columns):
    values = frame.loc[:, columns].to_numpy(float)
    if np.isinf(values).any() or ((values <= 0) & np.isfinite(values)).any():
        raise ValueError("Observed market prices must be positive and finite")


def _clean_measurements(table):
    """Invalid observations remain missing at their original calendar positions."""
    cleaned = table.loc[:, [*MEASURES, "ci_width"]].astype(float).copy()
    for name in MEASURES:
        cleaned[name] = cleaned[name].where(np.isfinite(cleaned[name]) & cleaned[name].gt(0))
    cleaned["ci_width"] = cleaned.ci_width.where(np.isfinite(cleaned.ci_width) & cleaned.ci_width.ge(0))
    return cleaned


def read_measurements(path, cutoff=SOURCE_END):
    """Check the pinned raw bytes before the frozen date-first provider parser."""
    end = _cutoff(cutoff)
    path = Path(path)
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError("Risk Lab source hash differs from the audited fixed response")
    parsed, audit = parse_equity_response(raw, cutoff=str(end.date()), symbol="SPY", identifier="84398")
    _dates(parsed.index)
    selected = parsed.loc[:, list(PROVIDER_COLUMNS.values())].rename(
        columns={value: key for key, value in PROVIDER_COLUMNS.items()})
    cleaned = _clean_measurements(selected)
    audit = {**audit, "source_path": str(path),
             "selected_field_map": {"qmle": 2, "rv5": 5, "rv15": 6, "ci_width": 4},
             "invalid_selected_fields": {name: int(cleaned[name].isna().sum()) for name in cleaned},
             "units": "provider-native annualized volatility and reported interval half-width",
             "normalization": "nonpositive/nonfinite volatility or negative/nonfinite width becomes missing; no clipping or filling",
             "historical_publication_latency_verified": False}
    return cleaned, audit


def build_features(daily, iv, table):
    """Return sixteen predictors with dated cutoffs and three future measurements.

    Risk Lab inputs end two reference sessions before the origin. Other market
    inputs end one session before it. The next-session target is assigned a
    maturity date two more reference sessions later. Future date labels never
    enter a predictor, and missing source sessions are retained before rolling.
    """
    for frame in (daily, iv, table):
        _dates(frame.index)
    _positive_market(daily, ["adj close"])
    _positive_market(iv, ["vix", "vix9d", "vvix"])
    measurements = _clean_measurements(table).reindex(daily.index)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        squared = measurements.loc[:, MEASURES].pow(2)
    squared = squared.where(np.isfinite(squared) & squared.gt(0))
    negative = (-np.log(daily["adj close"]).diff()).clip(lower=0)
    features = pd.DataFrame(index=daily.index)
    features["const"] = 1.
    for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
        features["lq_"+suffix] = np.log(squared.qmle.rolling(width, min_periods=width).mean()).shift(2)
        features["neg_"+suffix] = negative.rolling(width, min_periods=width).mean().shift(1)
    aligned_iv = iv.reindex(daily.index)
    features["lvix"] = np.log(aligned_iv.vix).shift(1)
    features["livshape"] = np.log(aligned_iv.vix9d/aligned_iv.vix).shift(1)
    features["lvvix"] = np.log(aligned_iv.vvix).shift(1)
    for weekday in range(1, 5):
        features[f"entry_dow_{weekday}"] = (daily.index.weekday == weekday).astype(float)
    with np.errstate(over="ignore", invalid="ignore"):
        features["width"] = np.log1p(measurements.ci_width/measurements.qmle).shift(2)
    features["quality_memory"] = features.width*(features.lq_d-features.lq_m)
    features = features.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(daily.index, index=daily.index)
    features["market_cutoff_date"] = dates.shift(1)
    features["measurement_cutoff_date"] = dates.shift(2)
    targets = squared.shift(-1).rename(columns={name: "y_"+name for name in MEASURES})
    targets["target_end"] = dates.shift(-1)
    targets["available_date"] = dates.shift(-3)
    return features, targets


def _market_audit(path, frame):
    return {"source_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bounded_rows": len(frame),
            "first_date": str(frame.index.min().date()) if len(frame) else None,
            "last_date": str(frame.index.max().date()) if len(frame) else None}


def load_sources(protocol, root=ROOT):
    """Load only the five declared sources; numerical data stop at the fence."""
    root = Path(root)
    end = _cutoff(protocol["index"]["source_end"])
    paths = {name: root/protocol["sources"][name] for name in ["risklab", "daily", "vix", "vix9d", "vvix"]}
    table, source_audit = read_measurements(paths["risklab"], end)
    daily = pd.read_parquet(paths["daily"], columns=["adj close"], filters=[("date", "<=", end)]).loc[:end]
    _dates(daily.index)
    _positive_market(daily, ["adj close"])
    sources = {"daily": _market_audit(paths["daily"], daily)}
    series = []
    for name, column in [("vix", "CLOSE"), ("vix9d", "CLOSE"), ("vvix", "VVIX")]:
        raw = pd.read_csv(paths[name], usecols=["DATE", column], dtype=str)
        dates = pd.to_datetime(raw.DATE, format="%m/%d/%Y", errors="raise")
        selected = dates <= end
        values = pd.to_numeric(raw.loc[selected, column], errors="raise").to_numpy(float)
        one = pd.Series(values, index=pd.DatetimeIndex(dates[selected], name="date"), name=name)
        _dates(one.index)
        series.append(one)
        sources[name] = _market_audit(paths[name], one)
    iv = pd.concat(series, axis=1).sort_index()
    _positive_market(iv, ["vix", "vix9d", "vvix"])
    audit = {"risklab": source_audit, "market_sources": sources,
             "numerical_source_end": str(end.date()),
             "calendar": {"definition": "bounded observed QQQ reference sessions; no certified historic SPY calendar",
                          "sessions": len(daily),
                          "first_date": str(daily.index.min().date()) if len(daily) else None,
                          "last_date": str(daily.index.max().date()) if len(daily) else None},
             "measurement_calendar": {
                 "missing_source_dates": [str(date.date()) for date in daily.index.difference(table.index)],
                 "outside_reference_dates": [str(date.date()) for date in table.index.difference(daily.index)],
             },
             "timing_assumption": "Risk Lab inputs lag2; market inputs lag1; target next session; target maturity two additional reference sessions",
             "historical_publication_latency_verified": False}
    return daily, iv, table, audit
