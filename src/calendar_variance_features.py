"""Original-plan calendar replication with lagged SPX full-session risk inputs.

This module owns only source admission, causal predictors, and next-session
labels. The caller owns all fitting, maturity gates, scoring, and inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import index_hinge, macro_search
from .macro_plan_features import ZONE, _fomc_schedules, _local_date, validate_bls_plans

ROOT = index_hinge.ROOT
BASE = ("const", "lrv_d", "lrv_w", "lrv_m", "neg_d", "neg_w", "neg_m",
        "lvix", "term", "lvvix", "entry_dow_1", "entry_dow_2", "entry_dow_3",
        "entry_dow_4", "nominal_hours")
ADDITIONS = ("cpi_plan", "nfp_plan", "fomc_plan")
ALL_FEATURES = BASE+ADDITIONS
MODELS = ("mean", "baseline", "calendar")
GK_FLOOR = 1e-10


def nominal_window(origin):
    """Civil close-to-close interval; skip weekends, retain holidays/early closes."""
    origin = _local_date(origin)
    if origin.weekday() > 4:
        raise ValueError("Nominal entry must be a weekday")
    start = (origin+pd.Timedelta(hours=16)).tz_localize(ZONE)
    next_day = origin+pd.Timedelta(days=1)
    while next_day.weekday() > 4:
        next_day += pd.Timedelta(days=1)
    end = (next_day+pd.Timedelta(hours=16)).tz_localize(ZONE)
    return start, end


def build_plan_features(origins, cutoff_dates, bls_records, fomc_records):
    """Keep original plans and source-known month coverage; no realized calendar.

    The evidence schema matches the prior overnight plan builder. Only nominal
    end times and elapsed hours change. Future observed sessions are not inputs.
    """
    index_hinge._dates(origins)
    if not cutoff_dates.index.equals(origins):
        raise ValueError("Aligned origin/cutoff dates required")
    bls = validate_bls_plans(bls_records)
    schedules = _fomc_schedules(fomc_records)
    rows, audits = [], []
    for origin, cutoff in zip(origins, cutoff_dates, strict=True):
        start, end = nominal_window(origin)
        row = {"nominal_hours": (end.tz_convert("UTC")-start.tz_convert("UTC")).total_seconds()/3600}
        audit = {"origin": str(origin.date()), "nominal_start": start.isoformat(), "nominal_end": end.isoformat(),
                 "source_rule": "publication_date strictly before preceding observed market-session date"}
        cutoff_missing = pd.isna(cutoff)
        if not cutoff_missing:
            cutoff = _local_date(cutoff)
            if cutoff >= origin:
                raise ValueError("Source cutoff must precede entry")
        months = set(pd.period_range(start.tz_localize(None).to_period("M"),
                                     end.tz_localize(None).to_period("M"), freq="M").astype(str))
        for event in ("cpi", "nfp"):
            eligible = [record for record in bls if not cutoff_missing and record["event"] == event
                        and record["announced_date"] < cutoff and record["month"] in months]
            missing = sorted(months-{record["month"] for record in eligible})
            row[event+"_plan"] = np.nan if missing else float(any(start < record["planned"] <= end for record in eligible))
            audit[event+"_missing_months"] = missing
            audit[event+"_source_ids"] = sorted(record["source_id"] for record in eligible)
            audit[event+"_source_hashes"] = sorted(record["source_sha256"] for record in eligible)
        schedule = schedules.get(end.year)
        known = schedule is not None and not cutoff_missing and schedule["announced_date"] < cutoff
        row["fomc_plan"] = float(end.tz_localize(None).normalize() in schedule["dates"]) if known else np.nan
        audit["fomc_source_id"] = schedule["source_id"] if known else None
        audit["fomc_source_sha256"] = schedule["source_sha256"] if known else None
        rows.append(row)
        audits.append(audit)
    return pd.DataFrame(rows, index=origins, columns=["nominal_hours", *ADDITIONS]), audits


def build_features(daily, iv, plans):
    """Preserve the observed calendar; market features end at the previous close.

    The daily risk proxy is floored intraday Garman–Klass plus squared raw
    overnight log return. Both histories and labels retain native squared-log
    units. Missing measurements propagate through strict complete windows.
    """
    index_hinge._daily_contract(daily)
    for frame in (iv, plans):
        index_hinge._dates(frame.index)
    index_hinge._positive_market(iv, tuple(index_hinge.IV_FIELDS))
    calendar = plans.reindex(daily.index)
    for column in ADDITIONS:
        if not calendar[column].dropna().isin([0., 1.]).all():
            raise ValueError("Calendar plans must be zero, one, or unknown")
    duration = calendar.nominal_hours.dropna()
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Positive finite nominal duration required")
    aligned = iv.reindex(daily.index)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        returns = np.log(daily.close/daily.close.shift(1))
        negative = (-returns).clip(lower=0)
        day = np.log(daily.close/daily.open)
        gk = (.5*np.log(daily.high/daily.low)**2-(2*np.log(2)-1)*day**2).clip(lower=GK_FLOOR)
        variance = gk+np.log(daily.open/daily.close.shift(1))**2
        features = pd.DataFrame(index=daily.index)
        for suffix, width in [("d", 1), ("w", 5), ("m", 22)]:
            features["lrv_"+suffix] = np.log(variance.rolling(width, min_periods=width).mean())
            features["neg_"+suffix] = negative.rolling(width, min_periods=width).mean()
        features["lvix"] = np.log(aligned.vix)
        features["term"] = np.log(aligned.vix9d/aligned.vix)
        features["lvvix"] = np.log(aligned.vvix)
    features = features.shift(1)
    features["const"] = 1.
    for weekday in range(1, 5):
        features[f"entry_dow_{weekday}"] = (daily.index.weekday == weekday).astype(float)
    for column in ("nominal_hours", *ADDITIONS):
        features[column] = calendar[column]
    features = features.loc[:, ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(daily.index, index=daily.index)
    features["feature_cutoff_date"] = dates.shift(1)
    future_risk = variance.shift(-1)
    targets = pd.DataFrame({"y": future_risk.where(np.isfinite(future_risk)),
                            "target_end": dates.shift(-1), "available_date": dates.shift(-1)}, index=daily.index)
    return features, targets


def load_sources(protocol, root=ROOT):
    """Reuse frozen bounded market parsing and hash-verified original-plan sources.

    The historical plan adapter remains rooted in its original repository
    corpus. This wrapper does not acquire or rewrite any source documents.
    """
    daily, iv, market_audit = index_hinge.load_sources(protocol, root)
    bls, annual, plan_audit = macro_search.load_plan_records(protocol)
    cutoff = pd.Series(daily.index, index=daily.index).shift(1)
    plans, evidence = build_plan_features(daily.index, cutoff, bls, annual)
    market_audit = {key: value for key, value in market_audit.items() if key != "gap_interpretation"}
    market_audit["raw_columns"] = list(BASE[:-1])
    market_audit["target_interpretation"] = (
        "next observed full-session SPX daily OHLC risk proxy in native squared-log-return units; "
        "not high-frequency integrated variance or causal announcement contribution"
    )
    audit = {"market": market_audit, "plans": plan_audit, "plan_availability": evidence,
             "plan_counts_full_bounded_calendar": {column: int(plans[column].sum()) for column in ADDITIONS},
             "unknown_rows": int(plans.isna().any(axis=1).sum())}
    return daily, iv, plans, audit
