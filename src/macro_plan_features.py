"""Fixed civil-time announcement-plan features, separate from realized sessions.

The nominal window skips weekends only. It is not an exchange holding-period
schedule. All original-plan documents need a publication date strictly earlier
than the preceding observed market-session date supplied by the caller.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ZONE = "America/New_York"


def _local_date(value):
    date = pd.Timestamp(value)
    if date.tz is not None or pd.isna(date) or date != date.normalize():
        raise ValueError("Normalized timezone-naive civil date required")
    return date


def _aware(value):
    stamp = pd.Timestamp(value)
    if stamp.tz is None or pd.isna(stamp):
        raise ValueError("Explicit timezone required for stated source timestamps")
    return stamp.tz_convert(ZONE)


def _identity(record):
    if not record.get("source_id") or not re.fullmatch(r"[0-9a-f]{64}", record.get("source_sha256", "")):
        raise ValueError("Source identity and SHA256 are required")


def nominal_window(origin):
    origin = _local_date(origin)
    if origin.weekday() > 4:
        raise ValueError("Nominal entry must be a weekday")
    start = (origin+pd.Timedelta(hours=16)).tz_localize(ZONE)
    next_day = origin+pd.Timedelta(days=1)
    while next_day.weekday() > 4:
        next_day += pd.Timedelta(days=1)
    end = (next_day+pd.Timedelta(hours=9, minutes=30)).tz_localize(ZONE)
    return start, end


def validate_bls_plans(records):
    result, keys, source_ids = [], set(), set()
    for record in records:
        _identity(record)
        event = record["event_type"]
        if event not in {"cpi", "nfp"}:
            raise ValueError("Only the two registered BLS source types are accepted")
        announced, planned = _aware(record["announced_at"]), _aware(record["planned_at"])
        if announced >= planned:
            raise ValueError("Publication must precede the original plan")
        month = planned.strftime("%Y-%m")
        if (event, month) in keys:
            raise ValueError("Ambiguous original plans for the same event and calendar month")
        if record["source_id"] in source_ids:
            raise ValueError("Duplicate original source document")
        keys.add((event, month))
        source_ids.add(record["source_id"])
        result.append({"event": event, "month": month, "announced": announced,
                       "announced_date": announced.tz_localize(None).normalize(), "planned": planned,
                       "source_id": record["source_id"], "source_sha256": record["source_sha256"]})
    return result


def _fomc_schedules(records):
    result = {}
    for record in records:
        _identity(record)
        year = int(record["year"])
        announcement = _local_date(record["announced_date"])
        dates = [_local_date(date) for date in record["final_dates"]]
        if (year in result or len(dates) != 8 or len(set(dates)) != len(dates)
                or announcement >= pd.Timestamp(year=year, month=1, day=1)
                or any(date.year != year or date <= announcement for date in dates)):
            raise ValueError("Invalid or ambiguous original annual FOMC schedule")
        result[year] = {"announced_date": announcement, "dates": set(dates),
                        "source_id": record["source_id"], "source_sha256": record["source_sha256"]}
    return result


def build_plan_features(origins, cutoff_dates, bls_records, fomc_records):
    """Create plan features and evidence rows without accepting future outcomes."""
    if (not isinstance(origins, pd.DatetimeIndex) or origins.has_duplicates
            or origins.tz is not None or origins.hasnans or not origins.is_monotonic_increasing
            or not origins.equals(origins.normalize()) or not cutoff_dates.index.equals(origins)):
        raise ValueError("Aligned, unique sorted civil origin/cutoff dates required")
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
    return pd.DataFrame(rows, index=origins, columns=["nominal_hours", "cpi_plan", "nfp_plan", "fomc_plan"]), audits
