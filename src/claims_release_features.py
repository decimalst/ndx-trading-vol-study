"""Causal date-only first-DOL-report features on an uncompressed origin calendar."""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from datetime import date, timedelta

import numpy as np
import pandas as pd

SOURCE_CEILING = date(2025, 10, 20)
VALUE_COLUMNS = ("claim_m4", "claim_age", "claim_x")
FEATURE_COLUMNS = (
    "claim_m4",
    "claim_age",
    "entry_dow_1",
    "entry_dow_2",
    "entry_dow_3",
    "entry_dow_4",
    "claim_x",
    "claim_reference_week",
    "claim_release_date",
    "claim_age_days",
    "claim_status",
)
_FIELDS = {
    "reference_week",
    "release_date",
    "first_report_value",
    "status",
    "source_url",
    "release_text_sha256",
    "source_comparison",
}


def _date(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Literal ISO ledger date required")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Valid ledger calendar date required") from error
    if parsed > SOURCE_CEILING:
        raise ValueError("Ledger date exceeds fixed source ceiling")
    return parsed


def _calendar(index):
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is not None
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError("Unique ordered naive midnight DatetimeIndex required")
    try:
        index.as_unit("ns")
    except (ValueError, OverflowError) as error:
        raise ValueError(
            "Calendar dates must be exactly representable in nanoseconds"
        ) from error
    if len(index) and index[-1].date() > SOURCE_CEILING:
        raise ValueError("Origin calendar exceeds fixed source ceiling")


def _ledger(rows):
    if type(rows) is not list:
        raise ValueError("Ledger rows must be a list")
    by_week, published = {}, {}
    for row in rows:
        if type(row) is not dict or not set(row) >= _FIELDS:
            raise ValueError("Required source-ledger fields missing")
        # Comparison metadata is never traversed: conflicting counts are not inputs.
        if type(row["source_comparison"]) is not dict:
            raise ValueError("Source comparison must remain an opaque audit dictionary")
        week = _date(row["reference_week"])
        released = None if row["release_date"] is None else _date(row["release_date"])
        value, status = row["first_report_value"], row["status"]
        if week.weekday() != 5 or week in by_week:
            raise ValueError("Unique Saturday reference weeks required")
        if released is not None and (released <= week or released in published):
            raise ValueError("Unique release dates strictly after reference weeks required")
        if status == "observed":
            if type(value) is not int or value <= 0 or released is None:
                raise ValueError(
                    "Observed first report requires a positive exact integer and date"
                )
        elif status == "unresolved_correction":
            if value is not None or released is None:
                raise ValueError(
                    "Unresolved report must retain its date and unavailable value"
                )
        elif status == "no_admitted_release":
            if released is not None or value is not None:
                raise ValueError("No-admitted-release row cannot invent a date or value")
        else:
            raise ValueError("Unknown source-ledger status")
        parsed = (week, released, value, status)
        by_week[week] = parsed
        if released is not None:
            published[released] = parsed
    return by_week, published


def _log_value(value):
    result = math.log(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite first-report logarithm")
    return result


def build_claim_features(
    reference_calendar: pd.DatetimeIndex, ledger_rows: list
) -> pd.DataFrame:
    """Retain every origin; compute claims only from complete prior released states."""
    _calendar(reference_calendar)
    by_week, published = _ledger(ledger_rows)
    release_dates = sorted(published)
    n = len(reference_calendar)
    output = {name: np.full(n, np.nan) for name in VALUE_COLUMNS}
    for weekday in range(1, 5):
        output[f"entry_dow_{weekday}"] = (reference_calendar.weekday == weekday).astype(float)
    weeks, dates = [pd.NaT] * n, [pd.NaT] * n
    ages = np.full(n, np.nan)
    reasons = ["no_prior_release"] * n
    for position, timestamp in enumerate(reference_calendar):
        origin = timestamp.date()
        selected = bisect_left(release_dates, origin) - 1
        if selected < 0:
            continue
        week, released, value, status = published[release_dates[selected]]
        weeks[position], dates[position] = pd.Timestamp(week), pd.Timestamp(released)
        age = (origin - released).days
        ages[position] = age
        if age > 7:
            reasons[position] = "stale_release"
            continue
        if status != "observed":
            reasons[position] = "latest_" + status
            continue
        prior_values = []
        reason = None
        for lag in range(1, 5):
            prior = by_week.get(week - timedelta(weeks=lag))
            if prior is None:
                reason = "missing_prior_week"
                break
            _, prior_release, prior_value, prior_status = prior
            if prior_release is None:
                reason = "prior_" + prior_status
                break
            if prior_release >= origin:
                reason = "prior_not_yet_released"
                break
            if prior_status != "observed":
                reason = "prior_" + prior_status
                break
            prior_values.append(prior_value)
        if reason is not None:
            reasons[position] = reason
            continue
        mean_log = math.fsum(_log_value(value) for value in prior_values) / 4
        difference = _log_value(value) - mean_log
        if not math.isfinite(mean_log) or not math.isfinite(difference):
            raise ValueError("Nonfinite first-report feature arithmetic")
        output["claim_m4"][position] = mean_log
        output["claim_age"][position] = age / 7
        output["claim_x"][position] = difference
        reasons[position] = "available"
    output["claim_reference_week"] = pd.Series(
        weeks, index=reference_calendar, dtype="datetime64[ns]"
    )
    output["claim_release_date"] = pd.Series(
        dates, index=reference_calendar, dtype="datetime64[ns]"
    )
    output["claim_age_days"] = ages
    output["claim_status"] = reasons
    return pd.DataFrame(output, index=reference_calendar.copy()).loc[:, list(FEATURE_COLUMNS)]
