"""Whole-attempt support counts without fitting, scoring, or source conversion."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from src.claims_release_pipeline import _dates, _iso
from src.treasury_dealer_pipeline import _configuration, _inputs

TENORS = (2, 3, 5, 7, 10, 30)
DEFAULT_CONFIG = {
    "forecast": {
        "origin_start": "2016-01-01",
        "origin_end": "2025-10-20",
        "source_end": "2025-10-20",
        "development": ["2016-01-01", "2019-12-31"],
        "evaluation": ["2020-01-01", "2025-10-20"],
        "minimum_train": 1000,
    },
    "stability": [["2020-01-01", "2022-12-31"], ["2023-01-01", "2025-10-20"]],
    "support": {
        "phase_daily": 505,
        "slice_daily": 252,
        "offset_daily": 63,
        "training_activation_dates": 200,
        "phase_activation_dates": 104,
        "slice_activation_dates": 52,
        "phase_active_dates_per_offset": 20,
        "training_events_per_tenor": 24,
        "phase_events_per_tenor": 12,
        "slice_events_per_tenor": 6,
    },
}
_EVENT_REQUIRED = {
    "event_id",
    "auction_date",
    "source_status",
    "origin",
    "status",
    "origin_masked",
    "mask_reason",
}


def _config(config):
    config = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
    if type(config) is not dict or set(config) != {"forecast", "stability", "support"}:
        raise ValueError("Exact forecast, stability and support configuration required")
    if type(config["forecast"]) is not dict:
        raise ValueError("Explicit forecast configuration dictionary required")
    forecast = _configuration(config["forecast"])
    support = config["support"]
    if type(support) is not dict or set(support) != set(DEFAULT_CONFIG["support"]):
        raise ValueError("Exact support floor names required")
    if any(type(value) is not int or value <= 0 for value in support.values()):
        raise ValueError("Every support floor must be a positive exact integer")
    ranges = config["stability"]
    if type(ranges) is not list or len(ranges) != 2:
        raise ValueError("Exactly two prospective stability intervals required")
    intervals = []
    for bounds in ranges:
        if type(bounds) is not list or len(bounds) != 2:
            raise ValueError("Two literal ISO dates required for each stability interval")
        start, end = map(_iso, bounds)
        if not forecast["evaluation"][0] <= start <= end <= forecast["evaluation"][1]:
            raise ValueError("Stability intervals must lie within the evaluation phase")
        intervals.append((start, end))
    if intervals[0][1] >= intervals[1][0]:
        raise ValueError("Stability intervals must be ordered and nonoverlapping")
    return forecast, intervals, support


def _event_groups(calendar, audits, tenors):
    """Read only source identity/status metadata, never shares or monetary values."""
    if type(audits) is not list or type(tenors) is not dict:
        raise ValueError("Explicit event audit list and exact event-tenor map required")
    seen, groups, blocked = set(), {}, set()
    for row in audits:
        if type(row) is not dict or not set(row) >= _EVENT_REQUIRED:
            raise ValueError("Required producer event audit metadata is missing")
        name = row["event_id"]
        if type(name) is not str or not name.strip() or name in seen or name not in tenors:
            raise ValueError("Unique event IDs and matching tenor membership required")
        seen.add(name)
        tenor = tenors[name]
        if tenor is not None and (type(tenor) is not int or tenor not in TENORS):
            raise ValueError("Exact supported original tenor or unknown required")
        lexeme = row["auction_date"]
        if type(lexeme) is not str or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", lexeme):
            raise ValueError("Exact ISO auction metadata date required")
        auction = _iso(lexeme)
        if auction < pd.Timestamp("2010-01-01"):
            raise ValueError("Auction date precedes the fixed source floor")
        source, status = row["source_status"], row["status"]
        allowed = {
            "KNOWN": {"SUPPORTED", "UNSUPPORTED_PRIOR", "OUTSIDE_CALENDAR"},
            "UNKNOWN": {"UNKNOWN", "OUTSIDE_CALENDAR"},
            "EXCLUDED": {"EXCLUDED"},
        }
        if (
            type(source) is not str
            or type(status) is not str
            or source not in allowed
            or status not in allowed[source]
        ):
            raise ValueError("Inconsistent source and producer event statuses")
        if source == "KNOWN" and tenor is None:
            raise ValueError("Known event requires confirmed original tenor metadata")
        if type(row["origin_masked"]) is not bool:
            raise ValueError("Exact boolean origin-mask audit required")
        origin = row["origin"]
        if origin is pd.NaT:
            if status not in {"EXCLUDED", "OUTSIDE_CALENDAR"}:
                raise ValueError("Active producer audit requires an origin")
            continue
        if (
            not isinstance(origin, pd.Timestamp)
            or origin.tz is not None
            or origin != origin.normalize()
            or origin not in calendar
            or origin <= auction
            or status in {"EXCLUDED", "OUTSIDE_CALENDAR"}
        ):
            raise ValueError("Exact full-calendar activation strictly after auction required")
        if status == "SUPPORTED" and row["mask_reason"] is not None:
            raise ValueError("Supported event cannot retain an unsupported reason")
        position = int(calendar.get_loc(origin))
        if status == "SUPPORTED" and not row["origin_masked"]:
            groups.setdefault(position, []).append((name, tenor))
        else:
            blocked.add(position)
    if seen != set(tenors):
        raise ValueError(
            "Event audit IDs must equal the full supplied ledger-tenor membership"
        )
    return groups, blocked


def _check_counts(features, groups, complete):
    names = ("auction_count",) + tuple(f"tenor_{tenor}" for tenor in TENORS[1:])
    values = features.loc[:, list(names)].to_numpy(dtype=float, na_value=np.nan)
    observed = np.isfinite(values)
    if np.any(observed & ((values < 0) | (values != np.floor(values)))):
        raise ValueError("Observed auction and tenor counts must be nonnegative integers")
    for position, row in enumerate(values):
        if not np.isfinite(row[0]):
            continue
        events = groups.get(position, ())
        expected = [len(events)] + [sum(t == tenor for _, t in events) for tenor in TENORS[1:]]
        if any(np.isfinite(value) and value != want for value, want in zip(row, expected)):
            raise ValueError("Auction feature counts disagree with supported event identities")
        if complete[position] and not np.isfinite(row).all():
            raise ValueError("Complete row cannot have missing auction counts")
    return values[:, 0] > 0


def audit_support(
    reference_calendar, features, targets, event_audit, *, event_tenors, config=None
):
    """Return all support failures and counts; never fit or shorten the calendar.

    Config overrides exist for generated tests. The empirical runner must use
    its immutable registered constants. Input validation is shared read-only
    with the fixed pipeline; all support counting and reporting is local.
    """
    forecast, stability, floors = _config(config)
    if not isinstance(reference_calendar, pd.DatetimeIndex):
        raise ValueError("Explicit native full reference calendar required")
    calendar = _dates(reference_calendar, forecast["source_end"])
    if (
        not len(calendar)
        or calendar.hasnans
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
    ):
        raise ValueError("Nonempty unique ordered full reference calendar required")
    groups, blocked = _event_groups(calendar, event_audit, event_tenors)
    inputs = _inputs(features, targets, forecast)
    full, _, complete, y, ends, _ = inputs
    if not full.equals(calendar):
        raise ValueError(
            "Feature/target frames must retain the explicit full reference calendar"
        )
    if any(complete[position] for position in blocked):
        raise ValueError("Masked or unsupported current event cannot be a complete source row")
    active = _check_counts(features, groups, complete)
    discrepancies = []

    def counts(mask):
        positions = np.flatnonzero(mask)
        by_tenor = {str(tenor): 0 for tenor in TENORS}
        for position in positions:
            for _, tenor in groups.get(int(position), ()):
                by_tenor[str(tenor)] += 1
        return {
            "daily_n": int(len(positions)),
            "activation_dates": int(np.count_nonzero(mask & active)),
            "event_n": sum(by_tenor.values()),
            "events_per_tenor": by_tenor,
        }

    def gate(scope, metric, observed, required, failures):
        if observed < required:
            failures.append(metric)
            discrepancies.append(
                {"scope": scope, "metric": metric, "observed": observed, "required": required}
            )

    def evaluated(mask, start, end, scope, kind, **extra):
        result = {
            "origin_start": start.date().isoformat(),
            "origin_end": end.date().isoformat(),
            **extra,
            **counts(mask),
        }
        failures = []
        gate(scope, f"{kind}_daily", result["daily_n"], floors[f"{kind}_daily"], failures)
        key = (
            "phase_active_dates_per_offset" if kind == "offset" else f"{kind}_activation_dates"
        )
        gate(scope, key, result["activation_dates"], floors[key], failures)
        if kind != "offset":
            key = f"{kind}_events_per_tenor"
            for tenor in TENORS:
                gate(
                    scope,
                    f"{key}:{tenor}",
                    result["events_per_tenor"][str(tenor)],
                    floors[key],
                    failures,
                )
        return {**result, "passed": not failures, "failures": failures}

    requested = (calendar >= forecast["origin_start"]) & (calendar <= forecast["origin_end"])
    months = calendar.to_period("M")
    monthly = []
    # Plan all monthly first feature-complete queries without reading query labels.
    for month in pd.period_range(forecast["origin_start"], forecast["origin_end"], freq="M"):
        positions = np.flatnonzero(requested & (months == month))
        applications = positions[complete[positions]]
        first = int(applications[0]) if len(applications) else None
        monthly.append(
            {
                "month": str(month),
                "status": "SUPPORTED"
                if first is not None
                else "NO_COMPLETE_ORIGIN"
                if len(positions)
                else "NO_REQUESTED_ORIGINS",
                "fit_origin": calendar[first].date().isoformat()
                if first is not None
                else None,
                "training_cutoff": calendar[first - 1].date().isoformat()
                if first is not None
                else None,
                "requested_n": int(len(positions)),
                "application_n": int(len(applications)),
                "train_n": None,
                "train_activation_dates": None,
                "train_event_n": None,
                "train_events_per_tenor": None,
                "passed": True,
                "failures": [],
            }
        )
    for row in monthly:
        if row["fit_origin"] is None:
            continue
        first, cutoff = pd.Timestamp(row["fit_origin"]), pd.Timestamp(row["training_cutoff"])
        train = complete & (calendar < first) & np.isfinite(y) & (ends <= cutoff)
        c = counts(train)
        row.update(
            train_n=c["daily_n"],
            train_activation_dates=c["activation_dates"],
            train_event_n=c["event_n"],
            train_events_per_tenor=c["events_per_tenor"],
        )
        scope, failures = "monthly:" + row["month"], row["failures"]
        gate(scope, "minimum_train", row["train_n"], forecast["minimum_train"], failures)
        gate(
            scope,
            "training_activation_dates",
            row["train_activation_dates"],
            floors["training_activation_dates"],
            failures,
        )
        for tenor in TENORS:
            gate(
                scope,
                f"training_events_per_tenor:{tenor}",
                row["train_events_per_tenor"][str(tenor)],
                floors["training_events_per_tenor"],
                failures,
            )
        row.update(
            passed=not failures, status="INSUFFICIENT_DATA" if failures else "SUPPORTED"
        )

    phases, offsets, masks = {}, {}, {}
    original_offsets = np.arange(len(calendar)) % 5
    for name in ("development", "evaluation"):
        start, end = forecast[name]
        fence = end if name == "development" else forecast["source_end"]
        mask = (
            complete
            & np.isfinite(y)
            & (calendar >= start)
            & (calendar <= end)
            & (ends <= fence)
        )
        masks[name] = mask
        phases[name] = evaluated(mask, start, end, "phase:" + name, "phase")
        offsets[name] = [
            evaluated(
                mask & (original_offsets == offset),
                start,
                end,
                f"offset:{name}:{offset}",
                "offset",
                offset=offset,
            )
            for offset in range(5)
        ]
    slices = [
        evaluated(
            masks["evaluation"] & (calendar >= start) & (calendar <= end),
            start,
            end,
            f"slice:{index}",
            "slice",
            slice_index=index,
        )
        for index, (start, end) in enumerate(stability)
    ]
    return {
        "status": "INSUFFICIENT_DATA" if discrepancies else "SUPPORT_PASS",
        "passed": not discrepancies,
        "monthly": monthly,
        "phases": phases,
        "slices": slices,
        "offsets": offsets,
        "discrepancies": discrepancies,
    }
