"""Causal auction-release pulses with fixed same-original-tenor donor means."""

import math
import re
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

FLOOR = date(2010, 1, 1)
CEILING = date(2025, 10, 20)
TENORS = {2, 3, 5, 7, 10, 30}
COLUMNS = (
    "auction_count",
    "tenor_3",
    "tenor_5",
    "tenor_7",
    "tenor_10",
    "tenor_30",
    "reopening_count",
    "log_offering_sum",
    "high_yield_sum",
    "bid_to_cover_sum",
    "prior_share_mean_sum",
    "dealer_surprise",
)
EVENT_FIELDS = {
    "event_id",
    "auction_date",
    "availability_after_date",
    "status",
    "original_tenor_years",
    "reopening",
    "primary_dealer_accepted",
    "competitive_accepted",
    "offering_amount_usd",
    "high_yield",
    "bid_to_cover",
}
VALUE_FIELDS = {
    "primary_dealer_accepted",
    "competitive_accepted",
    "offering_amount_usd",
    "high_yield",
    "bid_to_cover",
}


def _day(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Exact ISO source date required")
    day = date.fromisoformat(value)
    if not FLOOR <= day <= CEILING:
        raise ValueError("Auction source date outside fixed envelope")
    return day


def _envelope(reference_calendar, events):
    if (
        not isinstance(reference_calendar, pd.DatetimeIndex)
        or reference_calendar.tz is not None
        or reference_calendar.hasnans
        or not reference_calendar.is_unique
        or not reference_calendar.is_monotonic_increasing
        or not reference_calendar.equals(reference_calendar.normalize())
        or (len(reference_calendar) and reference_calendar[-1].date() > CEILING)
    ):
        raise ValueError("Unchanged unique ascending native daily calendar required")
    if type(events) is not list:
        raise ValueError("Explicit event list required")
    seen = set()
    validated = []
    # Finish the complete calendar/date/schema envelope before calling _financial.
    for event in events:
        if type(event) is not dict or set(event) not in (
            EVENT_FIELDS,
            EVENT_FIELDS | {"clock_upper_bound_date"},
        ):
            raise ValueError("Exact auction event schema with optional clock bound required")
        name = event["event_id"]
        if type(name) is not str or not name.strip() or name in seen:
            raise ValueError("Unique nonempty string event ID required")
        seen.add(name)
        auction = _day(event["auction_date"])
        status = event["status"]
        if type(status) is not str or status not in {"KNOWN", "UNKNOWN", "EXCLUDED"}:
            raise ValueError("Explicit known, unknown or excluded status required")
        release = event["availability_after_date"]
        if release is not None and _day(release) < auction:
            raise ValueError("Availability date precedes auction date")
        if status == "KNOWN" and release is None:
            raise ValueError("Known release requires an exact availability date")
        upper = event.get("clock_upper_bound_date")
        if upper is not None:
            if status != "UNKNOWN" or _day(upper) < auction:
                raise ValueError(
                    "Clock upper bound requires an unknown event and valid chronology"
                )
            if release is not None and _day(upper) < _day(release):
                raise ValueError("Clock upper bound precedes exact availability date")
        tenor = event["original_tenor_years"]
        if tenor is not None and (type(tenor) is not int or tenor not in TENORS):
            raise ValueError("Explicit supported original tenor required")
        if status == "KNOWN" and tenor is None:
            raise ValueError("Known event cannot have unknown original tenor")
        reopening = event["reopening"]
        if reopening is not None and type(reopening) is not bool:
            raise ValueError("Reopening must be a boolean or unknown")
        if status == "KNOWN" and reopening is None:
            raise ValueError("Known event requires reopening identity")
        if status == "UNKNOWN" and any(event[key] is not None for key in VALUE_FIELDS):
            raise ValueError("Unknown event cannot provide financial feature values")
        validated.append(event)
    return sorted(validated, key=lambda event: (event["auction_date"], event["event_id"]))


def _decimal(value, *, positive=False):
    if type(value) is not str or re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value) is None:
        raise ValueError("Finite plain decimal source string required")
    exact = Decimal(value)
    number = float(exact)
    if not math.isfinite(number) or (number == 0 and exact != 0) or (positive and number <= 0):
        raise ValueError("Financial decimal is nonfinite, underflowed or nonpositive")
    return number


def _financial(event):
    dealer = event["primary_dealer_accepted"]
    total = event["competitive_accepted"]
    offering = event["offering_amount_usd"]
    if (
        any(type(value) is not int for value in (dealer, total, offering))
        or total <= 0
        or offering <= 0
        or not 0 <= dealer <= total
    ):
        raise ValueError(
            "Exact valid integer dealer, competitive and offering amounts required"
        )
    share = dealer / total
    if not math.isfinite(share) or (dealer != 0 and share == 0):
        raise ValueError("Dealer share overflow or underflow")
    return {
        "share": share,
        "log_offering": math.log(offering),
        "high_yield": _decimal(event["high_yield"]),
        "bid_to_cover": _decimal(event["bid_to_cover"], positive=True),
    }


def _prior(event, origin, known_events, unknown_events):
    candidates = [
        prior
        for prior in known_events
        if prior["original_tenor_years"] == event["original_tenor_years"]
        and prior["auction_date"] < event["auction_date"]
    ]
    # known_events is already sorted chronologically, with deterministic ID ties.
    selected = candidates[-12:]
    if len(selected) < 12:
        return selected, "INSUFFICIENT_PRIOR_AUCTIONS"
    oldest = selected[0]["auction_date"]
    if len(candidates) > 12 and candidates[-13]["auction_date"] == oldest:
        return selected, "AMBIGUOUS_PRIOR_CUTOFF"
    if any(pd.Timestamp(prior["availability_after_date"]) >= origin for prior in selected):
        return selected, "UNRELEASED_PRIOR_EVENT"
    if any(
        prior["original_tenor_years"] in (None, event["original_tenor_years"])
        and oldest <= prior["auction_date"] < event["auction_date"]
        for prior in unknown_events
    ):
        return selected, "UNKNOWN_PRIOR_EVENT"
    return selected, None


def _pending(event, calendar):
    """Return the half-open mask slice and explicit clock evidence description."""
    start = int(calendar.searchsorted(pd.Timestamp(event["auction_date"]), side="right"))
    if event["status"] == "KNOWN":
        end = int(
            calendar.searchsorted(pd.Timestamp(event["availability_after_date"]), side="right")
        )
        return start, end, "KNOWN_DELAYED_RELEASE", event["availability_after_date"]
    exact = event["availability_after_date"]
    bound = exact if exact is not None else event.get("clock_upper_bound_date")
    kind = (
        "UNKNOWN_EXACT_CLOCK"
        if exact is not None
        else "UNKNOWN_UPPER_BOUND"
        if bound is not None
        else "UNKNOWN_UNBOUNDED_CLOCK"
    )
    if bound is None:
        return start, len(calendar), kind, None
    # A release could first be usable at the first origin strictly after the
    # bound. That origin itself remains unknown, even with no other new event.
    last_possible = int(calendar.searchsorted(pd.Timestamp(bound), side="right"))
    return start, min(last_possible + 1, len(calendar)), kind, bound


def build_auction_features(reference_calendar: pd.DatetimeIndex, events: list[dict]) -> dict:
    """Return the full daily feature frame and independent per-event donor audit."""
    ordered = _envelope(reference_calendar, events)
    known_events = [event for event in ordered if event["status"] == "KNOWN"]
    unknown_events = [event for event in ordered if event["status"] == "UNKNOWN"]
    numbers = {event["event_id"]: _financial(event) for event in known_events}
    frame = pd.DataFrame(0.0, index=reference_calendar.copy(), columns=COLUMNS)
    frame.loc[frame.index < pd.Timestamp(FLOOR), :] = np.nan
    audits = []
    groups = {}
    pending_mask = np.zeros(len(reference_calendar), dtype=bool)
    for event in ordered:
        name = event["event_id"]
        audit = {
            "event_id": name,
            "auction_date": event["auction_date"],
            "source_status": event["status"],
            "origin": pd.NaT,
            "donor_ids": [],
            "status": "EXCLUDED",
            "mask_reason": "EXCLUDED_EVENT",
            "origin_masked": False,
            "pending_clock_kind": None,
            "pending_bound_date": None,
            "pending_origin_start": pd.NaT,
            "pending_origin_end": pd.NaT,
            "pending_origin_count": 0,
        }
        audits.append(audit)
        if event["status"] == "EXCLUDED":
            continue
        start, end, kind, bound = _pending(event, reference_calendar)
        pending_mask[start:end] = True
        audit.update(
            pending_clock_kind=kind,
            pending_bound_date=bound,
            pending_origin_start=reference_calendar[start] if start < end else pd.NaT,
            pending_origin_end=reference_calendar[end - 1] if start < end else pd.NaT,
            pending_origin_count=end - start,
        )
        after = (
            event["availability_after_date"]
            if event["status"] == "KNOWN"
            else event["auction_date"]
        )
        index = reference_calendar.searchsorted(pd.Timestamp(after), side="right")
        if index == len(reference_calendar):
            audit.update(
                status="OUTSIDE_CALENDAR", mask_reason="NO_ORIGIN_IN_REFERENCE_CALENDAR"
            )
            continue
        origin = reference_calendar[index]
        audit["origin"] = origin
        groups.setdefault(origin, []).append((event, audit))
        if event["status"] == "UNKNOWN":
            audit.update(status="UNKNOWN", mask_reason="UNKNOWN_CURRENT_EVENT")
            continue
        selected, reason = _prior(event, origin, known_events, unknown_events)
        audit["donor_ids"] = [prior["event_id"] for prior in selected]
        audit["share"] = numbers[name]["share"]
        if reason is not None:
            audit.update(status="UNSUPPORTED_PRIOR", mask_reason=reason)
            continue
        mean = math.fsum(numbers[prior["event_id"]]["share"] for prior in selected) / 12
        audit.update(status="SUPPORTED", mask_reason=None, prior_mean=mean)
    frame.loc[pending_mask, :] = np.nan
    for origin, current in groups.items():
        if pending_mask[reference_calendar.get_loc(origin)] or any(
            audit["status"] != "SUPPORTED" for _, audit in current
        ):
            frame.loc[origin, :] = np.nan
            for _, audit in current:
                audit["origin_masked"] = True
            continue
        rows = []
        for event, audit in current:
            value = numbers[event["event_id"]]
            rows.append(
                [
                    1,
                    *[
                        int(event["original_tenor_years"] == tenor)
                        for tenor in (3, 5, 7, 10, 30)
                    ],
                    int(event["reopening"]),
                    value["log_offering"],
                    value["high_yield"],
                    value["bid_to_cover"],
                    audit["prior_mean"],
                    value["share"] - audit["prior_mean"],
                ]
            )
        try:
            summed = [math.fsum(row[column] for row in rows) for column in range(len(COLUMNS))]
        except OverflowError as error:
            raise ValueError("Nonfinite aggregated auction features") from error
        if not all(math.isfinite(value) for value in summed):
            raise ValueError("Nonfinite aggregated auction features")
        frame.loc[origin, :] = summed
    return {"features": frame, "events": audits}
