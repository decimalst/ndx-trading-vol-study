"""Independent Treasury clocks, donor histories, common calendar cohorts and QR fits."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from decimal import Decimal

import numpy as np
import pandas as pd

from src.verify_claims_release_forecasts import _compare, _market_targets

MARKET = (
    "const",
    "lrv_d",
    "lrv_w",
    "lrv_m",
    "lev_d",
    "lev_w",
    "lev_m",
    "liv",
    "lvix",
    "term",
    "xasset_stress",
    "market_stress",
)
HISTORY = ("tlt_ret", "tlt_r2", "tlt_lrv5", "tlt_lrv22") + tuple(
    f"weekday_{i}" for i in range(1, 5)
)
AUCTION = (
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
MATCHED = MARKET + HISTORY + AUCTION[:-1]
ALL = MATCHED + ("dealer_surprise",)
ARMS = {"market": MARKET, "matched": MATCHED, "candidate": ALL}
_FLOOR, _CEILING = pd.Timestamp("2010-01-01"), pd.Timestamp("2025-10-20")
_TENORS = (2, 3, 5, 7, 10, 30)
_EVENT = {
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
_VALUES = (
    "primary_dealer_accepted",
    "competitive_accepted",
    "offering_amount_usd",
    "high_yield",
    "bid_to_cover",
)
_CROSS = ("hyg", "tlt", "gld", "uso", "uup")
_APPLICATION = (
    "origin",
    "fit_origin",
    "training_cutoff",
    "train_n",
    "treasury_cutoff_date",
    "offset",
    "phase",
    "pred_market",
    "pred_matched",
    "pred_candidate",
)
_PANEL = (
    "origin",
    "model",
    "prediction",
    "y",
    "loss",
    "target_end",
    "phase",
    "treasury_cutoff_date",
    "offset",
    "fit_origin",
    "training_cutoff",
    "train_n",
)
_COVERAGE = (
    "origin",
    "phase",
    "feature_complete",
    "missing_features",
    "treasury_cutoff_date",
    "offset",
    "target_end",
    "target_observed",
    "target_within_phase",
    "scored",
    "status",
    "fit_origin",
    "training_cutoff",
)
_SCHEDULE = (
    "month",
    "status",
    "fit_origin",
    "training_cutoff",
    "requested_n",
    "application_n",
    "train_n",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _calendar(index, ceiling=_CEILING, empty=True):
    _need(isinstance(index, pd.DatetimeIndex), "Native calendar required")
    _need(
        (empty or len(index) > 0)
        and index.tz is None
        and not index.hasnans
        and index.is_unique
        and index.is_monotonic_increasing,
        "Invalid calendar identity",
    )
    _need(index.equals(index.normalize()), "Midnight calendar required")
    try:
        index.as_unit("ns")
    except (ValueError, OverflowError) as error:
        raise ValueError("Nanosecond-representable dates required") from error
    _need(
        bool((index <= ceiling).all()) and bool((index <= _CEILING).all()),
        "Calendar exceeds source ceiling",
    )


def _schema(frame, names, ceiling=_CEILING, empty=True):
    _need(
        isinstance(frame, pd.DataFrame)
        and frame.columns.is_unique
        and set(frame.columns) == set(names),
        "Exact input schema required",
    )
    _calendar(frame.index, ceiling, empty)


def _iso(value):
    _need(
        type(value) is str and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None,
        "Literal ISO config date required",
    )
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid config date") from error
    _need(result <= _CEILING, "Config exceeds fixed source ceiling")
    return result


def _config(config):
    if config is None:
        config = {
            "origin_start": "2016-01-01",
            "origin_end": "2025-10-20",
            "development": ["2016-01-01", "2019-12-31"],
            "evaluation": ["2020-01-01", "2025-10-20"],
            "source_end": "2025-10-20",
            "minimum_train": 1000,
        }
    _need(
        type(config) is dict
        and set(config)
        == {
            "origin_start",
            "origin_end",
            "development",
            "evaluation",
            "source_end",
            "minimum_train",
        },
        "Exact scheduling configuration required",
    )
    result = {
        name: _iso(config[name]) for name in ("origin_start", "origin_end", "source_end")
    }
    for phase in ("development", "evaluation"):
        _need(
            type(config[phase]) is list and len(config[phase]) == 2, "Phase date pair required"
        )
        result[phase] = tuple(_iso(value) for value in config[phase])
    a, b = result["development"]
    c, d = result["evaluation"]
    _need(
        _FLOOR
        <= result["origin_start"]
        <= a
        <= b
        < c
        <= d
        <= result["origin_end"]
        <= result["source_end"],
        "Ordered bounded phases and origin window required",
    )
    _need(
        type(config["minimum_train"]) is int and config["minimum_train"] > 0,
        "Positive exact training floor required",
    )
    result["minimum_train"] = config["minimum_train"]
    return result


def _real_dtype(series):
    dtype = series.dtype
    _need(
        (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype))
        and not pd.api.types.is_bool_dtype(dtype),
        "Real nonboolean numeric columns required",
    )


def _square(value):
    if math.isnan(value):
        return math.nan
    _need(math.isfinite(value), "Nonfinite history return")
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            result = float(np.float64(value) * np.float64(value))
    except FloatingPointError as error:
        raise ValueError("Nonzero square cannot be represented") from error
    _need(
        math.isfinite(result) and (value == 0 or result > 0),
        "Invalid squared-return arithmetic",
    )
    return result


def _event_envelope(events):
    """Validate every source identity/clock before any financial conversion."""
    _need(type(events) is list, "Explicit source event list required")
    names = set()
    for event in events:
        _need(
            type(event) is dict
            and set(event) in (_EVENT, _EVENT | {"clock_upper_bound_date"}),
            "Exact event schema required",
        )
        name = event["event_id"]
        _need(
            type(name) is str and bool(name.strip()) and name not in names,
            "Unique event identity required",
        )
        names.add(name)
        dates = {}
        for key in ("auction_date", "availability_after_date", "clock_upper_bound_date"):
            value = event.get(key)
            if value is not None:
                dates[key] = _iso(value)
                _need(dates[key] >= _FLOOR, "Source event precedes fixed floor")
        _need("auction_date" in dates, "Auction date required")
        status = event["status"]
        _need(
            type(status) is str and status in {"KNOWN", "UNKNOWN", "EXCLUDED"},
            "Source status required",
        )
        for key in ("availability_after_date", "clock_upper_bound_date"):
            _need(
                key not in dates or dates[key] >= dates["auction_date"],
                "Invalid event clock chronology",
            )
        _need(
            status != "KNOWN" or "availability_after_date" in dates,
            "Known release requires exact clock",
        )
        if "clock_upper_bound_date" in dates:
            _need(status == "UNKNOWN", "Only unknown events have upper clock bounds")
            _need(
                dates["clock_upper_bound_date"]
                >= dates.get("availability_after_date", dates["auction_date"]),
                "Clock upper bound precedes release",
            )
        tenor, reopened = event["original_tenor_years"], event["reopening"]
        _need(
            tenor is None or (type(tenor) is int and tenor in _TENORS),
            "Original tenor identity required",
        )
        _need(
            reopened is None or type(reopened) is bool, "Boolean reopening identity required"
        )
        _need(
            status != "KNOWN" or (tenor is not None and reopened is not None),
            "Known event has unresolved identity",
        )
        _need(
            status != "UNKNOWN" or all(event[key] is None for key in _VALUES),
            "Unknown financial fields cannot be supplied",
        )
    return sorted(events, key=lambda e: (e["auction_date"], e["event_id"]))


def _financial(event):
    dealer, total, offering = (event[key] for key in _VALUES[:3])
    _need(
        all(type(v) is int for v in (dealer, total, offering))
        and total > 0
        and offering > 0
        and 0 <= dealer <= total,
        "Exact positive accounting amounts required",
    )
    try:
        share = dealer / total
        logged = math.log(offering)
    except (OverflowError, ValueError) as error:
        raise ValueError("Unrepresentable financial arithmetic") from error
    _need(
        math.isfinite(share) and (dealer == 0 or share > 0) and math.isfinite(logged),
        "Unrepresentable financial arithmetic",
    )
    values = []
    for key in _VALUES[3:]:
        token = event[key]
        _need(
            type(token) is str
            and re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", token) is not None,
            "Plain source decimal required",
        )
        exact = Decimal(token)
        value = float(exact)
        _need(
            math.isfinite(value) and (value != 0 or exact == 0),
            "Nonfinite or underflowed financial decimal",
        )
        _need(key != "bid_to_cover" or value > 0, "Positive bid-to-cover required")
        values.append(value)
    return (share, logged, *values)


def _auction_features(index, events):
    """Scalar calendar/event reconstruction, independent of Treasury producers."""
    _calendar(index)
    ordered = _event_envelope(events)
    numbers = {e["event_id"]: _financial(e) for e in ordered if e["status"] == "KNOWN"}
    known = [e for e in ordered if e["status"] == "KNOWN"]
    unknown = [e for e in ordered if e["status"] == "UNKNOWN"]
    values = np.zeros((len(index), len(AUCTION)))
    values[index < _FLOOR] = np.nan
    sessions = list(index)
    pending = set()
    groups, audits = {}, []
    for event in ordered:
        name, status = event["event_id"], event["status"]
        row = dict(
            event_id=name,
            auction_date=event["auction_date"],
            source_status=status,
            origin=pd.NaT,
            donor_ids=[],
            status="EXCLUDED",
            mask_reason="EXCLUDED_EVENT",
            origin_masked=False,
            pending_clock_kind=None,
            pending_bound_date=None,
            pending_origin_start=pd.NaT,
            pending_origin_end=pd.NaT,
            pending_origin_count=0,
        )
        audits.append(row)
        if status == "EXCLUDED":
            continue
        auction = pd.Timestamp(event["auction_date"])
        release = event["availability_after_date"]
        bound = release if release is not None else event.get("clock_upper_bound_date")
        start = bisect_right(sessions, auction)
        if status == "KNOWN":
            kind = "KNOWN_DELAYED_RELEASE"
            masked = list(range(start, bisect_right(sessions, pd.Timestamp(bound))))
        else:
            kind = (
                "UNKNOWN_EXACT_CLOCK"
                if release is not None
                else "UNKNOWN_UPPER_BOUND"
                if bound is not None
                else "UNKNOWN_UNBOUNDED_CLOCK"
            )
            stop = (
                len(index)
                if bound is None
                else min(len(index), bisect_right(sessions, pd.Timestamp(bound)) + 1)
            )
            masked = list(range(start, stop))
        pending.update(masked)
        row.update(
            pending_clock_kind=kind,
            pending_bound_date=bound,
            pending_origin_start=index[masked[0]] if masked else pd.NaT,
            pending_origin_end=index[masked[-1]] if masked else pd.NaT,
            pending_origin_count=len(masked),
        )
        after = pd.Timestamp(release) if status == "KNOWN" else auction
        position = bisect_right(sessions, after)
        if position == len(index):
            row.update(
                status="OUTSIDE_CALENDAR", mask_reason="NO_ORIGIN_IN_REFERENCE_CALENDAR"
            )
            continue
        row["origin"] = index[position]
        groups.setdefault(position, []).append((event, row))
        if status == "UNKNOWN":
            row.update(status="UNKNOWN", mask_reason="UNKNOWN_CURRENT_EVENT")
            continue
        candidates = [
            e
            for e in known
            if e["original_tenor_years"] == event["original_tenor_years"]
            and e["auction_date"] < event["auction_date"]
        ]
        donors = candidates[-12:]
        row.update(donor_ids=[e["event_id"] for e in donors], share=numbers[name][0])
        reason = None
        if len(donors) < 12:
            reason = "INSUFFICIENT_PRIOR_AUCTIONS"
        elif (
            len(candidates) > 12
            and candidates[-13]["auction_date"] == donors[0]["auction_date"]
        ):
            reason = "AMBIGUOUS_PRIOR_CUTOFF"
        elif any(
            pd.Timestamp(e["availability_after_date"]) >= index[position] for e in donors
        ):
            reason = "UNRELEASED_PRIOR_EVENT"
        elif any(
            e["original_tenor_years"] in (None, event["original_tenor_years"])
            and donors[0]["auction_date"] <= e["auction_date"] < event["auction_date"]
            for e in unknown
        ):
            reason = "UNKNOWN_PRIOR_EVENT"
        if reason is not None:
            row.update(status="UNSUPPORTED_PRIOR", mask_reason=reason)
        else:
            row.update(
                status="SUPPORTED",
                mask_reason=None,
                prior_mean=math.fsum(numbers[e["event_id"]][0] for e in donors) / 12,
            )
    for position in pending:
        values[position] = np.nan
    for position, pairs in groups.items():
        if position in pending or any(row["status"] != "SUPPORTED" for _, row in pairs):
            values[position] = np.nan
            for _, row in pairs:
                row["origin_masked"] = True
            continue
        entries = []
        for event, row in pairs:
            share, logged, high_yield, cover = numbers[event["event_id"]]
            entries.append(
                [
                    1,
                    *[int(event["original_tenor_years"] == t) for t in _TENORS[1:]],
                    int(event["reopening"]),
                    logged,
                    high_yield,
                    cover,
                    row["prior_mean"],
                    share - row["prior_mean"],
                ]
            )
        try:
            values[position] = [
                math.fsum(entry[j] for entry in entries) for j in range(len(AUCTION))
            ]
        except OverflowError as error:
            raise ValueError("Nonfinite auction aggregation") from error
    _need(not np.isinf(values).any(), "Nonfinite auction feature arithmetic")
    return {"features": pd.DataFrame(values, index=index, columns=AUCTION), "events": audits}


def _market_context(index, cross):
    _calendar(index)
    _schema(cross, _CROSS)
    _real_dtype(cross.tlt)
    prices = cross.tlt.reindex(index).copy()
    prices.loc[index < _FLOOR] = np.nan
    raw = prices.to_numpy(dtype=float, na_value=np.nan)
    _need(
        not np.isinf(raw).any() and bool(np.all(np.isnan(raw) | (raw > 0))),
        "Positive finite TLT observations required",
    )
    values = np.full((len(index), len(HISTORY)), np.nan)
    for i in range(2, len(index)):
        if np.isfinite(raw[i - 2 : i]).all():
            values[i, 0] = math.log(raw[i - 1]) - math.log(raw[i - 2])
            values[i, 1] = _square(float(values[i, 0]))
    for i, day in enumerate(index):
        for j, window in ((2, 5), (3, 22)):
            sample = values[max(0, i + 1 - window) : i + 1, 1]
            if len(sample) == window and np.isfinite(sample).all():
                total = math.fsum(sample)
                mean = total / window
                _need(
                    math.isfinite(mean) and (total == 0 or mean > 0), "Invalid TLT strict mean"
                )
                if mean > 0:
                    values[i, j] = math.log(mean)
        values[i, 4:] = [float(day.weekday() == weekday) for weekday in range(1, 5)]
    result = pd.DataFrame(values, index=index, columns=HISTORY)
    result["treasury_cutoff_date"] = (
        pd.Series(
            [pd.NaT] + [d if d >= _FLOOR else pd.NaT for d in index[:-1]],
            index=index,
            dtype="datetime64[ns]",
        )
        if len(index)
        else pd.Series(index=index, dtype="datetime64[ns]")
    )
    return result


def _inputs(sources, events, features, targets, cfg):
    _need(
        type(sources) is dict and set(sources) == {"daily", "cross", "iv"},
        "Exact raw market source mapping required",
    )
    for name, names in (
        ("daily", ("open", "high", "low", "close", "adj close", "volume")),
        ("cross", _CROSS),
        ("iv", ("vxn", "vix", "vix9d")),
    ):
        _schema(sources[name], names, cfg["source_end"], empty=name != "daily")
    _event_envelope(events)
    _schema(
        features, ALL + ("rv_total", "treasury_cutoff_date"), cfg["source_end"], empty=False
    )
    _schema(targets, ("y", "target_end"), cfg["source_end"], empty=False)
    index = sources["daily"].index
    _need(
        features.index.equals(index) and targets.index.equals(index),
        "Full original market calendar must be preserved",
    )
    market, expected_targets = _market_targets(**sources)
    context = _market_context(index, sources["cross"])
    auction = _auction_features(index, events)
    expected = pd.concat([market, context, auction["features"]], axis=1)
    for actual, wanted, label in (
        (features, expected, "features"),
        (targets, expected_targets, "targets"),
    ):
        for column in wanted:
            if column not in ("treasury_cutoff_date", "target_end"):
                _real_dtype(actual[column])
            for i in range(len(index)):
                _compare(
                    actual[column].iloc[i], wanted[column].iloc[i], f"{label}[{i}].{column}"
                )
    return expected, expected_targets, auction["events"]


_SUPPORT_DEFAULTS = dict(
    phase_daily=505,
    slice_daily=252,
    offset_daily=63,
    training_activation_dates=200,
    phase_activation_dates=104,
    slice_activation_dates=52,
    phase_active_dates_per_offset=20,
    training_events_per_tenor=24,
    phase_events_per_tenor=12,
    slice_events_per_tenor=6,
)


def _support_config(config):
    if config is None:
        cfg = _config(None)
        return (
            cfg,
            [
                (_iso("2020-01-01"), _iso("2022-12-31")),
                (_iso("2023-01-01"), _iso("2025-10-20")),
            ],
            _SUPPORT_DEFAULTS.copy(),
        )
    _need(
        type(config) is dict and set(config) == {"forecast", "stability", "support"},
        "Exact support configuration required",
    )
    cfg = _config(config["forecast"])
    floors = config["support"]
    _need(
        type(floors) is dict
        and set(floors) == set(_SUPPORT_DEFAULTS)
        and all(type(v) is int and v > 0 for v in floors.values()),
        "Exact positive support floors required",
    )
    pairs = config["stability"]
    _need(type(pairs) is list and len(pairs) == 2, "Two stability intervals required")
    intervals = []
    for pair in pairs:
        _need(type(pair) is list and len(pair) == 2, "Stability date pair required")
        a, b = map(_iso, pair)
        _need(
            cfg["evaluation"][0] <= a <= b <= cfg["evaluation"][1],
            "Stability interval outside evaluation",
        )
        intervals.append((a, b))
    _need(intervals[0][1] < intervals[1][0], "Ordered disjoint stability intervals required")
    return cfg, intervals, floors.copy()


def _support_expected(features, targets, audits, events, cfg, intervals, floors):
    """Count explicit session/identity memberships without calling a fitter."""
    index = features.index
    complete = np.isfinite(features.loc[:, ALL].to_numpy(float)).all(axis=1)
    _need(
        not np.any(complete & features.treasury_cutoff_date.isna()),
        "Common rows require a previous-session cutoff",
    )
    tenors = {e["event_id"]: e["original_tenor_years"] for e in events}
    identities = {}
    for row in audits:
        if (
            row["status"] == "SUPPORTED"
            and row["source_status"] == "KNOWN"
            and not row["origin_masked"]
        ):
            identities.setdefault(index.get_loc(row["origin"]), []).append(
                tenors[row["event_id"]]
            )
    discrepancies = []

    def count(positions):
        by_tenor = {str(t): 0 for t in _TENORS}
        for position in positions:
            for tenor in identities.get(position, []):
                by_tenor[str(tenor)] += 1
        return dict(
            daily_n=len(positions),
            activation_dates=sum(bool(identities.get(i)) for i in positions),
            event_n=sum(by_tenor.values()),
            events_per_tenor=by_tenor,
        )

    def gates(scope, triples):
        failures = []
        for metric, observed, required in triples:
            if observed < required:
                failures.append(metric)
                discrepancies.append(
                    dict(scope=scope, metric=metric, observed=observed, required=required)
                )
        return failures

    def evaluate(positions, a, b, scope, kind, **extra):
        counts = count(positions)
        activekey = (
            "phase_active_dates_per_offset" if kind == "offset" else kind + "_activation_dates"
        )
        triples = [
            (kind + "_daily", counts["daily_n"], floors[kind + "_daily"]),
            (activekey, counts["activation_dates"], floors[activekey]),
        ]
        if kind != "offset":
            key = kind + "_events_per_tenor"
            triples.extend(
                (key + ":" + str(t), counts["events_per_tenor"][str(t)], floors[key])
                for t in _TENORS
            )
        failures = gates(scope, triples)
        return dict(
            origin_start=a.strftime("%Y-%m-%d"),
            origin_end=b.strftime("%Y-%m-%d"),
            **extra,
            **counts,
            passed=not failures,
            failures=failures,
        )

    requested = [
        i for i, day in enumerate(index) if cfg["origin_start"] <= day <= cfg["origin_end"]
    ]
    monthly = []
    for month in pd.period_range(cfg["origin_start"], cfg["origin_end"], freq="M"):
        positions = [i for i in requested if index[i].to_period("M") == month]
        applications = [i for i in positions if complete[i]]
        row = dict(
            month=str(month),
            status="NO_COMPLETE_ORIGIN" if positions else "NO_REQUESTED_ORIGINS",
            fit_origin=None,
            training_cutoff=None,
            requested_n=len(positions),
            application_n=len(applications),
            train_n=None,
            train_activation_dates=None,
            train_event_n=None,
            train_events_per_tenor=None,
            passed=True,
            failures=[],
        )
        if applications:
            first = applications[0]
            _need(first > 0, "Support schedule requires previous session")
            origin, cutoff = index[first], index[first - 1]
            positions = [
                i
                for i in range(first)
                if complete[i]
                and math.isfinite(targets.y.iloc[i])
                and pd.notna(targets.target_end.iloc[i])
                and targets.target_end.iloc[i] <= cutoff
            ]
            counts = count(positions)
            triples = [
                ("minimum_train", counts["daily_n"], cfg["minimum_train"]),
                (
                    "training_activation_dates",
                    counts["activation_dates"],
                    floors["training_activation_dates"],
                ),
            ]
            triples.extend(
                (
                    "training_events_per_tenor:" + str(t),
                    counts["events_per_tenor"][str(t)],
                    floors["training_events_per_tenor"],
                )
                for t in _TENORS
            )
            failures = gates("monthly:" + str(month), triples)
            row.update(
                status="INSUFFICIENT_DATA" if failures else "SUPPORTED",
                fit_origin=origin.strftime("%Y-%m-%d"),
                training_cutoff=cutoff.strftime("%Y-%m-%d"),
                train_n=counts["daily_n"],
                train_activation_dates=counts["activation_dates"],
                train_event_n=counts["event_n"],
                train_events_per_tenor=counts["events_per_tenor"],
                passed=not failures,
                failures=failures,
            )
        monthly.append(row)
    phases, offsets, scored = {}, {}, {}
    for name in ("development", "evaluation"):
        a, b = cfg[name]
        fence = b if name == "development" else cfg["source_end"]
        positions = [
            i
            for i, day in enumerate(index)
            if complete[i]
            and a <= day <= b
            and math.isfinite(targets.y.iloc[i])
            and pd.notna(targets.target_end.iloc[i])
            and targets.target_end.iloc[i] <= fence
        ]
        scored[name] = positions
        phases[name] = evaluate(positions, a, b, "phase:" + name, "phase")
        offsets[name] = [
            evaluate(
                [i for i in positions if i % 5 == offset],
                a,
                b,
                f"offset:{name}:{offset}",
                "offset",
                offset=offset,
            )
            for offset in range(5)
        ]
    slices = [
        evaluate(
            [i for i in scored["evaluation"] if a <= index[i] <= b],
            a,
            b,
            f"slice:{j}",
            "slice",
            slice_index=j,
        )
        for j, (a, b) in enumerate(intervals)
    ]
    return dict(
        status="INSUFFICIENT_DATA" if discrepancies else "SUPPORT_PASS",
        passed=not discrepancies,
        monthly=monthly,
        phases=phases,
        slices=slices,
        offsets=offsets,
        discrepancies=discrepancies,
    )


def verify_inputs_and_support(
    sources, ledger_events, features, targets, event_audit, support, config=None
):
    """Verify input reconstruction and all support gates, including a failed attempt."""
    cfg, intervals, floors = _support_config(config)
    expected, wanted_targets, audits = _inputs(sources, ledger_events, features, targets, cfg)
    _compare(event_audit, audits, "event_audit")
    wanted = _support_expected(
        expected, wanted_targets, audits, ledger_events, cfg, intervals, floors
    )
    _compare(support, wanted, "support")
    return dict(
        status="VERIFIED",
        support_passed=wanted["passed"],
        support_status=wanted["status"],
        calendar_rows=len(expected),
        events_verified=len(audits),
        feature_columns_verified=len(expected.columns),
        monthly_support_rows=len(wanted["monthly"]),
        limitations=[
            "Saved source ledger identity and provenance are authenticated separately; no new original-document or release-timestamp certification."
        ],
    )


def _fit_expected(train, y, application):
    """Three independent QR fits with arm-specific residual smearing and audits."""
    for frame in (train, application):
        _need(
            isinstance(frame, pd.DataFrame)
            and frame.index.is_unique
            and frame.columns.is_unique
            and set(ALL) <= set(frame.columns),
            "Unique common32 design required",
        )
        for name in ALL:
            _real_dtype(frame[name])
        values = frame.loc[:, ALL].to_numpy(dtype=float, na_value=np.nan)
        _need(
            np.isfinite(values).all() and bool((values[:, 0] == 1).all()),
            "Finite common32 rows and unit constant required",
        )
    if isinstance(y, pd.Series):
        _need(y.index.equals(train.index), "Target index mismatch")
        _real_dtype(y)
    target = np.asarray(y)
    _need(
        target.dtype.kind in "fiu"
        and target.ndim == 1
        and len(target) == len(train)
        and len(train) >= len(ALL),
        "Positive aligned target vector and supported design required",
    )
    target = target.astype(float)
    _need(
        np.isfinite(target).all() and bool((target > 0).all()),
        "Positive finite target required",
    )
    logged = np.log(target)
    predictions, audits = {}, {}
    for arm, columns in ARMS.items():
        raw, query = (
            frame.loc[:, columns].to_numpy(dtype=float) for frame in (train, application)
        )
        means = np.array([math.fsum(raw[:, j]) / len(raw) for j in range(1, len(columns))])
        scales = np.sqrt(np.mean((raw[:, 1:] - means) ** 2, axis=0))
        _need(
            np.isfinite(means).all()
            and np.isfinite(scales).all()
            and bool((scales > 1e-12).all()),
            "Fixed feature scale failure",
        )
        design = np.column_stack((np.ones(len(raw)), (raw[:, 1:] - means) / scales))
        query_design = np.column_stack((np.ones(len(query)), (query[:, 1:] - means) / scales))
        _need(
            np.isfinite(design).all() and np.isfinite(query_design).all(),
            "Invalid scaled design",
        )
        try:
            singular = np.linalg.svd(design, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
            _need(rank == len(columns), "Fixed model rank failure")
            q, r = np.linalg.qr(design, mode="reduced")
            beta = np.linalg.solve(r, q.T @ logged)
        except np.linalg.LinAlgError as error:
            raise ValueError("Independent QR fit failed") from error
        residual = logged - design @ beta
        top = float(residual.max())
        smear = top + math.log(
            math.fsum(math.exp(float(v) - top) for v in residual) / len(raw)
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            forecast = np.exp(query_design @ beta + smear)
        _need(
            np.isfinite(beta).all()
            and math.isfinite(smear)
            and np.isfinite(forecast).all()
            and bool((forecast > 0).all()),
            "Invalid forecast or smearing arithmetic",
        )
        predictions[arm] = forecast
        audits[arm] = {
            "feature_names": list(columns),
            "coefficients": dict(zip(columns, map(float, beta))),
            "feature_means": dict(zip(columns[1:], map(float, means))),
            "feature_scales": dict(zip(columns[1:], map(float, scales))),
            "singular_values": list(map(float, singular)),
            "rank": rank,
            "rank_relative_cutoff": 1e-12,
            "log_smearing": smear,
            "normal_equation_max_abs": float(np.abs(design.T @ residual / len(raw)).max()),
            "n_train": len(raw),
            "n_application": len(query),
        }
    return predictions, audits


def _frame(actual, rows, columns, name):
    _need(
        isinstance(actual, pd.DataFrame)
        and tuple(actual.columns) == columns
        and len(actual) == len(rows),
        "Frame schema/count mismatch: " + name,
    )
    for i, row in enumerate(rows):
        for column in columns:
            _compare(actual.iloc[i][column], row[column], f"{name}[{i}].{column}")


def verify_forecasts(sources, ledger_events, features, targets, produced, config=None):
    """Reconstruct every feature, target, successful fit, application and coverage row."""
    cfg = _config(config)
    expected, expected_targets, _ = _inputs(sources, ledger_events, features, targets, cfg)
    features, targets = expected, expected_targets
    _need(
        type(produced) is dict
        and set(produced) == {"applications", "panel", "coverage", "schedules", "fits"},
        "Exact produced payload required",
    )
    matrix = features.loc[:, ALL].to_numpy(float)
    _need(not np.isinf(matrix).any(), "Invalid reconstructed feature arithmetic")
    complete = np.isfinite(matrix).all(axis=1)
    index = features.index
    cutoff_dates = features.treasury_cutoff_date
    _need(
        not bool(np.any(complete & cutoff_dates.isna())),
        "Complete row without Treasury cutoff",
    )
    requested = [
        i for i, day in enumerate(index) if cfg["origin_start"] <= day <= cfg["origin_end"]
    ]

    def phase(day):
        for name in ("development", "evaluation"):
            if cfg[name][0] <= day <= cfg[name][1]:
                return name
        return "outside_phase"

    fits, schedules, applications, by_position = [], [], [], {}
    for month in pd.period_range(cfg["origin_start"], cfg["origin_end"], freq="M"):
        positions = [i for i in requested if index[i].to_period("M") == month]
        query_positions = [i for i in positions if complete[i]]
        schedule = {
            "month": str(month),
            "status": "no_requested_origins" if not positions else "no_complete_origin",
            "fit_origin": pd.NaT,
            "training_cutoff": pd.NaT,
            "requested_n": len(positions),
            "application_n": len(query_positions),
            "train_n": None,
        }
        if query_positions:
            first = query_positions[0]
            _need(first > 0, "Previous full-session cutoff required")
            origin, cutoff = index[first], index[first - 1]
            train_positions = [
                i
                for i in range(first)
                if complete[i]
                and pd.notna(targets.target_end.iloc[i])
                and targets.target_end.iloc[i] <= cutoff
                and math.isfinite(targets.y.iloc[i])
                and targets.y.iloc[i] > 0
            ]
            _need(
                len(train_positions) >= cfg["minimum_train"],
                "INSUFFICIENT_DATA: monthly common training floor",
            )
            prediction, audit = _fit_expected(
                features.iloc[train_positions],
                targets.y.iloc[train_positions],
                features.iloc[query_positions],
            )
            fits.append(
                {
                    "month": str(month),
                    "fit_origin": origin.strftime("%Y-%m-%d"),
                    "training_cutoff": cutoff.strftime("%Y-%m-%d"),
                    "train_origins": [index[i].strftime("%Y-%m-%d") for i in train_positions],
                    "application_origins": [
                        index[i].strftime("%Y-%m-%d") for i in query_positions
                    ],
                    "train_n": len(train_positions),
                    "model_audits": audit,
                }
            )
            schedule.update(
                status="fitted",
                fit_origin=origin,
                training_cutoff=cutoff,
                train_n=len(train_positions),
            )
            for j, i in enumerate(query_positions):
                app = {
                    "origin": index[i],
                    "fit_origin": origin,
                    "training_cutoff": cutoff,
                    "train_n": len(train_positions),
                    "treasury_cutoff_date": cutoff_dates.iloc[i],
                    "offset": i % 5,
                    "phase": phase(index[i]),
                    **{"pred_" + arm: float(prediction[arm][j]) for arm in ARMS},
                }
                applications.append(app)
                by_position[i] = app
        schedules.append(schedule)
    coverage, panel = [], []
    for i in requested:
        day, assigned = index[i], phase(index[i])
        endpoint, value = targets.target_end.iloc[i], targets.y.iloc[i]
        observed = bool(math.isfinite(value) and value > 0)
        fence = cfg["development"][1] if assigned == "development" else cfg["source_end"]
        within = bool(assigned != "outside_phase" and pd.notna(endpoint) and endpoint <= fence)
        if not complete[i]:
            status = "incomplete_features"
        elif assigned == "outside_phase":
            status = "outside_phase"
        elif pd.isna(endpoint):
            status = "target_not_mature"
        elif not observed:
            status = "missing_target"
        elif not within:
            status = "target_after_phase_cutoff"
        else:
            status = "scored"
        app = by_position.get(i)
        coverage.append(
            {
                "origin": day,
                "phase": assigned,
                "feature_complete": bool(complete[i]),
                "missing_features": "|".join(
                    name for j, name in enumerate(ALL) if not math.isfinite(matrix[i, j])
                ),
                "treasury_cutoff_date": cutoff_dates.iloc[i],
                "offset": i % 5,
                "target_end": endpoint,
                "target_observed": observed,
                "target_within_phase": within,
                "scored": status == "scored",
                "status": status,
                "fit_origin": pd.NaT if app is None else app["fit_origin"],
                "training_cutoff": pd.NaT if app is None else app["training_cutoff"],
            }
        )
        if status == "scored":
            for arm in ARMS:
                forecast = app["pred_" + arm]
                ratio = value / forecast
                _need(math.isfinite(ratio) and ratio > 0, "Invalid positive QLIKE ratio")
                loss = ratio - math.log(ratio) - 1
                _need(math.isfinite(loss), "Invalid QLIKE loss")
                panel.append(
                    {
                        "origin": day,
                        "model": arm,
                        "prediction": forecast,
                        "y": float(value),
                        "loss": loss,
                        "target_end": endpoint,
                        "phase": assigned,
                        "treasury_cutoff_date": cutoff_dates.iloc[i],
                        "offset": i % 5,
                        "fit_origin": app["fit_origin"],
                        "training_cutoff": app["training_cutoff"],
                        "train_n": app["train_n"],
                    }
                )
    _compare(produced["fits"], fits, "fits")
    for name, rows, columns in (
        ("applications", applications, _APPLICATION),
        ("panel", panel, _PANEL),
        ("coverage", coverage, _COVERAGE),
        ("schedules", schedules, _SCHEDULE),
    ):
        _frame(produced[name], rows, columns, name)
    return {
        "status": "VERIFIED",
        "application_origins": len(applications),
        "application_forecasts_verified": 3 * len(applications),
        "scored_origins": len(panel) // 3,
        "forecasts_verified": len(panel),
        "monthly_fits": len(fits),
        "model_fits": 3 * len(fits),
        "coverage_origins": len(coverage),
        "calendar_rows": len(index),
    }
