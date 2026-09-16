"""Independent exact reconstruction of the fixed original-DOL claims ledger.

The caller authenticates and pins input artifacts. No producer ledger, source
parser, feature or model function is imported to establish expected values.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from urllib.parse import urlsplit

_FIRST = date(2009, 5, 30)
_LAST = date(2025, 10, 18)
_CEILING = date(2025, 10, 20)
_TAIL = frozenset(("2025-09-27", "2025-10-04", "2025-10-11", "2025-10-18"))
_REPORT_KEYS = frozenset(
    (
        "statistic",
        "observation_date",
        "actual_release_date",
        "value",
        "source_url",
        "release_text_sha256",
        "body_release_dates",
        "body_release_date_verified",
    )
)
_ALFRED_KEYS = frozenset(
    ("observation_date", "alfred_realtime_start_date", "value", "status", "source_line")
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _date(value):
    _require(
        type(value) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
        "Literal ISO date required",
    )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Valid calendar date required") from error
    _require(parsed <= _CEILING, "Source date exceeds ceiling")
    return parsed


def _integer(value, minimum=1):
    _require(type(value) is int and value >= minimum, "Positive exact integer required")


def _keys(row, keys):
    _require(type(row) is dict and set(row) == keys, "Exact source schema required")


def _identity(url, released):
    _require(type(url) is str and "?" not in url and "#" not in url, "Dated URL required")
    try:
        parts = urlsplit(url)
    except ValueError as error:
        raise ValueError("Valid official source URL required") from error
    _require(parts.scheme == "https", "Official HTTPS source required")
    if parts.netloc == "www.dol.gov":
        match = re.fullmatch(
            r"/sites/dolgov/files/OPA/newsreleases/ui-claims/([0-9]{4})[0-9]+\.pdf",
            parts.path,
        )
        _require(
            match is not None and match[1] == str(released.year), "OPA identity year mismatch"
        )
        return
    _require(parts.netloc == "oui.doleta.gov", "Exact official source host required")
    if url == "https://oui.doleta.gov/press/2019/010318.pdf" and released == date(2019, 1, 3):
        return
    match = re.fullmatch(r"/press/([0-9]{4})/([0-9]{6})\.(asp|pdf)", parts.path)
    _require(
        match is not None
        and match[1] == str(released.year)
        and match[2] == f"{released.month:02}{released.day:02}{released.year % 100:02}",
        "Archive filename does not match release date",
    )


def _sources(reports, records, first, last):
    _require(type(reports) is list and type(records) is list, "Complete source lists required")
    by_report, by_record, release_dates = {}, {}, set()
    for row in reports:
        _keys(row, _REPORT_KEYS)
        _require(
            row["statistic"] == "advance_seasonally_adjusted_initial_claims", "Wrong statistic"
        )
        week, released = _date(row["observation_date"]), _date(row["actual_release_date"])
        _require(
            first <= week <= last and week.weekday() == 5 and week < released,
            "Invalid report week/date",
        )
        _integer(row["value"])
        _identity(row["source_url"], released)
        _require(
            type(row["release_text_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", row["release_text_sha256"]) is not None,
            "Exact lowercase text SHA256 required",
        )
        body = row["body_release_dates"]
        _require(
            type(body) is list
            and len(body) == 1
            and type(body[0]) is str
            and body[0] == row["actual_release_date"]
            and row["body_release_date_verified"] is True,
            "Explicit matching body-date evidence required",
        )
        _require(
            week not in by_report and released not in release_dates,
            "Duplicate DOL week/release",
        )
        by_report[week] = row
        release_dates.add(released)
    for row in records:
        _keys(row, _ALFRED_KEYS)
        week, released = (
            _date(row["observation_date"]),
            _date(row["alfred_realtime_start_date"]),
        )
        _require(
            first <= week <= last and week.weekday() == 5 and week <= released,
            "Invalid ALFRED week/date",
        )
        _require(
            row["status"] == "observed", "Unavailable ALFRED observation cannot be admitted"
        )
        _integer(row["value"])
        _integer(row["source_line"], minimum=2)
        _require(week not in by_record, "Duplicate ALFRED reference week")
        by_record[week] = row
    return by_report, by_record


def _same(actual, expected, context):
    _require(type(actual) is type(expected), f"Type mismatch at {context}")
    if type(expected) is dict:
        _require(set(actual) == set(expected), f"Schema mismatch at {context}")
        for key, value in expected.items():
            _same(actual[key], value, f"{context}.{key}")
    elif type(expected) is list:
        _require(len(actual) == len(expected), f"Length mismatch at {context}")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{context}[{index}]")
    else:
        _require(actual == expected, f"Value mismatch at {context}")


def verify_ledger(ledger, reports, alfred_records, start="2009-05-30", end="2025-10-18"):
    """Reconstruct every fixed disposition from pinned records, or raise mismatch."""
    first, last = _date(start), _date(end)
    _require(
        _FIRST <= first <= last <= _LAST and first.weekday() == last.weekday() == 5,
        "Saturday bounds inside the fixed source interval required",
    )
    dol, alfred = _sources(reports, alfred_records, first, last)
    rows = []
    counts = {"observed": 0, "unresolved_correction": 0, "no_admitted_release": 0}
    cursor = first
    while cursor <= last:
        week = cursor.isoformat()
        source, crosscheck = dol.get(cursor), alfred.get(cursor)
        if week in _TAIL:
            _require(
                source is None and crosscheck is None,
                "Fixed tail gap contains a source observation",
            )
            row = {
                "reference_week": week,
                "release_date": None,
                "first_report_value": None,
                "status": "no_admitted_release",
                "source_url": None,
                "release_text_sha256": None,
                "source_comparison": {
                    "kind": "no_admitted_release",
                    "dol_value": None,
                    "alfred_value": None,
                    "alfred_realtime_start_date": None,
                    "alfred_source_line": None,
                    "value_agrees": None,
                    "release_date_agrees": None,
                },
            }
        else:
            _require(
                source is not None and crosscheck is not None,
                "Both sources required for every non-tail week",
            )
            values_agree = source["value"] == crosscheck["value"]
            dates_agree = (
                source["actual_release_date"] == crosscheck["alfred_realtime_start_date"]
            )
            first_value, status, kind = source["value"], "observed", "exact_agreement"
            if week == "2017-03-18":
                _require(
                    source["value"] == 261000
                    and source["actual_release_date"] == "2017-03-23"
                    and crosscheck["value"] == 258000
                    and crosscheck["alfred_realtime_start_date"] == "2017-03-23",
                    "Fixed unresolved correction evidence changed",
                )
                first_value, status, kind = (
                    None,
                    "unresolved_correction",
                    "unresolved_correction",
                )
            elif week == "2018-03-17":
                _require(
                    source["value"] == 229000
                    and source["actual_release_date"] == "2018-03-22"
                    and crosscheck["value"] == 227000
                    and crosscheck["alfred_realtime_start_date"] == "2018-03-29",
                    "Fixed original-DOL recovery evidence changed",
                )
                kind = "documented_original_dol_disagreement"
            else:
                _require(values_agree and dates_agree, "Unapproved source disagreement")
            row = {
                "reference_week": week,
                "release_date": source["actual_release_date"],
                "first_report_value": first_value,
                "status": status,
                "source_url": source["source_url"],
                "release_text_sha256": source["release_text_sha256"],
                "source_comparison": {
                    "kind": kind,
                    "dol_value": source["value"],
                    "alfred_value": crosscheck["value"],
                    "alfred_realtime_start_date": crosscheck["alfred_realtime_start_date"],
                    "alfred_source_line": crosscheck["source_line"],
                    "value_agrees": values_agree,
                    "release_date_agrees": dates_agree,
                },
            }
        rows.append(row)
        counts[row["status"]] += 1
        cursor += timedelta(days=7)
    expected = {
        "status": "FIRST_REPORT_LEDGER_BUILT_NOT_INDEPENDENTLY_VERIFIED",
        "start": start,
        "end": end,
        "rows": rows,
        "counts": counts,
    }
    _same(ledger, expected, "ledger")
    return {
        "status": "VERIFIED",
        "start": start,
        "end": end,
        "reference_weeks": len(rows),
        "counts": counts,
        "reports_checked": len(reports),
        "alfred_records_checked": len(alfred_records),
    }
