"""Pure, independent DOL initial-headline extraction and ALFRED reconciliation.

Callers own official acquisition and text extraction. No network, historical
source loader, producer value parser, feature construction or scoring is used.
"""

from __future__ import annotations

import calendar
import hashlib
import re
from datetime import date, timedelta
from urllib.parse import urlsplit

SOURCE_CEILING = date(2025, 10, 20)
STATISTIC = "advance_seasonally_adjusted_initial_claims"
RECORD_FIELDS = {
    "observation_date",
    "alfred_realtime_start_date",
    "value",
    "status",
    "source_line",
}
REPORT_FIELDS = {
    "statistic",
    "observation_date",
    "actual_release_date",
    "value",
    "source_url",
    "release_text_sha256",
    "body_release_dates",
    "body_release_date_verified",
}
MONTHS = {
    label.lower(): number
    for number in range(1, 13)
    for label in (calendar.month_name[number], calendar.month_abbr[number])
}
MONTHS["sept"] = 9
WEEKDAYS = {label.lower(): number for number, label in enumerate(calendar.day_name)}
HEADLINE = re.compile(
    r"\bIn\s+the\s+week\s+ending\s+(?P<month>[A-Za-z]+)\.?\s+"
    r"(?P<day>[0-9]{1,2})(?:,\s*(?P<year>[0-9]{4}),\s*|,\s*|\s+)"
    r"the\s+a\s*d\s*v\s*a\s*n\s*c\s*e\s+figure\s+for\s+seasonally\s+adjusted\s+"
    r"initial\s+claims\s+was\s+",
    re.IGNORECASE,
)
RELEASE_HEADER = re.compile(
    r"(?:\bfor\s+release\b|\brelease\s+date\s*:|\bembargoed\s+until\b)"
    r".{0,180}?"
    r"(?:(?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
    r"(?P<day>[0-9]{1,2})\s*,?\s+(?P<year>[0-9]{4})\b",
    re.IGNORECASE,
)
COUNT_PREFIX = re.compile(r"(?P<count>[0-9][0-9,\s]*)(?:,\s*(?=[A-Za-z])|[.;](?=\s|$)|$)")


def _iso(value):
    if type(value) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Literal ISO calendar date required")
    try:
        result = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Valid calendar date required") from error
    if result > SOURCE_CEILING:
        raise ValueError("Date exceeds fixed source ceiling")
    return result


def _dated_url(source_url, released):
    """Validate identity and return whether explicit agreeing body dates are required."""
    if type(source_url) is not str:
        raise ValueError("Official dated source URL required")
    parts = urlsplit(source_url)
    if (
        parts.scheme != "https"
        or parts.netloc not in ("oui.doleta.gov", "www.dol.gov")
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Official dated source URL required")
    if parts.netloc == "www.dol.gov":
        match = re.fullmatch(
            r"/sites/dolgov/files/OPA/newsreleases/ui-claims/([0-9]{4})[0-9]+\.pdf",
            parts.path,
        )
        if match is None or match[1] != str(released.year):
            raise ValueError("Official numeric OPA identity must match actual release year")
        return True
    if source_url == "https://oui.doleta.gov/press/2019/010318.pdf" and released == date(
        2019, 1, 3
    ):
        return True
    match = re.fullmatch(r"/press/([0-9]{4})/([0-9]{6})\.(asp|pdf)", parts.path)
    if (
        match is None
        or match[1] != str(released.year)
        or match[2] != released.strftime("%m%d%y")
    ):
        raise ValueError("Dated URL and actual release date disagree")
    return False


def _calendar_date(month, day, year):
    try:
        return date(int(year), MONTHS[month.lower()], int(day))
    except (ValueError, KeyError) as error:
        raise ValueError("Valid stated month/day/year required") from error


def _week(match, released):
    year = match["year"]
    if year is not None:
        result = _calendar_date(match["month"], match["day"], year)
    else:
        result = _calendar_date(match["month"], match["day"], released.year)
        if result > released:
            result = _calendar_date(match["month"], match["day"], released.year - 1)
    if result.weekday() != 5 or result >= released or result > SOURCE_CEILING:
        raise ValueError("Headline week must be Saturday strictly before actual release")
    return result


def _integer_text(token):
    token = re.sub(r"\s*,\s*", ",", token.strip())
    valid = (
        re.fullmatch(r"[0-9]+", token)
        or re.fullmatch(r"[1-9][0-9]{0,2}(?:,[0-9]{3})+", token)
        or re.fullmatch(r"[1-9][0-9]{0,2}(?: [0-9]{3})+", token)
    )
    if not valid:
        raise ValueError("One positive integer headline count with valid grouping required")
    value = int(token.replace(",", "").replace(" ", ""))
    if value <= 0:
        raise ValueError("Positive headline count required")
    return value


def parse_release_text(text, release_date, source_url):
    """Parse one advance SA initial-claims headline without revised-value fallback."""
    released = _iso(release_date)
    body_date_required = _dated_url(source_url, released)
    if type(text) is not str or not text.strip():
        raise ValueError("Nonempty extracted release text required")
    normalized = " ".join(text.split())
    headlines = list(HEADLINE.finditer(normalized))
    if len(headlines) != 1:
        raise ValueError(
            "Exactly one advance seasonally adjusted initial-claims headline required"
        )
    headline = headlines[0]
    week = _week(headline, released)
    body_dates = []
    for header in RELEASE_HEADER.finditer(normalized):
        body_date = _calendar_date(header["month"], header["day"], header["year"])
        if body_date != released:
            raise ValueError("Explicit body release date disagrees with dated source")
        if header["weekday"] and WEEKDAYS[header["weekday"].lower()] != body_date.weekday():
            raise ValueError("Explicit body weekday disagrees with release date")
        body_dates.append(body_date.isoformat())
    if body_date_required and not body_dates:
        raise ValueError(
            "Noncalendar source identity requires an explicit agreeing body release date"
        )
    number = COUNT_PREFIX.match(normalized[headline.end() :])
    if number is None:
        raise ValueError("Malformed advance initial-claims count")
    value = _integer_text(number["count"])
    return {
        "statistic": STATISTIC,
        "observation_date": week.isoformat(),
        "actual_release_date": released.isoformat(),
        "value": value,
        "source_url": source_url,
        "release_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "body_release_dates": sorted(set(body_dates)),
        "body_release_date_verified": bool(body_dates),
    }


def _positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Positive exact integer required")


def _record(row):
    if type(row) is not dict or set(row) != RECORD_FIELDS:
        raise ValueError("Exact source-record schema required")
    week = _iso(row["observation_date"])
    vintage = _iso(row["alfred_realtime_start_date"])
    if week.weekday() != 5 or vintage < week:
        raise ValueError("Saturday observation and non-earlier ALFRED date required")
    _positive_integer(row["source_line"])
    if row["source_line"] < 2:
        raise ValueError("Source CSV data line must follow its header")
    if row["status"] == "observed":
        _positive_integer(row["value"])
    elif row["status"] != "missing" or row["value"] is not None:
        raise ValueError("Observed integer or explicit unavailable source value required")
    return dict(row)


def _report(row):
    if type(row) is not dict or set(row) != REPORT_FIELDS or row["statistic"] != STATISTIC:
        raise ValueError("Exact independent advance-initial-claims report schema required")
    week = _iso(row["observation_date"])
    released = _iso(row["actual_release_date"])
    if week.weekday() != 5 or week >= released:
        raise ValueError("Report Saturday must precede actual release")
    body_date_required = _dated_url(row["source_url"], released)
    _positive_integer(row["value"])
    if (
        type(row["release_text_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", row["release_text_sha256"]) is None
    ):
        raise ValueError("Exact extracted-text SHA-256 required")
    body = row["body_release_dates"]
    if (
        type(body) is not list
        or body not in ([], [released.isoformat()])
        or type(row["body_release_date_verified"]) is not bool
        or row["body_release_date_verified"] != bool(body)
        or (body_date_required and not body)
    ):
        raise ValueError("Explicit body-date evidence must remain separate and consistent")
    return dict(row)


def _unique(rows, date_field):
    if len({row["observation_date"] for row in rows}) != len(rows) or len(
        {row[date_field] for row in rows}
    ) != len(rows):
        raise ValueError("Duplicate reference weeks or release dates are ambiguous")


def reconcile_reports(records, reports):
    """Return complete agreement or explicit whole-scope mismatches; never impute."""
    if type(records) is not list or type(reports) is not list:
        raise ValueError("Complete source-record and independent-report lists required")
    records = [_record(row) for row in records]
    reports = [_report(row) for row in reports]
    _unique(records, "alfred_realtime_start_date")
    _unique(reports, "actual_release_date")
    by_record = {row["observation_date"]: row for row in records}
    by_report = {row["observation_date"]: row for row in reports}
    weeks = sorted(set(by_record) | set(by_report))
    issues, matched = [], []
    if not weeks:
        issues.append({"kind": "empty_scope"})
    else:
        cursor, last = date.fromisoformat(weeks[0]), date.fromisoformat(weeks[-1])
        while cursor <= last:
            if cursor.isoformat() not in weeks:
                issues.append(
                    {"kind": "missing_reference_week", "observation_date": cursor.isoformat()}
                )
            cursor += timedelta(days=7)
    for week in weeks:
        source, release = by_record.get(week), by_report.get(week)
        if source is None:
            issues.append({"kind": "extra_dol_report", "observation_date": week})
            continue
        if source["status"] == "missing":
            issues.append({"kind": "alfred_value_unavailable", "observation_date": week})
        if release is None:
            issues.append({"kind": "missing_dol_report", "observation_date": week})
            continue
        dates_match = source["alfred_realtime_start_date"] == release["actual_release_date"]
        values_match = source["status"] == "observed" and source["value"] == release["value"]
        if not dates_match:
            issues.append(
                {
                    "kind": "release_date_mismatch",
                    "observation_date": week,
                    "alfred_realtime_start_date": source["alfred_realtime_start_date"],
                    "actual_release_date": release["actual_release_date"],
                }
            )
        if source["status"] == "observed" and not values_match:
            issues.append(
                {
                    "kind": "value_mismatch",
                    "observation_date": week,
                    "alfred_value": source["value"],
                    "dol_value": release["value"],
                }
            )
        if dates_match and values_match:
            matched.append(
                {
                    "observation_date": week,
                    "alfred_realtime_start_date": source["alfred_realtime_start_date"],
                    "actual_release_date": release["actual_release_date"],
                    "value": source["value"],
                    "source_line": source["source_line"],
                    "source_url": release["source_url"],
                    "release_text_sha256": release["release_text_sha256"],
                }
            )
    issues.sort(key=lambda row: (row.get("observation_date", ""), row["kind"]))
    return {
        "status": "RECONCILIATION_MISMATCH" if issues else "RECONCILED",
        "records_count": len(records),
        "reports_count": len(reports),
        "matched_count": len(matched),
        "matched": matched,
        "issues": issues,
    }
