"""Build the fixed original-DOL source disposition without forecasting or I/O."""

from __future__ import annotations

from datetime import date, timedelta

from src.claims_release_reconcile import _iso, _record, _report

START = date(2009, 5, 30)
END = date(2025, 10, 18)
STATUS = "FIRST_REPORT_LEDGER_BUILT_NOT_INDEPENDENTLY_VERIFIED"
TAIL = frozenset(("2025-09-27", "2025-10-04", "2025-10-11", "2025-10-18"))
EXCEPTIONS = {
    "2017-03-18": {
        "dol_value": 261000,
        "dol_release": "2017-03-23",
        "alfred_value": 258000,
        "alfred_date": "2017-03-23",
        "status": "unresolved_correction",
        "kind": "unresolved_correction",
    },
    "2018-03-17": {
        "dol_value": 229000,
        "dol_release": "2018-03-22",
        "alfred_value": 227000,
        "alfred_date": "2018-03-29",
        "status": "observed",
        "kind": "documented_original_dol_disagreement",
    },
}


def _index(rows, validator, start, end, *, reports):
    if type(rows) is not list:
        raise ValueError("Complete report and ALFRED record lists required")
    result, release_dates = {}, set()
    for original in rows:
        row = validator(original)
        week = row["observation_date"]
        if not start <= _iso(week) <= end:
            raise ValueError(f"Source reference week outside requested scope: {week}")
        if week in result:
            raise ValueError(f"Duplicate source reference week: {week}")
        if reports:
            released = row["actual_release_date"]
            if (
                row["body_release_dates"] != [released]
                or row["body_release_date_verified"] is not True
            ):
                raise ValueError(
                    "Every report requires an explicit matching body release date"
                )
            if released in release_dates:
                raise ValueError(f"Duplicate DOL release date: {released}")
            release_dates.add(released)
        result[week] = row
    return result


def _comparison(kind, report=None, record=None):
    if report is None:
        return {
            "kind": kind,
            "dol_value": None,
            "alfred_value": None,
            "alfred_realtime_start_date": None,
            "alfred_source_line": None,
            "value_agrees": None,
            "release_date_agrees": None,
        }
    return {
        "kind": kind,
        "dol_value": report["value"],
        "alfred_value": record["value"],
        "alfred_realtime_start_date": record["alfred_realtime_start_date"],
        "alfred_source_line": record["source_line"],
        "value_agrees": report["value"] == record["value"],
        "release_date_agrees": report["actual_release_date"]
        == record["alfred_realtime_start_date"],
    }


def build_first_report_ledger(reports, alfred_records, start="2009-05-30", end="2025-10-18"):
    """Return every requested Saturday, rejecting any undeclared source difference.

    Inputs are already parsed source records. This builder checks their declared
    provenance and fixed disposition; callers separately authenticate source
    bytes and independently verify this output. No historical source is loaded.
    """
    first, last = _iso(start), _iso(end)
    if not START <= first <= last <= END or first.weekday() != 5 or last.weekday() != 5:
        raise ValueError("Ordered Saturday bounds within the fixed source scope required")
    by_report = _index(reports, _report, first, last, reports=True)
    by_record = _index(alfred_records, _record, first, last, reports=False)
    rows = []
    counts = {"observed": 0, "unresolved_correction": 0, "no_admitted_release": 0}
    cursor = first
    while cursor <= last:
        week = cursor.isoformat()
        report, record = by_report.get(week), by_record.get(week)
        if week in TAIL:
            if report is not None or record is not None:
                raise ValueError(
                    f"Declared no-admitted-release week has a source record: {week}"
                )
            row = {
                "reference_week": week,
                "release_date": None,
                "first_report_value": None,
                "status": "no_admitted_release",
                "source_url": None,
                "release_text_sha256": None,
                "source_comparison": _comparison("no_admitted_release"),
            }
        else:
            if report is None or record is None:
                raise ValueError(f"Undeclared source gap: {week}")
            if record["status"] != "observed":
                raise ValueError(f"Unavailable ALFRED comparison: {week}")
            exception = EXCEPTIONS.get(week)
            if exception is not None:
                if (
                    report["value"] != exception["dol_value"]
                    or report["actual_release_date"] != exception["dol_release"]
                    or record["value"] != exception["alfred_value"]
                    or record["alfred_realtime_start_date"] != exception["alfred_date"]
                ):
                    raise ValueError(
                        f"Source exception does not match documented evidence: {week}"
                    )
                status, kind = exception["status"], exception["kind"]
            else:
                if (
                    report["value"] != record["value"]
                    or report["actual_release_date"] != record["alfred_realtime_start_date"]
                ):
                    raise ValueError(
                        f"Undeclared source value or release-date disagreement: {week}"
                    )
                status, kind = "observed", "exact_agreement"
            row = {
                "reference_week": week,
                "release_date": report["actual_release_date"],
                "first_report_value": report["value"] if status == "observed" else None,
                "status": status,
                "source_url": report["source_url"],
                "release_text_sha256": report["release_text_sha256"],
                "source_comparison": _comparison(kind, report, record),
            }
        rows.append(row)
        counts[row["status"]] += 1
        cursor += timedelta(days=7)
    return {"status": STATUS, "start": start, "end": end, "rows": rows, "counts": counts}
