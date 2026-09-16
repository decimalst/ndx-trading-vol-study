"""Authenticate and bound original Cboe OVX/GVZ history bytes before admission.

Parsing is not independent verification, historical-vintage authentication, or
permission to run a forecast. Outside-window cells never enter numeric parsing.
"""

import csv
import hashlib
import io
import math
import re
from datetime import date

SOURCE_START = "2009-01-02"
SOURCE_END = "2025-10-20"
_ISO = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_US = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})\Z")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _iso_date(token):
    if not isinstance(token, str) or _ISO.fullmatch(token) is None:
        raise ValueError("Bounds must be literal ISO calendar dates")
    return date.fromisoformat(token)


def _source_date(token):
    if _ISO.fullmatch(token):
        return date.fromisoformat(token)
    match = _US.fullmatch(token)
    if match is None:
        raise ValueError("Source date must be ISO or US month/day/four-digit-year")
    month, day, year = map(int, match.groups())
    return date(year, month, day)


def _parse_value(token):
    if token in ("", "."):
        return None
    if _DECIMAL.fullmatch(token) is None:
        raise ValueError("In-scope index value is not an ordinary decimal")
    value = float(token)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Observed index values must be finite and positive")
    return value


def parse_cboe_history(
    payload: bytes,
    symbol: str,
    expected_sha256: str,
    start: str = SOURCE_START,
    end: str = SOURCE_END,
) -> dict:
    """Return bounded records after a complete date-only preflight.

    The caller must retain an independent byte receipt and run the separate
    source verifier before admitting any returned data to a new experiment.
    """
    if not isinstance(payload, bytes):
        raise ValueError("Original source payload must be bytes")
    if symbol not in ("OVX", "GVZ"):
        raise ValueError("Source symbol must be OVX or GVZ")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("Original payload SHA-256 authentication failed")
    lower, upper = _iso_date(start), _iso_date(end)
    if lower < _iso_date(SOURCE_START) or upper > _iso_date(SOURCE_END) or lower > upper:
        raise ValueError("Requested bounds lie outside the fixed source window")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Source is not strict UTF-8") from exc
    reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
    rows = []
    previous = None
    try:
        if next(reader, None) != ["DATE", symbol]:
            raise ValueError("Source header must be exactly DATE and selected symbol")
        for cells in reader:
            if not cells:
                continue
            if len(cells) != 2:
                raise ValueError("Every logical source row must have exactly two fields")
            observed_date = _source_date(cells[0])
            if previous is not None and observed_date <= previous:
                raise ValueError("Complete source dates must be ascending and unique")
            previous = observed_date
            rows.append((observed_date, cells[1], reader.line_num))
    except csv.Error as exc:
        raise ValueError("Malformed source CSV") from exc

    # No value conversion is permitted above this boundary. Rows outside the
    # caller's narrower window are counted by date alone and never returned.
    records = []
    before = after = observed = missing = 0
    for observed_date, token, source_line in rows:
        if observed_date < lower:
            before += 1
            continue
        if observed_date > upper:
            after += 1
            continue
        value = _parse_value(token)
        if value is None:
            missing += 1
        else:
            observed += 1
        records.append(
            {
                "date": observed_date.isoformat(),
                "value": value,
                "status": "missing" if value is None else "observed",
                "source_line": source_line,
            }
        )
    return {
        "status": "SOURCE_HISTORY_PARSED_NOT_INDEPENDENTLY_VERIFIED",
        "symbol": symbol,
        "source_sha256": expected_sha256,
        "start": start,
        "end": end,
        "records": records,
        "metadata": {
            "total_rows": len(rows),
            "before_start_rows": before,
            "after_end_rows": after,
            "retained_rows": len(records),
            "observed_rows": observed,
            "missing_rows": missing,
            "first_retained_date": records[0]["date"] if records else None,
            "last_retained_date": records[-1]["date"] if records else None,
        },
    }
