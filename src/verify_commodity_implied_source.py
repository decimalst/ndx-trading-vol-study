"""Independent, standard-library verification of bounded Cboe history parsing.

No producer imports, data access, numerical admission, or vintage certification.
"""

import csv
import hashlib
import io
import math
import re
from datetime import date

_ISO = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_US = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SHA = re.compile(r"[0-9a-f]{64}")
_LOWER = date(2009, 1, 2)
_UPPER = date(2025, 10, 20)


def _iso_date(token):
    if type(token) is not str or _ISO.fullmatch(token) is None:
        raise ValueError("Expected a literal ISO date")
    try:
        return date.fromisoformat(token)
    except ValueError as exc:
        raise ValueError("Invalid ISO calendar date") from exc


def _source_date(token):
    if _ISO.fullmatch(token) is not None:
        return _iso_date(token)
    match = _US.fullmatch(token)
    if match is None:
        raise ValueError("Unsupported source date spelling")
    month, day, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError("Invalid source calendar date") from exc


def _parse_value(token):
    """Called only after every file row passes date/structure preflight."""
    if token in ("", "."):
        return None
    if _NUMBER.fullmatch(token) is None:
        raise ValueError("Retained value is not an ASCII decimal")
    value = float(token)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Retained value must be finite and strictly positive")
    return value


def _exact(actual, expected, path="parsed"):
    """Compare schema and scalar types explicitly, including bool/int aliases."""
    if type(actual) is not type(expected):
        raise ValueError(f"Incorrect type at {path}")
    if type(expected) is dict:
        if actual.keys() != expected.keys():
            raise ValueError(f"Incorrect keys at {path}")
        for key, value in expected.items():
            _exact(actual[key], value, f"{path}.{key}")
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(f"Incorrect length at {path}")
        for position, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
            _exact(observed, wanted, f"{path}[{position}]")
    elif actual != expected:
        raise ValueError(f"Incorrect value at {path}")


def verify_history(
    payload,
    symbol,
    expected_sha256,
    parsed,
    start="2009-01-02",
    end="2025-10-20",
):
    """Independently reconstruct a complete parsed source result or raise."""
    if type(payload) is not bytes:
        raise ValueError("Source payload must be bytes")
    if type(symbol) is not str or symbol not in ("OVX", "GVZ"):
        raise ValueError("Unsupported source symbol")
    if type(expected_sha256) is not str or _SHA.fullmatch(expected_sha256) is None:
        raise ValueError("Expected source SHA-256 is malformed")
    signature = hashlib.sha256(payload).hexdigest()
    if signature != expected_sha256:
        raise ValueError("Source payload SHA-256 mismatch")
    first, last = _iso_date(start), _iso_date(end)
    if not _LOWER <= first <= last <= _UPPER:
        raise ValueError("Requested window exceeds fixed source scope or is reversed")
    try:
        content = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Source payload is not UTF-8") from exc

    # Retain lexical tokens only here. Structural problems anywhere in the file,
    # including excluded future rows, are found before any value is converted.
    staged = []
    before = after = total = 0
    previous = None
    try:
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        if next(reader, None) != ["DATE", symbol]:
            raise ValueError("Source header identity mismatch")
        for row in reader:
            if row == []:
                continue
            if len(row) != 2:
                raise ValueError("Every source row must have exactly two fields")
            current = _source_date(row[0])
            if previous is not None and current <= previous:
                raise ValueError("Source dates must be unique and strictly ascending")
            previous = current
            total += 1
            if current < first:
                before += 1
            elif current > last:
                after += 1
            else:
                staged.append((current.isoformat(), row[1], reader.line_num))
    except csv.Error as exc:
        raise ValueError("Malformed source CSV") from exc

    records = []
    observed = missing = 0
    for day, token, source_line in staged:
        value = _parse_value(token)
        if value is None:
            missing += 1
            status = "missing"
        else:
            observed += 1
            status = "observed"
        records.append(
            {"date": day, "value": value, "status": status, "source_line": source_line}
        )
    expected = {
        "status": "SOURCE_HISTORY_PARSED_NOT_INDEPENDENTLY_VERIFIED",
        "symbol": symbol,
        "source_sha256": signature,
        "start": start,
        "end": end,
        "records": records,
        "metadata": {
            "total_rows": total,
            "before_start_rows": before,
            "after_end_rows": after,
            "retained_rows": len(records),
            "observed_rows": observed,
            "missing_rows": missing,
            "first_retained_date": records[0]["date"] if records else None,
            "last_retained_date": records[-1]["date"] if records else None,
        },
    }
    _exact(parsed, expected)
    return {
        "status": "VERIFIED",
        "symbol": symbol,
        "retained_rows": len(records),
        "observed_rows": observed,
        "missing_rows": missing,
    }
