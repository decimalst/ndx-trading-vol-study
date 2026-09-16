"""Conservative metadata-only probe of extracted Treasury PDF text."""

import re
from datetime import date

MONTHS = {
    name: index
    for index, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def probe_pdf_header(text: str) -> dict:
    if type(text) is not str or len(text) > 2 * 1024 * 1024:
        raise ValueError("bounded extracted PDF text required")
    header = "\n".join(text.splitlines()[:12])
    dates = []
    pattern = r"^\s*(" + "|".join(MONTHS) + r")\s+([0-9]{1,2}),\s+([0-9]{4})(?![0-9])"
    for match in re.finditer(pattern, header, re.M):
        m, d, y = match.groups()
        value = date(int(y), MONTHS[m], int(d)).isoformat()
        if value not in dates:
            dates.append(value)
    cusips = sorted(set(re.findall(r"\bCUSIP(?:\s+Number)?\s*[:#]?\s+([A-Z0-9]{9})\b", text)))
    return {
        "has_release_label": bool(re.search(r"\bFor\s+Immediate\s+Release\b", header, re.I)),
        "release_dates": dates,
        "cusips": cusips,
    }


def check_pdf_identity(metadata: dict, *, expected_date=None, expected_cusip=None) -> str:
    if type(metadata) is not dict or set(metadata) != {
        "has_release_label",
        "release_dates",
        "cusips",
    }:
        raise ValueError("PDF metadata schema")
    if (
        metadata["has_release_label"] is not True
        or type(metadata["release_dates"]) is not list
        or len(metadata["release_dates"]) != 1
    ):
        raise ValueError("unique explicit release header date required")
    value = metadata["release_dates"][0]
    if (
        type(value) is not str
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None
        or not "2010-01-01" <= date.fromisoformat(value).isoformat() <= "2025-10-20"
    ):
        raise ValueError("PDF release date outside source scope")
    if expected_date is not None and value != expected_date:
        raise ValueError("PDF release date differs from selected event date")
    if expected_cusip is not None and metadata["cusips"] != [expected_cusip]:
        raise ValueError("PDF CUSIP does not uniquely match selected event")
    return value
