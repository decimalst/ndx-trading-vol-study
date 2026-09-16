"""Metadata-only support for dated offering headers and separately labeled CUSIPs."""

import re
from datetime import date

from src.treasury_pdf_identity import MONTHS, check_pdf_identity

__all__ = ["check_pdf_identity", "probe_pdf_header"]


def probe_pdf_header(text: str) -> dict:
    if type(text) is not str or len(text) > 2 * 1024 * 1024:
        raise ValueError("bounded extracted PDF text required")
    lines = text.splitlines()[:12]
    marker = re.compile(r"\b(?:For\s+I\.?mmediate\s+Release|Embargoed\s+Until)\b", re.I)
    datepattern = re.compile(
        r"^\s*(" + "|".join(MONTHS) + r")\s+([0-9]{1,2}),\s+([0-9]{4})(?![0-9])"
    )
    found = [i for i, line in enumerate(lines) if marker.search(line)]
    dates = []
    if len(found) == 1:
        # The date belongs immediately beneath the release/embargo marker.
        # Stop at a title or other non-date content; do not search body dates.
        following = lines[found[0] + 1 : found[0] + 5]
        for line in following:
            if not line.strip():
                continue
            match = datepattern.match(line)
            if match is None:
                break
            m, d, y = match.groups()
            value = date(int(y), MONTHS[m], int(d)).isoformat()
            if value not in dates:
                dates.append(value)
    cusips = sorted(
        set(re.findall(r"^\s*CUSIP(?:\s+Number)?\s*[:#]?\s+([A-Z0-9]{9})\b", text, re.M))
    )
    return {"has_release_label": len(found) == 1, "release_dates": dates, "cusips": cusips}
