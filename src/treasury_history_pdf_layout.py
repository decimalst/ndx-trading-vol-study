"""Bounded Treasury PDF metadata extraction from rows or reviewed column order."""

import re
from datetime import date

from src.treasury_pdf_header_layout import probe_pdf_header
from src.treasury_pdf_identity import MONTHS, check_pdf_identity

FIELDS = (
    "Term and Type of Security",
    "Auction Date",
    "Dated Date",
    "Issue Date",
    "Maturity Date",
    "Original Issue Date",
    "Series",
)
TITLES = {
    "TREASURY OFFERING ANNOUNCEMENT": (False, False),
    "AMENDED ANNOUNCEMENT": (False, True),
    "TREASURY AUCTION RESULTS": (True, False),
}
COLUMN_LABELS = (
    "Term and Type of Security",
    "Offering Amount",
    "Currently Outstanding",
    "CUSIP Number",
    "Auction Date",
    "Original Issue Date",
    "Issue Date",
    "Maturity Date",
    "Dated Date",
    "Series",
    "Yield",
    "Interest Rate",
)


def _calendar(value):
    match = re.fullmatch(r"(" + "|".join(MONTHS) + r") ([0-9]{1,2}), ([0-9]{4})", value)
    if match is None:
        raise ValueError("Exact PDF calendar date required")
    month, day, year = match.groups()
    return date(int(year), MONTHS[month], int(day))


def _label(line):
    # Supplemental STRIPS/interest identifiers are never the security's own CUSIP.
    if re.fullmatch(r"CUSIP Number\(s\)(?:[ \t]+.*)?", line) is not None:
        return None
    for label in ("CUSIP Number", "CUSIP", *FIELDS):
        match = re.fullmatch(re.escape(label) + r"(?:[ \t]+(.*))?", line)
        if match is not None:
            return ("CUSIP Number" if label == "CUSIP" else label, match.group(1))
    if re.match(r"CUSIP\b", line) is not None:
        raise ValueError("Unsupported own CUSIP label cannot be ignored")
    return None


def _column_fields(lines, title_index, first_label):
    """Use the exact observed fourteen-cell block; numeric cells stay opaque."""
    start = title_index + 1
    if start < len(lines) and lines[start] == "1":
        start += 1
    values = lines[start:first_label]
    labels = lines[first_label : first_label + 14]
    if (
        len(values) != 14
        or any(not value for value in values)
        or len(labels) != 14
        or tuple(labels[:12]) != COLUMN_LABELS
        or labels[12]
        not in {
            "Interest Payment Dates",
            "Interest Payment Dates4",
            "Interest Payment Dates 4",
        }
        or re.fullmatch(
            r"Accrued Interest from [0-9]{2}/[0-9]{2}/[0-9]{4} to [0-9]{2}/[0-9]{2}/[0-9]{4}",
            labels[13],
        )
        is None
    ):
        raise ValueError("Unsupported or incomplete PDF column structure")
    wanted = set(FIELDS) | {"CUSIP Number"}
    extracted = {
        label: values[index] for index, label in enumerate(COLUMN_LABELS) if label in wanted
    }
    remaining = lines[:start] + lines[first_label + 14 :]
    if any(_label(line) is not None for line in remaining):
        raise ValueError("Duplicate metadata field outside PDF column block")
    return extracted


def _row_fields(lines):
    found = {}
    for line in lines:
        pair = _label(line)
        if pair is None:
            continue
        label, value = pair
        if label in found:
            raise ValueError("Duplicate own CUSIP or PDF metadata field")
        if value is None or not value.strip():
            raise ValueError("Empty own CUSIP or PDF metadata field")
        found[label] = value.strip()
    return found


def inspect_pdf(text: str, *, result: bool, expected_cusip: str) -> dict:
    """Extract metadata without altering source text or authorizing an amendment."""
    if type(text) is not str or len(text) > 2 * 1024 * 1024:
        raise ValueError("Bounded extracted PDF text required")
    if type(result) is not bool:
        raise ValueError("Explicit boolean PDF role required")
    if type(expected_cusip) is not str or re.fullmatch(r"[A-Z0-9]{9}", expected_cusip) is None:
        raise ValueError("Exact uppercase expected CUSIP required")
    lines = [line.strip() for line in text.splitlines()]
    headings = [(index, line) for index, line in enumerate(lines) if line in TITLES]
    if len(headings) != 1:
        raise ValueError("Unique supported PDF document title required")
    title_index, title = headings[0]
    role, amended = TITLES[title]
    if role != result:
        raise ValueError("PDF document title differs from requested role")
    bare_terms = [index for index, line in enumerate(lines) if line == FIELDS[0]]
    if bare_terms:
        if result or len(bare_terms) != 1:
            raise ValueError("Unsupported PDF column layout or duplicate term field")
        extracted = _column_fields(lines, title_index, bare_terms[0])
    else:
        extracted = _row_fields(lines)
    own = extracted.pop("CUSIP Number", None)
    if type(own) is not str or re.fullmatch(r"[A-Z0-9]{9}", own) is None:
        raise ValueError("Unique actual own CUSIP field required")
    if own != expected_cusip:
        raise ValueError("Extracted own CUSIP differs from expected identity")
    required = set(FIELDS) - {"Original Issue Date", "Series"}
    if result:
        required.remove("Auction Date")
    if not required <= set(extracted):
        raise ValueError("Missing required PDF metadata field")
    for label, value in extracted.items():
        if label.endswith("Date"):
            day = _calendar(value)
            if label == "Auction Date" and not date(2010, 1, 1) <= day <= date(2025, 10, 20):
                raise ValueError("PDF auction date outside source scope")
    # Own identity comes solely from the field extraction above, never the supplied
    # expected value. The frozen probe supplies only its established header facts.
    header = probe_pdf_header(text)
    header = {**header, "cusips": [own]}
    release = check_pdf_identity(header, expected_cusip=own)
    return {
        "header": header,
        "release_date": release,
        "cusip": own,
        "amended": amended,
        "fields": {label: extracted[label] for label in FIELDS if label in extracted},
    }
