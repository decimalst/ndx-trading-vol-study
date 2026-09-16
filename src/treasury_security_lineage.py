"""Reconcile stated security lineage without reading financial amounts."""

import re
import xml.etree.ElementTree as ET
from datetime import date

from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header
from src.treasury_pdf_identity import MONTHS
from src.treasury_xml_identity import check_event_identity, read_xml_identity


def _iso(value):
    if type(value) is not str:
        raise ValueError("Missing source lineage date")
    value = value.removesuffix("T00:00:00")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("Invalid source lineage date")
    return date.fromisoformat(value)


def _pdf_field(text, name, *, optional=False):
    matches = re.findall(r"^" + re.escape(name) + r"[ \t]+([^\n\r]+)$", text, re.M)
    if optional and not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous PDF lineage field: {name}")
    return matches[0].strip()


def _pdf_date(text, name, *, optional=False):
    value = _pdf_field(text, name, optional=optional)
    if value is None:
        return None
    match = re.fullmatch(r"(" + "|".join(MONTHS) + r") ([0-9]{1,2}), ([0-9]{4})", value)
    if match is None:
        raise ValueError(f"Invalid PDF lineage date: {name}")
    month, day, year = match.groups()
    return date(int(year), MONTHS[month], int(day))


def reconcile_lineage(raw, pdf_text, *, auction_date, announcement_date, cusip):
    """Use exact stated dated-to-maturity years; never round remaining life.

    This establishes paired-document lineage descriptors, not regularity or
    historical first-vintage admission. Missing/ambiguous dates fail closed.
    """
    metadata = read_xml_identity(raw)
    check_event_identity(
        metadata, auction_date=auction_date, announcement_date=announcement_date, cusip=cusip
    )
    check_pdf_identity(
        probe_pdf_header(pdf_text), expected_date=auction_date, expected_cusip=cusip
    )
    if len(re.findall(r"^TREASURY AUCTION RESULTS[ \t]*$", pdf_text, re.M)) != 1:
        raise ValueError("Lineage requires the competitive results document")
    a = metadata["announcement"]
    if a.get("SecurityType") not in {"NOTE", "BOND"} or a.get("InflationIndexSecurity") != "N":
        raise ValueError("Nominal note or bond identity required")
    if a.get("FloatingRate") != "N" and not (
        "FloatingRate" not in a and auction_date < "2014-01-29"
    ):
        raise ValueError("Fixed-rate identity required")
    reopening = a.get("ReOpeningIndicator")
    if reopening not in {"Y", "N"}:
        raise ValueError("Explicit reopening identity required")

    root = ET.fromstring(raw.decode("utf-8"))
    parents = {child: parent for parent in root.iter() for child in parent}
    fields = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "DatedDate":
            continue
        parent = parents.get(node)
        if (
            node.tag != "DatedDate"
            or parent is None
            or parent.tag != "AuctionAnnouncement"
            or len(node)
        ):
            raise ValueError("DatedDate must be a direct exact announcement leaf")
        fields.append(node.text)
    if len(fields) != 1:
        raise ValueError("Expected one DatedDate")
    dated, issue, maturity = (
        _iso(fields[0]),
        _iso(a.get("IssueDate")),
        _iso(a.get("MaturityDate")),
    )
    for label, expected in (
        ("Dated Date", dated),
        ("Issue Date", issue),
        ("Maturity Date", maturity),
    ):
        if _pdf_date(pdf_text, label) != expected:
            raise ValueError(f"PDF/XML lineage date disagreement: {label}")
    if not dated <= issue < maturity:
        raise ValueError("Inconsistent dated, issue and maturity ordering")
    if date.fromisoformat(auction_date) > issue:
        raise ValueError("Issue date precedes the selected auction")
    original_dated = _iso(a["OriginalDatedDate"]) if a.get("OriginalDatedDate") else None
    original_issue = _iso(a["OriginalIssueDate"]) if a.get("OriginalIssueDate") else None
    pdf_original_issue = _pdf_date(pdf_text, "Original Issue Date", optional=reopening == "N")
    if reopening == "Y":
        if original_dated != dated or original_issue is None:
            raise ValueError("Reopening lacks consistent original date lineage")
        if not dated <= original_issue < issue or pdf_original_issue != original_issue:
            raise ValueError("Reopening original issue date does not reconcile")
    elif (
        original_dated not in (None, dated)
        or original_issue not in (None, issue)
        or pdf_original_issue not in (None, issue)
    ):
        raise ValueError("New issue conflicts with original date lineage")

    tenor = maturity.year - dated.year
    if (maturity.month, maturity.day) != (dated.month, dated.day) or tenor not in {
        2,
        3,
        5,
        7,
        10,
        30,
    }:
        raise ValueError("No supported exact original tenor from stated dates")
    term = _pdf_field(pdf_text, "Term and Type of Security")
    match = re.fullmatch(r"([0-9]+)-Year(?: ([0-9]+)-Month)? (Note|Bond)", term)
    if match is None:
        raise ValueError("Unrecognized nominal PDF security term")
    years, months, kind = match.groups()
    if (
        a.get("SecurityTermWeekYear") != f"{years}-YEAR"
        or a.get("SecurityTermDayMonth") != f"{months or '0'}-MONTH"
        or a.get("SecurityType") != kind.upper()
    ):
        raise ValueError("PDF/XML remaining term or security type disagreement")
    if reopening == "N" and (int(years) != tenor or int(months or "0") != 0):
        raise ValueError("New issue term disagrees with dated-to-maturity years")
    return {
        "original_tenor_years": tenor,
        "remaining_term": term,
        "reopening": reopening == "Y",
        "dated_date": dated.isoformat(),
        "issue_date": issue.isoformat(),
        "maturity_date": maturity.isoformat(),
        "original_issue_date": original_issue.isoformat() if original_issue else None,
        "basis": "exact stated dated-to-maturity calendar years with PDF/XML lineage agreement",
    }
