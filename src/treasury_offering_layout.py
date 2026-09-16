"""Narrow offering-label adapter preserving frozen XML and amount arithmetic.

The supplemental ``CUSIP Number(s)`` field is not the offered security's
``CUSIP Number``. Original text is inspected intact; no identity is inferred
from a corpus or supplemental identifier and no source is admitted here.
"""

import re
from datetime import date

from src import treasury_offering_amounts as frozen
from src.treasury_pdf_header_layout import check_pdf_identity, probe_pdf_header

parse_xml_offering = frozen.parse_xml_offering


def _pdf_token(text, identity):
    frozen._expected_identity(identity, xml=False)
    if type(text) is not str or len(text) > frozen._MAX_BYTES:
        raise ValueError("Bounded extracted announcement PDF text required")
    try:
        if len(text.encode("utf-8")) > frozen._MAX_BYTES or "\x00" in text:
            raise ValueError("Invalid or oversized announcement text")
    except UnicodeEncodeError as error:
        raise ValueError("Invalid Unicode announcement text") from error
    check_pdf_identity(
        probe_pdf_header(text),
        expected_date=identity["announcement_date"],
        expected_cusip=identity["cusip"],
    )
    lines = [line.strip() for line in text.splitlines()]
    if lines.count("TREASURY OFFERING ANNOUNCEMENT") != 1:
        raise ValueError("Unique offering-announcement title required")
    # Exempt only the observed exact plural label from the frozen ambiguity guard.
    # The own label is still mandatory, unique, nonempty and exactly matched.
    cusips = [
        line
        for line in lines
        if re.match(r"CUSIP(?:[ \t]+Number)?\b", line)
        and re.fullmatch(r"CUSIP Number\(s\)(?:[ \t]+.*)?", line) is None
    ]
    if len(cusips) != 1 or frozen._row(cusips, "CUSIP Number") != identity["cusip"]:
        raise ValueError("Unique labelled announcement CUSIP required")
    auction_text = frozen._row(lines, "Auction Date")
    matched = frozen._WRITTEN_DATE.fullmatch(auction_text)
    if matched is None:
        raise ValueError("Exact labelled PDF auction date required")
    month, day, year = matched.groups()
    if date(int(year), frozen.MONTHS[month], int(day)) != frozen._iso(
        identity["auction_date"]
    ):
        raise ValueError("PDF auction date differs from its expected identity")
    return frozen._row(lines, "Offering Amount")


def parse_announcement_offering(text, *, identity):
    """Parse exact labelled dollars after original-text, own-security checks."""
    return frozen._answer(_pdf_token(text, identity), "USD")


def reconcile_offering(pdf_text, announcement_xml, result_xml, *, identities, xml_unit):
    """Apply the label amendment while preserving frozen checks and return schema."""
    if type(identities) is not dict or set(identities) != {
        "announcement_pdf",
        "announcement_xml",
        "result_xml",
    }:
        raise ValueError("Three exact document-specific identity contracts required")
    if type(xml_unit) is not str or xml_unit != frozen._XML_UNIT:
        raise ValueError("Explicit fixed USD_BILLIONS mapping contract required")
    # Finish all document identity/container checks before any amount conversion.
    tokens = {
        "announcement_pdf": _pdf_token(pdf_text, identities["announcement_pdf"]),
        "announcement_xml": frozen._xml_token(
            announcement_xml, "announcement", identities["announcement_xml"], xml_unit
        ),
        "result_xml": frozen._xml_token(
            result_xml, "result", identities["result_xml"], xml_unit
        ),
    }
    sources = {
        name: frozen._answer(token, "USD" if name == "announcement_pdf" else xml_unit)
        for name, token in tokens.items()
    }
    values = {parsed["offering_amount_usd"] for parsed in sources.values()}
    if len(values) != 1:
        raise ValueError("Stated offering amounts disagree across PDF and both XML documents")
    return {
        "status": "VERIFIED_STATED_OFFERING_AGREEMENT",
        "offering_amount_usd": sources["announcement_pdf"]["offering_amount_usd"],
        "unit": "USD",
        "sources": sources,
        "xml_scale_basis": (
            "Fixed 1e9 mapping hypothesis checked against the explicitly USD-labelled PDF "
            "on this record; not a schema-authenticated universal XML unit."
        ),
        "source_admitted": False,
    }
